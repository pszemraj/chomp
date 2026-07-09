# Training Loop

This doc summarizes the training step behavior and the metrics logged in
`metrics.jsonl`.

## Scope

This page is the home for runtime training-loop behavior.

- For field-by-field config defaults and types: [Config Reference](config-reference.md)
- For optimizer internals and sweep guidance: [Optimization and Optimizers](optimization.md)
- For data stream, packing, and eval-set construction: [Data Pipeline](data_pipeline.md)
- For boundary masking semantics: [Packing and Boundary Semantics](packing.md)
- For save/restore/resume policy: [Checkpointing and Resume](checkpointing.md)

## Development notes

For linting, formatting, and the module-based test layout, see [Development Guide](dev.md).
In particular, training-loop and checkpoint/resume behaviors now live in
[`tests/test_training.py`](../tests/test_training.py).

## Train step contract

The compiled `train_step`:

- consumes fixed-shape `Batch` objects (`[A,B,T]`)
- performs gradient accumulation inside `jax.lax.scan`
- applies exactly one optimizer update per outer step

Grad accumulation is **token-weighted**: microbatch losses are scaled by the
count of valid (non-masked) tokens to keep updates correct with padding or
boundary masks.

## Optimizer selection

`optim.name` selects the optimizer:

- `adamw` (default)
- `muon`: applies Muon to selected matrix parameters and AdamW elsewhere.

The train loop treats both as one optimizer step per outer iteration; details
about Muon parameter partitioning, `optim.muon.*` behavior, and sweep-backed
defaults live in [Optimization and Optimizers](optimization.md).
For exact knob definitions, see [Config Reference](config-reference.md) (`optim.*`).

## Determinism

`train.deterministic` controls dropout behavior:

- `None`: derived from dropout rates (deterministic if all zero)
- `True`: force deterministic
- `False`: force stochastic

Deterministic runs are recommended for resume and regression tests. Note that
in `megalodon-jax`, activation checkpointing is disabled when
`train.deterministic=true`. If you want checkpointing with deterministic math,
set `train.deterministic=false` and keep all dropout rates at `0.0`.

## GPU environment notes

Two environment flags are helpful on newer GPUs:

- `XLA_PYTHON_CLIENT_PREALLOCATE=false` to avoid pre-allocating all GPU memory.
- `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false` if Triton GEMM causes
  `CUDA_ERROR_OUT_OF_MEMORY` on RTX 5090 with `jax/jaxlib 0.8.2`.

When `chomp train` detects an RTX 50xx (Blackwell) GPU via `nvidia-smi`, it
automatically appends `--xla_gpu_enable_triton_gemm=false` to `XLA_FLAGS`
(before JAX initializes) and warns if `XLA_PYTHON_CLIENT_PREALLOCATE` is not
set to `false`. On other GPUs, the helper stays quiet (debug log only).

XLA kernel selection on GPU is nondeterministic by default — the fast path,
and what production training uses. For debugging runs that need bit-exact
GPU numerics (e.g. comparing an interrupted+resumed run against a continuous
one at atol=0), opt in with `XLA_FLAGS=--xla_gpu_deterministic_ops=true`.
This is a different knob from `train.deterministic` above (dropout/RNG vs
kernel selection), and it is expensive: measured ~25-35% slower steps on a
100M-param Megalodon smoke benchmark (RTX 5090, seq_len 2048, bf16). The
effective setting is recorded in checkpoint meta, and resume warns if it
drifted across the boundary — see
[Checkpointing — Scope of exactness](checkpointing.md#scope-of-exactness).

## Attention and loss masking

The training step consumes already-packed fixed-shape batches. Stream semantics,
segment IDs, and boundary-related masking behavior are defined in
[Packing and Boundary Semantics](packing.md), and their placement in the data
path is defined in [Data Pipeline](data_pipeline.md).

When `data.packing_mode` is `bin` or `multipack` and
`data.packing_strict_segments=true` (the default), the step also forwards
`segment_ids` and `position_ids` into the backend model for strict packed
semantics. With megalodon-jax >= 0.1.2 this means full state isolation per
packed document (attention, RoPE positions, ComplexEMA, and TimestepNorm all
reset at segment boundaries); chomp verifies the backend's
`supports_segment_reset` capability flag at startup and fails fast otherwise.
See [Packing — Segment-Isolation Semantics](packing.md#segment-isolation-semantics).

## Loss-stability recipe

This is default hygiene for **domain-ordered or long-document streaming
corpora** — any stream whose document order is not globally mixed — not a
dataset-specific fix. Validated on Comma (2026-07, 100m 5k-step ablation; see
[Packing — Window shuffling](packing.md#window-shuffling-batch-decorrelation)):

- `data.window_shuffle_windows: 4096` (the default) — decorrelates batches from
  raw packer order; cut step-to-step |Δloss| ~31% and worst grad-norm spikes
  ~5x, at zero throughput cost.
- `data.shuffle_buffer_size: 200000` — fights domain ordering of the source
  stream; this is what removes the train-loss-below-eval memorization
  signature on domain-ordered corpora (Common Pile family). Pre-mixed corpora
  (Zyda-2, SmolLM2 mixes) are healthy even at 10k, but the large buffer is
  cheap insurance.
- Watch `docs_added_this_batch` and per-interval `docs_seen` deltas in
  metrics.jsonl; sustained near-zero pulls mean the stream is draining
  buffered content from few documents.

Smoke configs intentionally keep a small `shuffle_buffer_size`: the HF shuffle
buffer must fill before the first batch, so 200k docs would dominate smoke-test
startup time.

## Evaluation

If `train.eval_every > 0`, chomp runs a full pass over the eval token set and
logs `eval_loss`. The set is collected once when the run is created, persisted
to `run_dir/eval_tokens.json.gz`, and reloaded on every later start — resumed
runs evaluate on exactly the same tokens even if the upstream dataset drifted.
Eval text selection policy (eval split vs train fallback) is documented in
[Data Pipeline](data_pipeline.md).

Eval batches are assembled once and cached host-side for the whole run; the
device transfer happens per batch each eval, so no device memory is held
between evals.

For `bin` and `multipack`, the packers flush their remaining pending documents
into padded windows once the eval doc set is exhausted, so small eval sets
still evaluate. Eval fails fast at runtime if the set cannot fill even one
complete `[A, B, T]` batch (too few packed windows for `grad_accum *
batch_size` rows) or if every emitted label is masked out (zero valid loss
tokens — broken boundary masking or pathological short docs). There is no
silent `eval_loss: null` outcome.

## Generation samples

If `train.generate_every > 0`, chomp periodically samples a prompt from a
separate stream of the training split and runs `megalodon_jax.generate`,
printing both the prompt and generated continuation to the console (Rich panels
when enabled).

Default behaviors (when the `generate_*` fields are `null`):

- `train.generate_input_len`: half of `train.seq_len`
- `train.generate_max_tokens`: `model.chunk_size + 16`
- prompt selection: if a sample is longer than `generate_input_len`, randomly
  use the first or last `generate_input_len` tokens; otherwise use the full
  sample (no EOS token appended)

Optional sampling controls (`train.generate_temperature`, `train.generate_top_k`,
`train.generate_top_p`) are passed through when set; otherwise the Megalodon
defaults apply. Generation is currently only enabled for the `megalodon`
backend (dummy runs skip it silently).

## Dry run

Use `chomp train <config.yaml> --dry-run` to validate config, build the tokenizer/model/data
pipeline, and compile one step before exiting. W&B logging is skipped in dry-run
mode to avoid creating noisy runs.

`config_resolved.json` includes a small `derived` section; for example
`derived.optim.decay_steps_effective` records the effective LR schedule horizon.

## Gradient checkpointing

Megalodon supports activation checkpointing via `model.use_checkpoint`. This is
orthogonal to gradient accumulation and does not change the batch contract.
In `megalodon-jax`, checkpointing is gated on `train.deterministic=false`.

## Metrics

Metrics are written to `logging.metrics_file` every `train.log_every` steps
(and on eval steps) and include:

- `loss`
- `grad_norm`
- `lr`
- `loss_tokens` (exact compiled `token_sum` for that step)
- `tokens_seen` (cumulative exact compiled `token_sum`)
- `tokens_per_sec` (actual valid tokens / step_time_s)
- `packing_mode`, `packing_utilization` (when iterator stats are enabled)
- `loss_tokens_host` (host recomputed valid-loss tokens from labels + masks)
- `boundary_transitions` (count of in-batch segment transitions)
- `docs_per_seq_mean`, `docs_per_seq_min`, `docs_per_seq_max` (document-density summary)
- `first_step_compile_time_s` (first logged step after compile)
- `peak_memory_gb` (best-effort, device-dependent)
- `eval_loss` (only when eval runs)

If `logging.wandb.enabled=true`, the same rows are also logged to Weights & Biases.
chomp also uploads `config_original.yaml` as a W&B artifact at run start, and W&B
logs go to the default `./wandb` directory (or `WANDB_DIR` if set).

Console output is throttled by `train.log_every` and prints a compact
one-line summary (loss, grad norm, LR, step time, throughput, optional eval
loss, packing utilization, and best-effort device memory). Full logs from
third-party libraries are written to `logging.log_file` under the run directory.

`tokens_seen` resumes from checkpoint metadata when available.
