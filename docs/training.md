# Training Loop

Training step behavior and metrics written to `metrics.jsonl`.

Related: [Config Reference](config-reference.md),
[Optimization and Optimizers](optimization.md), [Data Pipeline](data_pipeline.md),
[Packing and Boundary Semantics](packing.md), [Checkpointing and Resume](checkpointing.md).

## Train step contract

The compiled `train_step`:

- consumes fixed-shape `Batch` objects (`[A,B,T]`)
- performs gradient accumulation inside `jax.lax.scan`
- applies exactly one optimizer update per outer step

Grad accumulation is **token-weighted**: microbatch losses are scaled by the
count of valid (non-masked) tokens to keep updates correct with padding or
boundary masks.

## Optimizer selection

The train loop treats `adamw` and `muon` as one optimizer step per outer
iteration. Muon parameter partitioning, `optim.muon.*` behavior, and
sweep-backed defaults live in [Optimization and Optimizers](optimization.md).
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
Enable activation checkpointing with `model.use_checkpoint`; it is orthogonal
to gradient accumulation.

## GPU environment notes

Two environment flags are helpful on newer GPUs:

- `XLA_PYTHON_CLIENT_PREALLOCATE=false` to avoid pre-allocating all GPU memory.
- `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false` if Triton GEMM causes
  `CUDA_ERROR_OUT_OF_MEMORY` on RTX 5090 with `jax/jaxlib 0.8.2`.

When `chomp train` detects an RTX 50xx (Blackwell) GPU via `nvidia-smi`, it
automatically appends `--xla_gpu_enable_triton_gemm=false` to `XLA_FLAGS`
(before JAX initializes) and warns if `XLA_PYTHON_CLIENT_PREALLOCATE` is not
set to `false`. On other GPUs, the helper stays quiet (debug log only).

XLA kernel selection on GPU is nondeterministic by default. This is the fast
path used for production training. For debugging runs that need bit-exact
GPU numerics (e.g. comparing an interrupted+resumed run against a continuous
one at atol=0), opt in with `XLA_FLAGS=--xla_gpu_deterministic_ops=true`.
This is a different knob from `train.deterministic` above (dropout/RNG vs
kernel selection), and it is expensive: measured ~25-35% slower steps on a
100M-param Megalodon smoke benchmark (RTX 5090, seq_len 2048, bf16). The
effective setting is recorded in checkpoint meta, and resume warns if it
drifted across the boundary; see
[Checkpointing: scope of exactness](checkpointing.md#scope-of-exactness).

## Packed batches

Packing modes, segment isolation, position IDs, and loss masking are defined in
[Packing and Boundary Semantics](packing.md). The stream-to-batch path is
described in [Data Pipeline](data_pipeline.md).

## Input-order stability

Guidance for domain-ordered and long-document streams lives in
[Packing: window shuffling](packing.md#window-shuffling-batch-decorrelation).
The supporting ablation is recorded in the
[Comma stability study](comma_stability_matrix.md).

## Evaluation

If `train.eval_every > 0` and `data.max_eval_samples > 0`, chomp runs a full
pass over the pinned eval token set and logs `eval_loss`. Eval text selection,
cache identity checks, packed eval flushing, and zero-batch/zero-token failures are documented in
[Data Pipeline validation set](data_pipeline.md#validation-set).

Eval batches are assembled once and cached host-side for the whole run; device
transfer happens per batch each eval, so no device memory is held between evals.

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

### Standalone generation

`chomp generate` accepts a run directory or checkpoint step directory, restores
the stored parameters and resolved config, and uses the run-pinned tokenizer
described in [Data Pipeline: tokenization](data_pipeline.md#tokenization) when
available. Set `--temperature 0` for greedy decoding; seeded sampling is the
default. Run `chomp generate --help` for the option list.

## Dry run

Use `chomp train <config.yaml> --dry-run` to validate config, build the
tokenizer/model/data pipeline, and execute one step before exiting. The step
compiles when `train.jit` is enabled. W&B logging is skipped in dry-run mode.

`config_resolved.json` includes a small `derived` section; for example
`derived.optim.decay_steps_effective` records the effective LR schedule horizon.

## Metrics

Metrics are written to `logging.metrics_file` every `train.log_every` steps
(and on eval steps) and include:

- `loss`
- `grad_norm`
- `lr`
- `loss_tokens` (exact compiled `token_sum` for that step)
- `tokens_seen` (cumulative exact compiled `token_sum`)
- `step_time_s`, `data_wait_s`
- `tokens_per_sec` (model step) and `tokens_per_sec_e2e` (including data wait)
- `packing_mode`, `packing_utilization` (when iterator stats are enabled)
- `docs_seen`, `docs_truncated`, `docs_added_this_batch` (when available)
- `loss_tokens_host` (host recomputed valid-loss tokens from labels + masks)
- `boundary_transitions` (count of in-batch segment transitions)
- `docs_per_seq_mean`, `docs_per_seq_min`, `docs_per_seq_max` (document-density summary)
- `first_step_compile_time_s` (first logged step after compile)
- `peak_memory_gb` (best-effort, device-dependent)
- `eval_loss` (only when eval runs)
- `lr_muon` (Muon runs only)

Data exhaustion and crashes append event rows to the same file.

If `logging.wandb.enabled=true`, Weights & Biases receives the training rows plus
detailed wall-clock, packing-capacity, eval-token, and current-device-memory
metrics. The local file instead retains the process peak-memory value and its
explicit `step` field. chomp also uploads `config_original.yaml` as a W&B
artifact at run start, and W&B logs go to the default `./wandb` directory (or
`WANDB_DIR` if set). Set `logging.wandb.enabled=false` to disable W&B; `mode`
selects online or offline logging only.

Console output is throttled by `train.log_every` and prints a compact
one-line summary (loss, grad norm, LR, step time, throughput, optional eval
loss, packing utilization, and best-effort device memory). Full logs from
third-party libraries are written to `logging.log_file` under the run directory
when that field is not `null`.
