# Training Loop

Training step behavior and metrics written to `metrics.jsonl`.

Related: [Config Reference](config-reference.yaml), [Optimization and Optimizers](optimization.md), [Data Pipeline](data_pipeline.md), [Packing and Boundary Semantics](packing.md), [Checkpointing and Resume](checkpointing.md).

## Maintained recipes

Checked-in scenarios are separated by intent. [`configs/dev/`](../configs/dev/) contains short infrastructure checks, while [`configs/pretrain/`](../configs/pretrain/) contains the four Megalodon scale recipes. Exact parameter counts, measurements, and fit qualifications are in the [README recipe table](../README.md#shipped-recipes-and-measured-expectations); the recipes are executable examples, and their per-field contracts live only in the [Config Reference](config-reference.yaml).

The configured maximum target budget is `steps * grad_accum * batch_size * (seq_len - 1)`; boundary, EOS, and padding masks reduce the realized `tokens_seen`. The maintained schedules provide at least roughly 20 maximum target positions per parameter, but that is a starting budget rather than a claim of compute optimality for a specific corpus or research objective.

## Train step contract

The compiled `train_step`:

- consumes fixed-shape `Batch` objects (`[A,B,T]`)
- performs gradient accumulation inside `jax.lax.scan`
- applies exactly one optimizer update per outer step

Grad accumulation is **token-weighted** without reconstructing numerators from rounded means. Each backend returns an FP32 loss sum and exact integer valid-target count. Chomp differentiates and accumulates those sums in FP32, adds the integer counts, then divides the logical-batch loss and gradients once by the total count.

Batch assembly computes the corresponding shifted valid-target count on the host and pairs it with the batch through prefetch/device transfer. The model-provided count is authoritative for loss and gradient normalization; the host count supports exact accounting and a consistency check. Every optimizer step queues its compiled int32 counter without forcing a device barrier. At logging, evaluation, checkpoint, first-compile, and final boundaries, the trainer synchronizes and requires every queued logical-batch count to equal its paired host count. At save and final boundaries, the finite check also covers the post-update parameters and optimizer state. Evaluation aggregates the same backend FP32 sums and integer counts on device and synchronizes once per pass.

Full packing diagnostics scan segment arrays only for batches whose global step will log or evaluate. Non-observable steps still compute the exact shifted loss-token count required for accounting, but skip the redundant Python-side diagnostic scan.

The canonical [`model.chunk_size` and `model.attention_window` contracts](config-reference.yaml) explain why chunk-local attention is the efficient training path and the current dense O(L²) sliding-window path is rejected only for training.

## Determinism

[`train.deterministic` and `model.use_checkpoint`](config-reference.yaml) define dropout/rematerialization resolution and resume treatment. This model-level control is separate from GPU kernel determinism; maintained recipes use zero dropout with explicit stochastic mode so rematerialization remains active.

## GPU environment notes

The supported JAX 0.10.x CUDA 13 stack runs RTX 50xx GPUs with its default kernel selection; Chomp does not rewrite `XLA_FLAGS`. In particular, do not carry over `--xla_gpu_enable_triton_gemm=false` from older RTX 50xx workaround advice: measured on jax 0.10.2 / RTX 5090 (200M Megalodon, seq_len 2048, bf16), disabling Triton GEMM is ~4.5% slower than the default, and `--xla_gpu_verify_triton_fusion_numerics=true` passes on the same workload. Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` when sharing a GPU or when allocator preallocation interferes with another process.

XLA kernel selection on GPU is nondeterministic by default. This is the fast path used for production training. For debugging runs that need bit-exact GPU numerics (e.g. comparing an interrupted+resumed run against a continuous one at atol=0), opt in with `XLA_FLAGS=--xla_gpu_deterministic_ops=true`. This is a different knob from `train.deterministic` above (dropout/RNG vs kernel selection), and it is expensive: measured ~25-35% slower steps on a 100M-param Megalodon smoke benchmark (RTX 5090, seq_len 2048, bf16).

## Evaluation

Evaluation activation, cadence, failure policy, and persistent status are defined by [`train.eval_*` and `data.max_eval_samples`](config-reference.yaml). Operationally, a fatal evaluation recorded at an aligned checkpoint remains owed before resumed training, while deliberate disablement remains visible in later metrics and checkpoints. Eval text selection and packed flushing are documented under [Data Pipeline validation set](data_pipeline.md#validation-set).

Eval batches are assembled once and cached host-side for the whole run; device transfer happens per batch each eval, so no device memory is held between evals.

## Generation samples

Periodic Megalodon generation samples prompts from a bounded pool of up to 16 unshuffled training-split documents, avoiding a second production-sized shuffle window. Long prompts randomly use their first or last configured span, and no EOS is appended. Cadence, computed lengths, sampling controls, and backend applicability are canonical under [`train.generate_*`](config-reference.yaml).

### Standalone generation

`chomp generate` accepts a run directory or checkpoint step directory, restores the stored parameters and resolved config, and uses the run-pinned tokenizer described in [Data Pipeline: tokenization](data_pipeline.md#tokenization) when available. For schema-2+ checkpoints, generation recomputes that tokenizer's effective manifest identity and requires it to match the selected checkpoint before encoding the prompt; metadata-free legacy checkpoints retain the previous source/snapshot fallback. Set `--temperature 0` for greedy decoding; seeded sampling is the default. Run `chomp generate --help` for the option list.

## Dry run

Use `chomp train <config.yaml> --dry-run` to validate config, build the tokenizer/model/data pipeline, and execute one step before exiting. The step compiles when `train.jit` is enabled. When `debug.nan_check` is enabled, success also requires finite loss, gradient norm, learning rate, post-update parameters, and optimizer state. W&B logging is skipped in dry-run mode.

`config_resolved.json` includes a small `derived` section. `derived.optim.decay_steps_effective` records the effective LR schedule horizon, and `derived.megalodon_jax` records the installed distribution version plus PEP 610 source identity when available.

Training also binds the effective tokenizer program to `tokenizer/identity.json` and stores its digest in each checkpoint. Resume validates that local snapshot before it constructs evaluation/training streams, so it does not contact the configured tokenizer source.

## Metrics

Metrics are written to `logging.metrics_file` on the first process-local training step, every `train.log_every` steps, and on eval steps. They include:

- `loss`
- `grad_norm`
- `lr`
- `loss_tokens` (exact host count; every queued logical-batch count is checked against compiled `token_sum` at sync points)
- `tokens_seen` (cumulative exact host count)
- `step_time_s`, `data_wait_s` (per-step averages over the last sync interval)
- `tokens_per_sec` (end-to-end throughput over the last sync interval)
- `packing_mode`, `packing_utilization` (when iterator stats are enabled)
- `docs_seen`, `docs_truncated` (when available)
- `source_tokens_observed`, `source_tokens_retained`, `source_tokens_discarded`, `source_truncation_fraction`
- `shuffle_window_docs`, `shuffle_window_bytes`, `shuffle_peak_window_docs`, `shuffle_peak_window_bytes`, `shuffle_replayed_window_docs`, and `shuffle_replayed_window_bytes`
- `loss_tokens_host` (same host count exposed with packing diagnostics)
- `boundary_transitions` (count of in-batch segment transitions)
- `segments_per_seq_mean`, `segments_per_seq_min`, `segments_per_seq_max` (packed-segment density summary)
- `first_step_compile_time_s` (first logged step after compile)
- `peak_memory_gb` (best-effort, device-dependent)
- `eval_loss` (only when eval runs)
- `lr_muon` (Muon runs only)
- `preemption_requested`, `preemption_signal` (one row on graceful SIGTERM/SIGUSR1 stop)

Data exhaustion and crashes append event rows to the same file.

When enabled, Weights & Biases receives the training rows plus detailed wall-clock, packing-capacity, eval-token, and current-device-memory metrics and uploads `config_original.yaml` for file-backed runs. Local metrics retain the process peak-memory value and explicit `step`; W&B enablement, mode, naming, and failure behavior are defined under [`logging.wandb.*`](config-reference.yaml).

Console output is a compact one-line summary of loss, gradient norm, LR, timing, throughput, optional evaluation, packing utilization, and best-effort device memory. Cadence and file destinations are defined under [`train.log_every` and `logging.*`](config-reference.yaml).
