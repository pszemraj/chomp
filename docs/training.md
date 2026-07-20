# Training Loop

Training step behavior and metrics written to `metrics.jsonl`.

Related: [Config Reference](config-reference.yaml), [Optimization and Optimizers](optimization.md), [Data Pipeline](data_pipeline.md), [Packing and Boundary Semantics](packing.md), [Checkpointing and Resume](checkpointing.md), [Comma stability study](comma_stability_matrix.md).

## Train step contract

The compiled `train_step`:

- consumes fixed-shape `Batch` objects (`[A,B,T]`)
- performs gradient accumulation inside `jax.lax.scan`
- applies exactly one optimizer update per outer step

Grad accumulation is **token-weighted**: microbatch losses are scaled by the count of valid (non-masked) tokens to keep updates correct with padding or boundary masks.

Batch assembly computes the same shifted valid-target count on the host and pairs it with the batch through prefetch/device transfer. The trainer updates `tokens_seen` from that Python integer without synchronizing every optimizer step. Logging, evaluation, checkpoint, first-compile, and finite-check cadences synchronize queued work and verify the current host count against the compiled int32 counter. At save and final boundaries, the finite check also covers the post-update parameters and optimizer state. Evaluation likewise accumulates all batch totals on device and synchronizes once per pass.

Full packing diagnostics scan segment arrays only for batches whose global step will log or evaluate. Non-observable steps still compute the exact shifted loss-token count required for accounting, but skip the redundant Python-side diagnostic scan.

`model.loss_chunk_size` optionally bounds Megalodon vocabulary-head intermediates in both training and evaluation. Its complete memory/throughput contract and starting recommendation are inline in the [Config Reference](config-reference.yaml).

## Determinism

`train.deterministic` controls dropout behavior:

- `None`: derived from dropout rates (deterministic if all zero)
- `True`: force deterministic
- `False`: force stochastic

Deterministic runs are recommended for resume and regression tests. Note that in `megalodon-jax`, activation checkpointing is disabled when `train.deterministic=true`. If you want checkpointing with deterministic math, set `train.deterministic=false` and keep all dropout rates at `0.0`. Enable activation checkpointing with `model.use_checkpoint`; it is orthogonal to gradient accumulation.

## GPU environment notes

The supported JAX 0.10.x CUDA 13 stack runs RTX 50xx GPUs with its default kernel selection; Chomp does not rewrite `XLA_FLAGS`. Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` when sharing a GPU or when allocator preallocation interferes with another process.

XLA kernel selection on GPU is nondeterministic by default. This is the fast path used for production training. For debugging runs that need bit-exact GPU numerics (e.g. comparing an interrupted+resumed run against a continuous one at atol=0), opt in with `XLA_FLAGS=--xla_gpu_deterministic_ops=true`. This is a different knob from `train.deterministic` above (dropout/RNG vs kernel selection), and it is expensive: measured ~25-35% slower steps on a 100M-param Megalodon smoke benchmark (RTX 5090, seq_len 2048, bf16). The effective setting is recorded in checkpoint meta, and resume warns if it drifted across the boundary; see [Checkpointing: scope of exactness](checkpointing.md#scope-of-exactness).

## Evaluation

If `train.eval_every > 0` and `data.max_eval_samples > 0`, chomp runs a full pass over the process-local eval token set and logs `eval_loss`. A non-finite loss reduction fails the run before a metric is logged. Eval text selection, packed eval flushing, and zero-batch/zero-token failures are documented in [Data Pipeline validation set](data_pipeline.md#validation-set).

Eval batches are assembled once and cached host-side for the whole run; device transfer happens per batch each eval, so no device memory is held between evals.

## Generation samples

If `train.generate_every > 0`, chomp periodically samples a prompt from a bounded pool of up to 16 unshuffled training-split documents and runs `megalodon_jax.generate`, printing both the prompt and generated continuation to the console (Rich panels when enabled). The pool avoids retaining a second production-sized document-shuffle window during training.

Default behaviors (when the `generate_*` fields are `null`):

- `train.generate_input_len`: half of `train.seq_len`
- `train.generate_max_tokens`: `model.chunk_size + 16`
- prompt selection: if a sample is longer than `generate_input_len`, randomly use the first or last `generate_input_len` tokens; otherwise use the full sample (no EOS token appended)

Optional sampling controls (`train.generate_temperature`, `train.generate_top_k`, `train.generate_top_p`) are passed through when set; otherwise the Megalodon defaults apply. Generation is currently only enabled for the `megalodon` backend (dummy runs skip it silently).

### Standalone generation

`chomp generate` accepts a run directory or checkpoint step directory, restores the stored parameters and resolved config, and uses the run-pinned tokenizer described in [Data Pipeline: tokenization](data_pipeline.md#tokenization) when available. Set `--temperature 0` for greedy decoding; seeded sampling is the default. Run `chomp generate --help` for the option list.

## Dry run

Use `chomp train <config.yaml> --dry-run` to validate config, build the tokenizer/model/data pipeline, and execute one step before exiting. The step compiles when `train.jit` is enabled. When `debug.nan_check` is enabled, success also requires finite loss, gradient norm, learning rate, post-update parameters, and optimizer state. W&B logging is skipped in dry-run mode.

`config_resolved.json` includes a small `derived` section; for example `derived.optim.decay_steps_effective` records the effective LR schedule horizon.

## Metrics

Metrics are written to `logging.metrics_file` on the first process-local training step, every `train.log_every` steps, and on eval steps. They include:

- `loss`
- `grad_norm`
- `lr`
- `loss_tokens` (exact host count, checked against compiled `token_sum` at sync points)
- `tokens_seen` (cumulative exact host count)
- `step_time_s`, `data_wait_s` (per-step averages over the last sync interval)
- `tokens_per_sec` (end-to-end throughput over the last sync interval)
- `packing_mode`, `packing_utilization` (when iterator stats are enabled)
- `docs_seen`, `docs_truncated` (when available)
- `source_tokens_observed`, `source_tokens_retained`, `source_tokens_discarded`, `source_truncation_fraction`
- `source_doc_tokens_p50_upper`, `source_doc_tokens_p90_upper`, `source_doc_tokens_p99_upper` (log2-histogram upper bounds)
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

If `logging.wandb.enabled=true`, Weights & Biases receives the training rows plus detailed wall-clock, packing-capacity, eval-token, and current-device-memory metrics. The local file instead retains the process peak-memory value and its explicit `step` field. When the run was started from a config file, chomp also uploads `config_original.yaml` as a W&B artifact. W&B logs go to the default `./wandb` directory (or `WANDB_DIR` if set). Set `logging.wandb.enabled=false` to disable W&B; `mode` selects online or offline logging only.

Console output is throttled by `train.log_every` and prints a compact one-line summary (loss, grad norm, LR, step time, throughput, optional eval loss, packing utilization, and best-effort device memory). Full logs from third-party libraries are written to `logging.log_file` under the run directory when that field is not `null`.
