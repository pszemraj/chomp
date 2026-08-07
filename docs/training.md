# Training Loop

Training step behavior and metrics written to `metrics.jsonl`.

Related: [Config Reference](config-reference.yaml), [Optimization and Optimizers](optimization.md), [Data Pipeline](data_pipeline.md), [Packing and Boundary Semantics](packing.md), [Checkpointing and Resume](checkpointing.md).

## Maintained recipes

Checked-in scenarios are separated by intent. [`configs/dev/`](../configs/dev/) contains short infrastructure checks, while [`configs/pretrain/`](../configs/pretrain/) contains the four Megalodon scale recipes. Exact parameter counts, measurements, and fit qualifications are in the [README recipe table](../README.md#shipped-recipes-and-measured-expectations); the recipes are executable examples, and their per-field contracts live only in the [Config Reference](config-reference.yaml).

The configured maximum target budget is `steps * grad_accum * batch_size * (seq_len - 1)`; boundary, EOS, and padding masks reduce the realized `tokens_seen`. The maintained schedules provide at least roughly 20 maximum target positions per parameter, but that is a starting budget rather than a claim of compute optimality for a specific corpus or research objective.

## Gated FFN and `ffn_hidden_dim`

All four `configs/pretrain/` recipes set `model.swiglu: true`, matching the upstream `megalodon-jax` paper presets. The gated path computes `silu(fc1(h)) * fc3(h) -> fc2`; with `swiglu: false` the FFN is a bare `silu(fc1(h)) -> fc2` with no gate.

**`model.swiglu` does not rescale `model.ffn_hidden_dim`.** Upstream counts FFN parameters as `2*d*f + (d*f if swiglu)`, so enabling the flag adds a third `d x f` matrix at the *same* width and inflates the model by 50% of its FFN mass. Param-matching therefore requires setting `ffn_hidden_dim` to two-thirds of the non-gated width. The maintained recipes already do this; the widths are chosen to stay on multiples of 64 for GEMM tiling:

| recipe | non-gated `f` | gated `f` | parameters |
| --- | ---: | ---: | ---: |
| 100M | 3072 | **2048** | 113,854,464 (exact match) |
| 200M | 2560 | **1728** | 188,777,472 (+0.42%) |
| 500M | 3840 | **2560** | 513,672,192 (exact match) |
| 1B | 4864 | **3264** | 976,978,944 (+0.24%) |

Two measured consequences (RTX 5090, 200M, `seq_len` 2048, bf16, param-matched):

- **With `model.use_checkpoint: true` the gate is free on memory** — 18.93 GB gated versus 18.9 GB non-gated — because `fc3`'s activation is rematerialized rather than stored. It costs about 1.7% throughput, which is the recompute.
- **With checkpointing off it costs memory but not speed** — +2.96 GB, and measured marginally *faster* (73,143 versus 72,247 valid tok/s), since the parameter-matched shapes have equal FLOPs and 1728 tiles better than 2560.

Muon coverage rises with the gate (84 to 96 tensors at the 200M shape) because `fc3` is a genuine GEMM weight; see [Optimization and Optimizers](optimization.md). Changing `model.swiglu` changes the parameter tree, so it is **not** resume-compatible with a checkpoint trained under the other setting.

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

### Device memory ceiling

The usable ceiling is **not** the card's physical memory. JAX preallocates a fraction of the device at startup — `XLA_PYTHON_CLIENT_MEM_FRACTION`, default `0.75` — and every allocation is served from that pool. Chomp does not set the variable, so the default applies unless you export it. Read the actual limit rather than deriving it from the card:

```python
import jax
jax.devices()[0].memory_stats()["bytes_limit"]
```

On a 32.6 GB RTX 5090 that default yields a **25.25 GB** pool; `0.88` yields 29.63 GB and `0.92` yields 30.98 GB. Compare `peak_memory_gb` in `metrics.jsonl` against that number, not against the card. A configuration whose peak sits comfortably under the card can still be several gigabytes over the pool.

**`bytes_limit` reports the pool XLA was configured to want, not the pool it got.** The fraction is applied to the card's *total* memory, ignoring whatever else is already resident, so the request can exceed what is physically available. When that happens XLA does not fail — it logs a single line to stderr and falls back to an allocator that grows on demand, while `bytes_limit` continues to report the full nominal figure:

```
E... cuda_executor.cc:1182] [0] Failed to allocate device memory of 28.85GiB
  (30979129344 bytes): RESOURCE_EXHAUSTED: : CUDA_ERROR_OUT_OF_MEMORY
```

An on-demand pool is exactly the fragmentation-prone state the fraction was raised to avoid, so the run proceeds and dies later. Measured on the 32.6 GB card with 2144 MiB held by an unrelated process: `0.92` requests 28.85 GiB, fails to reserve, and aborts at the step following the first generation sample; `0.90` requests 28.22 GiB, reserves cleanly, and completes with a 29.14 GB peak. **Grep the startup log for `CUDA_ERROR_OUT_OF_MEMORY` to confirm the reservation actually happened** — that line, not `bytes_limit`, is the ground truth. Leave room for other GPU processes when choosing the fraction, and note that a probe which only queries `bytes_limit` without allocating will not surface the failure, because the reservation is attempted lazily on first use.

**Generation samples raise the peak of the *next* training step**, by a measured +1.65 GB at two different training shapes (bs8×ga8 and bs16×ga4), so the cost belongs to the generation program rather than the training shape. Evaluation costs nothing: it is forward-only over the same `[A,B,T]` shape and reuses the training arena.

**Size the pool from a long run's `peak_memory_gb`, not a short probe's.** The generation spike depends on how fragmented the pool happens to be when the sample lands, so its worst case is a tail event that a 32-step probe rarely reaches. The 200M recipe peaked at 29.14 GB across several 32-step probes and at **29.69 GB** over 100,000 steps — the difference is half the slack under a `0.90` pool. Take the maximum over a completed run when one exists.

The mechanism is fragmentation rather than capacity. Between steps only a few GB is live, but the compiled `jit_train_step` requires one large contiguous scratch arena; the generation program allocates and frees around it, and the arena no longer fits. The failure surfaces during training immediately after a successful sample:

```
RESOURCE_EXHAUSTED: Out of memory while trying to allocate 18.09GiB
  [executable_name='jit_train_step']
```

If a recipe's peak plus that generation headroom exceeds the default pool, either raise the fraction at launch:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 chomp train <config.yaml> --run-dir runs/<name>
```

or reduce demand — narrow `train.batch_size` while raising `train.grad_accum` to hold tokens-per-step fixed, enable `model.use_checkpoint`, or set `train.generate_every: 0` to remove the spike. Raising the fraction does **not** fail loudly if the pool cannot be claimed; see the fallback behaviour above, and verify the reservation from the log. Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` instead when sharing a GPU with another process.

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
