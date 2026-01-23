# Training Loop

This doc summarizes the training step behavior and the metrics logged in
`metrics.jsonl`.

## Train step contract

The compiled `train_step`:

- consumes fixed-shape `Batch` objects (`[A,B,T]`)
- performs gradient accumulation inside `jax.lax.scan`
- applies exactly one optimizer update per outer step

Grad accumulation is **token-weighted**: microbatch losses are scaled by the
count of valid (non-masked) tokens to keep updates correct with padding or
boundary masks.

## Determinism

`train.deterministic` controls dropout behavior:

- `None`: derived from dropout rates (deterministic if all zero)
- `True`: force deterministic
- `False`: force stochastic

Deterministic runs are recommended for resume and regression tests.

## GPU environment notes

Two environment flags are helpful on newer GPUs:

- `XLA_PYTHON_CLIENT_PREALLOCATE=false` to avoid pre-allocating all GPU memory.
- `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false` if Triton GEMM causes
  `CUDA_ERROR_OUT_OF_MEMORY` on RTX 5090 with `jax/jaxlib 0.8.2`.

When `chomp-train` detects an RTX 50xx (Blackwell) GPU, it automatically appends
`--xla_gpu_enable_triton_gemm=false` to `XLA_FLAGS` and warns if
`XLA_PYTHON_CLIENT_PREALLOCATE` is not set to `false`. On other GPUs, the helper
stays quiet (debug log only).

## Attention and loss masking

chomp uses **stream semantics**, treating the corpus as a continuous token
stream. This is standard for simple pretraining. Segment IDs are still emitted
for boundary loss masking but are not used to alter attention.

Loss masking is handled in the data pipeline:

- `data.mask_boundary_loss`: mask labels at segment boundaries
- `data.train_on_eos`: mask EOS labels if desired

## Evaluation

If `train.eval_every > 0`, chomp runs a full pass over the validation texts
selected at run start and logs `eval_loss`.

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

Use `chomp-train <config.yaml> --dry-run` to validate config, build the tokenizer/model/data
pipeline, and compile one step before exiting. W&B logging is skipped in dry-run
mode to avoid creating noisy runs.

## Gradient checkpointing

Megalodon supports activation checkpointing via `model.use_checkpoint`. This is
orthogonal to gradient accumulation and does not change the batch contract.

## Metrics

Metrics are written per step to `logging.metrics_file` and include:

- `loss`
- `grad_norm`
- `lr`
- `tokens_seen`
- `packing_mode`, `packing_utilization`
- `first_step_compile_time_s` (step 0 only)
- `peak_memory_gb` (best-effort, device-dependent)
- `eval_loss` (only when eval runs)

If `logging.wandb.enabled=true`, the same rows are also logged to Weights & Biases.
chomp also uploads `config_original.yaml` as a W&B artifact at run start, and W&B
logs go to the default `./wandb` directory (or `WANDB_DIR` if set).

Console output is throttled by `train.log_every` and prints a compact
one-line summary (loss, grad norm, LR, step time, throughput, optional eval
loss, packing utilization, and best-effort device memory). Full logs from
third-party libraries are written to `logging.log_file` under the run directory.
