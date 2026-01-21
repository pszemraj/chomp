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

## Segment masking

When `model.segment_masking=true`, the Megalodon patch builds a block-diagonal
attention mask from `segment_ids`. This prevents attention across packed
document boundaries.

Loss masking is handled in the data pipeline:

- `data.mask_boundary_loss`: mask labels at segment boundaries
- `data.train_on_eos`: mask EOS labels if desired

## Evaluation

If `train.eval_every > 0`, chomp runs a full pass over the cached validation
texts and logs `eval_loss` and `eval_tokens`. The validation set is fixed for
the entire run and cached with pre-tokenized docs to avoid re-tokenization.

## Gradient checkpointing

Megalodon supports activation checkpointing via `model.use_checkpoint`. This is
orthogonal to gradient accumulation and does not change the batch contract.

## Metrics

Metrics are written per step to `logging.metrics_file` and include:

- `loss`
- `grad_norm`
- `lr`
- `tokens_seen`
- `wall_time_s`
- `packing_mode`, `packing_tokens`, `packing_capacity`, `packing_utilization`
- `first_step_compile_time_s` (step 0 only)
- `peak_memory_gb` (best-effort, device-dependent)
- `eval_loss`, `eval_tokens` (only when eval runs)

If `logging.wandb_enabled=true`, the same rows are also logged to Weights & Biases.
