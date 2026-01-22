# Checkpointing and Resume

chomp uses Orbax to checkpoint **both** training state and data iterator state.
Resume is treated as a contract, not a best-effort feature.

## What is saved

Each checkpoint stores three items:

1) `train_state`: model parameters, optimizer state, step, RNG
2) `data_state`: iterator state (HF cursor + packer buffer)
3) `meta`: JSON metadata (config snapshot + data fingerprint + versions)

The run directory also includes a tokenizer snapshot under `tokenizer/`.

## Save cadence

Checkpoint frequency is controlled by:

- `checkpoint.enabled`
- `checkpoint.save_every`
- `checkpoint.max_to_keep`
- `checkpoint.max_save_checkpoints`
- `checkpoint.async_save`

If async saving is enabled, the manager waits on exit to avoid partial writes.

Before saving a new checkpoint, chomp prunes the oldest checkpoint if the
directory already contains `checkpoint.max_save_checkpoints` entries.

## Resume compatibility checks

On resume, chomp compares the checkpoint metadata against the current config.
Hard failures include:

- data source identity (`hf_dataset`, `hf_name`, `split`, `text_key`)
- tokenizer settings and vocab rounding
- packing mode and packing buffer sizes
- batch shape invariants (`seq_len`, `batch_size`, `grad_accum`)
- model and optimizer config

Some changes are warnings only (e.g., shuffle buffer size), but they are logged
so you can make an informed decision.

## Typical usage

```bash
# Start a run
chomp-train --config configs/debug_smoke.yaml --run-dir runs/chomp/debug_run

# Resume latest
chomp-train --config configs/debug_smoke.yaml --run-dir runs/chomp/debug_run --resume latest
```

If a mismatch is detected, resume fails fast with a detailed error.
