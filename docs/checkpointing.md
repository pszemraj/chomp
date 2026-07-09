# Checkpointing and Resume

chomp uses Orbax to checkpoint **both** training state and data iterator state.
Resume is treated as a contract, not a best-effort feature.

## Scope

This page is the home for save/restore/resume semantics.

- For checkpoint field defaults/types: [Config Reference](config-reference.md) (`checkpoint.*`)
- For training-loop runtime behavior after resume: [Training Loop](training.md)
- For data iterator state details: [Data Pipeline](data_pipeline.md)

## What is saved

Each checkpoint stores three items:

1) `train_state`: model parameters, optimizer state, step, RNG
2) `data_state`: iterator state (HF cursor + packer buffer) via Grain's
   checkpoint handler
3) `meta`: JSON metadata (config snapshot + data fingerprint + versions)

The run directory also includes a tokenizer snapshot under `tokenizer/` and
the pinned eval token set `eval_tokens.json.gz` (created once at run start,
reloaded on every resume so eval losses stay comparable even if the upstream
dataset drifts). If that file is missing when resuming, resume fails hard;
`data.recreate_eval_cache: true` is the explicit override (eval curves across
the boundary are then not comparable).

## Save cadence

Checkpoint frequency is controlled by:

- `checkpoint.enabled`
- `checkpoint.save_every`
- `checkpoint.max_to_keep`
- `checkpoint.async_save`

If async saving is enabled, the manager waits on exit to avoid partial writes.
Orbax enforces `checkpoint.max_to_keep` for retained checkpoints.

Save steps force a metrics sync so the finite-loss/grad-norm check
(`debug.nan_check`) always runs before the write — a step with non-finite
metrics is never persisted as a resume point, even when the save cadence does
not land on a logging step.

## Final checkpoint policy

On exit (clean, crash, or Ctrl-C), a final checkpoint of the last completed
step is written only when it is safe to resume from:

- **Alignment**: the train state and data iterator must correspond to the same
  completed step. A crash between batch fetch and step completion — or the
  stream running dry partway through assembling a batch (`repeat: false`) —
  leaves the iterator ahead of the train state, so the final save is skipped
  (loudly) and resume uses the last periodic checkpoint.
- **Validity**: the last step's metrics are re-checked for finiteness before
  the write; "latest" can never be a NaN tombstone.

A final save that fails on an otherwise clean exit fails the run — training
never exits successfully with an unwritten checkpoint.

## Resume compatibility checks

On resume, chomp compares the checkpoint metadata against the current config.
Hard failures include:

- data source identity (`hf_dataset`, `hf_name`, `split`, `text_key`)
- stream-order and termination semantics (`shuffle`, `shuffle_buffer_size`,
  `repeat`, `window_shuffle_windows`, `grain_prefetch`)
- tokenizer settings and vocab rounding
- packing mode, packing buffer sizes, and strict-segment settings
- objective knobs (`mask_boundary_loss`, `train_on_eos`) and eval knobs
- batch shape invariants (`seq_len`, `batch_size`, `grad_accum`)
- model and optimizer config, `train.deterministic`

The packing, model, and optimizer sections are compared over the union of
keys recorded on either side, so a knob present in only one version's
fingerprint is a hard mismatch — never silently skipped.

`data.device_put` drift is a warning (it changes where the host-to-device
transfer happens, not sample order) — except when `grain_prefetch > 0` on
either side, where it hardens to an error because it changes the prefetch
mechanics around the serialized iterator state.

Remaining warnings are logged so you can make an informed decision, but
anything that changes what data the resumed run sees — or what it optimizes —
is an error, not a warning.

## Typical usage

```bash
# Start a run
chomp train configs/debug_smoke.yaml --run-dir runs/chomp/debug_run

# Resume latest
chomp train configs/debug_smoke.yaml --run-dir runs/chomp/debug_run --resume latest
```

If a mismatch is detected, resume fails fast with a detailed error.

## Scope of exactness

The exact-resume contract covers checkpoint save/restore. In-run recovery
from transient HF streaming errors is a separate, best-effort mechanism that
can replay up to `data.state_update_interval` recent documents — see
[Data Pipeline — Transient stream recovery](data_pipeline.md#transient-stream-recovery-best-effort).

Bit-exactness on GPU additionally requires deterministic XLA kernels: without
`--xla_gpu_deterministic_ops=true`, XLA may pick nondeterministic GPU kernels
(atomic-reduction scatters, algorithm choices sensitive to prior GPU state),
and a resumed run drifts from the continuous one in the low-order bits of the
optimizer state even though losses match. `chomp train` (and the test suite)
therefore appends that flag to `XLA_FLAGS` at startup whenever NVIDIA GPUs
are present; `chomp generate` does not, since generation makes no
bit-exactness promise. An explicit `--xla_gpu_deterministic_ops=false` in
`XLA_FLAGS` is respected and trades the bit-exact contract for the
nondeterministic kernels' throughput (measured ~25-35% faster steps on a
100M-param Megalodon smoke benchmark, RTX 5090, seq_len 2048 — see
[Training Loop — GPU environment notes](training.md#gpu-environment-notes)).
