# Checkpointing and Resume

chomp uses Orbax to checkpoint **both** training state and data iterator state.
Resume is treated as a contract, not a best-effort feature.

Related: [Config Reference](config-reference.md) (`checkpoint.*`),
[Training Loop](training.md), [Data Pipeline](data_pipeline.md).

## What is saved

Each checkpoint stores three items:

1) `train_state`: model parameters, optimizer state, step, RNG
2) `data_state`: the checkpointable data path described in
   [Data Pipeline: iterator state and resume](data_pipeline.md#iterator-state-and-resume)
3) `meta`: JSON metadata (config snapshot, data fingerprint, required
   non-negative `tokens_seen`, and versions)

The run directory also includes a tokenizer snapshot under `tokenizer/` and
the pinned eval token set `eval_tokens.json.gz`. Eval cache creation, drift
checks, and the `data.recreate_eval_cache` override are covered in
[Data Pipeline validation set](data_pipeline.md#validation-set).

## Save cadence

Checkpoint frequency is controlled by:

- `checkpoint.enabled`
- `checkpoint.save_every`
- `checkpoint.max_to_keep`
- `checkpoint.async_save`

If async saving is enabled, the manager waits on exit to avoid partial writes.
Orbax enforces `checkpoint.max_to_keep` for retained checkpoints.

When `debug.nan_check` is enabled, save steps force a metrics sync so the
finite-loss/grad-norm check runs before the write. A non-finite step is then
rejected even when the save cadence does not land on a logging step.

## Final checkpoint policy

On exit (clean, crash, or Ctrl-C), a final checkpoint of the last completed
step is written only when it is safe to resume from:

- **Alignment**: the train state and data iterator must correspond to the same
  completed step. A crash between batch fetch and step completion, or the
  stream running dry partway through assembling a batch (`repeat: false`),
  leaves the iterator ahead of the train state. The final save is skipped
  (loudly) and resume uses the last periodic checkpoint. If finite data ends
  exactly at a batch boundary before any new packed window is consumed, the
  iterator is still aligned with the last completed step and the final
  checkpoint is written.
- **Validity**: when `debug.nan_check` is enabled, the last step's metrics are
  re-checked for finiteness before the write, so "latest" cannot become a NaN
  tombstone.

A final save that fails on an otherwise clean exit fails the run; training
never exits successfully with an unwritten checkpoint.

## Resume compatibility checks

On resume, chomp compares the checkpoint metadata against the current config.
Missing or invalid `tokens_seen` metadata is also rejected so cumulative token
accounting resumes exactly.
Hard failures include:

- data source identity (`hf_dataset`, `hf_name`, `split`, `text_key`)
- stream-order and termination semantics (`shuffle`, `shuffle_buffer_size`,
  `repeat`, `window_shuffle_windows`, its effective seed, and `grain_prefetch`)
- tokenizer settings and vocab rounding
- packing mode, packing buffer sizes, and strict-segment settings
- objective knobs (`mask_boundary_loss`, `train_on_eos`) and eval knobs
- batch shape invariants (`seq_len`, `batch_size`, `grad_accum`)
- model and optimizer config, `train.deterministic`

The packing, model, and optimizer sections are compared over the union of
keys recorded on either side, so a knob present in only one version's
fingerprint is a hard mismatch and is never silently skipped.

`data.device_put` drift is a warning (it changes where the host-to-device
transfer happens, not sample order), except when `grain_prefetch > 0` on
either side, where it hardens to an error because it changes the prefetch
mechanics around the serialized iterator state.

The effective `xla_gpu_deterministic_ops` setting (parsed from `XLA_FLAGS`,
recorded at save time) is also compared and warns on drift: kernel
determinism is opt-in and only affects low-order step numerics; see
[Scope of exactness](#scope-of-exactness).

Remaining warnings are logged so you can make an informed decision, but
anything that changes what data the resumed run sees or what it optimizes
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
can replay up to `data.state_update_interval` recent documents; see
[Data Pipeline: transient stream recovery](data_pipeline.md#transient-stream-recovery-best-effort).

What resume guarantees is exact **state and data replay**: parameters,
optimizer state, RNG, and the data iterator position restore exactly, so the
resumed run optimizes the same objective over the same batches in the same
order as the continuous run.

GPU step arithmetic is not bit-identical by default because production uses
XLA's fast nondeterministic kernels. Setup, cost, and the opt-in deterministic
flag are documented in
[Training: GPU environment notes](training.md#gpu-environment-notes). The
effective setting is recorded in checkpoint metadata, and resume warns if it
changes.
