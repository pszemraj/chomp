# Checkpointing and Resume

chomp uses Orbax to checkpoint **both** training state and data iterator state. Resume is treated as a contract, not a best-effort feature.

Related: [Config Reference](config-reference.yaml) (`checkpoint.*`), [Training Loop](training.md), [Data Pipeline](data_pipeline.md).

## What is saved

Each checkpoint stores three items:

1) `train_state`: model parameters, optimizer state, step, RNG
2) `data_state`: the checkpointable data path described in [Data Pipeline: iterator state and resume](data_pipeline.md#iterator-state-and-resume)
3) `meta`: JSON metadata (config snapshot, data fingerprint, and required non-negative `tokens_seen`)

The run directory also includes the tokenizer snapshot under `tokenizer/`; resumed runs load that snapshot instead of the configured remote tokenizer source.

## Save cadence

Checkpoint frequency is controlled by:

- `checkpoint.enabled`
- `checkpoint.save_every`
- `checkpoint.max_to_keep`
- `checkpoint.async_save`

The manager and data iterator close on every exit path. Orbax waits for asynchronous writes and releases its checkpointer, metadata stores, and deleter; Grain stops prefetch workers and closes the underlying Hugging Face stream. Datasets 5.0.0 is pinned because it includes the remote-Parquet thread-shutdown cleanup for successful processes that stop mid-shard. For a single-source Parquet stream, Chomp also observes Datasets' builder flag and applies its bounded Arrow thread-shutdown grace after closing the generator. Local and non-Parquet streams do not wait. Orbax enforces `checkpoint.max_to_keep` for retained checkpoints.

A save succeeds only when Orbax explicitly accepts it. Before save and after restore, Chomp requires the checkpoint directory step, metadata step, and `TrainState.step` to agree; any mismatch is treated as corruption.

`--resume latest` continues the newest finalized checkpoint. An explicit step may select that newest step, but Chomp rejects an older retained step in the same checkpoint root because subsequent saves would collide with the already finalized future. To branch from an older step, copy it into a new run directory first.

When `debug.nan_check` is enabled, save steps force a metrics sync and validate loss, gradient norm, learning rate, post-update parameters, and optimizer state before the write. A non-finite step is rejected even when the save cadence does not land on a logging step.

## Preemption

On `SIGTERM` or `SIGUSR1`, the main-thread handler records only a stop flag. The loop does no IO inside the signal handler: it finishes an optimizer step already in flight, stops at the next aligned model/data boundary, writes a `preemption_requested` metrics row with `preemption_signal`, forces the final checkpoint, and closes Orbax before exiting. A fresh run stopped before its first batch saves the aligned step-zero state, so `--resume latest` remains available. A request received between steps stops before another batch is consumed; a request during the final step's evaluation, generation, or logging tail is recorded before finalization. The stop flag is checked again after checkpoint, resource, and telemetry finalization so a request during teardown cannot be reported as success. The Python API raises `TrainingPreempted` only after finalization; the CLI exits with `128 + signal` (143 for SIGTERM, 138 for SIGUSR1), so schedulers can distinguish a completed preemption from success or a training failure.

## Final checkpoint policy

On exit (clean, crash, or Ctrl-C), a final checkpoint of the last completed step is written only when it is safe to resume from:

- **Alignment**: the train state and data iterator must correspond to the same completed step. A crash between batch fetch and step completion leaves the iterator ahead of the train state, so the final save is skipped (loudly) and resume uses the last periodic checkpoint. Finite streams right-pad their final window and missing batch rows; after that batch completes, exact EOF is aligned and the final checkpoint is written.
- **Validity**: when `debug.nan_check` is enabled, the last step's metrics, parameters, and optimizer state are re-checked for finiteness, so "latest" cannot become a NaN tombstone. Final validity is a run invariant even when checkpointing is disabled.

A final save that fails on an otherwise clean exit fails the run; training never exits successfully with an unwritten checkpoint.

## Resume compatibility checks

On resume, chomp compares the checkpoint metadata against the current config. Missing or invalid `tokens_seen` metadata is also rejected so cumulative token accounting resumes exactly. Hard failures include:

- data source identity (`hf_dataset`, `hf_name`, `split`, `hf_revision`, `text_key`)
- data-pipeline implementation schema version
- effective stream-order and termination semantics (`shuffle`, its active buffer limits and seed, `repeat`, derived token- and row-bounded packed-window shuffle rows/effective seed, and `grain_prefetch`); inert raw limits do not block resume
- tokenizer settings and vocab rounding
- packing mode, packing buffer sizes, and strict-segment settings
- objective knobs (`mask_boundary_loss`, `train_on_eos`) and eval knobs
- batch shape invariants (`seq_len`, `batch_size`, `grad_accum`)
- active model and optimizer config, `train.deterministic`

Resume comparisons ignore backend- or mode-specific settings only when execution cannot consume them, such as Megalodon fields under DummyLM or Muon settings under AdamW. Muon remains a hybrid optimizer whose non-Muon leaves use AdamW, so `optim.adam` is active in both optimizer modes. Active fingerprint sections are compared over the union of keys recorded on either side, so a newly recorded data or packing knob is never silently skipped.

Remaining warnings are logged so you can make an informed decision, but anything that changes what data the resumed run sees or what it optimizes is an error, not a warning.

## Typical usage

```bash
# Start a run
chomp train configs/debug_smoke.yaml --run-dir runs/chomp/debug_run

# Resume latest
chomp train configs/debug_smoke.yaml --run-dir runs/chomp/debug_run --resume latest
```

If a mismatch is detected, resume fails fast with a detailed error.

## Scope of exactness

The exact-resume contract covers checkpoint save and restore.

What resume guarantees is exact **state and data replay**: parameters, optimizer state, RNG, and the data iterator position restore exactly, so the resumed run optimizes the same objective over the same batches in the same order as the continuous run.

GPU step arithmetic is bit-identical only with the opt-in setting described in [Training: GPU environment notes](training.md#gpu-environment-notes).
