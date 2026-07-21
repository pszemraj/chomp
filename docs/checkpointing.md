# Checkpointing and Resume

chomp uses Orbax to checkpoint **both** training state and data iterator state. An unchanged run resumes exactly; intentional config changes use the compatibility policy below.

Related: [Config Reference](config-reference.yaml) (`checkpoint.*`), [Training Loop](training.md), [Data Pipeline](data_pipeline.md).

## What is saved

Each checkpoint stores three items:

1) `train_state`: model parameters, optimizer state, step, RNG
2) `data_state`: the checkpointable data path described in [Data Pipeline: iterator state and resume](data_pipeline.md#iterator-state-and-resume)
3) `meta`: JSON metadata (config snapshot, data fingerprint, and required non-negative `tokens_seen`)

Runs using a Hugging Face tokenizer include its files under `tokenizer/`; resumed runs load them instead of the configured remote source. The built-in byte tokenizer has no files and is reconstructed from the resolved config.

## Save cadence

Checkpoint frequency is controlled by:

- `checkpoint.enabled`
- `checkpoint.save_every`
- `checkpoint.max_to_keep`
- `checkpoint.async_save`

The manager and data iterator close on every exit path. Orbax waits for asynchronous writes and releases its checkpointer, metadata stores, and deleter; Grain stops prefetch workers and closes the underlying Hugging Face stream. Datasets 5.0.0 is pinned because it includes the remote-Parquet thread-shutdown cleanup for successful processes that stop mid-shard. For a single-source Parquet stream, Chomp also observes Datasets' builder flag and applies its bounded Arrow thread-shutdown grace after closing the generator. Local and non-Parquet streams do not wait. Orbax enforces `checkpoint.max_to_keep` for retained checkpoints.

A save succeeds only when Orbax explicitly accepts it. Before save and after restore, Chomp requires the checkpoint directory step, metadata step, and `TrainState.step` to agree; any mismatch is treated as corruption.

`--resume latest` continues the newest finalized checkpoint. An explicit step may select that newest step, but Chomp rejects an older retained step in the same checkpoint root because subsequent saves would collide with the already finalized future. To branch from an older step, copy it into a new run directory first.

A run directory is single-writer: do not start concurrent training processes against the same `logging.run_dir`. Use a separate copied run directory when branching or running another continuation.

When `debug.nan_check` is enabled, save steps force a metrics sync and validate loss, gradient norm, learning rate, post-update parameters, and optimizer state before the write. A non-finite step is rejected even when the save cadence does not land on a logging step.

## Preemption

On `SIGTERM` or `SIGUSR1`, the main-thread handler records only a stop flag. The loop does no IO inside the signal handler: it finishes an optimizer step already in flight, stops at the next aligned model/data boundary, writes a `preemption_requested` metrics row with `preemption_signal`, forces the final checkpoint, and closes Orbax before exiting. A fresh run stopped before its first batch saves the aligned step-zero state, so `--resume latest` remains available. A request received between steps stops before another batch is consumed; a request during the final step's evaluation, generation, or logging tail is recorded before finalization. The stop flag is checked again after checkpoint, resource, and telemetry finalization so a request during teardown cannot be reported as success. The Python API raises `TrainingPreempted` only after finalization; the CLI exits and W&B finishes with `128 + signal` (143 for SIGTERM, 138 for SIGUSR1), so schedulers and telemetry agree that a completed preemption was not a successful training completion.

## Final checkpoint policy

On exit (clean, crash, or Ctrl-C), a final checkpoint of the last completed step is written only when it is safe to resume from:

- **Alignment**: the train state and data iterator must correspond to the same completed step. A crash between batch fetch and step completion leaves the iterator ahead of the train state, so the final save is skipped (loudly) and resume uses the last periodic checkpoint. Finite streams right-pad their final window and missing batch rows; after that batch completes, exact EOF is aligned and the final checkpoint is written.
- **Validity**: when `debug.nan_check` is enabled, the last step's metrics, parameters, and optimizer state are re-checked for finiteness, so "latest" cannot become a NaN tombstone. Final validity is a run invariant even when checkpointing is disabled.

A final save that fails on an otherwise clean exit fails the run; training never exits successfully with an unwritten checkpoint.

## Resume compatibility checks

On resume, chomp compares the checkpoint metadata against the current config. `checkpoint.resume_compat` controls semantic mismatches:

- `warn` (default) logs each data, objective, batch-shape, model-runtime, optimizer-value, schedule, or eval-selection change and continues. This supports ordinary workflows such as extending `train.steps`, lowering the LR, or changing eval size.
- `strict` rejects those semantic changes before restore when exact continuation is required.

Both modes always reject missing/invalid checkpoint metadata, invalid `tokens_seen`, model parameter-tree changes, and optimizer-state structure changes such as switching `optim.name`. Muon routing flags and enabling/disabling its optional `consistent_rms` transform are structural as well. These cannot consume the saved arrays.

The metadata-only compatibility check runs before evaluation or training datasets are constructed, so strict mismatches fail without opening the configured remote source.

After restoring model parameters, optimizer state, RNG, and step, chomp requires Grain to restore the matching iterator state. A data-state restore failure aborts resume in both compatibility modes; restarting the corpus behind a restored optimizer would produce a contradictory training history.

Resume comparisons ignore settings that cannot affect restored execution, including fresh-model `model.init_mode`, activation-checkpoint/segmented-scan implementation choices, tokenizer download settings, and vocab rounding once the resolved model vocabulary is already checked. Keys absent from older checkpoint metadata are skipped; the actual array restore remains the authority on structural compatibility.

`train.deterministic` is compared by its effective dropout behavior, so an inferred `null` and explicit `true` are resume-equivalent when all active dropout rates are zero. The maintained 100k-step recipes select strict compatibility explicitly.

For Hugging Face data, a checkpointed run records both the requested branch/tag and the immutable commit it resolved to. Resume reads that identity from the selected checkpoint metadata and reuses the commit without a Hub request only when the repository and requested ref still match. A deliberate new ref or commit is honored and then handled by the configured `warn` or `strict` compatibility policy.

## Typical usage

```bash
# Start a run
chomp train configs/debug_smoke.yaml --run-dir runs/chomp/debug_run

# Resume latest
chomp train configs/debug_smoke.yaml --run-dir runs/chomp/debug_run --resume latest
```

Add `-o checkpoint.resume_compat=strict` when exact semantic continuation is required.

## Scope of exactness

The exact-resume contract covers checkpoint save and restore.

With unchanged config, resume guarantees exact **state and data replay**: parameters, optimizer state, RNG, and the data iterator position restore exactly, so the resumed run optimizes the same objective over the same batches in the same order as the continuous run. Warn mode permits declared semantic mismatches, but it does not permit a failed iterator-state restore.

GPU step arithmetic is bit-identical only with the opt-in setting described in [Training: GPU environment notes](training.md#gpu-environment-notes).
