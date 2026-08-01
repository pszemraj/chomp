# Checkpointing and Resume

chomp uses Orbax to checkpoint **both** training state and data iterator state. Under the same code and runtime, an unchanged run resumes exactly; intentional config changes use the compatibility policy below.

Related: [Config Reference](config-reference.yaml) (`checkpoint.*`), [Training Loop](training.md), [Data Pipeline](data_pipeline.md).

## Design intent

Checkpointing is recovery for a researcher-operated, single-process, single-GPU training run. It preserves a coherent saved training history without turning ordinary experimentation into a production deployment or environment-certification workflow.

The policy boundary is deliberate:

- **State coherence fails closed.** Missing or structurally incompatible train state, invalid step/accounting metadata, or an iterator state that cannot be restored aborts resume. `warn` never means swallowing restore exceptions or restarting the corpus behind restored optimizer state.
- **Known semantic changes remain usable.** `warn` is the default because extending a run, changing evaluation, or making a shape-compatible research adjustment is a normal workflow when the saved state remains restorable. Every detected mismatch is reported. `strict` is available when the experiment requires unchanged config and data semantics; maintained long-run recipes select it explicitly.
- **A selected checkpoint owns its recorded semantics.** Its embedded config and source identity are the prior-state authority for resume compatibility and the default for generation. `config_resolved.json` is the immutable run-start record and a legacy fallback, not an override for later checkpoints. The requested resume config still flows through `warn` or `strict`, and an explicit generation config remains the highest-priority user choice.
- **Strict is not general environment attestation.** Chomp records and compares the executed Megalodon-JAX distribution because its model/loss program defines saved-state semantics, and records the packages directly used by the effective tokenizer. It does not hash the checkout, inspect dirty files, inventory unrelated packages, pin a device identity, or hard-gate XLA flags. Version control and experiment tracking own that broader provenance. GPU arithmetic is bit-exact only when the user opts into deterministic kernels.
- **One writer is the operating model.** Concurrent writers and distributed checkpoint coordination are outside this harness; use another run directory for a branch or continuation.

## What is saved

Each checkpoint stores three items:

1) `train_state`: model parameters, optimizer state, step, RNG
2) `data_state`: the checkpointable data path described in [Data Pipeline: iterator state and resume](data_pipeline.md#iterator-state-and-resume)
3) `meta`: schema-versioned JSON metadata (config snapshot, data fingerprint, required non-negative `tokens_seen`, evaluation failure-policy state, the executed Megalodon-JAX distribution identity, and the tokenizer-manifest identity)

The per-step metadata is self-describing. The backend identity always includes the installed distribution version. When the installation has PEP 610 `direct_url.json` metadata, it also includes the URL, VCS kind, requested revision, and resolved commit. Standalone generation uses the selected checkpoint's config before considering the run-start config, while resume compares the requested config and installed backend against that selected step. A retained step therefore continues to mean the model semantics that produced it without preventing deliberate continuation changes.

Every run writes `tokenizer/identity.json`. It records the effective tokenizer class, fast/slow status, directly relevant package versions, exact saved-file sizes and SHA-256 digests, and outputs for a versioned canary corpus. Hugging Face assets live beside it and are reloaded locally before fresh execution; resume loads only those run-pinned files with Hub access disabled. The byte tokenizer has no asset files, but its implementation and canary outputs are bound by the same manifest. Every checkpoint stores the canonical manifest digest so a selected checkpoint, rather than mutable run-start metadata, remains the semantic authority.

## Save cadence

Checkpoint frequency is controlled by:

- `checkpoint.enabled`
- `checkpoint.save_every`
- `checkpoint.max_to_keep`
- `checkpoint.async_save`

The manager and data iterator close on every exit path. Orbax waits for asynchronous writes and releases its checkpointer, metadata stores, and deleter; Grain stops prefetch workers and closes the underlying Hugging Face stream. Datasets 5.0.0 is pinned because it provides the remote-Parquet thread-shutdown workaround for successful processes that stop mid-shard. For a single-source Parquet stream, Chomp closes the generator and applies Datasets' bounded shutdown grace when the builder requests it, releases and collects its Arrow-backed dataset and iterator references while CPython is still live, then gives any native destructors initiated by that release the same bounded grace. Local and non-Parquet streams do not wait. Orbax enforces `checkpoint.max_to_keep` for retained checkpoints.

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
- `strict` rejects those semantic changes before restore when unchanged config and data semantics are required.

Both modes always reject missing/invalid checkpoint metadata, invalid `tokens_seen`, model parameter-tree changes, and optimizer-state structure changes such as switching `optim.name`. Muon routing flags and enabling/disabling its optional `consistent_rms` transform are structural as well. These cannot consume the saved arrays.

The resume compatibility preflight validates the tokenizer snapshot and selected checkpoint metadata before evaluation or training datasets are constructed, so strict mismatches fail without opening the configured remote data or tokenizer source.

After restoring model parameters, optimizer state, RNG, and step, chomp requires Grain to restore the matching iterator state. A data-state restore failure aborts resume in both compatibility modes; restarting the corpus behind a restored optimizer would produce a contradictory training history.

Resume comparisons ignore settings that cannot affect restored execution, including fresh-model `model.init_mode` and vocab rounding once the resolved model vocabulary is already checked. `model.loss_chunk_size` is active Megalodon resume semantics. `model.use_checkpoint` is active when effective `train.deterministic=false` allows upstream rematerialization, and inert when deterministic execution disables it. `train.eval_failure_policy` follows the configured warn/strict resume policy. `data.tokenizer.hf_use_fast` and `data.tokenizer.hf_trust_remote_code` are evaluated through the effective tokenizer identity: changing a request is harmless only when the loaded implementation and canary outputs remain equal. `model.use_associative_segment_scan` is active resume semantics when strict bin/multipack segment resets execute, and inert when segment metadata does not reach the model. Every active field in the current canonical config and data fingerprint must be present in checkpoint metadata: absence is a compatibility mismatch, distinct from a recorded `null` value. Strict mode rejects that mismatch; warn mode reports it before restore.

For Megalodon runs, strict resume rejects a changed or missing structured backend identity; warn mode reports it clearly. A legacy flat version or a checkpoint without this field is not treated as proven equal. The selected checkpoint is authoritative: `config_resolved.json` records the run-start identity for provenance but never fills a missing checkpoint field.

Checkpoint metadata schema 2 adds the tokenizer identity. Schema 3 also persists whether evaluation was disabled, its failure count and last failure step/type, and the last successful evaluation step, so a resumed run does not silently re-enable evaluation or reset its telemetry history. An enabled evaluator whose latest recorded attempt failed is still pending: resume repeats that evaluation against the restored parameters before another optimizer step or a successful return. Strict resume rejects a missing or unsupported schema marker and missing required schema-3 state. It also rejects a changed or missing run manifest, saved-file record, effective implementation, canary output, or checkpoint digest. Warn mode reports that equality is unproven, continues with the observed local tokenizer identity and fresh defaults for unavailable evaluation status, and records those observed values in later checkpoints. Schema-2 checkpoints and older run directories are therefore not strictly resumable.

Checkpoint compatibility deliberately does not fingerprint source trees, dirty or untracked files, the rest of the package environment, devices, or XLA flags. Those remain external experiment provenance rather than saved-state alignment.

`train.deterministic` is compared by its effective dropout behavior, so an inferred `null` and explicit `true` are resume-equivalent when all active dropout rates are zero. The maintained 100k-step recipes select strict compatibility explicitly.

For Hugging Face data, a checkpointed run records both the requested branch/tag and the immutable commit it resolved to. Resume reads that identity from the selected checkpoint metadata and reuses the commit without a Hub request only when the repository and requested ref still match. A deliberate new ref or commit is honored and then handled by the configured `warn` or `strict` compatibility policy.

## Typical usage

```bash
# Start a run
chomp train configs/debug_smoke.yaml --run-dir runs/chomp/debug_run

# Resume latest
chomp train configs/debug_smoke.yaml --run-dir runs/chomp/debug_run --resume latest
```

Add `-o checkpoint.resume_compat=strict` when unchanged config and data semantics are required.

## Scope of exactness

The exact-resume contract covers the state Chomp saves and restores. With unchanged config under the same code and runtime, parameters, optimizer state, RNG, and the data iterator position restore exactly, so the resumed run optimizes the same objective over the same batches in the same order as the continuous run. Warn mode permits declared semantic mismatches, but it does not permit a failed iterator-state restore.

Chomp does not prove that the executable, general dependency environment, hardware, or external process state remained identical. `strict` tightens config/data compatibility, binds Megalodon runs to the recorded backend distribution identity, and binds tokenizer execution to its manifest; it is not a claim of cross-environment bitwise reproducibility. GPU step arithmetic is bit-identical only with the opt-in setting described in [Training: GPU environment notes](training.md#gpu-environment-notes).
