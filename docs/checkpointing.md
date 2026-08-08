# Checkpointing and Resume

chomp uses Orbax to checkpoint **both** training state and data iterator state. Under the same code and runtime, an unchanged run resumes exactly; intentional config changes use the compatibility policy below.

Related: [Config Reference](config-reference.yaml) (`checkpoint.*`), [Training Loop](training.md), [Data Pipeline](data_pipeline.md), [Export](export.md).

A checkpoint is for continuing a run, not for shipping a model: it is chomp-specific, carries optimizer state, and is roughly twice the size of the weights. To hand the model to something else, see [Export](export.md).

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

Tokenizer snapshot construction and preflight validation are described under [Data Pipeline tokenization](data_pipeline.md#tokenization). Every checkpoint stores the canonical manifest digest, making the selected checkpoint rather than mutable run-start metadata authoritative.

## Save and shutdown behavior

Enablement, cadence, retention, and asynchronous-save semantics are defined under [`checkpoint.*`](config-reference.yaml).

The manager and data iterator close on every exit path. Orbax waits for asynchronous writes and releases its checkpointer, metadata stores, and deleter; Grain stops prefetch workers and closes the underlying Hugging Face stream. Datasets 5.0.0 is pinned because it provides the remote-Parquet thread-shutdown workaround for successful processes that stop mid-shard. For a single-source Parquet stream, Chomp closes the generator and applies Datasets' bounded shutdown grace when the builder requests it, releases and collects its Arrow-backed dataset and iterator references while CPython is still live, then gives any native destructors initiated by that release the same bounded grace. Local and non-Parquet streams do not wait.

A save succeeds only when Orbax explicitly accepts it. Before save and after restore, Chomp requires the checkpoint directory step, metadata step, and `TrainState.step` to agree; any mismatch is treated as corruption.

`--resume latest` continues the newest finalized checkpoint. An explicit step may select that newest step, but Chomp rejects an older retained step in the same checkpoint root because subsequent saves would collide with the already finalized future. To continue an older step under unchanged semantics, copy it into a new run directory first; to start a new training history from it, use `--init-from` below.

A run directory is single-writer. Use a separate copied run directory when branching or running another continuation.

Save-time validity follows [`debug.nan_check`](config-reference.yaml). When active, a save forces metrics synchronization and rejects non-finite metrics, parameters, or optimizer state even off the normal logging cadence.

## Preemption

On `SIGTERM` or `SIGUSR1`, the main-thread handler records only a stop flag. The loop does no IO inside the signal handler: it finishes an optimizer step already in flight, stops at the next aligned model/data boundary, writes a `preemption_requested` metrics row with `preemption_signal`, forces the final checkpoint, and closes Orbax before exiting. A fresh run stopped before its first batch saves the aligned step-zero state, so `--resume latest` remains available. A request received between steps stops before another batch is consumed; a request during the final step's evaluation, generation, or logging tail is recorded before finalization. The stop flag is checked again after checkpoint, resource, and telemetry finalization so a request during teardown cannot be reported as success. The Python API raises `TrainingPreempted` only after finalization; the CLI exits and W&B finishes with `128 + signal` (143 for SIGTERM, 138 for SIGUSR1), so schedulers and telemetry agree that a completed preemption was not a successful training completion.

## Process exit

The `chomp` console script ends the process with `os._exit` after flushing stdout, stderr, and logging handlers, so interpreter finalization never runs. Native threads inside dependencies — Apache Arrow's Parquet readers behind Hugging Face streaming, [arrow#45214](https://github.com/apache/arrow/issues/45214) — can call back into CPython once `Py_FinalizeEx` has started and abort the process with `Fatal Python error: PyGILState_Release`. That aborts *after* the run is complete and every checkpoint is durable, but it reports SIGABRT (134), which makes a finished run indistinguishable from a crashed one to a shell, an `&&` chain, or a scheduler. `datasets` ships a sleep-based workaround and `HFStreamingTextStream.close` applies it twice; sleeping narrows the race without closing it, and a completed 100,000-step run still hit the abort.

Skipping finalization is safe here because training closes everything it owns first — checkpoint manager, data iterator, metrics writer, W&B run, logging handlers — so no `atexit` hook is load-bearing. The exit status is the one the CLI computed: `0` on success, `1` on finalization failure, `128 + signal` on preemption, `2` on usage errors. Library callers importing `chomp.cli.cli` (and the test suite) get ordinary interpreter shutdown; only the process entry point hard-exits.

## Final checkpoint policy

On exit (clean, crash, or Ctrl-C), a final checkpoint of the last completed step is written only when it is safe to resume from:

- **Alignment**: the train state and data iterator must correspond to the same completed step. A crash between batch fetch and step completion leaves the iterator ahead of the train state, so the final save is skipped (loudly) and resume uses the last periodic checkpoint. Finite streams right-pad their final window and missing batch rows; after that batch completes, exact EOF is aligned and the final checkpoint is written.
- **Validity**: when configured finite-state checking is enabled, the last step is re-checked so "latest" cannot become a NaN tombstone. This final check applies even when checkpointing is disabled.

A final save that fails on an otherwise clean exit fails the run; training never exits successfully with an unwritten checkpoint.

## Resume compatibility checks

[`checkpoint.resume_compat`](config-reference.yaml) selects the contextual semantic-comparison policy. Its rationale is the distinction between coherent saved state and deliberate research changes described under [Design intent](#design-intent).

The policy compares configurations that conform to the current schema; it is not a migration system. Before the first stable release, obsolete config or checkpoint fields are not translated or silently discarded, and snapshots containing removed keys fail deserialization in both modes.

Both modes reject missing or invalid checkpoint metadata and any parameter-tree or optimizer-state structural incompatibility; those arrays cannot be restored coherently.

The resume compatibility preflight validates the tokenizer snapshot and selected checkpoint metadata before evaluation or training datasets are constructed, so strict mismatches fail without opening the configured remote data or tokenizer source.

After restoring model parameters, optimizer state, RNG, and step, chomp requires Grain to restore the matching iterator state. A data-state restore failure aborts resume in both compatibility modes; restarting the corpus behind a restored optimizer would produce a contradictory training history.

Comparisons use effective execution semantics: inactive or request-only settings may be equivalent when resolved behavior is unchanged, while active objective/runtime settings follow the configured policy. Per-key classifications live in the Config Reference. Every active current fingerprint field must be present in checkpoint metadata; missing is distinct from a recorded `null`.

For Megalodon runs, strict resume rejects a changed or missing structured backend identity; warn mode reports it clearly. A legacy flat version or a checkpoint without this field is not treated as proven equal. The selected checkpoint is authoritative: `config_resolved.json` records the run-start identity for provenance but never fills a missing checkpoint field.

Checkpoint metadata schema 2 adds tokenizer identity. Schema 3 also persists evaluation disablement and failure/success history, so resume cannot silently re-enable evaluation or erase telemetry. An enabled evaluator whose latest recorded attempt failed remains pending and must run against restored parameters before another optimizer step or successful return. A successful retry during a no-op resume does not create a replacement checkpoint because model and data state did not advance, so another no-op resume from the same selected checkpoint repeats the evaluation. Missing or unsupported schema state cannot prove strict equality; warn mode reports that fact, uses the observed local tokenizer identity and fresh defaults for unavailable evaluation status, then records them in later checkpoints. Schema-2 checkpoints and older run directories are therefore not strictly resumable.

Checkpoint compatibility deliberately does not fingerprint source trees, dirty or untracked files, the rest of the package environment, devices, or XLA flags. Those remain external experiment provenance rather than saved-state alignment.

Hugging Face source identity is checkpoint-bound; source replay mechanics are documented under [Data Pipeline iterator state and resume](data_pipeline.md#iterator-state-and-resume), and field precedence is canonical in the Config Reference.

## Typical usage

```bash
# Start a run
chomp train configs/dev/offline_cpu_smoke.yaml --run-dir runs/chomp/debug_run

# Resume latest
chomp train configs/dev/offline_cpu_smoke.yaml --run-dir runs/chomp/debug_run --resume latest
```

Add `-o checkpoint.resume_compat=strict` when unchanged config and data semantics are required.

## Warm start (`--init-from`)

`--init-from` seeds a **new** run's parameters from an existing checkpoint. It is not resume: optimizer state, RNG, step counter, token accounting, and corpus position all start fresh, and the new run gets its own warmup and decay schedule because the learning rate is a function of `TrainState.step`.

```bash
# Phase 2 of a staged run: same model, different data semantics
chomp train configs/custom/phase2.yaml --run-dir runs/chomp/phase2 --init-from runs/chomp/phase1
```

The argument accepts a run directory, a checkpoint root, or an explicit step directory, resolved exactly as `chomp generate` resolves its checkpoint argument. `--init-from` and `--resume` are mutually exclusive; the two operations disagree about which training history the optimizer state belongs to.

Because no data or optimizer state is restored, warm start deliberately imposes **no** config compatibility check: packing mode, corpus, batch shape, optimizer, and schedule may all differ. Two things are still enforced. The saved parameter tree must match the new config's architecture, which Orbax validates structurally during restore. The source checkpoint's tokenizer identity must equal the new run's, which is checked explicitly because it is the one mismatch a structural check cannot see — a different tokenizer of the same vocabulary size restores cleanly and then trains on embedding rows that mean something else. Checkpoints with no recorded tokenizer identity are refused rather than assumed compatible.

Each warm-started run records `warm_start.json` in its run directory with the source step directory, step, token count, and tokenizer identity, so a warm-started run remains distinguishable from a fresh one after the fact.

## Scope of exactness

The exact-resume contract covers the state Chomp saves and restores. With unchanged config under the same code and runtime, parameters, optimizer state, RNG, and the data iterator position restore exactly, so the resumed run optimizes the same objective over the same batches in the same order as the continuous run. Warn mode permits declared semantic mismatches, but it does not permit a failed iterator-state restore.

Chomp does not prove that the executable, general dependency environment, hardware, or external process state remained identical. `strict` tightens config/data compatibility, binds Megalodon runs to the recorded backend distribution identity, and binds tokenizer execution to its manifest; it is not a claim of cross-environment bitwise reproducibility. GPU step arithmetic is bit-identical only with the opt-in setting described in [Training: GPU environment notes](training.md#gpu-environment-notes).
