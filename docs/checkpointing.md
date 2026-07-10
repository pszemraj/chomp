# Checkpointing and Resume

chomp uses Orbax to checkpoint **both** training state and data iterator state.
Resume is treated as a contract, not a best-effort feature.

Related: [Config Reference](config-reference.yaml) (`checkpoint.*`),
[Training Loop](training.md), [Data Pipeline](data_pipeline.md).

## What is saved

Each checkpoint stores three items:

1) `train_state`: model parameters, optimizer state, step, RNG
2) `data_state`: the checkpointable data path described in
   [Data Pipeline: iterator state and resume](data_pipeline.md#iterator-state-and-resume)
3) `meta`: JSON metadata (config snapshot, data fingerprint, required
   non-negative `tokens_seen`, parameter-manifest hash, and strict runtime
   identity)

FFD packer queues in `data_state` use flat little-endian int32 payloads plus
row offsets. Grain's JSON handler stores those payloads as base64 rather than
materializing nested Python integer lists, which bounds serialization object
overhead for long contexts and large pending buffers.

The run directory also includes a required tokenizer snapshot under `tokenizer/` and
the pinned eval token set `eval_tokens.json.gz`. It also contains
`parameter-manifest.json`, which lists every trainable parameter and fixed
buffer with its optimizer group and decay policy. Eval cache creation, drift
checks, and the `data.recreate_eval_cache` override are covered in
[Data Pipeline validation set](data_pipeline.md#validation-set).

Tokenizer snapshots are written to a temporary sibling, loaded back for
validation, and atomically renamed. A failed or interrupted save never leaves
an incomplete `tokenizer/` directory that a later resume could mistake for a
valid snapshot.

## Save cadence

Checkpoint frequency is controlled by:

- `checkpoint.enabled`
- `checkpoint.save_every`
- `checkpoint.max_to_keep`
- `checkpoint.async_save`

The manager closes on every exit path, which waits for asynchronous writes and
releases its checkpointer, metadata stores, and deleter. Orbax enforces
`checkpoint.max_to_keep` for retained checkpoints.

A save succeeds only when Orbax explicitly accepts it. Before save and after
restore, Chomp requires the checkpoint directory step, metadata step, and
`TrainState.step` to agree; any mismatch is treated as corruption.

When `debug.nan_check` is enabled, save steps force a metrics sync so the
finite-loss/grad-norm check runs before the write. A non-finite step is then
rejected even when the save cadence does not land on a logging step.

## Run ownership and preemption

Each resolved run directory has a nonblocking sibling lock held from before
artifact setup until checkpoint-manager shutdown. A second fresh or resumed
process targeting that run fails before it can write config, tokenizer, eval,
metrics, manifest, or checkpoint artifacts. Lock files persist so every
process contends on the same inode; the operating-system lock, not file
existence, determines ownership.

On `SIGTERM` or `SIGUSR1`, the main-thread handler records only a stop flag.
The loop does no IO inside the signal handler: it finishes an optimizer step
already in flight, stops at the next aligned model/data boundary, writes a
`preemption_requested` metrics row with `preemption_signal`, forces the final
checkpoint, and closes Orbax before exiting. A request received between
steps stops before another batch is consumed. The Python API raises
`TrainingPreempted` only after finalization; the CLI exits with `128 + signal`
(143 for SIGTERM, 138 for SIGUSR1), so schedulers can distinguish a completed
preemption from success or a training failure.

## Final checkpoint policy

On exit (clean, crash, or Ctrl-C), a final checkpoint of the last completed
step is written only when it is safe to resume from:

- **Alignment**: the train state and data iterator must correspond to the same
  completed step. A crash between batch fetch and step completion leaves the
  iterator ahead of the train state, so the final save is skipped (loudly) and
  resume uses the last periodic checkpoint. Finite streams right-pad their
  final window and missing batch rows; after that batch completes, exact EOF is
  aligned and the final checkpoint is written.
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

- data source identity (`hf_dataset`, `hf_name`, `split`, `hf_revision`, `text_key`)
- data-pipeline implementation schema version
- stream-order and termination semantics (`shuffle`, `shuffle_buffer_size`,
  `shuffle_buffer_bytes`, `repeat`, `window_shuffle_tokens`, its derived
  rows/effective seed, and `grain_prefetch`)
- tokenizer settings and vocab rounding
- packing mode, packing buffer sizes, and strict-segment settings
- objective knobs (`mask_boundary_loss`, `train_on_eos`) and eval knobs
- batch shape invariants (`seq_len`, `batch_size`, `grad_accum`)
- model and optimizer config, `train.deterministic`
- the complete parameter/fixed-buffer, optimizer-group, and decay assignment
  through the parameter-manifest hash
- Python/platform, accelerator backend and device, Chomp source revision, and
  exact JAX/JAXlib/Equinox/Optax/Orbax/Grain/Datasets/Transformers/tokenizers/
  Megalodon-JAX versions

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

The exact-resume contract covers checkpoint save/restore and in-run HF retry
reconstruction. Retry recovery restores the last compact source state and
fast-forwards over the exact count already yielded; see
[Data Pipeline: transient stream recovery](data_pipeline.md#transient-stream-recovery).

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
