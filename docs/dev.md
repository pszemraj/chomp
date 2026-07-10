# Development Guide

Common developer tasks: linting, formatting, running tests, and finding the
right test file.

> [!IMPORTANT]
> Backward compatibility is **not** a requirement right now. It is acceptable to
> break older configs, checkpoints, or resume metadata when it simplifies the
> system or improves correctness.

## Environment

All project commands should run inside the `mega-jax` conda environment.

```bash
conda run --name mega-jax <command>
```

Runtime and development dependencies are exactly pinned in
[`pyproject.toml`](../pyproject.toml), including the Megalodon and Optax source
commits. Upgrade them deliberately with the full resume/checkpoint suite, not
as unconstrained installs. JAX accelerator plugins must match the pinned
`jax==jaxlib==0.8.2` core.

## Lint and format

Run Ruff in fix mode, then format:

```bash
conda run --name mega-jax ruff check --fix .
conda run --name mega-jax ruff format .
```

## Tests

Run the full suite:

```bash
conda run --name mega-jax pytest -q
```

Run a single module-focused test file:

```bash
conda run --name mega-jax pytest -q tests/test_training.py
```

Slow tests are marked with `@pytest.mark.slow`.

Run slow tests only:

```bash
conda run --name mega-jax pytest -q -m slow
```

Skip slow tests explicitly:

```bash
conda run --name mega-jax pytest -q -m "not slow"
```

## Test layout

Tests are organized by source module (not by micro-feature):

- [`tests/test_config.py`](../tests/test_config.py): config validation, variables, tokenizer-derived updates,
  and generate-time config loading
- [`tests/test_config_reference.py`](../tests/test_config_reference.py): annotated config key, default,
  type, and inline-contract drift checks
- [`tests/test_data_pipeline.py`](../tests/test_data_pipeline.py): packing, segment IDs, HF streaming/state,
  tokenizer decode, and tokenizer snapshot behavior
- [`tests/test_training.py`](../tests/test_training.py): training loop behavior, crash handling, dry-run,
  checkpointing, and resume behavior
- [`tests/test_optimizer.py`](../tests/test_optimizer.py): Muon optimizer labeling and grad accumulation
  equivalence
- [`tests/test_utils.py`](../tests/test_utils.py): device placement, init sanity,
  parameter counting, finite-metric checks, and XLA environment helpers
- [`tests/test_cli.py`](../tests/test_cli.py): CLI banner and generate command behavior
- [`tests/test_eval.py`](../tests/test_eval.py): eval logging and eval text selection
- [`tests/test_ckpt_paths.py`](../tests/test_ckpt_paths.py): checkpoint path/config resolution

Shared helper modules:

- [`tests/helpers/assertions.py`](../tests/helpers/assertions.py): structured JAX
  equality checks
- [`tests/helpers/config_factories.py`](../tests/helpers/config_factories.py): reusable tiny
  train/checkpoint configs
- [`tests/helpers/hf_fakes.py`](../tests/helpers/hf_fakes.py): reusable fake HF streaming iterables
- [`tests/helpers/hf_resume_worker.py`](../tests/helpers/hf_resume_worker.py): fresh-process real-HF
  resume continuation worker
- [`tests/helpers/io.py`](../tests/helpers/io.py): test artifact readers

High-risk invariants remain isolated for visibility:

- [`tests/test_compile_stability.py`](../tests/test_compile_stability.py)
- [`tests/test_cache_policy.py`](../tests/test_cache_policy.py)
- [`tests/test_gpu_smoke.py`](../tests/test_gpu_smoke.py)

## Tracked follow-ups

- **TODO(provenance): carry original-document and source identity through the
  host pipeline.** Replace the text-only source item with a compact record,
  retain document identity across long-document chunks and FFD rows, and keep
  per-row segment-to-document/source mappings beside (not inside) device
  batches. Log unique original documents, dominant-document token fraction,
  maximum rows from one document, source counts, and source entropy per logged
  optimizer batch. This belongs with multi-source mixing because source fields
  and stable IDs are dataset-schema decisions. Acceptance requires exact
  interrupted/resumed provenance metrics, bounded host memory, and no dense
  document-ID device array. Until then, `docs_added_this_batch` is only a
  bursty source-pull diagnostic and must not be interpreted as batch
  homogeneity.

- **TODO(grain-shuffle): replace the pinned private Grain workaround with an
  owned packed-window iterator.** The verified runtime pins `grain==0.2.15`
  and fails fast if `_WindowShuffleDatasetIterator._init` changes. An owned
  transform should store only parent state at the window boundary, window
  index, and permutation cursor, then remove the private attribute patch.
  Acceptance requires the full continuous-versus-resumed suite with prefetch,
  mid-window restore, and a deliberate Grain upgrade before changing the pin.

- **TODO(batch-transfer): reduce the device batch to tokens and compact segment
  metadata.** Attention masks and position IDs are functions of segment IDs;
  labels are functions of tokens, boundaries, EOS policy, and padding. Derive
  them on-device only after preserving the host zero-objective guard, exact
  loss-token envelope, eval behavior, strict/non-strict segment semantics, and
  backend capability checks. Benchmark host-to-device bytes and step time at
  2K, 8K, and 32K contexts; keep the change only when it improves measured
  throughput or memory without complicating those contracts.

- **TODO(megalodon-rope): move fixed rotary frequencies into the model's static
  contract.** The pinned Megalodon-JAX release exposes `rotary.inv_freq` as an
  ordinary floating array, so Chomp currently classifies that exact path as a
  fixed buffer and hard-gates the parameter-manifest hash. Update the dependency
  to derive frequencies from static dimension/base values in the forward path
  (or expose an upstream trainable-leaf API), then delete the path exception.
  Acceptance requires the fixed-buffer optimizer-state/update regression and
  all real-model parameter-manifest variants to remain green.
