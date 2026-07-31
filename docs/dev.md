# Development Guide

Common developer tasks: linting, formatting, running tests, and finding the right test file.

> [!IMPORTANT]
> Backward compatibility is **not** a requirement right now. It is acceptable to break older configs, checkpoints, or resume metadata when it simplifies the system or improves correctness.

## Environment

All project commands should run inside the `mega-jax` conda environment.

```bash
conda run --name mega-jax <command>
```

Grain, Datasets, and Orbax are pinned in [`pyproject.toml`](../pyproject.toml) because their iterator/checkpoint behavior is part of the harness. The model stack uses bounded compatible release lines starting at JAX 0.10.2, Equinox 0.13.8, Optax 0.2.8, and Megalodon-JAX 0.2.2. Other runtime and development packages use minimum versions rather than acting as a lockfile. The base JAX dependency includes its `cuda13` extra, which resolves matching jaxlib and CUDA plugin versions plus the pip-managed CUDA 13 runtime. CPU-only installations are unsupported.

Personal experiment configs belong under `configs/custom/`. That directory is recursively gitignored so local recipes do not enter commits accidentally.

## Reviewing checkpoint changes

The [checkpoint design intent](checkpointing.md#design-intent) is part of the project scope. Preserve the distinction between saved-state coherence and environment provenance:

- Hard-fail when model/optimizer/data state cannot form one coherent continuation.
- Keep known, shape-compatible research changes visible and usable through the default `warn` policy; use `strict` for unchanged config/data experiments.
- Do not reinterpret `strict` as general source-tree, dependency, device, or kernel attestation. Megalodon-JAX is the narrow exception: its recorded distribution identity is active model semantics. Do not add broader hard gates without an explicit project-scope change.
- Do not restore automatic fresh-stream fallback after train-state restore. A deliberate branch uses a separate run directory.
- Assume one researcher-controlled writer per run directory rather than adding distributed locking or ownership machinery.

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

## Test layout

Tests are organized by source module (not by micro-feature):

- [`tests/test_config.py`](../tests/test_config.py): config validation, variables, tokenizer-derived updates, and generate-time config loading
- [`tests/test_data_pipeline.py`](../tests/test_data_pipeline.py): packing, segment IDs, HF streaming/state, and tokenizer behavior
- [`tests/test_training.py`](../tests/test_training.py): training loop behavior, crash handling, dry-run, checkpointing, and resume behavior
- [`tests/test_optimizer.py`](../tests/test_optimizer.py): Muon optimizer labeling and grad accumulation equivalence
- [`tests/test_utils.py`](../tests/test_utils.py): device backend validation, init sanity, parameter counting, and finite-metric checks
- [`tests/test_cli.py`](../tests/test_cli.py): CLI parsing and generate command behavior
- [`tests/test_eval.py`](../tests/test_eval.py): eval logging and eval text selection
- [`tests/test_ckpt_paths.py`](../tests/test_ckpt_paths.py): checkpoint path/config resolution

Shared helper modules:

- [`tests/helpers/config_factories.py`](../tests/helpers/config_factories.py): reusable tiny model, pipeline, and run configs
- [`tests/helpers/gpu.py`](../tests/helpers/gpu.py): NVIDIA discovery and GPU subprocess workers
- [`tests/helpers/hf_fakes.py`](../tests/helpers/hf_fakes.py): reusable fake HF streaming iterables
- [`tests/helpers/hf_resume_worker.py`](../tests/helpers/hf_resume_worker.py): fresh-process real-HF resume continuation worker
- [`tests/helpers/io.py`](../tests/helpers/io.py): test artifact readers

GPU invariants remain isolated in [`tests/test_gpu_smoke.py`](../tests/test_gpu_smoke.py).

Resume compatibility behavior is documented in [Checkpointing and Resume](checkpointing.md#resume-compatibility-checks). The default warns on semantic drift while structural train-state mismatches remain hard errors; strict config/data compatibility is opt-in.
