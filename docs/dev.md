# Development Guide

Common developer tasks: linting, formatting, running tests, and finding the right test file.

> [!IMPORTANT]
> Backward compatibility is **not** a requirement right now. It is acceptable to break older configs, checkpoints, or resume metadata when it simplifies the system or improves correctness.

## Environment

All project commands should run inside the `mega-jax` conda environment.

```bash
conda run --name mega-jax <command>
```

Grain, Datasets, and Orbax are pinned in [`pyproject.toml`](../pyproject.toml) because their iterator/checkpoint behavior is part of the harness. The model stack uses bounded compatible release lines starting at JAX 0.10.2, Equinox 0.13.8, Optax 0.2.8, and Megalodon-JAX 0.2.1. Other runtime and development packages use minimum versions rather than acting as a lockfile. The base JAX dependency includes its `cuda13` extra, which resolves matching jaxlib and CUDA plugin versions plus the pip-managed CUDA 13 runtime. CPU-only installations are unsupported.

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

## Planned: best-effort resume (`checkpoint.resume_compat`)

Not implemented yet. Resume is currently all-or-nothing: `check_resume_compat`
([`src/chomp/ckpt.py`](../src/chomp/ckpt.py)) hard-errors on any config/fingerprint drift
before restore. The planned relaxation:

- Add `checkpoint.resume_compat: "strict" | "warn"` to `CheckpointConfig`
  ([`src/chomp/config.py`](../src/chomp/config.py)), default `strict`.
- Tier the checks in `check_resume_compat`. Always error: drift that makes the array
  restore structurally impossible (model architecture, optimizer structure) — Orbax would
  fail anyway, just later and cryptically. Downgrade to warnings in `warn` mode:
  data-order/objective drift (resolved source revision, shuffle knobs, packing, seed,
  `mask_boundary_loss`, `train_on_eos`) and diagnostic drift (eval selection).
- At the resume site in [`src/chomp/train.py`](../src/chomp/train.py), wrap the data-state
  restore: in `warn` mode a Grain `set_state` failure (the saved state no longer fits a
  changed pipeline shape) logs loudly and falls back to a fresh stream. Train state —
  params, opt_state, RNG, step, and therefore the LR schedule — always restores fully.
- Tests in [`tests/test_training.py`](../tests/test_training.py): warn-mode resume across a
  shuffle-knob change (state restored, warning logged), across a packing-mode change
  (fresh-stream fallback), and strict mode unchanged.
