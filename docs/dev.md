# Development Guide

Common developer tasks: linting, formatting, running tests, and finding the right test file.

> [!IMPORTANT]
> Backward compatibility is **not** a requirement right now. It is acceptable to break older configs, checkpoints, or resume metadata when it simplifies the system or improves correctness.

## Environment

All project commands should run inside the `mega-jax` conda environment.

```bash
conda run --name mega-jax <command>
```

Dependency declarations and supported release lines live in [`pyproject.toml`](../pyproject.toml); installation and accelerator expectations are in [Requirements and installation](../README.md#requirements-and-installation). Stateful iterator/checkpoint pins require behavior-specific review before they change. CPU-only execution is for tests and debugging, not Megalodon training.

## Configuration layout

- [`configs/dev/`](../configs/dev/) contains short smoke scenarios. `offline_cpu_smoke.yaml` is deterministic and network-free; `hf_streaming_smoke.yaml` deliberately exercises Hub streaming and the saved HF tokenizer path with a narrow DummyLM.
- [`configs/pretrain/`](../configs/pretrain/) contains maintained Megalodon recipes from approximately 100M through 1B parameters. Scale choices and measured evidence are in the [recipe table](../README.md#shipped-recipes-and-measured-expectations); field defaults and interactions live in the [Config Reference](config-reference.yaml).
- `configs/custom/` is recursively gitignored for personal experiments so local recipes do not enter commits accidentally.

Use a pretrain recipe with `--dry-run` for a real Megalodon compile and optimizer update; the Hub streaming smoke intentionally does not duplicate that expensive check.

## Reviewing checkpoint changes

Treat [checkpoint design intent](checkpointing.md#design-intent) and [resume compatibility](checkpointing.md#resume-compatibility-checks) as normative. Never pair restored train state with a fresh data stream, do not broaden strict resume into general environment attestation without a scope change, and retain the single-writer run-directory assumption.

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
