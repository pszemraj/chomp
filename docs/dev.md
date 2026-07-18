# Development Guide

Common developer tasks: linting, formatting, running tests, and finding the right test file.

> [!IMPORTANT]
> Backward compatibility is **not** a requirement right now. It is acceptable to break older configs, checkpoints, or resume metadata when it simplifies the system or improves correctness.

## Environment

All project commands should run inside the `mega-jax` conda environment.

```bash
conda run --name mega-jax <command>
```

Stateful data/checkpoint dependencies and development tools are exactly pinned in [`pyproject.toml`](../pyproject.toml). The model stack uses bounded compatible release lines starting at JAX 0.10.2, Equinox 0.13.8, Optax 0.2.8, and Megalodon-JAX 0.2.1. The base JAX dependency includes its `cuda13` extra, which resolves matching jaxlib and CUDA plugin versions plus the pip-managed CUDA 13 runtime. CPU-only installations are unsupported. Upgrade a bound only with the full resume/checkpoint and escalated GPU suites.

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
- [`tests/test_data_pipeline.py`](../tests/test_data_pipeline.py): packing, segment IDs, HF streaming/state, tokenizer decode, and tokenizer snapshot behavior
- [`tests/test_training.py`](../tests/test_training.py): training loop behavior, crash handling, dry-run, checkpointing, and resume behavior
- [`tests/test_optimizer.py`](../tests/test_optimizer.py): Muon optimizer labeling and grad accumulation equivalence
- [`tests/test_utils.py`](../tests/test_utils.py): device placement, init sanity, parameter counting, finite-metric checks, and XLA environment helpers
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

## Tracked follow-ups

- **TODO(provenance): carry original-document and source identity through the host pipeline.** Replace the text-only source item with a compact record, retain document identity across long-document chunks and FFD rows, and keep per-row segment-to-document/source mappings beside (not inside) device batches. Log unique original documents, dominant-document token fraction, maximum rows from one document, source counts, and source entropy per logged optimizer batch. This belongs with multi-source mixing because source fields and stable IDs are dataset-schema decisions. Acceptance requires exact interrupted/resumed provenance metrics, bounded host memory, and no dense document-ID device array.

- **TODO(grain-shuffle): replace the pinned private Grain workaround with an owned packed-window iterator.** The verified runtime pins `grain==0.2.15` and fails fast if `_WindowShuffleDatasetIterator._init` changes. An owned transform should store only parent state at the window boundary, window index, and permutation cursor, then remove the private attribute patch. Acceptance requires the full continuous-versus-resumed suite with prefetch, mid-window restore, and a deliberate Grain upgrade before changing the pin.

- **TODO(batch-labels): evaluate deriving labels on-device.** Labels are a function of tokens, segment boundaries, EOS policy, and padding. Remove the transferred label array only if the design preserves the host zero-objective guard, exact loss-token envelope, eval behavior, and strict/non-strict segment semantics. Benchmark host-to-device bytes and step time at 2K, 8K, and 32K contexts before keeping the added device-side logic.
