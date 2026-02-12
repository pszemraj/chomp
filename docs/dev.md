# Development Guide

This page covers common developer tasks: linting, formatting, running tests, and
finding the right test file.

> [!IMPORTANT]
> Backward compatibility is **not** a requirement right now. It is acceptable to
> break older configs, checkpoints, or resume metadata when it simplifies the
> system or improves correctness.

## Environment

All project commands should run inside the `mega-jax` conda environment.

```bash
conda run --name mega-jax <command>
```

## Lint And Format

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

## Test Layout

Tests are organized by source module (not by micro-feature):

- [`tests/test_config.py`](../tests/test_config.py): config validation, variables, tokenizer-derived updates,
  and generate-time config loading
- [`tests/test_data_pipeline.py`](../tests/test_data_pipeline.py): packing, segment IDs, HF streaming/state,
  tokenizer decode, and tokenizer snapshot behavior
- [`tests/test_training.py`](../tests/test_training.py): training loop behavior, crash handling, dry-run,
  checkpointing, and resume behavior
- [`tests/test_optimizer.py`](../tests/test_optimizer.py): Muon optimizer labeling and grad accumulation
  equivalence
- [`tests/test_utils.py`](../tests/test_utils.py): device placement, init sanity, param counting,
  token counters, finite-metric checks, and XLA env helpers
- [`tests/test_cli.py`](../tests/test_cli.py): CLI banner and generate command behavior
- [`tests/test_eval.py`](../tests/test_eval.py): eval logging and eval text selection

High-risk invariants remain isolated for visibility:

- [`tests/test_compile_stability.py`](../tests/test_compile_stability.py)
- [`tests/test_cache_policy.py`](../tests/test_cache_policy.py)
- [`tests/test_gpu_smoke.py`](../tests/test_gpu_smoke.py)

## Nice-to-Haves (Later)

- Make `tokens_seen` exact when iterator stats are missing (e.g., `device_put=True`),
  by accumulating `token_sum` in-step or storing a counter in `TrainState`.
- Optional perf knob: skip grad-norm computation when clipping is disabled and
  grad-norm logging is off.
