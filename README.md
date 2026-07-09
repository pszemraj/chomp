# chomp

A minimal, single-GPU JAX/Equinox pretraining harness for [Megalodon-JAX](https://github.com/pszemraj/megalodon-jax) models.

## Install

1. Install [JAX](https://docs.jax.dev/en/latest/installation.html) for your platform/CUDA version
2. Install chomp:

```bash
git clone https://github.com/pszemraj/chomp.git && cd chomp
pip install -e .
```

## Quick Start

```bash
# Smoke test (CPU, offline)
chomp train configs/debug_smoke.yaml

# Dry run (validate config, execute one step, exit)
chomp train configs/debug_smoke.yaml --dry-run

# Train with checkpoints
chomp train configs/zyda2_100m_2048.yaml --run-dir runs/my_run

# Resume
chomp train configs/zyda2_100m_2048.yaml --run-dir runs/my_run --resume latest

# Generate
chomp generate runs/my_run --prompt "Hello world" --max-tokens 64
```

## Configs

| Config                         | Description                |
| ------------------------------ | -------------------------- |
| [`configs/debug_smoke.yaml`](configs/debug_smoke.yaml)     | Tiny local-text smoke test |
| [`configs/zyda2_100m_2048.yaml`](configs/zyda2_100m_2048.yaml) | 100M Megalodon on Zyda-2   |
| [`configs/zyda2_200m_2048.yaml`](configs/zyda2_200m_2048.yaml) | 200M Megalodon on Zyda-2   |

Personal top-level configs in `configs/custom/` are gitignored. Reusable
experiment suites can be tracked in named subdirectories there.

## Key Features

- **Fixed shapes**: compile once, no dynamic padding
- **Resumable**: checkpoints train state + data iterator position
- **Streaming**: HF datasets with sequential/bin/multipack packing and boundary-aware loss masking
- **Tokenizer alignment**: auto-rounds vocab size and sets special token IDs

## Docs

- [Config Reference](docs/config-reference.md) - field/type/default reference
- [Training](docs/training.md) - train step behavior, generation, and metrics
- [Data Pipeline](docs/data_pipeline.md) - stream-to-batch path and eval-set construction
- [Packing](docs/packing.md) - packing strategy and boundary-masking semantics
- [Optimization](docs/optimization.md) - optimizer behavior and Muon sweep guidance
- [Checkpointing](docs/checkpointing.md) - save/restore/resume contract
- [Development Guide](docs/dev.md) - lint, format, test workflow
