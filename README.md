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

# Dry run (validate config, compile one step, exit)
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
| `configs/debug_smoke.yaml`     | Tiny local-text smoke test |
| `configs/zyda2_100m_2048.yaml` | 100M Megalodon on Zyda-2   |
| `configs/zyda2_200m_2048.yaml` | 200M Megalodon on Zyda-2   |

Local configs go in `configs/custom/` (gitignored).

## Key Features

- **Fixed shapes**: compile once, no dynamic padding
- **Resumable**: checkpoints train state + data iterator position
- **Streaming**: HF datasets with bin packing and boundary-aware loss masking
- **Tokenizer alignment**: auto-rounds vocab size and sets special token IDs

## Docs

- [Config Reference](docs/config-reference.qmd) - full field-by-field reference
- [Data Pipeline](docs/data_pipeline.md) - HF streaming and Grain batching
- [Packing](docs/packing.md) - packing modes and loss masking
- [Checkpointing](docs/checkpointing.md) - save/restore and resume
- [Training](docs/training.md) - train loop and metrics
