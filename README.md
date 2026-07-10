# chomp

A minimal, single-GPU JAX/Equinox pretraining harness for [Megalodon-JAX](https://github.com/pszemraj/megalodon-jax) models.

## Install

The project pins the runtime versions covered by its checkpoint/resume suite.
For NVIDIA CUDA 13, install the matching JAX plugin first; CPU users can skip
the first command. Then install Chomp without changing the pinned core versions:

```bash
pip install "jax[cuda13]==0.8.2"
git clone https://github.com/pszemraj/chomp.git && cd chomp
pip install -e .
```

Other accelerators need the JAX 0.8.2 plugin for that platform.

## Quick start

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

## Configurations

| Config                         | Description                |
| ------------------------------ | -------------------------- |
| [`configs/debug_smoke.yaml`](configs/debug_smoke.yaml)     | Tiny local-text smoke test |
| [`configs/zyda2_100m_2048.yaml`](configs/zyda2_100m_2048.yaml) | 100M Megalodon on Zyda-2   |
| [`configs/zyda2_200m_2048.yaml`](configs/zyda2_200m_2048.yaml) | 200M Megalodon on Zyda-2   |

Personal top-level configs in `configs/custom/` are gitignored. Reusable
experiment suites can be tracked in named subdirectories there.

## Key features

- **Fixed shapes**: compile once, no dynamic padding
- **Resumable**: checkpoints train state + data iterator position
- **Streaming**: HF datasets with sequential/bin/multipack packing and boundary-aware loss masking
- **Tokenizer alignment**: auto-rounds vocab size and sets special token IDs

## Documentation

- [Config Reference](docs/config-reference.yaml): annotated, copyable field/type/default reference
- [Training](docs/training.md): train step behavior, generation, and metrics
- [Data Pipeline](docs/data_pipeline.md): stream-to-batch path and eval-set construction
- [Packing](docs/packing.md): packing strategy and boundary-masking semantics
- [Optimization](docs/optimization.md): optimizer behavior and Muon sweep guidance
- [Checkpointing](docs/checkpointing.md): save/restore/resume contract
- [Development Guide](docs/dev.md): lint, format, test workflow
