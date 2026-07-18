# chomp

A minimal, single-GPU JAX/Equinox pretraining harness for [Megalodon-JAX](https://github.com/pszemraj/megalodon-jax) models.

## Install

Chomp targets Linux systems with NVIDIA GPUs and directly depends on JAX's pip-managed CUDA 13 stack. A compatible NVIDIA driver is required; the base install supplies JAX, jaxlib, the CUDA plugin, and CUDA/cuDNN runtime wheels:

```bash
git clone https://github.com/pszemraj/chomp.git && cd chomp
pip install -e .
```

CPU-only and non-NVIDIA installations are not supported. Chomp accepts compatible fixes along the JAX 0.10.x and Megalodon-JAX 0.2.x release lines from the minimum versions declared in `pyproject.toml`; checkpoint metadata records the exact resolved runtime and rejects version drift on resume.

## Quick start

```bash
# Offline smoke test
chomp train configs/debug_smoke.yaml

# Dry run (validate config, execute one step, exit)
chomp train configs/debug_smoke.yaml --dry-run

# Train the recommended 100k-step recipe with checkpoints
chomp train configs/smoldata_mix_100m_2048.yaml --run-dir runs/my_run

# Resume
chomp train configs/smoldata_mix_100m_2048.yaml --run-dir runs/my_run --resume latest

# Generate
chomp generate runs/my_run --prompt "Hello world" --max-tokens 64
```

## Configurations

| Config | Description |
| ------ | ----------- |
| [`configs/debug_smoke.yaml`](configs/debug_smoke.yaml) | Tiny local-text smoke test |
| [`configs/smoldata_mix_100m_2048.yaml`](configs/smoldata_mix_100m_2048.yaml) | Recommended 100M, 100k-step mixed-corpus run |
| [`configs/zyda2_100m_2048.yaml`](configs/zyda2_100m_2048.yaml) | 100M, 100k-step Zyda-2 comparison |
| [`configs/zyda2_200m_2048.yaml`](configs/zyda2_200m_2048.yaml) | 200M, 100k-step Zyda-2 capacity run |

The 100k recipes use 2 sequences per microbatch, 8 accumulation slices, and
2,048-token rows: about 3.28 billion packed token slots before boundary and
padding masks. The Smol-Data recipe is the default choice for source diversity;
the Zyda-2 recipes provide a stable corpus comparison and a larger model option.

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
