# chomp

Chomp is a compact, single-GPU JAX/Equinox pretraining harness for [Megalodon-JAX](https://github.com/pszemraj/megalodon-jax). [Megalodon](https://arxiv.org/abs/2404.08801) combines recurrent CEMA memory with gated attention for efficient long-context sequence modeling.

Chomp sits between a toy training script and a distributed framework: it keeps the readable, hackable shape of nanoGPT-style code while adding resumable Hugging Face streaming, document packing, checkpointed iterator position, evaluation, generation, and the Megalodon-JAX model adapter. It deliberately does not provide multi-host training, distributed orchestration, general environment attestation, or a Transformers export stack; use a distributed trainer such as [Levanter](https://github.com/stanford-crfm/levanter) when those are the actual requirements.

The project is alpha software. Correctness takes priority over backward compatibility until the first stable release, so older configs and checkpoints may occasionally require migration.

## Requirements and installation

| Component | Supported configuration |
| --- | --- |
| OS | Linux |
| Python | 3.11 or newer; 3.11 and 3.12 are declared project targets |
| Accelerator | One NVIDIA CUDA GPU for real training; CPU is supported only by the offline debug config |
| JAX | `0.10.x` with the pip-managed CUDA 13 runtime |
| Megalodon-JAX | `>=0.2.2,<0.3` |

A 24 GB consumer GPU is the planning target for the shipped long-run recipes. Short measured probes use substantially less in-use memory, but a completed 24 GB / RTX 4090 baseline is still pending; the recipe table separates the measured fit checks from that target.

Create an isolated environment and install the editable package:

```bash
mamba create -n chomp python=3.12 pip -y
conda activate chomp
git clone https://github.com/pszemraj/chomp.git
cd chomp
pip install -e .
```

The project dependency `jax[cuda13]` installs matching JAX, jaxlib, CUDA plugin, CUDA runtime, and cuDNN wheels. A separate local CUDA toolkit is not required. The NVIDIA driver must support CUDA 13; see the [official JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for current driver requirements.

Verify the installation before starting a run:

```bash
nvidia-smi
python -c "import jax; print('JAX', jax.__version__); print(jax.devices()); assert jax.devices()[0].platform == 'gpu'"
```

Success means `nvidia-smi` lists the intended GPU and JAX prints a device similar to `[CudaDevice(id=0)]` without raising the assertion. If it prints `CpuDevice`, stop and fix the installation rather than enabling CPU fallback for a real run.

Chomp does not shard across multiple visible GPUs. Unsharded arrays use JAX's first default device and the others remain idle; select one explicitly when needed:

```bash
CUDA_VISIBLE_DEVICES=1 chomp train configs/zyda2_smoke.yaml
```

## Quick start and success criteria

- `chomp train configs/debug_smoke.yaml` — expect `65,536` parameters, ten finite steps in about three CPU seconds, loss near `5.69 → 5.58`, and a final run directory.
- `chomp train configs/smoldata_mix_100m_2048.yaml --run-dir runs/my_run` — expect `113,854,464` parameters, a 1-3 minute first compile, initial loss around `10-11`, then metrics every 25 steps; add `--resume latest` to continue a checkpoint.

Additional checks:

- Add `--dry-run` to either command to execute one finite update without W&B or checkpoint saving; success prints `[chomp] dry-run complete`.
- `chomp train configs/zyda2_smoke.yaml` checks the real Hub/tokenizer/GPU path in five steps; the measured RTX 5090 run took 84 seconds, peaked at `0.4 GB`, and held loss near `10.48` as expected for this sanity-only run.

With unchanged config under the same code/runtime, resume restores parameters, optimizer state, RNG, and data position exactly, so the resumed run sees the same batches as an uninterrupted run. Bit-identical GPU arithmetic additionally requires the deterministic-kernel setting described in [Checkpointing](docs/checkpointing.md#scope-of-exactness).

## Shipped recipes and measured expectations

| Config | Parameters | Effective loss tokens | Peak VRAM | Wall time | Final loss | Data requirements |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| [`debug_smoke.yaml`](configs/debug_smoke.yaml) | 65.5K | 4,560 measured | CPU | 3 s measured | 5.576 measured | Offline local text and byte tokenizer |
| [`zyda2_smoke.yaml`](configs/zyda2_smoke.yaml) | 8.19M dummy | 2,530 measured | 0.4 GB measured | 84 s measured on RTX 5090 | 10.478 sanity-only | Public Zyda-2 `sample-100BT`; network required |
| [`smoldata_mix_100m_2048.yaml`](configs/smoldata_mix_100m_2048.yaml) | 113.85M | up to 3.2752B | 3.6 GB one-step probe | **TBD:** completed RTX 4090 baseline | **TBD:** completed baseline | Public 50/30/20 FinePDFs-Edu, DCLM, FineWeb-Edu mixture; network required |
| [`zyda2_200m_2048.yaml`](configs/zyda2_200m_2048.yaml) | 187.99M | up to 3.2752B | 4.9 GB one-step probe | **TBD:** completed RTX 4090 baseline | **TBD:** completed baseline | Public Zyda-2 `sample-100BT`; network required |

Measurements were taken on 2026-07-21 with JAX 0.10.2. GPU smoke and dry-run probes used an RTX 5090; the offline smoke used the development host CPU. The 114M and 188M VRAM values are reported in-use peaks from one-step local-data probes with the shipped model and batch shapes, not completed Hub-backed runs. They are useful fit checks, not guarantees.

The long recipes process up to ~3.28B target positions; `tokens_seen` in the final metrics row is authoritative.[^target-positions] Full-run RTX 4090 wall time and final loss remain visibly pending until a completed baseline exists rather than being extrapolated from a short probe.

Both production datasets use `datasets.load_dataset(..., streaming=True)`: Chomp reads remote Parquet shards on demand and does not download the complete corpus before training. The mixed corpus is roughly 268 GB of remote Parquet, but a run transfers only the shards and byte ranges it consumes. Expect sustained network use plus small tokenizer/metadata caches under `~/.cache/huggingface`; inspect local cache growth with `du -sh ~/.cache/huggingface`.

[^target-positions]: Config-derived maximum: `100,000 steps × 8 accumulation slices × 2 sequences × (2,048 - 1) = 3,275,200,000`; boundary, EOS, and padding masks reduce the effective count.

## Monitoring and run directories

Training prints a compact line to stdout at the first step, every `train.log_every` steps, and evaluation steps. The durable machine-readable record is append-only JSONL:

```bash
tail -f runs/my_run/metrics.jsonl
```

See [Training metrics](docs/training.md#metrics) for the JSONL schema and console cadence.

A typical run directory is:

```text
runs/my_run/
├── config_original.yaml       # authored input config
├── config_resolved.json       # resolved config plus executed Megalodon-JAX identity
├── metrics.jsonl              # append-only training/eval/event rows
├── train.log                  # Python and dependency logs
├── tokenizer/                 # run-pinned tokenizer files and execution manifest
└── checkpoints/
    └── 2500/
        ├── train_state/       # parameters, optimizer state, RNG, step
        ├── data_state/        # exact iterator position
        └── meta/              # schema, config/data, tokens_seen, backend/tokenizer identities
```

Checkpoint cadence, retention, disk behavior, and resume policy are documented in [Checkpointing](docs/checkpointing.md). W&B is optional; after `wandb login`, enable it with `-o logging.wandb.enabled=true` while retaining `metrics.jsonl` locally.

## Writing a custom config

Every section has defaults. This minimal experiment overrides only its model shape, source, duration, context, and optimizer:

```yaml
model:
  model_dim: 512
  num_layers: 8
  num_heads: 1
  z_dim: 128
  value_dim: 1024
  ffn_hidden_dim: 2048

data:
  hf_dataset: HuggingFaceFW/finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT-shuffled
  hf_name: default

train:
  steps: 1000
  seq_len: 1024

optim:
  name: muon
  lr: 3.0e-4
  warmup_steps: 100
```

Save it as `my_experiment.yaml` and run `chomp train my_experiment.yaml --run-dir /tmp/my_experiment_dry --dry-run`; expect a parameter count, one finite update, and `[chomp] dry-run complete`. This exact example has been executed, not only parsed. See the annotated [Config Reference](docs/config-reference.yaml) for every remaining key and default.

## Generation and export

Pass a run directory to generate from its latest retained checkpoint:

```bash
chomp generate runs/my_run --prompt "Hello world" --max-tokens 64 --temperature 0
```

Success prints the resolved checkpoint path, restores model parameters, and emits prompt/generated panels. `--temperature 0` is greedy; sampling defaults to temperature `1.0` and seed `42`, with optional `--top-k` and `--top-p`. You may also pass a checkpoint root or an exact step directory instead of the run directory.

Chomp currently has no built-in Hugging Face Transformers or safetensors weight export. Training checkpoints are Orbax pytrees interpreted by Megalodon-JAX; the tokenizer itself is saved in Hugging Face format under the run directory. Export requires a model-specific conversion path and should not be assumed from the presence of `save_pretrained` tokenizer files.

## Troubleshooting

- **JAX sees CPU or no device:** Run the install verification from the same environment. If `nvidia-smi` works but JAX prints `CpuDevice`, inspect `pip show jax jaxlib jax-cuda13-plugin`, reinstall with `pip install --upgrade "jax[cuda13]>=0.10.2,<0.11"`, and rerun the device assertion. Never use `train.allow_cpu=true` for a production recipe.
- **Driver, CUDA plugin, or cuDNN failure:** Fix `nvidia-smi` or update the driver first. If JAX reports incompatible CUDA/cuDNN libraries, try a clean shell with `LD_LIBRARY_PATH` unset; do not mix `jax[cuda13]` with a separately installed CPU-only jaxlib.
- **Out of memory:** Lower the microbatch and raise accumulation with `-o train.batch_size=1 -o train.grad_accum=16`; try `-o model.loss_chunk_size=256` if the vocabulary projection is the peak. The long recipes already use activation checkpointing. `XLA_PYTHON_CLIENT_PREALLOCATE=false` may help when sharing a GPU but does not reduce true peak memory.
- **Hub startup is slow or fails:** Real-data runs need outbound HTTPS throughout training. `HF_TOKEN` is optional but can help with rate limits; first startup may spend over a minute resolving revisions, downloading the tokenizer, and opening remote Parquet. Chomp reports failures rather than substituting another dataset.
- **Run directory exists:** Fresh runs refuse to clobber it. Choose another `--run-dir`, or use `--resume latest` only to continue its checkpoints; branch an older checkpoint into a separate, single-writer directory.

## Documentation

- [Config Reference](docs/config-reference.yaml): per-key types, defaults, constraints, and interactions
- [Training](docs/training.md): train step behavior, generation, and metrics
- [Data Pipeline](docs/data_pipeline.md): stream-to-batch path and eval-set construction
- [Packing](docs/packing.md): packing strategy and boundary-masking semantics
- [Optimization](docs/optimization.md): optimizer behavior and Muon sweep guidance
- [Checkpointing](docs/checkpointing.md): save/restore/resume contract and exactness scope
- [Development Guide](docs/dev.md): lint, format, test workflow, and local config conventions

## License and citation

Chomp is licensed under the [Apache License 2.0](LICENSE).

If Chomp is useful in published work, cite the software and the [Megalodon architecture paper](https://arxiv.org/abs/2404.08801):

```bibtex
@software{chomp2026,
  title  = {chomp: A Single-GPU Megalodon-JAX Pretraining Harness},
  author = {{chomp contributors}},
  year   = {2026},
  url    = {https://github.com/pszemraj/chomp},
  license = {Apache-2.0}
}
```
