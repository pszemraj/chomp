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

A 24 GB consumer GPU is the planning target for the maintained 100M and 200M recipes. The 500M and 1B files are upper-capacity templates: their exact forward/backward dry runs fit the 32 GB RTX 5090 used here, but sustained training and a 24 GB fit are not established. Measure on the intended GPU before starting a long run.

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
CUDA_VISIBLE_DEVICES=1 chomp train configs/dev/hf_streaming_smoke.yaml
```

## Quick start and success criteria

- `chomp train configs/dev/offline_cpu_smoke.yaml` — expect `65,536` parameters, ten finite steps in about three CPU seconds, loss near `5.69 → 5.58`, and a final run directory without network or GPU access.
- `chomp train configs/pretrain/megalodon_100m_2048.yaml --run-dir runs/my_run` — expect `113,854,464` parameters, a 1-3 minute first compile, initial loss around `10-11`, then metrics every 25 steps; add `--resume latest` to continue a checkpoint.

Additional checks:

- Add `--dry-run` to either command to execute one finite update without W&B or checkpoint saving; success prints `[chomp] dry-run complete`.
- `chomp train configs/dev/hf_streaming_smoke.yaml` checks the real Hub/tokenizer/GPU path in five steps while keeping DummyLM narrow; use a pretrain recipe with `--dry-run` when the real Megalodon compile/update path is the target.

With unchanged config under the same code/runtime, resume restores parameters, optimizer state, RNG, and data position exactly, so the resumed run sees the same batches as an uninterrupted run. Bit-identical GPU arithmetic additionally requires the deterministic-kernel setting described in [Checkpointing](docs/checkpointing.md#scope-of-exactness).

## Shipped recipes and measured expectations

Development checks live under `configs/dev/`; maintained training recipes live under `configs/pretrain/`. The scale labels are intentionally approximate, while the parameter column is the exact constructed count at the authored 32,128-token vocabulary.

| Config | Parameters | Maximum target positions | Peak VRAM | Wall time / evidence | Final loss | Data requirements |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| [`offline_cpu_smoke.yaml`](configs/dev/offline_cpu_smoke.yaml) | 65.5K | 5,080 | CPU | ~3 s measured | 5.576 measured | Offline local text and byte tokenizer |
| [`hf_streaming_smoke.yaml`](configs/dev/hf_streaming_smoke.yaml) | 2.048M dummy after tokenizer resolution | 2,540 | 0.192 GB measured | First update 9.765s; five steps passed | 10.542 sanity-only | Public Zyda-2 `sample-100BT`; network required |
| [`megalodon_100m_2048.yaml`](configs/pretrain/megalodon_100m_2048.yaml) | 113.85M | 3.2752B | 4.657 GB measured | 12m19s end-to-end / 500-step validation on RTX 5090 | 5.187 eval at step 500 | Public 50/30/20 FinePDFs-Edu, DCLM, FineWeb-Edu mixture; network required |
| [`megalodon_200m_2048.yaml`](configs/pretrain/megalodon_200m_2048.yaml) | 187.99M | 4.0940B | 5.0 GB measured | First dry-run step 92.869s | 10.847 sanity-only | Same streamed mixture |
| [`megalodon_500m_2048.yaml`](configs/pretrain/megalodon_500m_2048.yaml) | 513.67M | 10.4806B | 11.1 GB measured | First dry-run step 128.258s | 10.845 sanity-only | Same streamed mixture |
| [`megalodon_1b_2048.yaml`](configs/pretrain/megalodon_1b_2048.yaml) | 974.62M | 19.6512B | 20.9 GB measured | First dry-run step 145.535s | 10.808 sanity-only | Same streamed mixture |

The 200M model preserves the released `mega200M` depth and projection ratios. The 500M model scales those ratios to width 1,536 and 16 layers. The 1B model uses the released 1.3B widths and projection ratios with 18 rather than 24 layers. All four tie the input/output embedding and use 512-token fixed attention chunks, so their exact counts and local-attention geometry intentionally differ from the released untied 2,048-chunk presets. The 500M/1B templates use the memory-saving settings shown in their linked configs; read the [field contracts](docs/config-reference.yaml) and [resume compatibility policy](docs/checkpointing.md#resume-compatibility-checks) before changing them.

Measurements use JAX 0.10.2 and an RTX 5090; the offline smoke uses the development host CPU. The clean-exit 500-step 114M validation was measured on 2026-08-01 with the shipped `2048 × batch 2 × accumulation 8` model/data geometry, BF16 compute, FP32 parameters and accumulation, strict bin-packed segment resets, Muon, real streamed mixture data, scheduled 64-document evaluation, and schema-3 async checkpoints at steps 250 and 500. Only the duration, warmup, eval/checkpoint cadence, and eval sample cap were shortened. The source shuffle retained its production 200,000-document / 512 MiB limits and reached 64,503 documents / 536.9 MB; the packed-row shuffle retained 4,096 rows / 8.39M tokens. The first update took 120.2 seconds, including 85.5 seconds of compile/device work and 34.7 seconds of data wait. Steady steps had 1.100-second median wall time and 29,675 median valid tokens/s, process allocator peak was 4.657 GB, and host/device valid-target counts agreed through 16,318,387 targets. Training loss moved from 10.787 to 4.767 and eval loss from 5.563 at step 250 to 5.187 at step 500. The run exited cleanly and a strict same-target restore from step 500 also succeeded. Those losses belong to the deliberately compressed validation schedule, not the maintained 100M schedule.

The 200M/500M/1B rows are exact-config `--dry-run` measurements from 2026-08-02 on the same RTX 5090, with production-sized streamed shuffle buffers and `XLA_PYTHON_CLIENT_PREALLOCATE=false`. Each executed data loading, packing, the complete forward/backward graph, gradient checks, and optimizer-state update; the reported first step includes compilation and has zero scheduled learning rate at the start of warmup, so it is fit evidence rather than steady-state throughput or a nonzero parameter trajectory. End-to-end dry-run wall spans were about 2m40s, 3m05s, and 3m30s respectively. XLA reported a 17.10 GiB unoptimized and 16.81 GiB rematerialized compiler peak estimate for the 1B graph; Chomp observed a 20.9 GB process allocator peak. These measurements are evidence on one 32 GB machine, not guarantees. Sustained baselines, 24 GB/RTX 4090 fits for the larger templates, and full-schedule losses remain pending rather than being extrapolated.

The maintained schedules span 3.28B to 19.65B maximum target positions, with at least roughly 20 positions per parameter before packing masks. `tokens_seen` in the final metrics row is authoritative.[^target-positions]

Every checked-in HF scenario uses `datasets.load_dataset(..., streaming=True)`: Chomp reads remote Parquet shards on demand and does not download the complete corpus before training. The pretrain mixture is roughly 268 GB of remote Parquet, but a run transfers only the shards and byte ranges it consumes. Expect sustained network use plus small tokenizer/metadata caches under `~/.cache/huggingface`; inspect local cache growth with `du -sh ~/.cache/huggingface`.

[^target-positions]: Config-derived maximum: `steps × grad_accum × batch_size × (seq_len - 1)`; boundary, EOS, and padding masks reduce the effective count.

## Monitoring and run directories

Training prints compact progress to stdout and keeps an append-only JSONL record:

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

Evaluation behavior is summarized under [Training evaluation](docs/training.md#evaluation), with continuation rules under [resume compatibility](docs/checkpointing.md#resume-compatibility-checks). Use the annotated [Config Reference](docs/config-reference.yaml) for the canonical evaluation and Megalodon attention contracts.

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
- **Out of memory:** Lower the microbatch and raise accumulation with `-o train.batch_size=1 -o train.grad_accum=16`, keeping their product fixed so the tokens per optimizer step do not change. Every pretrain recipe already uses activation checkpointing. `XLA_PYTHON_CLIENT_PREALLOCATE=false` may help when sharing a GPU but does not reduce true peak memory.
- **Hub startup is slow or fails:** Real-data runs need outbound HTTPS throughout training. `HF_TOKEN` is optional but can help with rate limits; first startup may spend over a minute resolving revisions, downloading the tokenizer, and opening remote Parquet. Chomp reports failures rather than substituting another dataset.
- **Run directory exists:** Fresh runs refuse to clobber it. Choose another `--run-dir`, or use `--resume latest` only to continue its checkpoints; branch an older checkpoint into a separate, single-writer directory.
- **Resume rejects a config change you meant to make:** `--resume` continues one training history and requires the data pipeline to match. To start a new run from an existing model instead, use `--init-from <run-or-step-dir>`, which loads parameters only and leaves optimizer state, schedule, and corpus position fresh. See [Checkpointing: warm start](docs/checkpointing.md#warm-start---init-from).

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
