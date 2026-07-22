# chomp

Chomp is a compact, single-GPU JAX/Equinox pretraining harness for [Megalodon-JAX](https://github.com/pszemraj/megalodon-jax). [Megalodon](https://arxiv.org/abs/2404.08801) combines recurrent CEMA memory with gated attention for efficient long-context sequence modeling.

Chomp sits between a toy training script and a distributed framework: it keeps the readable, hackable shape of nanoGPT-style code while adding resumable Hugging Face streaming, document packing, checkpointed iterator position, evaluation, generation, and the Megalodon-JAX model adapter. It deliberately does not provide multi-host training, distributed orchestration, environment attestation, or a Transformers export stack; use a distributed trainer such as [Levanter](https://github.com/stanford-crfm/levanter) when those are the actual requirements.

The project is alpha software. Correctness takes priority over backward compatibility until the first stable release, so older configs and checkpoints may occasionally require migration.

## Requirements and installation

| Component | Supported configuration |
| --- | --- |
| OS | Linux |
| Python | 3.11 or newer; 3.11 and 3.12 are declared project targets |
| Accelerator | One NVIDIA CUDA GPU for real training; CPU is supported only by the offline debug config |
| JAX | `0.10.x` with the pip-managed CUDA 13 runtime |
| Megalodon-JAX | `0.2.x` |

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

### 1. Verify the offline training path

```bash
chomp train configs/debug_smoke.yaml
```

This exercises config loading, byte tokenization, bin packing, optimization, metrics, and synchronous checkpoints without network access or a GPU. A passing run prints `65,536` parameters, ten finite step lines, and a final run directory. On the reference development host it completes in about 3 seconds and loss moves from approximately `5.69` to `5.58`:

```text
[chomp] params: 65,536
step 1 | loss 5.6931 | ... | pack 0.898
...
step 10 | loss 5.5758 | ... | pack 0.898
[chomp] run_dir: runs/chomp/<timestamp>_debug_smoke
```

Small CPU timing differences are normal. A traceback, non-finite loss, missing step 10, or missing final run directory is not.

### 2. Verify config, compilation, and one update

```bash
chomp train configs/debug_smoke.yaml --dry-run
```

A dry run builds the model and data path, executes one optimizer step, checks finite state, and exits without W&B or a checkpoint save. Success includes `[chomp] dry-run complete` and one finite step line; the debug config takes about 3 seconds on the reference host. Use a new `--run-dir` for each dry run because Chomp refuses to overwrite an existing run.

### 3. Verify the real Hub and GPU input path

```bash
chomp train configs/zyda2_smoke.yaml
```

This is a five-step GPU smoke test with the real 32K tokenizer and streamed Zyda-2 sample, but a dummy model. Both referenced datasets are currently public and ungated, so `HF_TOKEN` is optional. On an RTX 5090 with a warm local tokenizer cache, the measured run took 84 seconds end-to-end, reported a `0.4 GB` peak, and produced five finite losses around the random-init baseline:

```text
[chomp] params: 8,192,000
step 1 | loss 10.4776 | ... | peak 0.3GB
...
step 5 | loss 10.4781 | ... | peak 0.4GB
```

Allow roughly 1-2 minutes with a healthy network. Most elapsed time is Hub resolution, streaming startup, and first-step compilation; the five optimizer steps themselves are intentionally tiny. Loss is not expected to improve meaningfully in five steps.

### 4. Preflight a real recipe

```bash
chomp train configs/smoldata_mix_100m_2048.yaml --run-dir /tmp/chomp_100m_dry --dry-run
```

Success prints `113,854,464` parameters, `[chomp] dry-run complete`, and one finite step. The first compile is expected to take roughly 1-3 minutes on a high-end consumer GPU; do not use first-step throughput to estimate the full run. The command also opens the configured Hub stream, so network and evaluation-data startup add variability.

### 5. Start and resume the recommended run

```bash
chomp train configs/smoldata_mix_100m_2048.yaml --run-dir runs/my_run
chomp train configs/smoldata_mix_100m_2048.yaml --run-dir runs/my_run --resume latest
```

The fresh run should resolve the dataset and tokenizer, print `113,854,464` parameters, compile the first step, then emit compact metrics every 25 steps. Initial cross-entropy around `10-11` is plausible for a 32K-token random initialization; `NaN`, `Inf`, or repeated crashes before step 1 are failures. The resume command should print `[chomp] resumed from checkpoint step N` and continue beyond that step.

With unchanged config under the same code/runtime, resume restores parameters, optimizer state, RNG, and data position exactly, so the resumed run sees the same batches as an uninterrupted run. Bit-identical GPU arithmetic additionally requires the deterministic-kernel setting described in [Checkpointing](docs/checkpointing.md#scope-of-exactness).

## Shipped recipes and measured expectations

| Config | Parameters | Effective loss tokens | Peak VRAM | Wall time | Final loss | Data requirements |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| [`debug_smoke.yaml`](configs/debug_smoke.yaml) | 65.5K | 4,560 measured | CPU | 3 s measured | 5.576 measured | Offline local text and byte tokenizer |
| [`zyda2_smoke.yaml`](configs/zyda2_smoke.yaml) | 8.19M dummy | 2,530 measured | 0.4 GB measured | 84 s measured on RTX 5090 | 10.478 sanity-only | Public Zyda-2 `sample-100BT`; network required |
| [`smoldata_mix_100m_2048.yaml`](configs/smoldata_mix_100m_2048.yaml) | 113.85M | up to 3.2752B | 3.6 GB one-step probe | **TBD:** completed RTX 4090 baseline | **TBD:** completed baseline | Public 50/30/20 FinePDFs-Edu, DCLM, FineWeb-Edu mixture; network required |
| [`zyda2_200m_2048.yaml`](configs/zyda2_200m_2048.yaml) | 187.99M | up to 3.2752B | 4.9 GB one-step probe | **TBD:** completed RTX 4090 baseline | **TBD:** completed baseline | Public Zyda-2 `sample-100BT`; network required |

Measurements were taken on 2026-07-21 with JAX 0.10.2. GPU smoke and dry-run probes used an RTX 5090; the offline smoke used the development host CPU. The 114M and 188M VRAM values are reported in-use peaks from one-step local-data probes with the shipped model and batch shapes, not completed Hub-backed runs. They are useful fit checks, not guarantees.

The long recipes have at most `steps × grad_accum × batch_size × (seq_len - 1) = 3,275,200,000` causal target positions. Boundary, EOS, and padding masks make the effective count slightly lower; `tokens_seen` in the final metrics row is the authoritative value. Full-run RTX 4090 wall time and final loss remain visibly pending until a completed baseline exists rather than being extrapolated from a short probe.

Both production datasets use `datasets.load_dataset(..., streaming=True)`: Chomp reads remote Parquet shards on demand and does not download the complete corpus before training. The mixed corpus is roughly 268 GB of remote Parquet, but a run transfers only the shards and byte ranges it consumes. Expect sustained network use plus small tokenizer/metadata caches under `~/.cache/huggingface`; inspect local cache growth with `du -sh ~/.cache/huggingface`.

## Monitoring and run directories

Training prints a compact line to stdout at the first step, every `train.log_every` steps, and evaluation steps. The durable machine-readable record is append-only JSONL:

```bash
tail -f runs/my_run/metrics.jsonl
```

Each new line is a JSON object containing loss, gradient norm, learning rate, exact `loss_tokens`, cumulative `tokens_seen`, timing, throughput, packing utilization, and best-effort device memory. For a compact live view with `jq` installed:

```bash
watch -n 5 'tail -n 1 runs/my_run/metrics.jsonl | jq {step,loss,grad_norm,lr,tokens_seen,tokens_per_sec,peak_memory_gb}'
```

A typical run directory is:

```text
runs/my_run/
├── config_original.yaml       # authored input config
├── config_resolved.json       # resolved run-start config and derived values
├── metrics.jsonl              # append-only training/eval/event rows
├── train.log                  # Python and dependency logs
├── tokenizer/                 # run-pinned Hugging Face tokenizer
└── checkpoints/
    └── 2500/
        ├── train_state/       # parameters, optimizer state, RNG, step
        ├── data_state/        # exact iterator position
        └── meta/              # config, data fingerprint, tokens_seen
```

The long recipes save every 2,500 steps and retain the newest three checkpoints. Orbax permanently removes older checkpoints beyond `checkpoint.max_to_keep`; increase it before the run if deeper rollback matters. Full-recipe checkpoint size has not yet been baselined, and it should not be inferred from peak VRAM because serialized optimizer state has a different footprint. After the first save, measure it directly with:

```bash
du -sh runs/my_run/checkpoints/*
```

For disk planning, multiply the observed per-step size by the retention count and leave room for one save in progress.

W&B is optional and disabled by default. After `wandb login`, enable it without editing the recipe:

```bash
chomp train configs/smoldata_mix_100m_2048.yaml --run-dir runs/my_run_wandb -o logging.wandb.enabled=true
```

Success includes a W&B run URL, while `metrics.jsonl` remains the local source of truth. W&B files use `WANDB_DIR` or `./wandb`, not the Chomp run directory.

## Writing a custom config

Every section has defaults. Start by specifying only the model, source, batch geometry, and optimizer choices that define the experiment:

```yaml
model:
  model_dim: 512
  num_layers: 8
  num_heads: 1
  z_dim: 128
  value_dim: 1024
  ffn_hidden_dim: 2048
  chunk_size: 512
  use_checkpoint: true

data:
  hf_dataset: HuggingFaceFW/finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT-shuffled
  hf_name: default
  tokenizer:
    hf_name_or_path: pszemraj/bytebpe-tokenizer-32k-mlm

train:
  steps: 1000
  batch_size: 1
  seq_len: 1024
  grad_accum: 8

optim:
  name: muon
  lr: 3.0e-4
  warmup_steps: 100

checkpoint:
  save_every: 250
  max_to_keep: 2
```

Save it as `my_experiment.yaml`, then verify one complete update before committing GPU time:

```bash
chomp train my_experiment.yaml --run-dir /tmp/my_experiment_dry --dry-run
chomp train my_experiment.yaml --run-dir runs/my_experiment
```

The dry run should print a parameter count, `[chomp] dry-run complete`, and one finite step. See the annotated [Config Reference](docs/config-reference.yaml) for every key, computed default, constraint, and interaction. Use the shipped 114M recipe as the maintained full-run starting point rather than growing a custom model before the pipeline is verified.

## Generation and export

Pass a run directory to generate from its latest retained checkpoint:

```bash
chomp generate runs/my_run --prompt "Hello world" --max-tokens 64 --temperature 0
```

Success prints the resolved checkpoint path, restores model parameters, and emits prompt/generated panels. `--temperature 0` is greedy; sampling defaults to temperature `1.0` and seed `42`, with optional `--top-k` and `--top-p`. You may also pass a checkpoint root or an exact step directory instead of the run directory.

Chomp currently has no built-in Hugging Face Transformers or safetensors weight export. Training checkpoints are Orbax pytrees interpreted by Megalodon-JAX; the tokenizer itself is saved in Hugging Face format under the run directory. Export requires a model-specific conversion path and should not be assumed from the presence of `save_pretrained` tokenizer files.

## Troubleshooting

### JAX sees CPU or no device

Run the verification command from the same environment as Chomp. If `nvidia-smi` works but JAX prints `CpuDevice`, inspect the installed packages and reinstall the CUDA extra:

```bash
pip show jax jaxlib jax-cuda13-plugin
pip install --upgrade "jax[cuda13]>=0.10.2,<0.11"
```

Then rerun the device assertion. Do not set `train.allow_cpu=true` for a production recipe; it only makes the expensive silent fallback harder to notice.

### Driver, CUDA plugin, or cuDNN initialization fails

- If `nvidia-smi` fails or reports no GPU, fix device visibility or update the NVIDIA driver first.
- The pip-managed CUDA wheels should not use a conflicting system toolkit. Start a clean shell with `LD_LIBRARY_PATH` unset if JAX reports incompatible CUDA or cuDNN libraries.
- Do not mix `jax[cuda13]` with a separately installed CPU-only jaxlib. Reinstall from the project environment if package resolution drifted.

### Out of memory

For a fresh run, lower the per-microbatch batch size and raise accumulation to preserve the same 16 sequences per optimizer update:

```bash
chomp train configs/smoldata_mix_100m_2048.yaml --run-dir runs/my_run_small_batch -o train.batch_size=1 -o train.grad_accum=16
```

If the vocabulary projection is the peak, also try `-o model.loss_chunk_size=256`. The shipped long recipes already enable activation checkpointing. `XLA_PYTHON_CLIENT_PREALLOCATE=false` can help when sharing a GPU, but it does not reduce the model's true peak requirement.

### Hub startup is slow or fails

The real-data smoke and long recipes need outbound HTTPS throughout training. The datasets are public, but setting `HF_TOKEN` can still help with Hub rate limits. A first run may spend a minute or more resolving revisions, downloading the tokenizer, and opening remote Parquet shards before step 1. Authentication, schema, or network failures are reported; Chomp does not silently substitute another dataset.

### Run directory already exists

Fresh runs refuse to clobber an existing directory. Choose a new `--run-dir`, or use `--resume latest` only when continuing the checkpoints already stored there. Run directories are single-writer; branch an older checkpoint into a separate directory.

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
