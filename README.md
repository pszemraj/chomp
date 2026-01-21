# chomp

A **minimal**, single-GPU-first JAX/Equinox pretraining harness designed to train
**Megalodon-JAX** models.

chomp's philosophy is intentionally boring:
- fixed shapes → compile once
- arrays-only TrainState → checkpointing is straightforward
- real streaming data → no synthetic crutches

Implemented so far (this draft):
- Phases 0–2: config + model integration + compiled train_step with `lax.scan` grad accumulation
- Phase 3: Orbax checkpointing + resume contract (train_state + data iterator state)
- Initial Phase 4 chunk: Hugging Face streaming → tokenize + pack → fixed [A,B,T] batches (no Grain yet)

## Install

### 1) Install JAX + jaxlib
JAX wheels are platform- and CUDA-specific. Follow the official instructions:
- https://docs.jax.dev/en/latest/installation.html

### 2) Install `chomp`

```bash
pip install -e .
```

### 3) Install `megalodon-jax` (optional, for real models)

```bash
pip install -e /path/to/megalodon-jax
```

or:

```bash
pip install -e '.[megalodon]'
```

### 4) Optional: Hugging Face tokenizer support

If you want to use `data.tokenizer.kind: hf`:

```bash
pip install -e '.[hf]'
```

## Run

### Smoke test (offline)

This uses `data.backend: local_text`, which still exercises tokenize+pack but avoids network.

```bash
chomp-train --config configs/debug_smoke.yaml
```

### Real streaming dataset: Zyphra/Zyda-2 (sample-100BT)

The default `DataConfig` points at Zyda-2 sample-100BT streaming:

```python
import datasets
ds = datasets.load_dataset("Zyphra/Zyda-2", name="sample-100BT", split="train", streaming=True)
```

In chomp YAML, that looks like:

```yaml
data:
  backend: hf
  hf_dataset: Zyphra/Zyda-2
  hf_name: sample-100BT
  hf_split: train
  text_key: text
```

### Resume from checkpoint

Resume requires setting `logging.run_dir` (either in YAML or via CLI) to an existing run directory.

```bash
# 1) Start a run and save checkpoints
chomp-train --config configs/debug_smoke.yaml --run-dir runs/chomp/debug_run

# 2) Resume from the latest checkpoint
chomp-train --config configs/debug_smoke.yaml --run-dir runs/chomp/debug_run --resume latest
```

## Design principles

- **Compile once**: fixed shapes; no dynamic padding; grad accumulation inside the compiled step.
- **Arrays-only TrainState**: checkpoint-friendly; no hidden Python objects in the jitted state.
- **Resume is a contract**: train_state *and* data iterator state are persisted.
- **Training never uses cache**: cache is an inference concern.
- **Tokenizer compatibility**: HF tokenizer vocab size + special token IDs must match `model.*` (fail-fast).
