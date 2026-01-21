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
- Phases 4–5: HF streaming → tokenize → pack → Grain pipeline → fixed [A,B,T] batches

## Install

### 1) Install JAX + jaxlib
JAX wheels are platform- and CUDA-specific. Follow the official instructions:
- https://docs.jax.dev/en/latest/installation.html

### 2) Install `chomp`

```bash
pip install -e .
```

### 3) Install `megalodon-jax` (for real models)

```bash
pip install -e /path/to/megalodon-jax
```

or:

```bash
pip install -e '.[megalodon]'
```

### Tokenizer defaults + vocab rounding

- Default HF tokenizer: `BEE-spoke-data/bpe-tokenizer-32k-smolNeoX`
- When `tokenizer.kind: hf`, chomp reads tokenizer metadata and:
  - rounds `model.vocab_size` up to `data.tokenizer.vocab_size_multiple` (default: 128)
  - auto-sets `model.{bos,eos,pad}_token_id` from the tokenizer unless disabled

Example (round to multiple of 64 instead of 128):

```yaml
data:
  tokenizer:
    kind: hf
    vocab_size_multiple: 64
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
  hf_eval_split: validation
  text_key: text
  packing_mode: bin
  packing_buffer_docs: 256
  grain_prefetch: 2
  max_eval_samples: 1000
```

### Resume from checkpoint

Resume requires setting `logging.run_dir` (either in YAML or via CLI) to an existing run directory.

```bash
# 1) Start a run and save checkpoints
chomp-train --config configs/debug_smoke.yaml --run-dir runs/chomp/debug_run

# 2) Resume from the latest checkpoint
chomp-train --config configs/debug_smoke.yaml --run-dir runs/chomp/debug_run --resume latest
```

Resume performs strict config/data compatibility checks and fails fast on mismatches.
Each run directory includes a tokenizer snapshot under `tokenizer/`.

## Design principles

- **Compile once**: fixed shapes; no dynamic padding; grad accumulation inside the compiled step.
- **Token-weighted GA**: gradient accumulation scales by valid token count (correct with masks/padding).
- **Segment masking toggle**: set `model.segment_masking` to enable/disable block-diagonal attention for packed sequences.
- **Boundary-aware loss masking**: `data.mask_boundary_loss` sets labels at segment boundaries to `-100` to avoid cross-document loss; `data.train_on_eos` controls EOS supervision.
- **Bin packing optional**: set `data.packing_mode: bin` with `data.packing_buffer_docs` to enable FFD packing (pads to fixed length).
- **Grain pipeline**: tune `data.grain_prefetch` for threaded prefetch and log `packing_utilization`.
- **Arrays-only TrainState**: checkpoint-friendly; no hidden Python objects in the jitted state.
- **Resume is a contract**: train_state *and* data iterator state are persisted.
- **Training never uses cache**: cache is an inference concern.
- **Tokenizer compatibility**: tokenizer vocab + special tokens are detected and `model.vocab_size` is aligned.
- **Fixed eval set**: `data.max_eval_samples` caches validation texts for consistent eval.
- **W&B logging**: enable with `logging.wandb_enabled=true`.

## Docs

- `docs/packing.md`: packing modes, segment IDs, and loss masking
- `docs/data_pipeline.md`: HF streaming → Grain → batch contract
- `docs/checkpointing.md`: Orbax save/restore + resume compatibility
- `docs/configuration.md`: config tree and key knobs
- `docs/dev.md`: dev log + deferred scope
- `docs/training.md`: train loop behavior and metrics
