# Configuration

chomp uses frozen dataclasses for configuration. YAML files map directly to the
config tree, and dot-path overrides are supported from the CLI.

## Loading and overrides

```bash
chomp-train --config configs/debug_smoke.yaml \
  --override train.steps=200 \
  --override train.batch_size=4
```

Unknown keys or invalid values fail fast during validation.

## Top-level config tree

- `model`: model architecture and dtype policy
- `data`: dataset, tokenizer, and packing behavior
- `train`: batch sizes, sequence length, steps
- `optim`: AdamW + schedule
- `checkpoint`: Orbax save/restore settings
- `logging`: run directory + metrics
- `debug`: NaN checks and device assertions

## Model

Key fields (megalodon backend):

- `model.backend`: `megalodon` or `dummy`
- `model.vocab_size`
- `model.model_dim`, `model.num_layers`, `model.num_heads`
- `model.chunk_size` (must divide `train.seq_len`)
- `model.segment_masking` (block-diagonal attention on packed segments)

Dtypes are configured as strings and validated:

- `model.param_dtype`, `model.compute_dtype`, `model.accum_dtype`, `model.softmax_dtype`

## Data + tokenizer

Data source:

- `data.backend`: `hf` or `local_text`
- `data.hf_dataset`, `data.hf_name`, `data.hf_split`, `data.text_key`
- `data.hf_eval_split`, `data.max_eval_samples`
- `data.shuffle`, `data.shuffle_buffer_size`, `data.seed`, `data.repeat`

Tokenizer:

- `data.tokenizer.kind`: `hf` or `byte`
- `data.tokenizer.hf_name_or_path`
- `data.tokenizer.vocab_size_multiple`
- `data.tokenizer.add_bos`, `data.tokenizer.add_eos`

Packing:

- `data.packing_mode`: `sequential` or `bin`
- `data.packing_buffer_docs`
- `data.packing_max_docs_per_bin`
- `data.mask_boundary_loss`, `data.train_on_eos`
- `data.grain_prefetch`

## Train and optimizer

Train:

- `train.steps`
- `train.batch_size`, `train.grad_accum`, `train.seq_len`
- `train.allow_cpu` (fail fast if CPU)
- `train.deterministic` (optional override)
- `train.eval_every` (0 disables periodic eval)

Optimizer:

- `optim.lr`, `optim.weight_decay`
- `optim.grad_clip_norm`
- `optim.warmup_steps`, `optim.total_steps`

## Checkpointing and logging

- `checkpoint.enabled`, `checkpoint.save_every`, `checkpoint.max_to_keep`,
  `checkpoint.async_save`
- `logging.run_dir`, `logging.metrics_file`, `logging.level`
- `logging.wandb_enabled`, `logging.wandb_project`, `logging.wandb_entity`
- `logging.wandb_run_name`, `logging.wandb_mode`, `logging.wandb_tags`
