# Configuration

chomp uses frozen dataclasses for configuration. YAML files map directly to the
config tree, and dot-path overrides are supported from the CLI.

For the full field-by-field reference (types, defaults, constraints, examples),
see `docs/config-reference.qmd`.

## Loading and overrides

```bash
chomp-train configs/debug_smoke.yaml \
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
- `model.init_mode` (default: `he`)
- `model.pad_token_id` must differ from `model.eos_token_id` (enforced). Megalodon
  zero-masks pad embeddings and the training loop masks pad positions in loss;
  if pad==eos, EOS tokens get zeroed and masked, breaking supervision.

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
- `optim.warmup_steps`, `optim.min_lr_ratio` (decay runs across `train.steps`)

## Checkpointing and logging

- `checkpoint.enabled`, `checkpoint.save_every`, `checkpoint.max_to_keep`,
  `checkpoint.async_save`
- `logging.run_dir`, `logging.metrics_file`, `logging.level`
- `logging.console_use_rich`, `logging.log_file`
- `logging.wandb.enabled`, `logging.wandb.project`, `logging.wandb.entity`
- `logging.wandb.run_name`, `logging.wandb.mode`, `logging.wandb.tags`

Console output is rate-limited by `train.log_every`. Noisy third-party INFO
logs are suppressed on the console but still captured in the run log file
(`logging.log_file`, created under the run directory).
