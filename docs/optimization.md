# Optimization and Optimizers

This document focuses on optimizer behavior in the training harness, with a
special emphasis on Muon support and recent sweep results.

## Supported optimizers

`optim.name` selects the optimizer:

- `adamw` (default): standard AdamW on all parameters.
- `muon`: Muon on a safe whitelist of projection weight matrices, AdamW on the
  rest.

For `optim.name=muon`, the harness uses explicit parameter partitioning:

- Muon is applied only to matmul-style projection weights (for Megalodon:
  `attn.wz/wv/wr/wh1/wh2`, `ffn.fc1/fc2/fc3`, and `lm_head`).
- AdamW is applied everywhere else (including embeddings, norms, and CEMA
  parameters).

This avoids the common footgun where non-Muon parameters silently become NadamW
(or receive Muon updates) due to optimizer coupling.

## Why Muon needs special handling

Optax's Muon lives in `optax.contrib` and is designed for matrix parameters.
Megalodon includes several 2D tensors that are not matmul weights, so Muon
selection must be path-aware rather than "all 2D tensors" by default.

Muon also typically operates in a very different step-size regime than AdamW.
In practice that means:

- `optim.lr` is treated as the AdamW learning rate.
- Muon's effective learning rate is `optim.lr * optim.muon.lr_scale`.
- Muon-specific scaling options (like `optim.muon.consistent_rms`) can materially
  change what `optim.muon.lr_scale` values are stable.

## Muon sweep: 1000-step comparison (current state)

To ground the defaults in something concrete, we ran a small but controlled
sweep on `configs/zyda2_100m_2048.yaml`:

- Train steps: 1000
- Eval every: 250
- W&B: disabled
- Checkpointing: disabled
- Eval split: `train` (for fast, consistent comparisons)

Command pattern (example):

```bash
conda run --name mega-jax chomp train configs/zyda2_100m_2048.yaml \
  -o optim.name=muon \
  -o optim.muon.lr_scale=100 \
  -o optim.muon.consistent_rms=null \
  -o train.steps=1000 \
  -o train.eval_every=250 \
  -o logging.wandb.enabled=false \
  -o checkpoint.enabled=false \
  -o data.hf_eval_split=train
```

### Results summary

All values below are from the final log at step 1000.

| Optimizer | Muon scale | consistent_rms | Final train loss | Eval loss | Avg tokens/sec | Run dir |
|---|---:|---:|---:|---:|---:|---|
| AdamW | - | - | 4.8252 | 4.7304 | 59,896 | `runs/chomp/20260126_183922_zyda2_100m_2048` |
| Muon | 200 | null | 4.4544 | 4.3743 | 53,598 | `runs/chomp/20260126_215603_zyda2_100m_2048` |
| Muon | 100 | null | **4.3762** | **4.2914** | 53,000 | `runs/chomp/20260126_222855_zyda2_100m_2048` |
| Muon | 200 | 0.2 | 4.8969 | 4.8033 | 52,669 | `runs/chomp/20260126_221507_zyda2_100m_2048` |
| Muon | 100 | 0.2 | 5.4423 | 5.3353 | 53,273 | `runs/chomp/20260126_224307_zyda2_100m_2048` |

### Takeaways

- Muon clearly beats AdamW at 1000 steps in this setup.
- `optim.muon.consistent_rms=0.2` looks harmful right now.
- A Muon scale of `100` with `consistent_rms=null` is the strongest of the
  tested settings.

## Current recommended defaults

Based on the sweep above, the defaults are now:

- `optim.muon.lr_scale: 100.0`
- `optim.muon.consistent_rms: null`

These defaults only matter when `optim.name=muon`.

## Next steps and good follow-up sweeps

If you want to iterate further, here are the highest-value next experiments:

1) Sweep Muon LR scale around the new default

- Try: 80, 90, 100, 110, 125, 150
- Keep `optim.muon.consistent_rms=null` while tuning scale.

2) Revisit `consistent_rms` after scale tuning

- Try enabling it (`0.2`) only after identifying a stable LR scale.
- If you enable it, re-tune `optim.muon.lr_scale` from scratch.

3) Adam-side tuning for the non-Muon group

Muon runs still depend heavily on the AdamW group. Consider experimenting with:

- `optim.adam.b2=0.95` (common for LLMs)
- Adam weight decay masking (for example, excluding embeddings)

4) Repeat the sweep on a second config

- For example: a slightly larger model or different packing mode
- Avoid drawing conclusions from a single config

## Notes and cautions

- These are short-horizon results (1000 steps). They are useful for direction
  finding but are not definitive pretraining conclusions.
- Optimizer behavior can change meaningfully when schedule horizons, packing
  policies, or parameter sharding strategies change.
- If you resume from checkpoints, treat schedule horizons and effective
  optimizer settings as part of the run identity.
