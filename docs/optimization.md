# Optimization and Optimizers

This document focuses on optimizer behavior in the training harness, with a
special emphasis on Muon support and recent sweep results.

## Scope

This page is the home for optimizer behavior and tuning guidance.

- For field-level defaults/types: `docs/config-reference.md` (`optim.*`)
- For train-step runtime behavior and metrics: `docs/training.md`

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
- When `optim.muon.consistent_rms=null`, we skip Muon shape scaling
  (`scale_by_shape`) to preserve the earlier Muon-only behavior.

## Muon sweep: 10k-step comparison (current state)

To ground the defaults in something concrete, we ran a controlled 10k-step
comparison on a 200M Megalodon config (see
`configs/custom/muon-lr-scale-10k/*.yaml`):

- Train steps: 10,000
- Eval every: 1,000
- `optim.muon.consistent_rms=null` (no shape scaling)
- W&B project: `muon-lr-scale-10k`

Command pattern (example):

```bash
conda run --name mega-jax chomp train configs/custom/muon-lr-scale-10k/muon_lr100_10k.yaml
```

### Results summary

All values below are eval loss at step 10,000 (lower is better).

| Optimizer | Muon scale | consistent_rms | Eval loss @ 10k |
|---|---:|---:|---:|
| AdamW | - | - | 3.50916 |
| Muon | 150 | null | 3.26316 |
| Muon | 100 | null | **3.25314** |

### Takeaways

- Muon clearly beats AdamW at 10k steps in this setup.
- `optim.muon.lr_scale=100` slightly edges out `150`.
- We continue to keep `optim.muon.consistent_rms=null` until a focused sweep
  shows a benefit.

## Current recommended defaults

Based on the sweep above, the defaults are now:

- `optim.muon.lr_scale: 100.0`
- `optim.muon.consistent_rms: null`

These defaults only matter when `optim.name=muon`.

## Next steps and good follow-up sweeps

If you want to iterate further, here are the highest-value next experiments:

1) Sweep Muon LR scale around the new default (if you want to refine)

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

- These are still short-horizon results (10k steps). They are useful for
  direction finding but are not definitive pretraining conclusions.
- Optimizer behavior can change meaningfully when schedule horizons, packing
  policies, or parameter sharding strategies change.
- If you resume from checkpoints, treat schedule horizons and effective
  optimizer settings as part of the run identity.
