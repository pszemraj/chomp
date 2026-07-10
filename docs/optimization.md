# Optimization and Optimizers

Optimizer behavior in the training harness, with emphasis on Muon support and
recent sweep results.

Related: [Config Reference](config-reference.yaml) (`optim.*`),
[Training Loop](training.md).

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

By default, the projection whitelist excludes embeddings and other non-matmul
matrices. `optim.muon.allow_tied_embed` adds the token embedding, while
`optim.muon.allow_all_2d` replaces the whitelist with every 2D tensor.

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

## Muon sweep: 10k-step comparison

A controlled 10k-step comparison used a 200M Megalodon config (see
[`configs/custom/muon-lr-scale-10k/`](../configs/custom/muon-lr-scale-10k/)):

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
| --------- | ---------: | -------------: | --------------: |
| AdamW     |          - |              - |         3.50916 |
| Muon      |        150 |           null |         3.26316 |
| Muon      |        100 |           null |       **3.25314** |

### Takeaways

- Muon reduced eval loss relative to AdamW in this setup.
- `optim.muon.lr_scale=100` slightly edges out `150`.
- We continue to keep `optim.muon.consistent_rms=null` until a focused sweep
  shows a benefit.

## Notes and cautions

- These are still short-horizon results (10k steps). They are useful for
  direction finding but are not definitive pretraining conclusions.
- Optimizer behavior can change meaningfully when schedule horizons, packing
  policies, or parameter sharding strategies change.
- If you resume from checkpoints, treat schedule horizons and effective
  optimizer settings as part of the run identity.
