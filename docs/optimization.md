# Optimization and Optimizers

Optimizer behavior in the training harness, with emphasis on Muon support and recent sweep results.

Related: [Config Reference](config-reference.yaml) (`optim.*`), [Training Loop](training.md).

## Supported optimizers

Chomp provides AdamW over all parameters or Muon over a safe projection whitelist with AdamW over the remainder. Selection, defaults, and structural-resume behavior are canonical under [`optim.*`](config-reference.yaml).

The model adapter classifies known model arrays for optimizer routing and weight decay. Unrecognized arrays use AdamW without weight decay.

Megalodon-JAX 0.2.2 derives RoPE frequencies from static dimension/base values during the model call, so no rotary array appears in the model tree, optimizer state, or checkpoint. Trainable CEMA coefficients remain parameters.

For `optim.name=muon`, the harness uses explicit parameter partitioning:

- Muon is applied only to matmul-style projection weights (for Megalodon: `attn.wz/wv/wr/wh1/wh2`, `ffn.fc1/fc2/fc3`, and `lm_head`).
- AdamW is applied everywhere else (including embeddings, norms, and CEMA parameters).

AdamW decay is also path-aware. It applies to token embeddings and dense projection weights. Biases, norm/scale parameters, normalized-FFN residual scales, attention affine offsets, and all CEMA coefficients receive no decoupled weight decay.

The normal route excludes embeddings and non-matmul matrices. Experimental eligibility expansions, the DummyLM exception, and their structural-resume implications are defined under [`optim.muon.*`](config-reference.yaml).

## Why Muon needs special handling

Optax's Muon lives in `optax.contrib` and is designed for matrix parameters. Megalodon includes several 2D tensors that are not matmul weights, so Muon selection must be path-aware rather than "all 2D tensors" by default.

Muon operates in a different step-size regime from AdamW: its effective learning rate is `optim.lr * optim.muon.lr_scale`, and optional RMS/shape scaling can materially change the stable multiplier. Exact defaults, recommendations, and optimizer-structure implications live in the [Config Reference](config-reference.yaml).

The maintained pretrain recipes make the measured policy explicit: `optim.lr=3e-4`, `optim.muon.lr_scale=100`, and `optim.muon.consistent_rms=null`. The 500M/1B configs inherit this policy as a documented starting point, not as evidence from a scale-specific sweep.

## Muon sweep: 10k-step comparison

A controlled 10k-step comparison used untracked local 200M Megalodon configs under `configs/custom/`:

- Train steps: 10,000
- Eval every: 1,000
- `optim.muon.consistent_rms=null` (no shape scaling)
- W&B project: `muon-lr-scale-10k`

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
- We continue to keep `optim.muon.consistent_rms=null` until a focused sweep shows a benefit.

## Notes and cautions

- These are still short-horizon results (10k steps). They are useful for direction finding but are not definitive pretraining conclusions.
- Optimizer behavior can change meaningfully when schedule horizons, packing policies, or parameter sharding strategies change.
- Schedule horizons and effective optimizer settings are run semantics; use their [per-key contracts](config-reference.yaml) together with [resume compatibility](checkpointing.md#resume-compatibility-checks).
