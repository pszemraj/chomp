# Comma Stability Matrix (Sequential Packing, Chomp-only)

Date: 2026-02-17 to 2026-02-18

## Goal

Measure loss stability under a 2x2 matrix on Comma with unchanged stream semantics:

- `mask_boundary_loss`: `true` vs `false`
- `shuffle_buffer_size`: `10_000` vs `200_000`

All variants use `packing_mode: sequential`, identical model/optimizer seed, and the same base config family.

## Variants

- A: `mask_boundary_loss=true`, `shuffle_buffer_size=10000`
- B: `mask_boundary_loss=false`, `shuffle_buffer_size=10000`
- C: `mask_boundary_loss=true`, `shuffle_buffer_size=200000`
- D: `mask_boundary_loss=false`, `shuffle_buffer_size=200000`

Local untracked config files:

- `scratch/local_stability/comma_stability_A.yaml`
- `scratch/local_stability/comma_stability_B.yaml`
- `scratch/local_stability/comma_stability_C.yaml`
- `scratch/local_stability/comma_stability_D.yaml`

## Commands

Pilot runs executed (100-step smoke to verify instrumentation and matrix wiring):

```bash
conda run --name mega-jax chomp train scratch/local_stability/comma_stability_A.yaml \
  --override train.steps=100 --override train.eval_every=0 --override optim.warmup_steps=10 \
  --run-dir runs/dataset-tests/200m-comma-stability-A-pilot100

conda run --name mega-jax chomp train scratch/local_stability/comma_stability_B.yaml \
  --override train.steps=100 --override train.eval_every=0 --override optim.warmup_steps=10 \
  --run-dir runs/dataset-tests/200m-comma-stability-B-pilot100

conda run --name mega-jax chomp train scratch/local_stability/comma_stability_C.yaml \
  --override train.steps=100 --override train.eval_every=0 --override optim.warmup_steps=10 \
  --run-dir runs/dataset-tests/200m-comma-stability-C-pilot100

conda run --name mega-jax chomp train scratch/local_stability/comma_stability_D.yaml \
  --override train.steps=100 --override train.eval_every=0 --override optim.warmup_steps=10 \
  --run-dir runs/dataset-tests/200m-comma-stability-D-pilot100
```

Full matrix commands (target 5k steps each, sequential execution):

```bash
conda run --name mega-jax chomp train scratch/local_stability/comma_stability_A.yaml \
  --run-dir runs/dataset-tests/200m-comma-stability-A-full5k

conda run --name mega-jax chomp train scratch/local_stability/comma_stability_B.yaml \
  --run-dir runs/dataset-tests/200m-comma-stability-B-full5k

conda run --name mega-jax chomp train scratch/local_stability/comma_stability_C.yaml \
  --run-dir runs/dataset-tests/200m-comma-stability-C-full5k

conda run --name mega-jax chomp train scratch/local_stability/comma_stability_D.yaml \
  --run-dir runs/dataset-tests/200m-comma-stability-D-full5k
```

## Pilot Results (step 100)

| Variant |   loss | grad_norm | loss_tokens | boundary_transitions | docs_per_seq_mean | tokens/sec |
| ------- | -----: | --------: | ----------: | -------------------: | ----------------: | ---------: |
| A       | 4.7158 |    0.9802 |      130856 |                  152 |            3.3750 |    46287.9 |
| B       | 4.7180 |    0.9803 |      131008 |                  152 |            3.3750 |    46232.0 |
| C       | 4.6834 |    1.3681 |      130860 |                  148 |            3.3125 |    46178.4 |
| D       | 4.6851 |    1.4087 |      131008 |                  148 |            3.3125 |    43862.3 |

Run IDs:

- `runs/dataset-tests/200m-comma-stability-A-pilot100`
- `runs/dataset-tests/200m-comma-stability-B-pilot100`
- `runs/dataset-tests/200m-comma-stability-C-pilot100`
- `runs/dataset-tests/200m-comma-stability-D-pilot100`

## Comparative Findings (Pilot)

- Boundary masking predictably reduces `loss_tokens` relative to `mask_boundary_loss=false` at matched shuffle settings.
- Increasing shuffle buffer from `10k` to `200k` changes document-density composition (`docs_per_seq_mean`, `boundary_transitions`) even under sequential packing.
- New diagnostics (`loss_tokens`, `loss_tokens_host`, boundary/docs-per-seq metrics) provide direct observability for objective-density drift.

These are early indicators from 100-step pilots. Use the 5k matrix above for stability conclusions.

## Recommended Default Knobs for Comma Workloads

Until full 5k matrix comparison is complete:

- Prefer `packing_mode: sequential` for clean utilization behavior.
- Track `loss_tokens` and `boundary_transitions` alongside loss to avoid misattributing objective-density shifts to optimizer instability.
- Start from larger shuffle (`shuffle_buffer_size=200000`) when memory allows to reduce local composition correlation.
- Make `mask_boundary_loss` an explicit experiment axis; do not treat it as a silent default.

## Resolution (2026-07)

The full 5k A–D matrix plus a follow-up window-shuffle ablation closed this
investigation. Findings:

- `mask_boundary_loss` has no measurable effect on stability (A≈B, C≈D over 5k
  steps).
- `shuffle_buffer_size` 10k→200k cuts post-warmup loss std 0.31→0.20. Root
  cause of the large slow excursions is **domain marching**: HF streaming
  walks the corpus source-by-source and a 10k-doc buffer cannot mix across
  domain blocks. This also produced the train-loss-below-eval memorization
  signature on Comma/Common Pile. (`max_doc_tokens: 8192` was active in all
  runs, ruling out single-giant-document batch domination as the mechanism.)
- The remaining fast oscillation came from batches being contiguous,
  unshuffled slices of packer output. Fixed by `data.window_shuffle_windows`
  (default 4096): in a 5k WS0-vs-WS4096 pair (both `shuffle_buffer_size:
  200000`, run dirs `runs/dataset-tests/100m-comma-stability-WS*-sb200k-5k-
  20260705_231323`), window shuffle cut step-to-step |Δloss| 31% (p95 −38%),
  worst grad-norm spike 19.6→4.1, improved final eval (3.137→3.113), at zero
  throughput cost.

**Final recipe for Comma-like (domain-ordered, long-tail) corpora:**
`packing_mode: sequential` + `window_shuffle_windows: 4096` +
`shuffle_buffer_size: 200000`. See [Training — Loss-stability
recipe](training.md#loss-stability-recipe) and
[Packing — Window shuffling](packing.md#window-shuffling-batch-decorrelation).
