# Run analysis: 20260128_215038_zyda2_200m_2k-cema16-20k-gnorm

Source: `runs/chomp/20260128_215038_zyda2_200m_2k-cema16-20k-gnorm/metrics.jsonl`

## Summary

- Eval loss is cleanly monotonic: **4.04 @ 2.5k → 3.02 @ 20k**. No regressions.
- Train loss trends down: median **~4.15 (≤5k)** → **~3.32 (≥12k)**.
- Late-run grad norm drift exists, but updates keep shrinking and eval stays smooth.

## Gradient clipping

`clip_frac` is a per-step indicator (1 if clipping happened, else 0).

- Overall clip rate: **35%** of logged steps.
- By phase:
  - Early (≤5k): **33.5%**
  - Mid (5k–12k): **13.2%**
  - Late (12k–20k): **55.6%**
- `grad_norm_pre_clip` median rises late (**0.86 → 1.03**), so more steps cross the 1.0 clip threshold.
- `grad_norm_post_clip` is pinned near 1.0 on clipped steps, as expected.

## Update size (stability signal)

- `update_ratio` declines strongly over time:
  - Median early: **0.0050**
  - Median late: **0.00123**
  - Linear trend slope is negative (r ≈ **−0.86**).
- `update_norm` median drops **7.31 → 1.56** while `param_norm` stays bounded (~1.3–1.6k).

## Interpretation

This run looks healthy. The late-run increase in `grad_norm_pre_clip` produces more clipping, but the update magnitudes still shrink as LR decays and eval loss continues to improve. There are no signs of instability or regression.

## Optional follow-ups

- If you want less late-stage clipping: raise `optim.grad_clip_norm` (e.g., 1.5–2.0) or reduce `optim.muon.lr_scale` slightly.
- If you are satisfied with stability, keep the current settings.
