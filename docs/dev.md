# Dev Log

This file tracks intentional scope decisions and future hooks.

## 2025-03-08: Multi-source mixing deferred

Decision: v0 uses a single HF dataset config (`data.hf_dataset`, `data.hf_name`,
`data.hf_split`) and does not support mixtures.

If/when we add multi-source mixing:

- Introduce `data.sources: list[HFDatasetSource]` and a stable `source_key()`.
- Build per-source iterators and mix them deterministically.
- Persist per-source iterator states in checkpoints.
- Update resume compatibility checks to compare source keys and weights.

## 2025-03-08: Single prefetch knob

Decision: keep a single `data.grain_prefetch` setting. Multi-tier host/cpu/device
prefetch is not needed for v0.

If we later add tiers, implement them in `chomp.data.grain` and keep the public
config surface minimal.

## 2025-03-08: Source of truth for sequence length

`train.seq_len` is the single source of truth for window length. Packers and
model chunk-size validation use this value, so it stays centralized and easy
to audit.
