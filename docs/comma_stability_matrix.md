# Comma Stability Study

The February-July 2026 experiments measured loss stability on Comma under
sequential packing. The initial 2x2 matrix varied:

- `data.mask_boundary_loss`: `true` or `false`
- `data.shuffle_buffer_size`: `10_000` or `200_000`

All variants used the same model, optimizer, seed, and base config family. A
follow-up ablation compared `data.window_shuffle_windows: 0` against `4096`
with the 200,000-document shuffle buffer.

## Results

- Boundary masking had no measurable effect on stability over 5,000 steps.
- Increasing the document shuffle buffer from 10,000 to 200,000 reduced
  post-warmup loss standard deviation from 0.31 to 0.20.
- The large slow excursions came from domain marching: HF streaming traversed
  source blocks that a 10,000-document buffer could not mix. The same runs
  showed the train-loss-below-eval signature associated with local
  memorization.
- `data.tokenizer.max_doc_tokens: 8192` was active throughout, ruling out
  unbounded single-document batches as the cause of those slow excursions.
- With the 200,000-document buffer, enabling a 4,096-window shuffle reduced
  step-to-step absolute loss changes by 31% (38% at p95), reduced the worst
  gradient-norm spike from 19.6 to 4.1, and improved final eval loss from
  3.137 to 3.113 without reducing throughput.

Operational guidance for document- and window-level shuffling is in
[Packing — Window shuffling](packing.md#window-shuffling-batch-decorrelation).
