# Comma Stability Study

The February-July 2026 experiments measured loss stability on Comma under sequential packing. The initial 2x2 matrix varied:

- `data.mask_boundary_loss`: `true` or `false`
- `data.shuffle_buffer_size`: `10_000` or `200_000`

All variants used the same model, optimizer, seed, and base config family. A follow-up ablation compared disabled packed-window shuffle against an 8,388,608-token budget (4,096 rows at context 2,048) with the 200,000-document shuffle buffer.

## Results

- Boundary masking had no measurable effect on stability over 5,000 steps.
- Increasing the document shuffle buffer from 10,000 to 200,000 reduced post-warmup loss standard deviation from 0.31 to 0.20.
- The large slow excursions came from domain marching: HF streaming traversed source blocks that a 10,000-document buffer could not mix. The same runs showed a train-loss-below-probe signature associated with local memorization. These historical runs sampled that probe from the training corpus, so it was not held-out generalization evidence.
- `data.tokenizer.max_doc_tokens: 8192` was active throughout. At the 2,048-token context this limited one source document to four full content windows plus a small EOS tail, so the experiment supports source/domain adjacency as the slow-excursion mechanism; it did not test documents spanning hundreds of optimizer steps.
- With the 200,000-document buffer, enabling a 4,096-window shuffle reduced step-to-step absolute loss changes by 31% (38% at p95), reduced the worst gradient-norm spike from 19.6 to 4.1, and improved the final train-corpus probe loss from 3.137 to 3.113 without reducing throughput. The decorrelation result remains useful, but those numbers do not measure generalization. New runs use either an explicit distinct split or a train-excluded content-hash holdout for `eval_loss`.

Operational guidance for document- and window-level shuffling is in [Packing: window shuffling](packing.md#window-shuffling-batch-decorrelation).
