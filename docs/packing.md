# Packing and Boundary Semantics

Packing turns variable-length documents into fixed training sequences and applies boundary-related loss masking.

Related: [Data Pipeline](data_pipeline.md), [Config Reference](config-reference.yaml) (`data.packing_*`, `data.mask_boundary_loss`, `data.train_on_eos`).

## Packing modes

chomp uses a Grain-backed input pipeline and supports three packing strategies, all emitting fixed-length windows of `seq_len`:

1) **Sequential packer** (`data.packing_mode: sequential`)
   - Appends tokenized documents into a rolling buffer and emits windows in stream order.
   - Exact resume retains the unconsumed tail of the current tokenized document, so one long document can make packer checkpoint state much larger than `seq_len`; truncation is an explicit data-policy choice defined by [`data.tokenizer.max_doc_tokens`](config-reference.yaml).

2) **Bin packer** (`data.packing_mode: bin`)
   - Buffers multiple documents, seeds bins from the oldest candidates, and uses a First-Fit-Decreasing heuristic to fill remaining capacity.
   - Useful for higher utilization when documents are short or variable length.
   - Note this is a **length-based local reorder with bounded lookahead**, not "stream order with less padding": fill candidates are length-sorted, but FIFO seeds guarantee that no old short candidate can starve.

3) **Multipack packer** (`data.packing_mode: multipack`)
   - Uses FIFO-seeded, grouped First-Fit-Decreasing packing over at least one cycle's rows and emits one complete cycle at a time.
   - Intended for strict packed training semantics with segment-aware attention.

Each window follows the [Data Pipeline batch contract](data_pipeline.md#batch-contract). Segment IDs identify packed document runs, and position IDs reset within each run. Bin and multipack output pad with `model.pad_token_id`, `segment_id=0`, and a false attention mask.

Lookahead, per-row segment caps, and strict-isolation field contracts are canonical under [`data.packing_*`](config-reference.yaml). Shared runtime behavior for both packed modes:

- Cycle size is `A*B` for training and `B` for evaluation (`A=1` there). Resume metadata records both effective lookaheads when evaluation data is enabled, because a configured change can be clamped away for training while still changing eval packing.
- FIFO progress: every cycle emits its oldest pending chunks as mandatory seeds before FFD fill. Document tails therefore make bounded progress even under an endless stream of full-size chunks.
- End-of-stream flush: when the upstream stream is exhausted (`data.repeat: false`, or the finite eval doc set), both packers flush their remaining sub-threshold pending documents into as many padded windows as needed instead of silently dropping them. Sequential packing likewise right-pads its final nonempty token tail. Batch assembly right-pads missing rows, so every usable final window reaches training or evaluation without changing the compiled `[A, B, T]` shape. Evaluation always uses `A=1`; changing `train.grad_accum` does not change finite eval coverage.

## Segment-isolation semantics

`sequential` keeps stream semantics: the corpus is treated as one continuous token stream, no segment metadata reaches the model, and segment IDs are used only for loss masking and diagnostics.

`bin` and `multipack` can enable strict packed semantics as defined by [`data.packing_strict_segments`](config-reference.yaml). Both modes place multiple unrelated documents in one sequence, and for a recurrent-state architecture cross-document bleed means CEMA/TimestepNorm contamination, not just attention leakage. With megalodon-jax >= 0.2.2, strict execution provides **full state isolation**: each packed document computes as if it were run alone:

- segment-isolated attention (masked by contiguous segment *runs*, so a reused segment id cannot attend back to an earlier same-id document),
- per-segment RoPE position resets derived from `segment_ids`,
- ComplexEMA recurrent state zeroed at every segment boundary,
- TimestepNorm running statistics (count/mean/M2) restarted at every boundary,
- cross-segment label pairs and pairs targeting padding excluded from the loss by the backend automatically.

chomp gates this path on the backend's `supports_segment_reset` capability flag and fails fast at startup if the installed megalodon-jax does not support it. The package dependency requires megalodon-jax >= 0.2.2.

Costs and notes:

- Strict mode bypasses the FFT CEMA path (it cannot express resets) and adds ~2x attention FLOPs on packed rows (per-document chunk re-anchoring), so it trades throughput for correctness.
- The segmented CEMA implementation is either a parallel associative scan or a compact sequential forward carry; its selection and contextual resume contract are defined by `model.use_associative_segment_scan`. Compiled backward peak memory must still be measured.
- Strict packed metadata is training-only upstream: passing segment_ids with a cache (generation/streaming) raises.
- Strict execution requires boundary-loss masking as specified by `data.mask_boundary_loss`. The backend excludes boundary pairs and supplies the authoritative normalization count; matching host accounting is verified at synchronization and checkpoint boundaries.

A measured forward/backward comparison on 2026-07-31 used the shipped 113.85M model geometry (`seq_len=2048`, batch 2, accumulation 8, BF16, activation checkpointing), an RTX 5090, JAX 0.10.2, real streamed mixture data, and `XLA_PYTHON_CLIENT_PREALLOCATE=false`:

| Model execution | Packing semantics | Warm step times | Valid tokens/s | Process allocator peak |
| --- | --- | --- | --- | --- |
| No segmented reset | `sequential` continuous-stream mode | 0.536 / 0.722 s | 61,097 / 45,374 | 4.735 GB |
| Associative segmented CEMA | Strict `bin` | 1.175 / 1.349 s | 18,596 / 18,153 | 4.700 GB |
| Sequential segmented CEMA | Strict `bin` | 4.593 / 4.802 s | 4,757 / 5,099 | 6.032 GB |

The two strict-bin rows are the paired scan comparison: they consumed identical batches, valid-target counts, and packing statistics. At this shape the sequential scan was 3.56–3.91× slower and used 1.332 GB more process-lifetime allocator peak than the associative scan. The continuous-stream row has different recurrent boundary semantics and valid-target utilization, so it is a separate non-segmented execution reference rather than a scan-only control. These one-device measurements include forward, backward, and optimizer execution and establish the observed compiled training-step behavior for this recipe geometry; they do not isolate a backward-only allocation or prove a general memory complexity or performance ratio.

### Non-strict operation: CEMA/TimestepNorm state crosses boundaries

Under `sequential`, and under `bin`/`multipack` without strict segment execution, no segment metadata reaches the model: recurrent/normalization state flows across packed document boundaries. For `sequential` this is the intended continuous-stream semantics; packed modes must opt into that cross-document bleed deliberately. When enabled, boundary loss masking still prevents cross-document next-token supervision.

For sequential packing internals, segment IDs are normalized to bounded values and reindexed per emitted window. Boundary semantics are unchanged, and this avoids long-run integer overflow hazards.

## Window shuffling (batch decorrelation)

Independent of packing mode, every batch used to be assembled from *consecutive* packer output: all `A*B` rows of a step were adjacent slices of one document stream. On long-document corpora (e.g. Common Pile), a single document spanning hundreds of `seq_len` windows would dominate several consecutive optimizer steps, producing near-single-document batches, choppy train loss, and quasi-online memorization (train loss diving below eval loss).

A seeded, bounded shuffle between the packer and batch assembly decorrelates packed `[T]` windows. Its token/row sizing and effective-cycle formula are defined by [`data.window_shuffle_tokens` and `data.window_shuffle_max_rows`](config-reference.yaml). Resume remains exact because only upstream state at the window start and permutation progress are checkpointed, then the window is reconstructed and replayed deterministically. Eval batches are never shuffled.

(In the 5,000-step Comma ablation, enabling the 4,096-row window reduced step-to-step absolute loss changes by 31% and the worst gradient-norm spike from 19.6 to 4.1 without reducing throughput; its probe sampled the training corpus, so these are stability rather than generalization results.)

At the maintained 8.39M-token shuffle budget, the raw int32 token and segment-ID arrays occupy about 64 MiB; this excludes each row's Python tuple, NumPy objects, and list references. Grain materializes each complete row window and its shuffled list before yielding from that window, so the independent row cap bounds that object count and the number of packed rows requested per fill. Startup logs report the effective row and token counts before data consumption. The compiled model derives position IDs from segment IDs after batching.

Guidance:

- This is default hygiene for any streaming corpus that is not known to be globally pre-mixed, not a fix for one dataset. High-risk inputs are **domain-ordered or long-document corpora**: source-concatenated mixes (Common Pile derivatives), arXiv/books/legal/PDF dumps, code dumps, or anything stored as "all of source A, then all of source B", or with documents spanning many `seq_len` windows.
- For long-document corpora, combine sequential packing with both document-level and packed-window shuffling. The former fights source/domain/shard-order homogeneity; the latter fights within-document adjacent-window homogeneity. Their sizing guidance is in the [Config Reference](config-reference.yaml), and neither replaces the other.
- `bin`/`multipack` mainly improve utilization on short-document mixes (Zyda-2, SmolLM2-style); on long-document corpora most windows are full-capacity chunks and bin packing adds no utilization benefit.

TODO: Replace the eager FFD long-document chunk queue with an owned token buffer plus offset and bound `bin` candidate selection. One enormous document currently creates every capacity-sized chunk up front, and `bin` repeatedly sorts the remaining pool; changing that order and serialized state requires an explicit pipeline compatibility transition.

## Boundary-aware loss masking

Boundary masking excludes cross-document next-token targets; EOS masking excludes labels equal to the effective EOS ID. Defaults, constraints, and interactions are canonical under [`data.mask_boundary_loss` and `data.train_on_eos`](config-reference.yaml).

These masks are applied inside the data pipeline before batching and do not affect shapes. If they leave a complete batch with zero valid shifted targets, assembly fails before the optimizer, schedule, model RNG, or training step can advance. This usually indicates one-token documents, a tokenizer special-token collision, or an over-restrictive masking configuration.
