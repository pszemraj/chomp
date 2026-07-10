# Packing and Boundary Semantics

Packing turns variable-length documents into fixed training sequences and
applies boundary-related loss masking.

Related: [Data Pipeline](data_pipeline.md), [Config Reference](config-reference.yaml)
(`data.packing_*`, `data.mask_boundary_loss`, `data.train_on_eos`).

## Packing modes

chomp uses a Grain-backed input pipeline and supports three packing strategies,
all emitting fixed-length windows of `seq_len`:

1) **Sequential packer** (`data.packing_mode: sequential`, default)
   - Appends tokenized documents into a rolling buffer and emits windows in
     stream order.

2) **Bin packer** (`data.packing_mode: bin`)
   - Buffers multiple documents, seeds bins from the oldest candidates, and
     uses a First-Fit-Decreasing heuristic to fill remaining capacity.
   - Useful for higher utilization when documents are short or variable length.
   - Note this is a **length-based local reorder with bounded lookahead**, not
     "stream order with less padding": fill candidates are length-sorted, but
     FIFO seeds guarantee that no old short candidate can starve.

3) **Multipack packer** (`data.packing_mode: multipack`)
   - Uses FIFO-seeded, grouped First-Fit-Decreasing packing over
     `max(data.packing_group_docs, A*B)` candidates and emits `A*B` packed
     sequences per cycle.
   - Intended for strict packed training semantics with segment-aware attention.

Each window follows the [Data Pipeline batch contract](data_pipeline.md#batch-contract).
Segment IDs identify packed document runs, and position IDs reset within each
run. Bin and multipack output pad with `model.pad_token_id`, `segment_id=0`,
and a false attention mask.

Key bin-packing knobs:

- `data.packing_buffer_docs`: number of documents to buffer before packing.
- `data.packing_max_docs_per_bin`: optional cap on documents per bin.

Key multipack knobs:

- `data.packing_group_docs`: grouped lookahead size for multipack packing.
- `data.packing_max_docs_per_bin`: optional cap on packed segments per sequence.

Shared by both packed modes:

- `data.packing_strict_segments` (default `true`): require full per-document
  state isolation in the backend.
- FIFO progress: every cycle emits the oldest `A*B` pending chunks as mandatory
  seeds before FFD fill. Document tails therefore make bounded progress even
  under an endless stream of full-size chunks.
- End-of-stream flush: when the upstream stream is exhausted (`data.repeat:
  false`, or the finite eval doc set), both packers flush their remaining
  sub-threshold pending documents into as many padded windows as needed
  instead of silently dropping them. The fixed-shape contract still applies
  downstream: windows that cannot fill one complete `[A, B, T]` batch are
  dropped at batch assembly, and eval raises if zero batches result.

## Segment-isolation semantics

`sequential` keeps stream semantics: the corpus is treated as one continuous
token stream, no segment metadata reaches the model, and segment IDs are used
only for loss masking and diagnostics.

`bin` and `multipack` with `data.packing_strict_segments: true` (the default)
enable strict packed semantics. Both modes place multiple unrelated documents
in one sequence, and for a recurrent-state architecture cross-document bleed
means CEMA/TimestepNorm contamination, not just attention leakage, so
isolation is required by default wherever documents are packed. With
megalodon-jax >= 0.1.2 this is **full state isolation**: each packed document
computes as if it were run alone:

- segment-isolated attention (masked by contiguous segment *runs*, so a reused
  segment id cannot attend back to an earlier same-id document),
- per-segment RoPE position resets via explicit `position_ids`,
- ComplexEMA recurrent state zeroed at every segment boundary,
- TimestepNorm running statistics (count/mean/M2) restarted at every boundary,
- cross-segment label pairs and pairs targeting padding excluded from the loss
  by the backend automatically.

chomp gates this path on the backend's `supports_segment_reset` capability
flag and fails fast at startup if the installed megalodon-jax predates it
(older versions accepted the same kwargs but only isolated attention).
Independently of packing mode, chomp requires **megalodon-jax >= 0.1.2 across
the board**: every megalodon model build (train and generate) enforces the
version floor, so a stale environment fails immediately rather than running
degraded semantics anywhere.

Costs and notes:

- Strict mode bypasses the FFT CEMA path (it cannot express resets) and adds
  ~2x attention FLOPs on packed rows (per-document chunk re-anchoring), so it
  trades throughput for correctness.
- `model.use_associative_segment_scan` selects the segmented CEMA
  implementation: `true` (default) is a parallel associative scan; `false` is
  a sequential low-memory fallback if the associative path OOMs.
- Strict packed metadata is training-only upstream: passing segment_ids with a
  cache (generation/streaming) raises.
- `data.mask_boundary_loss: true` is **required** in strict mode (config
  validation errors otherwise). The backend masks boundary pairs regardless,
  but chomp's token-weighted grad accumulation and `loss_tokens` counting
  happen host-side from the labels; unmasked labels would silently change the
  gradient normalization denominator.

### Non-strict operation: CEMA/TimestepNorm state crosses boundaries

Under `sequential`, and under `bin`/`multipack` with an explicit
`data.packing_strict_segments: false`, no segment metadata reaches the model:
recurrent/normalization state flows across packed document boundaries. For
`sequential` this is the intended continuous-stream semantics. For the packed
modes it is deliberate cross-document bleed that must be opted into; the
default refuses to do it silently. Boundary loss masking still prevents
cross-document next-token supervision in all cases.

For sequential packing internals, segment IDs are normalized to bounded values
and reindexed per emitted window. Boundary semantics are unchanged, and this
avoids long-run integer overflow hazards.

## Window shuffling (batch decorrelation)

Independent of packing mode, every batch used to be assembled from *consecutive*
packer output: all `A*B` rows of a step were adjacent slices of one document
stream. On long-document corpora (e.g. Common Pile), a single document spanning
hundreds of `seq_len` windows would dominate several consecutive optimizer
steps, producing near-single-document batches, choppy train loss, and
quasi-online memorization (train loss diving below eval loss).

`data.window_shuffle_windows` (default `4096`, `0` disables) inserts a seeded,
deterministic shuffle of packed `[T]` windows between the packer and batch
assembly. Disjoint blocks of that many windows are permuted, so a
244-window document contributes a few rows per batch instead of every row.
Resume remains exact: the shuffle checkpoints only the upstream state at the
window start plus permutation progress, then reconstructs and replays the
window deterministically. Eval batches are never shuffled.

The shuffle window stores only int32 token and segment-ID arrays. Position IDs
are derived from segment IDs during batch assembly after shuffling, saving one
`[window_shuffle_windows, seq_len]` array (about 32 MiB at the default
4096x2048 geometry).

Guidance:

- This is default hygiene for any streaming corpus that is not known to be
  globally pre-mixed, not a fix for one dataset. High-risk inputs are
  **domain-ordered or long-document corpora**: source-concatenated mixes
  (Common Pile derivatives), arXiv/books/legal/PDF dumps, code dumps, or
  anything stored as "all of source A, then all of source B", or with
  documents spanning many `seq_len` windows.
- Long-document corpora: `sequential` + `window_shuffle_windows` (default) +
  a large `shuffle_buffer_size` (e.g. `200_000`). The document-level shuffle
  window fights source/domain/shard-order homogeneity of the stream; the
  window shuffle fights within-document, adjacent-window homogeneity. They
  are complementary; neither replaces the other.
- `bin`/`multipack` mainly improve utilization on short-document mixes
  (Zyda-2, SmolLM2-style); on long-document corpora most windows are
  full-capacity chunks and bin packing adds no utilization benefit.

## Boundary-aware loss masking

Two config knobs control loss behavior at packed boundaries:

- `data.mask_boundary_loss` (default: true)
  - When enabled, labels at **segment transitions** are set to `-100`.
  - This prevents the model from learning cross-document next-token
    predictions (e.g., predicting the first token of the next document from
    the previous document's final token).

- `data.train_on_eos` (default: true)
  - When disabled, any label equal to `model.eos_token_id` is set to `-100`.
  - This suppresses EOS supervision even when `data.tokenizer.add_eos=true`.

These masks are applied inside the data pipeline before batching and do not
affect shapes.
If they leave a complete batch with zero valid shifted targets, assembly fails
before the optimizer, schedule, model RNG, or training step can advance. This
usually indicates one-token documents, a tokenizer special-token collision, or
an over-restrictive masking configuration.

## Position IDs

The batch contract now includes `position_ids` for all packing modes.
For `bin` and `multipack` under strict segment isolation, they are consumed by
the backend to reset positions at each packed segment boundary. Otherwise they
are informational.
