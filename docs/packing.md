# Packing and Boundary Semantics

This document describes how chomp packs variable-length documents into fixed
training sequences and how boundary-related loss masking works.

## Scope

This page is the home for packing strategy and boundary-masking behavior.

- For where packing sits in the end-to-end data path: [Data Pipeline](data_pipeline.md)
- For field-level defaults/types: [Config Reference](config-reference.md) (`data.packing_*`,
  `data.mask_boundary_loss`, `data.train_on_eos`)

## Packing modes

chomp uses a Grain-backed input pipeline and supports three packing strategies,
all emitting fixed-length windows of `seq_len`:

1) **Sequential packer** (`data.packing_mode: sequential`, default)
   - Appends tokenized documents into a rolling buffer and emits windows in
     stream order.

2) **Bin packer** (`data.packing_mode: bin`)
   - Buffers multiple documents and uses a First-Fit-Decreasing heuristic to
     pack documents into bins of size `seq_len`.
   - Useful for higher utilization when documents are short or variable length.

3) **Multipack packer** (`data.packing_mode: multipack`)
   - Uses grouped First-Fit-Decreasing packing over `data.packing_group_docs`
     candidates and emits `A*B` packed sequences per cycle.
   - Emits per-segment `position_ids` (reset to `0` at each packed segment).
   - Intended for strict packed training semantics with segment-aware attention.

From each window we derive:

- `input_ids`: tokens `[0..T-1]`
- `labels`: tokens `[0..T-1]` (model shifts internally)
- `segment_ids`: packed document IDs for each token
- `position_ids`: per-segment position IDs
- `attention_mask`: `True` for real tokens, `False` for padding

The bin packer pads to fixed length; pad positions use `model.pad_token_id` and
`segment_id=0`.

Key bin-packing knobs:

- `data.packing_buffer_docs`: number of documents to buffer before packing.
- `data.packing_max_docs_per_bin`: optional cap on documents per bin.

Key multipack knobs:

- `data.packing_group_docs`: grouped lookahead size for multipack packing.
- `data.packing_strict_attention`: require strict segment-aware attention path.
- `data.packing_max_docs_per_bin`: optional cap on packed segments per sequence.

## Attention Semantics

`sequential` and `bin` modes keep stream semantics by default: segment IDs are
used for loss masking and diagnostics, but attention remains stream-like.

`multipack` with `data.packing_strict_attention: true` enables strict packed
semantics. With megalodon-jax >= 0.1.2 this is **full state isolation** — each
packed document computes as if it were run alone:

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

Costs and notes:

- Strict mode bypasses the FFT CEMA path (it cannot express resets) and adds
  ~2x attention FLOPs on packed rows (per-document chunk re-anchoring), so it
  trades throughput for correctness.
- `model.use_associative_segment_scan` selects the segmented CEMA
  implementation: `true` (default) is a parallel associative scan; `false` is
  a sequential low-memory fallback if the associative path OOMs.
- Strict packed metadata is training-only upstream: passing segment_ids with a
  cache (generation/streaming) raises.
- Keep `data.mask_boundary_loss: true` in strict mode. The backend masks
  boundary pairs regardless, but chomp's token-weighted grad accumulation and
  `loss_tokens` counting happen host-side from the labels; pre-masking keeps
  the two consistent (config validation warns otherwise).

### Non-strict modes: CEMA/TimestepNorm state crosses boundaries

Under `sequential`, `bin`, and non-strict `multipack`, no segment metadata
reaches the model: recurrent/normalization state flows across packed document
boundaries by design (stream semantics). This is a semantics choice, not a
leak — boundary loss masking still prevents cross-document next-token
supervision.

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
window start plus two counters and replays deterministically. Eval batches are
never shuffled.

Guidance:

- Long-document corpora: `sequential` + `window_shuffle_windows` (default) +
  a large `shuffle_buffer_size` (e.g. `200_000`). The document-level shuffle
  buffer fights domain ordering of the source stream; the window shuffle
  fights single-document batch homogeneity. They are complementary.
- `bin`/`multipack` mainly improve utilization on short-document mixes
  (Zyda-2, SmolLM2-style); on long-document corpora most windows are
  full-capacity chunks and bin packing adds no utilization benefit.
- Watch `docs_added_this_batch` in the metrics: it collapses toward 0 when a
  single giant document is draining through consecutive batches (the failure
  mode window shuffling removes), and is bursty-but-nonzero when healthy.

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

## Position IDs

The batch contract now includes `position_ids` for all packing modes.
For `multipack`, they are consumed by strict attention mode to reset positions
at each packed segment boundary. For non-strict modes, they are informational.

## Future work

Near-term packing work focuses on:

- further multipack efficiency tuning and diagnostics
- benchmarking strict multipack (full isolation) vs sequential + window
  shuffle on short-document mixes, now that megalodon-jax 0.1.2 makes strict
  mode fully correct (previously "expensive partial correctness")
