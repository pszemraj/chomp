# Packing and Boundary Semantics

This document describes how chomp packs variable-length documents into fixed
training sequences and how boundary-related loss masking works.

## Packing modes

chomp uses a Grain-backed input pipeline and supports two packing strategies,
both emitting fixed-length windows of `seq_len`:

1) **Sequential packer** (`data.packing_mode: sequential`, default)
   - Appends tokenized documents into a rolling buffer and emits windows in
     stream order.

2) **Bin packer** (`data.packing_mode: bin`)
   - Buffers multiple documents and uses a First-Fit-Decreasing heuristic to
     pack documents into bins of size `seq_len`.
   - Useful for higher utilization when documents are short or variable length.

From each window we derive:

- `input_ids`: tokens `[0..T-1]`
- `labels`: tokens `[0..T-1]` (model shifts internally)
- `segment_ids`: packed document IDs for each token
- `attention_mask`: `True` for real tokens, `False` for padding

The bin packer pads to fixed length; pad positions use `model.pad_token_id` and
`segment_id=0`.

Key bin-packing knobs:

- `data.packing_buffer_docs`: number of documents to buffer before packing.
- `data.packing_max_docs_per_bin`: optional cap on documents per bin.

## Stream Semantics

chomp treats the corpus as a continuous token stream. This is the only supported
attention behavior and matches common pretraining setups.

**Rationale:**

- Attention-only segment masking would still leak across documents via
  Megalodon's ComplexEMA and TimestepNorm ("expensive partial correctness").
- Boundary loss masking (via `data.mask_boundary_loss`) handles the most
  important document-boundary concern by preventing cross-document loss.
- Stream semantics keeps the system minimal and predictable.

Segment IDs are still emitted by the data pipeline for boundary loss masking,
but they are not used to alter attention.

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

chomp does not emit position IDs today. Megalodon relies on RoPE internally and
does not accept explicit position IDs in its public API. If/when we add support
for position ID reset at segment boundaries, it will be gated by a new config
flag and will preserve the fixed-shape batch contract.

> [!NOTE]
> Packing quality today refers to segment IDs + boundary masking. Position ID
> resets remain deferred until Megalodon exposes a stable API for them.

## Future work

Near-term packing work focuses on:

- position ID resets at segment boundaries (if/when Megalodon exposes this)
- improving utilization stats and diagnostics
