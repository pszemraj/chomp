# Packing and Boundary Semantics

This document describes how chomp packs variable-length documents into fixed
training sequences and how boundary-related loss masking works.

## Current packing model

chomp uses a simple streaming packer (`TokenPacker`) that appends tokenized
documents into a rolling buffer and emits fixed-length windows of `seq_len + 1`.
From each window we derive:

- `input_ids`: tokens `[0..T-1]`
- `labels`: tokens `[1..T]` (causal shift)
- `segment_ids`: packed document IDs for each token
- `attention_mask`: all `True` (padding not used today)

The packing strategy is **sequential**, not bin-packing: documents are streamed
in order and packed until enough tokens exist for the next window. This keeps
the pipeline deterministic and streaming-friendly.

## Segment IDs and attention

Each document gets a monotonically increasing `segment_id`. When
`model.segment_masking=true`, the Megalodon patch uses these IDs to form a
block-diagonal attention mask so tokens only attend within their document.

If you disable `model.segment_masking`, segment IDs are still emitted but only
used for loss masking (see below).

## Boundary-aware loss masking

Two config knobs control loss behavior at packed boundaries:

- `data.mask_boundary_loss` (default: true)
  - When enabled, labels at **segment transitions** are set to `-100`.
  - This prevents the model from learning cross-document next-token
    predictions (e.g., predicting the first token of the next document from
    the previous document’s final token).

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

## Future work

If we adopt bin-packing or Grain-based pipelines, we will preserve the same
external `Batch` contract (`[A,B,T]` tensors with `segment_ids`) and keep loss
masking in the data pipeline so training remains compile-stable.
