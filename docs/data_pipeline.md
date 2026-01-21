# Data Pipeline

This document describes the streaming data path and the fixed-shape batch
contract that the trainer relies on.

## Overview

chomp always uses the same data path, even in debug mode:

1) **HF streaming** (`datasets`) or `local_text` (debug)
2) **Tokenizer** (`data.tokenizer.kind`)
3) **Packer** (sequential or bin)
4) **Grain iterator** (prefetch + checkpointable state)
5) **Batch** tensors `[A, B, T]`

The trainer only sees fixed-shape `Batch` objects and never handles ragged
sequences.

## Batch contract

All batches have **fixed shapes**:

- `input_ids`: `[A, B, T]` int32
- `labels`: `[A, B, T]` int32 (causal shift)
- `attention_mask`: `[A, B, T]` bool
- `segment_ids`: `[A, B, T]` int32

Where:

- `A = train.grad_accum`
- `B = train.batch_size`
- `T = train.seq_len`

Inside the compiled train step, the batch is sliced along the microbatch axis
to `[B, T]` views.

## Tokenization

`data.tokenizer.kind` selects the tokenizer:

- `hf`: `transformers.AutoTokenizer` (default)
- `byte`: a simple byte-level tokenizer for infrastructure bring-up

When using `hf`, chomp:

- detects the tokenizer vocab size
- rounds `model.vocab_size` up to `data.tokenizer.vocab_size_multiple`
- optionally updates `model.{bos,eos,pad}_token_id` from tokenizer metadata

Special token insertion is controlled by:

- `data.tokenizer.add_bos`
- `data.tokenizer.add_eos`

## Packing

Packer modes:

- `data.packing_mode: sequential` (stream order, rolling buffer)
- `data.packing_mode: bin` (First-Fit-Decreasing; better utilization)

Both packers emit fixed windows of length `seq_len + 1`, then split into:

- `input_ids = tokens[0..T-1]`
- `labels = tokens[1..T]`

Segment IDs are emitted for each token to support block-diagonal attention.

Loss masking controls:

- `data.mask_boundary_loss`: mask labels at segment boundaries
- `data.train_on_eos`: mask labels that equal `model.eos_token_id`

## Grain iterator

The Grain wrapper provides:

- deterministic iteration
- optional threaded prefetch (`data.grain_prefetch`)
- a checkpointable iterator state (`get_state` / `set_state`)

The packing iterator itself remains a small, explicit Python object; Grain only
wraps it for performance and checkpoint integration.

## Iterator state and resume

The iterator exposes a JSON-serializable state dict containing:

- HF stream cursor (`datasets` state dict)
- packer buffer contents

This is checkpointed alongside the model so resume does not rely on `.skip()`
or re-streaming.

## Key config knobs

- `data.backend`: `hf` or `local_text`
- `data.hf_dataset`, `data.hf_name`, `data.hf_split`, `data.text_key`
- `data.shuffle`, `data.shuffle_buffer_size`, `data.seed`, `data.repeat`
- `data.state_update_interval`, `data.max_retries`, `data.retry_delay_sec`
- `data.tokenizer.*`
- `data.packing_mode`, `data.packing_buffer_docs`,
  `data.packing_max_docs_per_bin`
- `data.mask_boundary_loss`, `data.train_on_eos`
- `data.grain_prefetch`
- `train.seq_len`, `train.batch_size`, `train.grad_accum`
