# Data Pipeline

This document describes the streaming data path and the fixed-shape batch
contract that the trainer relies on.

## Scope

This page is the home for stream-to-batch flow and eval-set construction.

- For field-level defaults/types: `docs/config-reference.md` (`data.*`)
- For packing strategy and masking semantics: `docs/packing.md`
- For how batches are consumed during training: `docs/training.md`

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
- `labels`: `[A, B, T]` int32 (aligned with `input_ids`)
- `attention_mask`: `[A, B, T]` bool
- `segment_ids`: `[A, B, T]` int32

Where:

- `A = train.grad_accum`
- `B = train.batch_size`
- `T = train.seq_len` (single source of truth)

Inside the compiled train step, the batch is sliced along the microbatch axis
to `[B, T]` views.

## Tokenization

`data.tokenizer.kind` selects the tokenizer:

- `hf`: `transformers.AutoTokenizer` (default)
- `byte`: a simple byte-level tokenizer for infrastructure bring-up

When using `hf`, chomp resolves tokenizer-dependent model settings
(`model.vocab_size`, special token IDs) before training starts.
Tokenizer knobs are defined in `docs/config-reference.md` under
`data.tokenizer.*`.

chomp saves a tokenizer snapshot under `run_dir/tokenizer` and will prefer that
snapshot on resume to keep tokenization reproducible.

## Packing

The pipeline supports `sequential` and `bin` packing modes and always emits
fixed windows of length `seq_len` before batching.
Packing trade-offs and boundary-masking behavior are documented in
`docs/packing.md`.

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

## Validation set

chomp builds a fixed validation set at startup:

- If `data.hf_eval_split` is set and the HF dataset has that split, it takes the
  first `data.max_eval_samples` examples from that split.
- Otherwise it takes the first `data.max_eval_samples` examples from the
  (shuffled) training split.
- Set `data.hf_eval_split: null` to skip eval-split lookup and always use train.
- For train-split fallback, if `data.seed` is left at `0` and `train.seed` is
  non-zero, the shuffle seed defaults to `train.seed`.

The selected texts are derived at run start (configured eval split preferred,
fallback to train) and are not cached on disk.

## Key config knobs

Use `docs/config-reference.md` as the canonical source for `data.*` and related
`train.*` shape knobs. The most operationally important fields for this page
are:

- `data.backend`, `data.hf_*`, `data.text_key`
- `data.hf_eval_split`, `data.max_eval_samples`
- `data.shuffle`, `data.shuffle_buffer_size`, `data.seed`, `data.repeat`
- `data.packing_mode`, `data.packing_buffer_docs`, `data.packing_max_docs_per_bin`
- `train.seq_len`, `train.batch_size`, `train.grad_accum`
