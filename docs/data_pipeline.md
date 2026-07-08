# Data Pipeline

This document describes the streaming data path and the fixed-shape batch
contract that the trainer relies on.

## Scope

This page is the home for stream-to-batch flow and eval-set construction.

- For field-level defaults/types: [Config Reference](config-reference.md) (`data.*`)
- For packing strategy and masking semantics: [Packing and Boundary Semantics](packing.md)
- For how batches are consumed during training: [Training Loop](training.md)

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
- `position_ids`: `[A, B, T]` int32

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
Tokenizer knobs are defined in [Config Reference](config-reference.md) under
`data.tokenizer.*`.

chomp saves a tokenizer snapshot under `run_dir/tokenizer` and will prefer that
snapshot on resume to keep tokenization reproducible.

## Packing

The pipeline supports `sequential`, `bin`, and `multipack` packing modes and
always emits fixed windows of length `seq_len` before batching.
Packing trade-offs and boundary-masking behavior are documented in
[Packing and Boundary Semantics](packing.md).

Between the packer and batch assembly, packed windows pass through a seeded
window shuffle (`data.window_shuffle_windows`, train iterator only) so batches
are decorrelated from raw stream order; see
[Window shuffling](packing.md#window-shuffling-batch-decorrelation).

## Grain iterator

The Grain pipeline is composed as: sequence producer (text stream + packer)
-> optional packed-window shuffle -> batch assembly -> optional prefetch.
The wrapper provides:

- deterministic iteration
- optional threaded prefetch (`data.grain_prefetch`)
- a checkpointable iterator state (`get_state` / `set_state`)
- host-side packing diagnostics per emitted batch (`get_stats`)

The packing iterator itself remains a small, explicit Python object; Grain only
wraps it for performance and checkpoint integration.

When stats are enabled (`data.device_put: false`), iterator stats include:

- `packing_tokens`, `packing_capacity`, `packing_utilization`
- `loss_tokens_host` (valid shifted labels after masking)
- `boundary_transitions` (segment boundary count)
- `docs_per_seq_mean`, `docs_per_seq_min`, `docs_per_seq_max`
- `docs_added_this_batch` (fresh stream documents consumed per assembled batch;
  collapses toward 0 while already-buffered content drains, bursty when a
  shuffle window refills. Measured below the prefetch layer, so with
  `data.grain_prefetch > 0` the value may belong to a batch up to
  prefetch-depth ahead of the one just consumed)

## Iterator state and resume

The iterator exposes a JSON-serializable state dict containing:

- HF stream cursor (`datasets` state dict)
- packer buffer contents

This is checkpointed alongside the model so resume does not rely on `.skip()`
or re-streaming.

### Transient stream recovery (best-effort)

Checkpoint resume is exact; in-run recovery from transient HF streaming
errors is not. The stream caches a last-known-good state every
`data.state_update_interval` documents (default 2000). When `next()` fails
(network hiccup), iteration rebuilds from that cached state and retries with
backoff — which can **replay up to `state_update_interval` recently yielded
documents**. This is a deliberate tradeoff for long unattended runs: a rare
duplicated slice beats a dead run. If bounding duplication matters more than
overhead, lower `data.state_update_interval`; if strict no-replay semantics
are required, treat any retry warning in the logs as a signal to stop and
resume from the last checkpoint instead (checkpointed state is unaffected by
recovery). Errors that survive `data.max_retries` still propagate and crash
the run.

## Validation set

chomp builds a fixed validation set when the run is created:

- If `data.hf_eval_split` is set and the HF dataset has that split, it takes the
  first `data.max_eval_samples` examples from that split.
- Otherwise it takes the first `data.max_eval_samples` examples from the
  (shuffled) training split.
- Set `data.hf_eval_split: null` to skip eval-split lookup and always use train.
- For train-split fallback, if `data.seed` is left at `0` and `train.seed` is
  non-zero, the shuffle seed defaults to `train.seed`.

The selected tokens are **persisted to `run_dir/eval_tokens.json.gz`** together
with an identity manifest (eval knobs) and a content hash. Every later start —
including resume — loads that exact set instead of re-collecting from the
stream, so eval losses stay comparable even if the upstream dataset revision or
split contents drift. A cache whose manifest or hash mismatches fails loudly;
delete the run directory (not just the cache) to change eval identity.

A **missing** cache on resume is also a hard error — silently recollecting
would compare post-resume eval losses against a different token set than every
earlier point on the curve. `data.recreate_eval_cache: true` is the explicit
one-shot override; it rebuilds the cache with a loud warning that eval curves
across the boundary are not comparable.

Config validation rejects `max_eval_samples` below the packer emission
threshold for `bin`/`multipack` (packers never flush a partial buffer at end of
stream). If eval still cannot emit any batch at runtime (for example an eval
split yielding fewer documents than promised), or emits batches whose labels
are entirely masked (zero valid loss tokens), training raises instead of
silently emitting a null eval loss.

## Key config knobs

Use [Config Reference](config-reference.md) as the canonical source for `data.*` and related
`train.*` shape knobs. The most operationally important fields for this page
are:

- `data.backend`, `data.hf_*`, `data.text_key`
- `data.hf_eval_split`, `data.max_eval_samples`
- `data.shuffle`, `data.shuffle_buffer_size`, `data.seed`, `data.repeat`
- `data.window_shuffle_windows`
- `data.packing_mode`, `data.packing_buffer_docs`, `data.packing_group_docs`
- `data.packing_max_docs_per_bin`, `data.packing_strict_segments`
- `train.seq_len`, `train.batch_size`, `train.grad_accum`
