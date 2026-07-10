# Data Pipeline

Streaming data path, eval-set construction, and the fixed-shape batch contract.

Related: [Config Reference](config-reference.yaml) (`data.*`),
[Packing and Boundary Semantics](packing.md), [Training Loop](training.md).

## Overview

chomp always uses the same data path, even in debug mode:

1) **HF streaming** (`datasets`) or `local_text` (debug)
2) **Tokenizer** (`data.tokenizer.kind`)
3) **Packer** (sequential, bin, or multipack)
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
- `T = train.seq_len`

Inside the compiled train step, the batch is sliced along the microbatch axis
to `[B, T]` views.

## Tokenization

`data.tokenizer.kind` selects the tokenizer:

- `byte` (default): a simple byte-level tokenizer for infrastructure bring-up
- `hf`: `transformers.AutoTokenizer` for real pretraining

When using `hf`, chomp resolves tokenizer-dependent model settings
(`model.vocab_size`, special token IDs) before training starts.
Tokenizer knobs are defined in [Config Reference](config-reference.yaml) under
`data.tokenizer.*`.

chomp saves a tokenizer snapshot under `run_dir/tokenizer`. Resume requires
that snapshot and fails before modifying the run directory when it is missing;
it never rebuilds a replacement from a potentially changed repository or path.

`data.tokenizer.max_doc_tokens: null` means no truncation. A positive cap is
explicit data loss applied after tokenization and before BOS/EOS insertion;
the iterator reports documents truncated, source tokens observed/retained/
discarded, discarded-token fraction, and approximate p50/p90/p99 document
length upper bounds. Quantiles come from a checkpointed fixed-size log2
histogram, so diagnostics remain resume-stable without retaining an unbounded
length list. The cap does not reduce the tokenizer's peak work because the full
document is encoded first.

## Packing

The pipeline supports `sequential`, `bin`, and `multipack` packing modes and
always emits fixed windows of length `seq_len` before batching.
Packing trade-offs and boundary-masking behavior are documented in
[Packing and Boundary Semantics](packing.md).

Between the packer and batch assembly, packed windows pass through a seeded
window shuffle (`data.window_shuffle_tokens`, train iterator only) so batches
are decorrelated from raw stream order; see
[Window shuffling](packing.md#window-shuffling-batch-decorrelation).

The HF document shuffle ends each window at the first of
`data.shuffle_buffer_size` documents or `data.shuffle_buffer_bytes` UTF-8
payload bytes. Runtime packing metrics expose current and peak document-window
counts/bytes plus replay totals. The byte limit may be exceeded by one
oversized document because reading ahead and carrying that document outside
the compact checkpoint state would break exact replay.

## Grain iterator

The training pipeline is composed as: unshuffled HF source -> optional
chomp-owned document-window shuffle -> tokenizer/packer -> optional Grain
packed-window shuffle -> batch assembly -> optional prefetch.
The wrapper provides:

- deterministic iteration
- optional threaded prefetch (`data.grain_prefetch`)
- a checkpointable iterator state (`get_state` / `set_state`)
- host-side packing diagnostics per emitted batch (`get_stats`)
- an exact host loss-token count paired with the batch through prefetch and
  device transfer (`get_loss_tokens`)

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

The iterator exposes checkpointable state containing the source cursor (HF or
local text), document- and packed-window shuffle replay progress, packer
buffered/ready queues, and enabled prefetch-wrapper state.

Bin and multipack queue rows are checkpointed as flat int32 payloads with row
offsets, not nested token lists. This changes serialization shape only; FIFO
queue order and emitted rows are identical.

This is checkpointed alongside the model so resume does not rely on `.skip()`
or re-streaming.

Hugging Face's streaming shuffle omits the contents of its read-ahead buffer
from `state_dict()`, so chomp never calls it in the checkpointed path. Chomp
instead permutes disjoint document windows. State stores the unshuffled source
position at the current window's start, its index, and the output cursor; a
restore reconstructs the window deterministically without inflating the
checkpoint with document text. Set `data.hf_revision` to an immutable commit
for production runs, because exact iterator replay cannot compensate for an
upstream repository changing beneath the same name. Chomp enforces a full
40-hex commit whenever `checkpoint.enabled: true`; mutable revisions remain
available only for explicitly non-checkpointed exploration.

### Transient stream recovery

In-run recovery from transient HF streaming errors preserves exact document
order. The stream retains a last-known-good compact state and the exact number
of documents yielded since it. On failure, it rebuilds from that state,
discards precisely those already-yielded documents, and retries with backoff.
The initial source state covers failures before the first document, and a
failed periodic state capture retains the preceding good state.

`data.state_update_interval` (default 2000) controls reconstruction work, not
data duplication: a recovery may reread and discard up to that many documents.
If state restore or fast-forward cannot reproduce the prior position, training
fails and must resume from the last Chomp checkpoint rather than continuing
from a partially reconstructed stream. Errors that survive `data.max_retries`
also propagate and fail the run.

## Validation set

chomp builds a fixed validation set when the run is created:

- If `data.hf_eval_split` is set, it is authoritative. Any loading,
  authentication, schema, decoding, or collection error fails startup. It must
  differ from `data.hf_split` while evaluation is enabled.
- The selected split uses the configured `data.shuffle` behavior and contributes
  at most `data.max_eval_samples` examples.
- With `data.hf_eval_split: null`, a stable BLAKE2 content hash reserves
  `data.hf_eval_holdout_fraction` of identities for eval and removes them from
  the training stream. Duplicate content always lands on the same side. The
  sparse hash selection does not also fill a document-shuffle window; doing so
  would multiply startup reads by roughly the inverse holdout fraction.
- A positive `data.max_eval_samples` that yields no documents fails startup;
  use `0` to disable evaluation intentionally.

The selected tokens are **persisted to `run_dir/eval_tokens.json.gz`** together
with an identity manifest (eval knobs) and a content hash. Every later start,
including resume, loads that exact set instead of re-collecting from the
stream, so eval losses stay comparable even if the upstream dataset revision or
split contents drift. A cache whose manifest or hash mismatches fails loudly;
delete the run directory (not just the cache) to change eval identity.

A **missing** cache on resume is also a hard error: silently recollecting
would compare post-resume eval losses against a different token set than every
earlier point on the curve. `data.recreate_eval_cache: true` is the explicit
one-shot override; after checkpoint compatibility validation succeeds, it
rebuilds the cache with a loud warning that eval curves across the boundary
are not comparable. A rejected resume never persists its recollected tokens.

At end of stream the `bin`/`multipack` packers flush their remaining pending
documents into padded windows, so an eval doc set below the pack threshold
still emits windows. Eval uses `A=1` and pads missing final rows independently
of `train.grad_accum`. If it yields no usable window or emits batches whose
labels are entirely masked (zero valid loss tokens), training raises instead
of silently emitting a null eval loss.
