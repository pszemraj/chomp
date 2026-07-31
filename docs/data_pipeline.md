# Data Pipeline

Streaming data path, eval-set construction, and the fixed-shape batch contract.

Related: [Config Reference](config-reference.yaml) (`data.*`), [Packing and Boundary Semantics](packing.md), [Training Loop](training.md).

## Overview

chomp always uses the same data path, even in debug mode:

1) **HF streaming** (`datasets`) or `local_text` (debug)
2) **Tokenizer** (`data.tokenizer.kind`)
3) **Packer** (sequential, bin, or multipack)
4) **Grain iterator** (prefetch + checkpointable state)
5) **Batch** tensors `[A, B, T]`

The trainer only sees fixed-shape `Batch` objects and never handles ragged sequences.

## Batch contract

All batches have **fixed shapes**:

- `input_ids`: `[A, B, T]` int32
- `labels`: `[A, B, T]` int32 (aligned with `input_ids`)
- `segment_ids`: `[A, B, T]` int32

The compiled model path derives the attention mask and segment-local positions from `segment_ids`. Keeping these deterministic arrays out of the batch avoids buffering and transferring duplicate data.

Where:

- `A = train.grad_accum`
- `B = train.batch_size`
- `T = train.seq_len`

Inside the compiled train step, the batch is sliced along the microbatch axis to `[B, T]` views.

## Tokenization

`data.tokenizer.kind` selects the tokenizer:

- `hf` (default): `transformers.AutoTokenizer` for real pretraining
- `byte`: a simple byte-level tokenizer for offline infrastructure bring-up

When using `hf`, chomp resolves tokenizer-dependent model settings (`model.vocab_size`, special token IDs) before training starts. Tokenizer knobs are defined in [Config Reference](config-reference.yaml) under `data.tokenizer.*`.

Every fresh run writes `run_dir/tokenizer/identity.json`. For Hugging Face tokenizers, chomp saves the tokenizer assets there, reloads them with `local_files_only=true`, and uses that reloaded instance for the run; resume never falls back to `hf_name_or_path`. The manifest records the effective class module and qualified name, fast/slow status, directly relevant package versions, SHA-256 and size for every saved asset, and outputs for versioned canaries covering ordinary text, whitespace, Unicode, byte-fallback-style text, newlines, and special-token-like strings. The byte tokenizer has no asset files but uses the same implementation/output manifest.

Resume validates the saved-file set, effective implementation, and canary outputs before evaluation or training streams are built and before Grain iterator state is restored. Each checkpoint stores the manifest digest. Under `checkpoint.resume_compat: strict`, a missing or changed manifest/digest is an error; `warn` reports that equivalence is unproven and records the observed identity in subsequent checkpoints.

`data.tokenizer.max_doc_tokens: null` means no truncation. A positive cap is explicit data loss applied after tokenization and before BOS/EOS insertion; the iterator reports documents truncated, source tokens observed/retained/discarded, and discarded-token fraction. The cap does not reduce the tokenizer's peak work because the full document is encoded first.

## Packing

The pipeline supports `sequential`, `bin`, and `multipack` packing modes and always emits fixed windows of length `seq_len` before batching. Packing trade-offs and boundary-masking behavior are documented in [Packing and Boundary Semantics](packing.md).

Between the packer and batch assembly, packed windows pass through a seeded, token- and row-bounded window shuffle (`data.window_shuffle_tokens` and `data.window_shuffle_max_rows`, train iterator only) so batches are decorrelated from raw stream order; see [Window shuffling](packing.md#window-shuffling-batch-decorrelation).

The HF document shuffle ends each window at the first of `data.shuffle_buffer_size` documents or `data.shuffle_buffer_bytes` UTF-8 payload bytes. Runtime packing metrics expose current and peak document-window counts/bytes plus replay totals. The byte limit may be exceeded by one oversized document because reading ahead and carrying that document outside the compact checkpoint state would break exact replay.

## Grain iterator

The training pipeline is composed as: unshuffled HF source -> optional chomp-owned document-window shuffle -> tokenizer/packer -> optional Grain packed-window shuffle -> batch assembly -> optional prefetch. The wrapper provides:

- deterministic iteration
- optional threaded prefetch (`data.grain_prefetch`)
- a checkpointable iterator state (`get_state` / `set_state`)
- host-side packing diagnostics per emitted batch (`get_stats`)
- an exact host loss-token count paired with the batch through prefetch and device transfer (`get_loss_tokens`)

The packing iterator itself remains a small, explicit Python object; Grain only wraps it for performance and checkpoint integration.

The iterator exposes packing utilization, exact host loss-token counts, and boundary/segment summaries. Their logged field names and meanings are listed under [Training metrics](training.md#metrics).

## Iterator state and resume

The iterator exposes checkpointable state containing the source cursor (HF or local text), document- and packed-window shuffle replay progress, packer buffered/ready queues, and enabled prefetch-wrapper state.

Bin and multipack queue rows are checkpointed as flat int32 payloads with row offsets, not nested token lists. This changes serialization shape only; FIFO queue order and emitted rows are identical.

This is checkpointed alongside the model so resume does not rely on `.skip()` or re-streaming.

Hugging Face's streaming shuffle omits the contents of its read-ahead buffer from `state_dict()`, so chomp never calls it in the checkpointed path. Chomp instead permutes disjoint document windows. State stores the unshuffled source position at the current window's start, its index, and the output cursor; a restore reconstructs the window deterministically without inflating the checkpoint with document text. Iterator state cannot compensate for an upstream repository changing beneath the same name, so checkpoint metadata records both the requested ref and the concrete commit used by the stream. Resume reuses that commit without querying the live ref only when the selected checkpoint has the same repository and requested ref; deliberate changes resolve normally and flow through resume compatibility checks.

The configured `data.text_key` must exist in every selected row and contain a string. Missing fields and non-string values are deterministic schema failures: Chomp does not stringify them or retry them.

## Validation set

chomp builds a fixed validation set when the run is created:

- If `data.hf_eval_split` is set, it is authoritative and never falls back to training data. It must differ from `data.hf_split` while evaluation is enabled.
- The selected split is read unshuffled in literal order (`data.shuffle` applies only to the training stream) and contributes at most `data.max_eval_samples` examples.
- With `data.hf_eval_split: null`, a stable BLAKE2 content hash reserves `data.hf_eval_holdout_fraction` of identities for eval and removes them from the training stream. Duplicate content always lands on the same side. The sparse hash selection does not also fill a document-shuffle window; doing so would multiply startup reads by roughly the inverse holdout fraction.
- A positive `data.max_eval_samples` that yields no documents, or any loading/authentication/schema/decoding failure, follows `train.eval_failure_policy`: `fatal` fails startup, while `disable` logs the failure and starts training with future evaluation disabled.

The selected documents are tokenized once when a process starts, then the resulting eval batches are reused for every evaluation in that process. For `bin` and `multipack`, evaluation's effective packing lookahead is the larger of the configured lookahead and `train.batch_size`; checkpoint compatibility tracks it separately from training's `train.batch_size * train.grad_accum` minimum.

After the first eval batch assembly, the original tokenized Python lists are released because the packed arrays own the payload. TODO: bound initial evaluation collection by tokens rather than only `data.max_eval_samples`, and replace the Python-list cache with a compact contiguous int32 ragged representation. This needs an explicit selection/fingerprint policy for the token budget.

At end of stream the `bin`/`multipack` packers flush their remaining pending documents into padded windows, so an eval doc set below the pack threshold still emits windows. Eval uses `A=1` and pads missing final rows independently of `train.grad_accum`. If a scheduled pass yields no usable window, emits a zero-loss-token batch, or otherwise fails, `fatal` fails the run and `disable` stops later evals. Disable mode persists `eval_disabled`, failure count, last failure step/type, and last successful eval step in subsequent local and W&B metrics. Training data assembly retains its strict zero-loss-token failure because advancing the optimizer without an objective would be invalid.
