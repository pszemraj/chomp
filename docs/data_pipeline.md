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

Tokenizer selection and every tokenizer-field contract are defined under [`data.tokenizer`](config-reference.yaml). Before training, Chomp resolves tokenizer-dependent vocabulary and special-token IDs; Hugging Face tokenization serves real pretraining, while the byte tokenizer is for offline infrastructure checks.

Every fresh run writes `run_dir/tokenizer/identity.json`. For Hugging Face tokenizers, chomp saves the tokenizer assets there, reloads them with `local_files_only=true`, and uses that reloaded instance for the run; resume never falls back to `hf_name_or_path`. The manifest records the effective class module and qualified name, fast/slow status, directly relevant third-party package versions, SHA-256 and size for every saved asset, and outputs for versioned canaries covering ordinary text, whitespace, Unicode, byte-fallback-style text, newlines, and special-token-like strings. The byte tokenizer has no asset files or third-party execution dependency but uses the same implementation/output manifest.

Resume validates the saved-file set, effective implementation, and canary outputs before evaluation or training streams are built and before Grain iterator state is restored. Each checkpoint stores the manifest digest; mismatch handling follows [resume compatibility](checkpointing.md#resume-compatibility-checks). Warn mode leaves a valid saved manifest unchanged when the observed identity differs, preserving the original evidence while later checkpoints record the observed identity.

When document truncation is enabled, iterator diagnostics report documents truncated and source tokens observed, retained, and discarded. The truncation stage, constraints, and memory implications are defined by [`data.tokenizer.max_doc_tokens`](config-reference.yaml).

## Packing

Every configured packer emits fixed `seq_len` windows before batching. Algorithms and boundary semantics are documented under [Packing modes](packing.md#packing-modes), with field contracts in the [Config Reference](config-reference.yaml).

Between the packer and batch assembly, packed windows pass through a seeded, token- and row-bounded window shuffle (`data.window_shuffle_tokens` and `data.window_shuffle_max_rows`, train iterator only) so batches are decorrelated from raw stream order; see [Window shuffling](packing.md#window-shuffling-batch-decorrelation).

Chomp owns deterministic bounded Hugging Face document-shuffle windows. Runtime metrics expose current and peak window counts/bytes plus replay totals; sizing and threshold behavior are defined by the [`data.shuffle_buffer_*` fields](config-reference.yaml), while reconstruction and compact-checkpoint rationale are described below.

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

Hugging Face's streaming shuffle omits the contents of its read-ahead buffer from `state_dict()`, so chomp never calls it in the checkpointed path. Chomp instead permutes disjoint document windows. State stores the unshuffled source position at the current window's start, its index, and the output cursor; a restore reconstructs the window deterministically without inflating the checkpoint with document text. Because that cursor has meaning only for the exact reconstructed window, both resume modes refuse changes to the active source-selection, partition, or document-shuffle recipe. Checkpoint metadata binds those fields; their continuation contracts are in the [Config Reference](config-reference.yaml) and [Checkpointing and Resume](checkpointing.md#resume-compatibility-checks).

Source-schema violations fail deterministically rather than being coerced or skipped; the selected-field contract is [`data.text_key`](config-reference.yaml).

## Validation set

When evaluation collection is active, process startup deterministically selects and tokenizes one fixed, unshuffled document set before training. Explicit-split and hash-holdout activation, sizing, and cadence rules are canonical under the [`data.hf_eval_*`, `data.max_eval_samples`, and `train.eval_every` fields](config-reference.yaml). Null-split selection hashes complete text with BLAKE2, so duplicate content cannot cross train/eval; selection never falls back across sources. Initialization and runtime failures share the workflow under [Training evaluation](training.md#evaluation).

Packed batches are materialized on first evaluation, cached host-side, and replace the original Python token lists. How those rows are laid out is [`data.eval_packing`](config-reference.yaml), which is independent of the training packer by default; effective eval packing lookahead is tracked separately from training because their cycle sizes differ, and the exact clamp rules live with the packing-field contracts in the Config Reference. Row layout never changes which documents were selected above.

TODO: bound initial evaluation collection by tokens rather than only `data.max_eval_samples`, and replace the Python-list cache with a compact contiguous int32 ragged representation. This needs an explicit selection/fingerprint policy for the token budget.

At end of stream the `bin`/`multipack` packers flush their remaining pending documents into padded windows, so an eval doc set below the pack threshold still emits windows. Eval uses `A=1` and pads missing final rows independently of `train.grad_accum`; failures follow that evaluation policy. Training data assembly retains its strict zero-loss-token failure because advancing the optimizer without an objective would be invalid.
