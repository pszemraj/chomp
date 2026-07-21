# SPDX-License-Identifier: Apache-2.0

"""Minimal data pipeline for chomp (Phases 0–4-ish).

This is intentionally *not* a framework.

Goal for v0:
- Consume Zyphra/Zyda-2 (streaming) by default
- Tokenize + pack into fixed-shape microbatches [A, B, T]
- Provide get_state/set_state hooks so checkpoint+resume is real

This module implements the core iterator used by the Grain wrapper in
`chomp.data.grain`. The Grain layer handles prefetching, but the packing and
state semantics live here.

Why remove synthetic batches?
Because synthetic batches turn into a crutch: people think the trainer works
when it doesn't survive contact with real streaming data.

This pipeline keeps debug sources (local_text) but *still* exercises tokenize+pack.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from chomp.config import Config, resolve_window_shuffle_rows, validate_config
from chomp.types import IGNORE_INDEX, Batch

from .hf import ContentPartition, HFStreamingTextStream, HFStreamSpec, LocalTextStream
from .pack import FFDPacker, TokenPacker

_WINDOW_SHUFFLE_SEED_OFFSET = 104_729
_UINT32_MODULUS = 2**32


def effective_window_shuffle_seed(cfg: Config) -> int:
    """Return the deterministic seed consumed by packed-window shuffling.

    :param Config cfg: Training configuration.
    :return int: Effective Grain window-shuffle seed.
    """
    return (int(cfg.data.seed) + _WINDOW_SHUFFLE_SEED_OFFSET) % _UINT32_MODULUS


class Tokenizer(Protocol):
    """Protocol for tokenizers that convert text to token ids."""

    def encode(self, text: str) -> list[int]:
        """Encode text string to a list of token ids."""
        ...

    def decode(self, tokens: list[int], *, skip_special_tokens: bool = True) -> str:
        """Decode token ids back into a text string.

        :param list[int] tokens: Token ids to decode.
        :param bool skip_special_tokens: If True, drop special tokens.
        :return str: Decoded text.
        """
        ...

    def __len__(self) -> int: ...


TextItem = str | list[int]


class TextStream(Protocol):
    """Protocol for text streams used by the packer."""

    def __next__(self) -> TextItem: ...

    def get_state(self) -> dict[str, Any]:
        """Return stream state for checkpointing."""
        ...

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore stream state from a checkpoint."""
        ...

    def close(self) -> None:
        """Release resources owned by the stream."""
        ...


logger = logging.getLogger(__name__)


class ZeroLossTokensError(RuntimeError):
    """Raised when a complete batch contains no valid causal targets."""


@dataclass
class ByteTokenizer:
    """A tiny byte-level tokenizer.

    It maps UTF-8 bytes to token ids.

    If `byte_offset>0`, it reserves ids [0..byte_offset-1] for special tokens
    and maps raw bytes 0..255 to [byte_offset..byte_offset+255].

    This is not intended for serious pretraining quality. It's an infrastructure tool.
    """

    byte_offset: int = 0

    def encode(self, text: str) -> list[int]:
        """Encode text to token ids by mapping UTF-8 bytes with offset.

        :param str text: Input text string.
        :return list[int]: Token ids (byte values + byte_offset).
        """
        b = text.encode("utf-8", errors="replace")
        off = int(self.byte_offset)
        return [off + int(x) for x in b]

    def decode(self, tokens: list[int], *, skip_special_tokens: bool = True) -> str:
        """Decode token ids back into UTF-8 text.

        :param list[int] tokens: Token ids to decode.
        :param bool skip_special_tokens: If True, drop tokens < byte_offset.
        :return str: Decoded text.
        """
        off = int(self.byte_offset)
        out = bytearray()
        for tok in tokens:
            val = int(tok)
            if val < off:
                if skip_special_tokens:
                    continue
                out.append(ord("?"))
                continue
            out.append(val - off)
        return bytes(out).decode("utf-8", errors="replace")

    def __len__(self) -> int:
        return int(self.byte_offset) + 256


class HFTokenizer:
    """Hugging Face tokenizer wrapper.

    Requires `transformers` (included in default install).
    """

    def __init__(self, name_or_path: str, *, use_fast: bool, trust_remote_code: bool):
        """Initialize HuggingFace tokenizer from name or local path.

        :param str name_or_path: HuggingFace model name or local path.
        :param bool use_fast: Whether to use fast Rust tokenizer.
        :param bool trust_remote_code: Whether to allow custom tokenizer code.
        """
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(
            name_or_path,
            use_fast=use_fast,
            trust_remote_code=trust_remote_code,
        )

        # Ensure we have a pad token to avoid weirdness.
        if self._tok.pad_token is None and self._tok.eos_token is not None:
            self._tok.pad_token = self._tok.eos_token

    def encode(self, text: str) -> list[int]:
        """Encode text to token ids without adding special tokens.

        :param str text: Input text string.
        :raises RuntimeError: If tokenizer does not return input_ids.
        :return list[int]: Token ids.
        """
        out = self._tok(text, add_special_tokens=False)
        ids = out.get("input_ids")
        if ids is None:
            raise RuntimeError("Tokenizer did not return input_ids")
        return list(ids)

    def decode(self, tokens: list[int], *, skip_special_tokens: bool = True) -> str:
        """Decode tokens back into text.

        :param list[int] tokens: Token ids to decode.
        :param bool skip_special_tokens: If True, drop special tokens.
        :return str: Decoded text.
        """
        return self._tok.decode(list(tokens), skip_special_tokens=skip_special_tokens)

    def __len__(self) -> int:
        return int(len(self._tok))

    @property
    def bos_token_id(self) -> int | None:
        """Beginning-of-sequence token ID, or None if not defined.

        :return int | None: BOS token ID.
        """
        return self._tok.bos_token_id

    @property
    def eos_token_id(self) -> int | None:
        """End-of-sequence token ID, or None if not defined.

        :return int | None: EOS token ID.
        """
        return self._tok.eos_token_id

    @property
    def pad_token_id(self) -> int | None:
        """Padding token ID, or None if not defined.

        :return int | None: PAD token ID.
        """
        return self._tok.pad_token_id

    def save_pretrained(self, path: str | Path) -> None:
        """Save tokenizer files to a directory.

        :param path: Directory path to save tokenizer files.
        """
        self._tok.save_pretrained(str(path))


def build_tokenizer(cfg: Config) -> Tokenizer:
    """Build a tokenizer instance from config.

    :param Config cfg: Configuration with tokenizer settings.
    :raises ValueError: If tokenizer kind is unknown.
    :return Tokenizer: Configured tokenizer instance.
    """
    tok = cfg.data.tokenizer
    if tok.kind == "byte":
        return ByteTokenizer(byte_offset=tok.byte_offset)
    if tok.kind == "hf":
        assert tok.hf_name_or_path is not None
        return HFTokenizer(
            tok.hf_name_or_path,
            use_fast=tok.hf_use_fast,
            trust_remote_code=tok.hf_trust_remote_code,
        )
    raise ValueError(f"Unknown tokenizer.kind: {tok.kind!r}")


def _hf_source_fields(
    cfg: Config,
    *,
    split: str,
    repeat: bool,
    content_partition: ContentPartition = "all",
) -> dict[str, Any]:
    """Resolve the effective HF source fields shared by runtime and artifacts.

    :param Config cfg: Training configuration.
    :param str split: Dataset split name.
    :param bool repeat: Whether to repeat the stream when exhausted.
    :param ContentPartition content_partition: All, training, or held-out documents.
    :return dict[str, Any]: Fields accepted by :class:`HFStreamSpec`.
    """
    return {
        "dataset": cfg.data.hf_dataset,
        "name": cfg.data.hf_name,
        "split": split,
        "text_key": cfg.data.text_key,
        "revision": cfg.data.hf_revision,
        # Hash selection already samples sparsely across the source. Filling a
        # second document-shuffle window would scan roughly
        # shuffle_buffer_size / holdout_fraction source documents before the
        # first eval row, for no disjointness benefit.
        "shuffle": cfg.data.shuffle and content_partition != "eval",
        "shuffle_buffer_size": cfg.data.shuffle_buffer_size,
        "shuffle_buffer_bytes": cfg.data.shuffle_buffer_bytes,
        "seed": int(cfg.data.seed),
        "repeat": repeat,
        "content_partition": content_partition,
        "eval_holdout_fraction": cfg.data.hf_eval_holdout_fraction,
    }


def _hf_source_identity(fields: dict[str, Any]) -> dict[str, Any]:
    """Return only source fields that affect effective stream behavior.

    :param dict[str, Any] fields: Resolved fields from :func:`_hf_source_fields`.
    :return dict[str, Any]: Stable source identity for caches and checkpoints.
    """
    identity = dict(fields)
    if not identity["shuffle"]:
        for key in ("shuffle_buffer_size", "shuffle_buffer_bytes", "seed"):
            identity.pop(key)
    if identity["content_partition"] == "all":
        identity.pop("eval_holdout_fraction")
    return identity


def _build_hf_stream(
    cfg: Config,
    *,
    split: str,
    repeat: bool,
    content_partition: ContentPartition = "all",
) -> HFStreamingTextStream:
    """Build an HF streaming text stream from config.

    :param Config cfg: Training configuration.
    :param str split: Dataset split name.
    :param bool repeat: Whether to repeat the stream when exhausted.
    :param ContentPartition content_partition: All, training, or held-out documents.
    :return HFStreamingTextStream: Streaming text stream wrapper.
    """
    spec = HFStreamSpec(
        **_hf_source_fields(
            cfg,
            split=split,
            repeat=repeat,
            content_partition=content_partition,
        )
    )
    return HFStreamingTextStream(spec)


def _round_up_to_multiple(value: int, multiple: int) -> int:
    """Round value up to the nearest multiple for aligned tensor shapes.

    :param int value: Value to round.
    :param int multiple: Multiple to round to.
    :return int: Rounded value.
    """
    if multiple <= 1:
        return value
    return int(((value + multiple - 1) // multiple) * multiple)


def resolve_tokenizer_config(cfg: Config, tok: Tokenizer) -> Config:
    """Resolve tokenizer-derived model fields (vocab size + special token IDs).

    :param Config cfg: Input configuration.
    :param Tokenizer tok: Tokenizer instance.
    :raises RuntimeError: If tokenizer doesn't expose vocab size.
    :raises ValueError: If vocab size is invalid or special tokens missing.
    :return Config: Updated config with tokenizer-derived fields.
    """

    try:
        tok_vocab = int(len(tok))
    except Exception as exc:
        raise RuntimeError("Tokenizer must expose vocab size via __len__") from exc

    if tok_vocab <= 0:
        raise ValueError(f"Tokenizer vocab size must be positive, got {tok_vocab}")

    multiple = int(cfg.data.tokenizer.vocab_size_multiple)
    if multiple <= 0:
        raise ValueError(f"data.tokenizer.vocab_size_multiple must be positive, got {multiple}")

    requested_vocab = int(cfg.model.vocab_size)
    base_vocab = max(requested_vocab, tok_vocab)
    rounded_vocab = _round_up_to_multiple(base_vocab, multiple)

    model_updates: dict[str, int] = {}
    if rounded_vocab != cfg.model.vocab_size:
        logger.info(
            "Adjusting model.vocab_size from %d to %d (tokenizer=%d, multiple=%d).",
            cfg.model.vocab_size,
            rounded_vocab,
            tok_vocab,
            multiple,
        )
        model_updates["vocab_size"] = rounded_vocab

    if cfg.data.tokenizer.kind == "hf" and cfg.data.tokenizer.auto_set_special_tokens:
        tok_bos = getattr(tok, "bos_token_id", None)
        tok_eos = getattr(tok, "eos_token_id", None)
        tok_pad = getattr(tok, "pad_token_id", None)

        if cfg.data.tokenizer.add_bos and tok_bos is None:
            raise ValueError("HF tokenizer has no bos_token_id but data.tokenizer.add_bos=true")
        if cfg.data.tokenizer.add_eos and tok_eos is None:
            raise ValueError("HF tokenizer has no eos_token_id but data.tokenizer.add_eos=true")

        def _maybe_update(field: str, value: int | None) -> None:
            """Update model field from tokenizer value if different.

            :param str field: Model config field name.
            :param value: Tokenizer-provided value, or None to skip.
            """
            if value is None:
                return
            cur = getattr(cfg.model, field)
            if cur != value:
                logger.info("Using tokenizer %s=%d (config had %d).", field, value, cur)
                model_updates[field] = int(value)

        _maybe_update("bos_token_id", tok_bos)
        _maybe_update("eos_token_id", tok_eos)
        _maybe_update("pad_token_id", tok_pad)

    updated_cfg = cfg
    if model_updates:
        updated_cfg = replace(updated_cfg, model=replace(updated_cfg.model, **model_updates))

    effective_vocab = int(updated_cfg.model.vocab_size)
    for field in ("pad_token_id", "bos_token_id", "eos_token_id"):
        token_id = int(getattr(updated_cfg.model, field))
        if not 0 <= token_id < effective_vocab:
            raise ValueError(
                f"model.{field} must be within [0, resolved vocab_size={effective_vocab}), "
                f"got {token_id}"
            )

    # Re-validate after tokenizer-derived updates (vocab rounding, special tokens).
    validate_config(updated_cfg)
    return updated_cfg


def prepare_tokenizer_and_config(
    cfg: Config, *, tokenizer: Tokenizer | None = None
) -> tuple[Config, Tokenizer]:
    """Build tokenizer and return an updated config with tokenizer-derived fields.

    :param Config cfg: Input configuration.
    :param Tokenizer | None tokenizer: Optional pre-built tokenizer override.
    :return tuple: (updated_config, tokenizer) tuple.
    """

    tok = tokenizer or build_tokenizer(cfg)
    cfg = resolve_tokenizer_config(cfg, tok)
    return cfg, tok


def save_tokenizer_snapshot(
    run_dir: Path, cfg: Config, tok: Tokenizer, *, allow_existing: bool
) -> None:
    """Persist a Hugging Face tokenizer with the run.

    :param Path run_dir: Run directory path.
    :param Config cfg: Training configuration.
    :param Tokenizer tok: Tokenizer instance to save.
    :param bool allow_existing: Whether this is an existing run.
    """
    if cfg.data.tokenizer.kind != "hf" or allow_existing:
        return

    tok_dir = Path(run_dir) / "tokenizer"
    tok_dir.mkdir()
    tok.save_pretrained(tok_dir)  # type: ignore[attr-defined]


def load_tokenizer_snapshot(run_dir: Path, cfg: Config) -> Tokenizer:
    """Load a tokenizer snapshot from a run directory.

    :param Path run_dir: Run directory containing tokenizer snapshot.
    :param Config cfg: Training configuration.
    :return Tokenizer: Restored Hugging Face tokenizer instance.
    """
    tok_dir = Path(run_dir) / "tokenizer"
    return HFTokenizer(
        str(tok_dir),
        use_fast=cfg.data.tokenizer.hf_use_fast,
        trust_remote_code=cfg.data.tokenizer.hf_trust_remote_code,
    )


def _collect_texts(stream: TextStream, max_samples: int) -> list[str]:
    """Collect up to max_samples texts from a stream.

    :param TextStream stream: Text stream to read from.
    :param int max_samples: Maximum number of samples to collect.
    :return list[str]: Collected text samples.
    """
    texts: list[str] = []
    try:
        for _ in range(int(max_samples)):
            try:
                texts.append(next(stream))
            except StopIteration:
                break
        return texts
    finally:
        stream.close()


def _content_holdout_enabled(cfg: Config) -> bool:
    """Return whether the HF training split is content-partitioned.

    :param Config cfg: Training configuration.
    :return bool: True when evaluation consumes a hash holdout from the train split.
    """
    return (
        cfg.data.backend == "hf"
        and cfg.data.hf_eval_split is None
        and cfg.data.max_eval_samples > 0
    )


def _eval_source_split(cfg: Config) -> str:
    """Resolve the exact source selector used to build evaluation documents.

    :param Config cfg: Training configuration.
    :return str: HF split name or the local-text source marker.
    """
    if cfg.data.backend == "local_text":
        return "local_text"
    return cfg.data.hf_eval_split or cfg.data.hf_split


def load_or_create_eval_tokens(cfg: Config, *, tokenizer: Tokenizer) -> list[list[int]]:
    """Build the evaluation token set from the configured source.

    :param Config cfg: Training configuration.
    :param Tokenizer tokenizer: Tokenizer used to pre-tokenize eval texts.
    :return list[list[int]]: Tokenized documents for evaluation.
    """
    max_samples = int(cfg.data.max_eval_samples)
    if max_samples <= 0:
        return []

    if cfg.data.backend == "hf":
        split = _eval_source_split(cfg)
        content_partition: ContentPartition = "eval" if _content_holdout_enabled(cfg) else "all"
        try:
            stream = _build_hf_stream(
                cfg,
                split=split,
                repeat=False,
                content_partition=content_partition,
            )
            texts = _collect_texts(stream, max_samples)
        except Exception as exc:
            selection = (
                "data.hf_eval_split" if cfg.data.hf_eval_split is not None else "data.hf_split"
            )
            raise RuntimeError(
                f"Failed to collect evaluation documents from {selection}={split!r}. "
                "Evaluation never falls back after an error. Use a valid explicit split, "
                "or set data.hf_eval_split=null for a disjoint content-hash holdout."
            ) from exc
    elif cfg.data.backend == "local_text":
        texts = [cfg.data.local_text] * max_samples
    else:
        raise RuntimeError(f"Unknown data.backend for eval: {cfg.data.backend!r}")

    if not texts:
        raise RuntimeError(
            "Evaluation collected zero documents while data.max_eval_samples is positive "
            f"({max_samples}) from source {_eval_source_split(cfg)!r}. Refusing to silently "
            "disable evaluation; fix the source/split or set data.max_eval_samples=0 explicitly."
        )

    return [tokenizer.encode(text) for text in texts]


def _build_backend_text_stream(cfg: Config) -> TextStream:
    """Build the configured backend's text stream over the training split.

    :param Config cfg: Training configuration.
    :return TextStream: Streaming text iterator.
    :raises ValueError: If data.backend is unknown.
    """
    if cfg.data.backend == "hf":
        content_partition: ContentPartition = "train" if _content_holdout_enabled(cfg) else "all"
        return _build_hf_stream(
            cfg,
            split=cfg.data.hf_split,
            repeat=cfg.data.repeat,
            content_partition=content_partition,
        )
    if cfg.data.backend == "local_text":
        return LocalTextStream(text=cfg.data.local_text, repeat=cfg.data.repeat)
    raise ValueError(f"Unknown data.backend: {cfg.data.backend!r}")


def load_generation_prompt_tokens(
    cfg: Config, *, tokenizer: Tokenizer, max_samples: int = 16
) -> list[list[int]]:
    """Load a small bounded prompt pool without a document-shuffle buffer.

    Periodic generation is diagnostic and not checkpointed. Reading a bounded,
    unshuffled pool avoids retaining a second production-sized HF shuffle
    window for the lifetime of the run.

    :param Config cfg: Training configuration.
    :param Tokenizer tokenizer: Tokenizer for prompt documents.
    :param int max_samples: Maximum prompt documents to retain.
    :return list[list[int]]: Tokenized training-source prompts.
    """
    if max_samples <= 0:
        return []
    prompt_cfg = replace(
        cfg,
        data=replace(cfg.data, shuffle=False, repeat=False),
    )
    texts = _collect_texts(_build_backend_text_stream(prompt_cfg), max_samples)
    return [tokenizer.encode(text) for text in texts]


def data_fingerprint(cfg: Config) -> dict[str, Any]:
    """A small, stable fingerprint that we store in checkpoint meta.

    :param Config cfg: Training configuration.
    :return dict[str, Any]: Fingerprint dict with source, tokenizer, and batch shape info.
    """

    d = cfg.data
    t = cfg.data.tokenizer
    if d.backend == "hf":
        content_partition: ContentPartition = "train" if _content_holdout_enabled(cfg) else "all"
        fields = _hf_source_fields(
            cfg,
            split=d.hf_split,
            repeat=d.repeat,
            content_partition=content_partition,
        )
        src = {"backend": "hf", **_hf_source_identity(fields)}
    else:
        src = {
            "backend": "local_text",
            "repeat": d.repeat,
            "local_text": d.local_text,
        }

    tok = {
        "kind": t.kind,
        "hf_name_or_path": t.hf_name_or_path,
        "hf_use_fast": t.hf_use_fast,
        "hf_trust_remote_code": t.hf_trust_remote_code,
        "byte_offset": t.byte_offset,
        "add_bos": t.add_bos,
        "add_eos": t.add_eos,
        "max_doc_tokens": t.max_doc_tokens,
        "vocab_size_multiple": t.vocab_size_multiple,
        "auto_set_special_tokens": t.auto_set_special_tokens,
    }
    # Record only active mode knobs and effective shuffle geometry so inert
    # defaults and raw budgets cannot reject a behaviorally identical resume.
    # Thread prefetch is deliberately absent: Grain serializes the parent
    # state paired with the last consumer-delivered batch, not its queued rows.
    window_shuffle_rows = resolve_window_shuffle_rows(cfg)
    packing = {
        "mode": d.packing_mode,
        "mask_boundary_loss": d.mask_boundary_loss,
        "train_on_eos": d.train_on_eos,
        "window_shuffle_rows": window_shuffle_rows,
    }
    if window_shuffle_rows > 0:
        # The shuffle reconstructs current and future windows from this seed
        # after restore. Keep the fingerprint tied to the effective value used
        # by Grain so changing either data.seed or the internal offset cannot
        # silently change replay order.
        packing["window_shuffle_seed"] = effective_window_shuffle_seed(cfg)
    if d.packing_mode in ("bin", "multipack"):
        packing["max_docs_per_bin"] = d.packing_max_docs_per_bin
        packing["strict_segments"] = d.packing_strict_segments
    if d.packing_mode == "bin":
        packing["buffer_docs"] = d.packing_buffer_docs
    if d.packing_mode == "multipack":
        packing["group_docs"] = d.packing_group_docs
    return {
        "source": src,
        "tokenizer": tok,
        "packing": packing,
        "seq_len": cfg.train.seq_len,
        "batch_size": cfg.train.batch_size,
        "grad_accum": cfg.train.grad_accum,
    }


def _mask_labels(
    labels: np.ndarray,
    segs: np.ndarray,
    *,
    mask_boundary_loss: bool,
    train_on_eos: bool,
    eos_id: int,
) -> np.ndarray:
    """Apply boundary and EOS masking to label array.

    :param np.ndarray labels: Label array of length T.
    :param np.ndarray segs: Segment IDs of length T.
    :param bool mask_boundary_loss: Mask labels at segment transitions.
    :param bool train_on_eos: Keep EOS positions in the loss.
    :param int eos_id: EOS token id used when train_on_eos is False.
    :return np.ndarray: Masked labels of length T.
    """
    if mask_boundary_loss:
        same = (segs[1:] == segs[:-1]) & (segs[1:] > 0) & (segs[:-1] > 0)
        labels[1:] = np.where(same, labels[1:], IGNORE_INDEX).astype(np.int32)
    if not train_on_eos:
        labels = np.where(labels == eos_id, IGNORE_INDEX, labels).astype(np.int32)
    return labels


@dataclass(frozen=True)
class _BatchAssemblySpec:
    """Batch-assembly knobs extracted from a Config.

    Single source of the config-to-assembly mapping shared by the Grain train
    path and the eval iterator, so the two paths cannot drift apart.
    """

    grad_accum: int
    batch_size: int
    seq_len: int
    mask_boundary_loss: bool
    train_on_eos: bool
    eos_id: int
    pad_id: int

    @staticmethod
    def from_config(cfg: Config, *, grad_accum: int | None = None) -> _BatchAssemblySpec:
        """Extract the batch-assembly knobs from a training config.

        :param Config cfg: Training configuration.
        :param grad_accum: Optional assembly-only accumulation axis override.
        :return _BatchAssemblySpec: Immutable assembly parameters.
        """
        return _BatchAssemblySpec(
            grad_accum=int(cfg.train.grad_accum if grad_accum is None else grad_accum),
            batch_size=int(cfg.train.batch_size),
            seq_len=int(cfg.train.seq_len),
            mask_boundary_loss=bool(cfg.data.mask_boundary_loss),
            train_on_eos=bool(cfg.data.train_on_eos),
            eos_id=int(cfg.model.eos_token_id),
            pad_id=int(cfg.model.pad_token_id),
        )


def _assemble_batch(
    next_window: Callable[[], tuple[np.ndarray, np.ndarray]],
    spec: _BatchAssemblySpec,
) -> tuple[Batch, int]:
    """Assemble a fixed-shape batch and its exact valid-target count.

    This is the single source for label masking, padding, zero-objective
    rejection, and host loss-token accounting used by train and eval.

    :param next_window: Callable yielding (tokens, segment_ids) [T] arrays.
    :param _BatchAssemblySpec spec: Assembly knobs extracted from the config.
    :return tuple[Batch, int]: Fixed-shape batch and valid shifted-target count.
    """
    grad_accum, batch_size, seq_len = spec.grad_accum, spec.batch_size, spec.seq_len
    need = grad_accum * batch_size
    inps = np.full((need, seq_len), spec.pad_id, dtype=np.int32)
    labs = np.full((need, seq_len), IGNORE_INDEX, dtype=np.int32)
    segs_out = np.zeros((need, seq_len), dtype=np.int32)

    for idx in range(need):
        try:
            seq, segs = next_window()  # [T]
        except StopIteration:
            if idx == 0:
                raise
            break
        # Labels align with input_ids; the model shifts internally.
        inp = np.asarray(seq, dtype=np.int32)
        labs[idx] = _mask_labels(
            inp.copy(),
            segs,
            mask_boundary_loss=spec.mask_boundary_loss,
            train_on_eos=spec.train_on_eos,
            eos_id=spec.eos_id,
        )
        inps[idx] = inp
        segs_out[idx] = np.asarray(segs, dtype=np.int32)

    valid_targets = (labs[:, 1:] != IGNORE_INDEX) & (segs_out[:, 1:] > 0)
    loss_tokens_host = int(np.count_nonzero(valid_targets))
    if loss_tokens_host == 0:
        raise ZeroLossTokensError(
            "Batch contains zero valid loss tokens after causal shift, boundary/EOS masking, "
            "and padding. Check for one-token documents, tokenizer special-token collisions, "
            "or an over-restrictive masking configuration. Refusing to advance the optimizer, "
            "schedule, RNG, or training step without an objective."
        )

    segs_abt = segs_out.reshape(grad_accum, batch_size, seq_len)
    batch = Batch(
        input_ids=inps.reshape(grad_accum, batch_size, seq_len),
        labels=labs.reshape(grad_accum, batch_size, seq_len),
        segment_ids=segs_abt,
    )
    return batch, loss_tokens_host


def _build_packer(cfg: Config, *, rows_per_pack: int | None = None) -> TokenPacker | FFDPacker:
    """Create the configured token packer.

    :param Config cfg: Training configuration.
    :param rows_per_pack: Optional FFD output-row count per packing cycle.
    :return TokenPacker | FFDPacker: Configured packer.
    """
    bins_per_pack = (
        int(cfg.train.grad_accum) * int(cfg.train.batch_size)
        if rows_per_pack is None
        else int(rows_per_pack)
    )
    common: dict[str, Any] = {
        "seq_len": cfg.train.seq_len,
        "add_bos": cfg.data.tokenizer.add_bos,
        "add_eos": cfg.data.tokenizer.add_eos,
        "bos_id": cfg.model.bos_token_id,
        "eos_id": cfg.model.eos_token_id,
        "max_doc_tokens": cfg.data.tokenizer.max_doc_tokens,
        "pad_id": cfg.model.pad_token_id,
    }
    if cfg.data.packing_mode in ("bin", "multipack"):
        lookahead_docs = (
            cfg.data.packing_buffer_docs
            if cfg.data.packing_mode == "bin"
            else cfg.data.packing_group_docs
        )
        return FFDPacker(
            **common,
            mode=cfg.data.packing_mode,
            bins_per_pack=bins_per_pack,
            lookahead_docs=lookahead_docs,
            max_docs_per_bin=cfg.data.packing_max_docs_per_bin,
        )
    if cfg.data.packing_mode == "sequential":
        return TokenPacker(**common)
    raise ValueError(f"Unsupported packing mode: {cfg.data.packing_mode!r}")


class _SequenceProducer:
    """Produces packed [T] windows from a text stream + packer.

    This is the single source of data-order truth: one text stream feeding one
    packer, popped one `[T]` window at a time. Batch assembly (and any window
    shuffling between the two) lives elsewhere.

    Implements `get_state`/`set_state` for resume correctness.
    """

    def __init__(
        self,
        cfg: Config,
        *,
        tokenizer: Tokenizer | None,
        text_stream: Iterator[TextItem] | None = None,
        rows_per_pack: int | None = None,
    ):
        """Initialize the sequence producer.

        :param Config cfg: Training configuration.
        :param Tokenizer | None tokenizer: Tokenizer for string items; None is
            valid only for pre-tokenized streams.
        :param text_stream: Optional text stream override (used for eval datasets).
        :param rows_per_pack: Optional FFD output-row count per packing cycle.
        :raises ValueError: If data.backend is unknown.
        """
        self._tok = tokenizer

        # Text stream
        if text_stream is not None:
            self._text_stream = text_stream
        else:
            self._text_stream = _build_backend_text_stream(cfg)

        self._packer = _build_packer(cfg, rows_per_pack=rows_per_pack)

    def _push_next_document(self) -> None:
        """Fetch one item from the text stream and add it to the packer."""
        item = next(self._text_stream)
        if isinstance(item, str):
            if self._tok is None:
                raise RuntimeError("A tokenizer is required for string stream items")
            ids = self._tok.encode(item)
        elif isinstance(item, list):
            ids = item
        else:
            raise TypeError(
                f"Text stream yielded unsupported item type {type(item).__name__}; "
                "expected str or list[int]."
            )
        self._packer.add_document(ids)

    def close(self) -> None:
        """Release the producer's underlying text stream."""
        close = getattr(self._text_stream, "close", None)
        if callable(close):
            close()

    def next_window(self) -> tuple[np.ndarray, np.ndarray]:
        """Pop the next [T] token and segment-ID arrays from the packer.

        :raises StopIteration: When the text stream is exhausted and the
            packer has nothing left to emit.
        :return tuple[np.ndarray, np.ndarray]: Token and segment-ID arrays.
        """
        while not self._packer.can_pop():
            try:
                self._push_next_document()
            except StopIteration:
                # Stream exhausted (data.repeat=false, or a finite eval doc
                # set): let the packer flush partially filled buffers rather
                # than silently dropping tail documents.
                self._packer.finish()
                if not self._packer.can_pop():
                    raise
                break
        seq, segs = self._packer.pop_seq_with_segments()
        return (
            np.asarray(seq, dtype=np.int32),
            np.asarray(segs, dtype=np.int32),
        )

    # -------- checkpoint hooks --------

    def get_state(self) -> dict[str, Any]:
        """Capture current producer state for checkpointing.

        :return dict[str, Any]: State dict with text stream and packer state.
        """
        return {
            "text": self._text_stream.get_state(),
            "packer": self._packer.get_state(),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore producer state from a checkpoint.

        :param dict[str, Any] state: State dict from get_state().
        """
        self._text_stream.set_state(state["text"])
        self._packer.set_state(state["packer"])

    def get_stats(self) -> dict[str, int | float]:
        """Return source- and packer-level document stats if available.

        :return dict[str, int | float]: Stream memory/replay and packing counters.
        """
        stats: dict[str, int | float] = {}
        get_stream_stats = getattr(self._text_stream, "get_stats", None)
        if callable(get_stream_stats):
            stats.update(get_stream_stats())
        stats.update(self._packer.get_stats())
        return stats


class _EvalBatchIterator:
    """One-pass iterator that assembles fixed-shape evaluation batches.

    This is the data side of the compile-once contract:
    - Every `__next__` yields arrays of exactly the same shape & dtype.

    Batches are assembled directly from pre-tokenized documents in source
    order; evaluation is finite, unshuffled, and never checkpointed.
    """

    def __init__(self, cfg: Config, *, tokens: list[list[int]]):
        """Initialize the evaluation batch iterator.

        :param Config cfg: Training configuration.
        :param list[list[int]] tokens: Ordered pre-tokenized documents.
        """
        self._producer = _SequenceProducer(
            cfg,
            tokenizer=None,
            text_stream=iter(tokens),
            rows_per_pack=int(cfg.train.batch_size),
        )
        # Evaluation has no optimizer accumulation requirement. Keeping A=1
        # prevents train.grad_accum from changing which finite eval rows fit.
        self._spec = _BatchAssemblySpec.from_config(cfg, grad_accum=1)

    def __iter__(self) -> _EvalBatchIterator:
        return self

    def __next__(self) -> Batch:
        batch, _loss_tokens_host = _assemble_batch(self._producer.next_window, self._spec)
        return batch


def build_train_iterator(cfg: Config, *, tokenizer: Tokenizer | None = None) -> Any:
    """Build the training batch iterator.

    :param Config cfg: Training configuration.
    :param tokenizer: Optional pre-built tokenizer; built from config if None.
    :return Any: Iterator yielding fixed-shape Batch objects.
    """
    if tokenizer is None:
        cfg, tokenizer = prepare_tokenizer_and_config(cfg)
    # TODO: multi-source mixing would be inserted here before packing.
    from chomp.data.grain import build_grain_iterator

    return build_grain_iterator(cfg, tokenizer=tokenizer)


def build_eval_iterator(cfg: Config, *, tokens: list[list[int]]) -> Any:
    """Build a one-pass evaluation iterator from tokenized docs.

    :param Config cfg: Training configuration.
    :param list[list[int]] tokens: Tokenized evaluation documents.
    :return Any: Iterator yielding fixed-shape Batch objects.
    """
    return _EvalBatchIterator(cfg, tokens=tokens)
