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

import hashlib
import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from chomp.config import Config, validate_config
from chomp.types import IGNORE_INDEX, Batch

from .hf import HFStreamingTextStream, HFStreamSpec, ListTokenStream, LocalTextStream
from .pack import BinPacker, TokenPacker


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


logger = logging.getLogger(__name__)


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
        :raises ImportError: If transformers is not installed.
        """
        try:
            from transformers import AutoTokenizer
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "HFTokenizer requires transformers. Install with: pip install transformers tokenizers"
            ) from e

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


def _build_hf_stream(
    cfg: Config,
    *,
    split: str,
    repeat: bool,
    seed_offset: int = 0,
    seed_override: int | None = None,
) -> HFStreamingTextStream:
    """Build an HF streaming text stream from config.

    :param Config cfg: Training configuration.
    :param str split: Dataset split name.
    :param bool repeat: Whether to repeat the stream when exhausted.
    :param int seed_offset: Optional seed offset for independent streams.
    :param int | None seed_override: Optional base shuffle seed override.
    :return HFStreamingTextStream: Streaming text stream wrapper.
    """
    base_seed = int(cfg.data.seed) if seed_override is None else int(seed_override)
    spec = HFStreamSpec(
        dataset=cfg.data.hf_dataset,
        name=cfg.data.hf_name,
        split=split,
        text_key=cfg.data.text_key,
        shuffle=cfg.data.shuffle,
        shuffle_buffer_size=cfg.data.shuffle_buffer_size,
        seed=base_seed + int(seed_offset),
        repeat=repeat,
        max_retries=cfg.data.max_retries,
        retry_delay_sec=cfg.data.retry_delay_sec,
        state_update_interval=cfg.data.state_update_interval,
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

    tok_cfg = updated_cfg.data.tokenizer
    if tok_cfg.max_doc_tokens is None:
        default_max = int(updated_cfg.train.seq_len) * 4
        logger.info(
            "data.tokenizer.max_doc_tokens is null; defaulting to %d (4 * seq_len). "
            "Set to 0 to disable truncation.",
            default_max,
        )
        tok_cfg = replace(tok_cfg, max_doc_tokens=default_max)
        updated_cfg = replace(updated_cfg, data=replace(updated_cfg.data, tokenizer=tok_cfg))
    elif tok_cfg.max_doc_tokens <= 0:
        logger.info(
            "data.tokenizer.max_doc_tokens=%d disables truncation; storing as null.",
            tok_cfg.max_doc_tokens,
        )
        tok_cfg = replace(tok_cfg, max_doc_tokens=None)
        updated_cfg = replace(updated_cfg, data=replace(updated_cfg.data, tokenizer=tok_cfg))

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
    """Persist the tokenizer to disk for reproducible resumes.

    :param Path run_dir: Run directory path.
    :param Config cfg: Training configuration.
    :param Tokenizer tok: Tokenizer instance to save.
    :param bool allow_existing: If True, skip if snapshot already exists.
    :raises RuntimeError: If snapshot exists and allow_existing=False.
    """

    tok_dir = Path(run_dir) / "tokenizer"
    if tok_dir.exists():
        if allow_existing:
            return
        raise RuntimeError(f"Tokenizer snapshot already exists: {tok_dir}")

    tok_dir.mkdir(parents=True, exist_ok=False)

    if hasattr(tok, "save_pretrained"):
        try:
            tok.save_pretrained(tok_dir)  # type: ignore[call-arg]
            return
        except Exception as exc:
            raise RuntimeError(f"Failed to save HF tokenizer to {tok_dir}") from exc

    record = {
        "kind": cfg.data.tokenizer.kind,
        "vocab_size": int(cfg.model.vocab_size),
        "byte_offset": cfg.data.tokenizer.byte_offset,
        "add_bos": cfg.data.tokenizer.add_bos,
        "add_eos": cfg.data.tokenizer.add_eos,
        "bos_token_id": int(cfg.model.bos_token_id),
        "eos_token_id": int(cfg.model.eos_token_id),
        "pad_token_id": int(cfg.model.pad_token_id),
    }
    (tok_dir / "tokenizer.json").write_text(
        json.dumps(record, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_tokenizer_snapshot(run_dir: Path, cfg: Config) -> Tokenizer:
    """Load a tokenizer snapshot from a run directory.

    :param Path run_dir: Run directory containing tokenizer snapshot.
    :param Config cfg: Training configuration (used to pick tokenizer kind).
    :return Tokenizer: Restored tokenizer instance.
    :raises FileNotFoundError: If tokenizer snapshot is missing.
    :raises RuntimeError: If snapshot is invalid or incompatible.
    """
    tok_dir = Path(run_dir) / "tokenizer"
    if not tok_dir.exists():
        raise FileNotFoundError(f"Tokenizer snapshot not found at {tok_dir}")

    tok_cfg = cfg.data.tokenizer
    if tok_cfg.kind == "byte":
        record_path = tok_dir / "tokenizer.json"
        if not record_path.exists():
            raise RuntimeError(f"Byte tokenizer snapshot missing {record_path}")
        record = json.loads(record_path.read_text(encoding="utf-8") or "{}")
        kind = record.get("kind")
        if kind != "byte":
            raise RuntimeError(f"Tokenizer snapshot kind mismatch: expected 'byte', found {kind!r}")
        return ByteTokenizer(byte_offset=int(record.get("byte_offset", 0)))

    if tok_cfg.kind == "hf":
        return HFTokenizer(
            str(tok_dir),
            use_fast=tok_cfg.hf_use_fast,
            trust_remote_code=tok_cfg.hf_trust_remote_code,
        )

    raise ValueError(f"Unknown tokenizer.kind: {tok_cfg.kind!r}")


def tokenizer_snapshot_hash(run_dir: Path) -> str | None:
    """Compute a stable hash of the tokenizer snapshot directory.

    :param Path run_dir: Run directory containing tokenizer snapshot.
    :return str | None: SHA256 hash hex digest, or None if snapshot missing.
    """
    tok_dir = Path(run_dir) / "tokenizer"
    if not tok_dir.exists():
        return None

    digest = hashlib.sha256()
    files = [p for p in tok_dir.rglob("*") if p.is_file()]
    for path in sorted(files, key=lambda p: p.relative_to(tok_dir).as_posix()):
        rel = path.relative_to(tok_dir).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _collect_texts(stream: TextStream, max_samples: int) -> list[str]:
    """Collect up to max_samples texts from a stream.

    :param TextStream stream: Text stream to read from.
    :param int max_samples: Maximum number of samples to collect.
    :return list[str]: Collected text samples.
    """
    texts: list[str] = []
    for _ in range(int(max_samples)):
        try:
            texts.append(next(stream))
        except StopIteration:
            break
    return texts


def _tokenize_eval_texts(texts: list[str], tok: Tokenizer) -> list[list[int]]:
    """Tokenize eval texts once for reuse across eval runs.

    :param list[str] texts: Raw text samples.
    :param Tokenizer tok: Tokenizer instance.
    :return list[list[int]]: Tokenized documents.
    """
    return [tok.encode(text) for text in texts]


def load_or_create_eval_texts(cfg: Config, *, tokenizer: Tokenizer) -> list[list[int]]:
    """Create an evaluation set without on-disk caching.

    :param Config cfg: Training configuration.
    :param Tokenizer tokenizer: Tokenizer used to pre-tokenize eval texts.
    :return list[list[int]]: Tokenized documents for evaluation.
    """
    max_samples = int(cfg.data.max_eval_samples)
    if max_samples <= 0:
        return []

    texts: list[str] = []
    split_used: str | None = None

    if cfg.data.backend == "hf":
        split_candidates: list[str] = []
        eval_split = cfg.data.hf_eval_split
        if eval_split is not None and str(eval_split).strip():
            split_candidates.append(str(eval_split))
        if cfg.data.hf_split not in split_candidates:
            split_candidates.append(cfg.data.hf_split)

        split_errors: list[str] = []
        for split in split_candidates:
            try:
                seed_override = None
                if (
                    split == cfg.data.hf_split
                    and int(cfg.data.seed) == 0
                    and int(cfg.train.seed) != 0
                ):
                    # Keep eval-train fallback deterministic across runs by defaulting
                    # to train.seed when data.seed is left at its default 0.
                    seed_override = int(cfg.train.seed)
                    logger.info(
                        "Using train.seed=%d for eval train-split shuffle (data.seed=0).",
                        cfg.train.seed,
                    )

                stream = _build_hf_stream(
                    cfg, split=split, repeat=False, seed_override=seed_override
                )
                texts = _collect_texts(stream, max_samples)
                split_used = split
                break
            except Exception as exc:
                split_errors.append(f"{split!r}: {type(exc).__name__}: {exc}")
                logger.warning("Eval split %r unavailable: %s", split, exc)
                continue
        if split_used is None:
            details = "; ".join(split_errors) if split_errors else "no split candidates"
            raise RuntimeError(
                "Failed to build eval dataset from HF streaming splits. "
                f"Tried: {split_candidates}. Errors: {details}"
            )
    elif cfg.data.backend == "local_text":
        texts = [cfg.data.local_text] * max_samples
        split_used = "local_text"
    else:
        raise RuntimeError(f"Unknown data.backend for eval: {cfg.data.backend!r}")

    if not texts:
        logger.warning("Eval text set is empty (max_eval_samples=%d).", max_samples)

    return _tokenize_eval_texts(texts, tokenizer)


def build_generation_text_stream(cfg: Config, *, seed_offset: int = 1) -> TextStream:
    """Build a text stream for periodic generation prompts.

    Uses the training split but with an optional seed offset so sampling stays
    independent from the training iterator.

    :param Config cfg: Training configuration.
    :param int seed_offset: Offset added to the dataset shuffle seed.
    :return TextStream: Streaming text iterator.
    """
    if cfg.data.backend == "hf":
        return _build_hf_stream(
            cfg,
            split=cfg.data.hf_split,
            repeat=cfg.data.repeat,
            seed_offset=seed_offset,
        )
    if cfg.data.backend == "local_text":
        return LocalTextStream(text=cfg.data.local_text, repeat=cfg.data.repeat)
    raise ValueError(f"Unknown data.backend for generation: {cfg.data.backend!r}")


def data_fingerprint(cfg: Config, *, tokenizer_snapshot_hash: str | None = None) -> dict[str, Any]:
    """A small, stable fingerprint that we store in checkpoint meta.

    :param Config cfg: Training configuration.
    :param str | None tokenizer_snapshot_hash: Optional tokenizer snapshot hash for resume checks.
    :return dict[str, Any]: Fingerprint dict with source, tokenizer, and batch shape info.
    """

    d = cfg.data
    t = cfg.data.tokenizer
    if d.backend == "hf":
        src = {
            "backend": "hf",
            "dataset": d.hf_dataset,
            "name": d.hf_name,
            "split": d.hf_split,
            "text_key": d.text_key,
            "shuffle": d.shuffle,
            "shuffle_buffer_size": d.shuffle_buffer_size,
            "seed": d.seed,
        }
    else:
        src = {
            "backend": "local_text",
            "repeat": d.repeat,
            "local_text_hash": hashlib.sha1(d.local_text.encode("utf-8")).hexdigest(),
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
    if tokenizer_snapshot_hash is not None:
        tok["snapshot_sha256"] = tokenizer_snapshot_hash

    packing = {
        "mode": d.packing_mode,
        "buffer_docs": d.packing_buffer_docs,
        "max_docs_per_bin": d.packing_max_docs_per_bin,
        "mask_boundary_loss": d.mask_boundary_loss,
        "train_on_eos": d.train_on_eos,
        "grain_prefetch": d.grain_prefetch,
    }
    eval_cfg = {
        "max_eval_samples": d.max_eval_samples,
        "hf_eval_split": d.hf_eval_split,
    }

    return {
        "source": src,
        "tokenizer": tok,
        "packing": packing,
        "eval": eval_cfg,
        "seq_len": cfg.train.seq_len,
        "batch_size": cfg.train.batch_size,
        "grad_accum": cfg.train.grad_accum,
    }


class TrainBatchIterator:
    """An iterator that yields fixed-shape `Batch` objects.

    This is the data side of the compile-once contract:
    - Every `__next__` yields arrays of exactly the same shape & dtype.

    It also implements `get_state`/`set_state` for resume correctness.
    """

    def __init__(self, cfg: Config, *, tokenizer: Tokenizer, text_stream: TextStream | None = None):
        """Initialize the training batch iterator.

        :param Config cfg: Training configuration.
        :param Tokenizer tokenizer: Tokenizer for encoding text.
        :param text_stream: Optional text stream override (used for eval datasets).
        :raises ValueError: If data.backend is unknown.
        """
        self._cfg = cfg
        self._tok = tokenizer

        # Text stream
        if text_stream is not None:
            self._text_stream = text_stream
        elif cfg.data.backend == "hf":
            self._text_stream = _build_hf_stream(
                cfg, split=cfg.data.hf_split, repeat=cfg.data.repeat
            )
        elif cfg.data.backend == "local_text":
            self._text_stream = LocalTextStream(text=cfg.data.local_text, repeat=cfg.data.repeat)
        else:
            raise ValueError(f"Unknown data.backend: {cfg.data.backend!r}")

        # Packer
        if cfg.data.packing_mode == "bin":
            self._packer = BinPacker(
                seq_len=cfg.train.seq_len,
                add_bos=cfg.data.tokenizer.add_bos,
                add_eos=cfg.data.tokenizer.add_eos,
                bos_id=cfg.model.bos_token_id,
                eos_id=cfg.model.eos_token_id,
                max_doc_tokens=cfg.data.tokenizer.max_doc_tokens,
                bins_per_pack=int(cfg.train.grad_accum) * int(cfg.train.batch_size),
                buffer_docs=cfg.data.packing_buffer_docs,
                max_docs_per_bin=cfg.data.packing_max_docs_per_bin,
                pad_id=cfg.model.pad_token_id,
            )
        else:
            self._packer = TokenPacker(
                seq_len=cfg.train.seq_len,
                add_bos=cfg.data.tokenizer.add_bos,
                add_eos=cfg.data.tokenizer.add_eos,
                bos_id=cfg.model.bos_token_id,
                eos_id=cfg.model.eos_token_id,
                max_doc_tokens=cfg.data.tokenizer.max_doc_tokens,
            )

        # Batch shape
        self._A = int(cfg.train.grad_accum)
        self._B = int(cfg.train.batch_size)
        self._T = int(cfg.train.seq_len)
        self._device_put = bool(cfg.data.device_put)
        self._mask_boundary_loss = bool(cfg.data.mask_boundary_loss)
        self._train_on_eos = bool(cfg.data.train_on_eos)
        self._eos_id = int(cfg.model.eos_token_id)

    def __iter__(self) -> TrainBatchIterator:
        return self

    def _push_next_document(self) -> None:
        """Fetch one item from the text stream and add it to the packer."""
        item = next(self._text_stream)
        if isinstance(item, str):
            ids = self._tok.encode(item)
        elif isinstance(item, list):
            ids = item
        else:
            ids = list(item)
        self._packer.add_document(ids)

    def _next_sequence(self) -> tuple[np.ndarray, np.ndarray]:
        """Pop the next [T] token/segment sequence from the packer.

        :return tuple[np.ndarray, np.ndarray]: Tokens and segment IDs (length T).
        """
        while not self._packer.can_pop():
            self._push_next_document()
        return self._packer.pop_seq_with_segments()

    def _mask_labels(self, labels: np.ndarray, segs: np.ndarray) -> np.ndarray:
        """Apply boundary and EOS masking to label array.

        :param np.ndarray labels: Label array of length T.
        :param np.ndarray segs: Segment IDs of length T.
        :return np.ndarray: Masked labels of length T.
        """
        if self._mask_boundary_loss:
            same = (segs[1:] == segs[:-1]) & (segs[1:] > 0) & (segs[:-1] > 0)
            if labels.size > 1:
                labels[1:] = np.where(same, labels[1:], IGNORE_INDEX).astype(np.int32)
        if not self._train_on_eos:
            labels = np.where(labels == self._eos_id, IGNORE_INDEX, labels).astype(np.int32)
        return labels

    def __next__(self) -> Batch:
        need = self._A * self._B
        inps = np.empty((need, self._T), dtype=np.int32)
        labs = np.empty((need, self._T), dtype=np.int32)
        segs_out = np.empty((need, self._T), dtype=np.int32)

        idx = 0
        while idx < need:
            seq, segs = self._next_sequence()  # [T]
            # Convert to input/labels [T]. Labels align with input_ids; model shifts internally.
            inp = np.asarray(seq, dtype=np.int32)
            lab = self._mask_labels(inp.copy(), segs)
            seg = np.asarray(segs, dtype=np.int32)
            inps[idx] = inp
            labs[idx] = lab
            segs_out[idx] = seg
            idx += 1

        # Reshape -> [A, B, T]
        inps = inps.reshape(self._A, self._B, self._T)
        labs = labs.reshape(self._A, self._B, self._T)
        segs = segs_out.reshape(self._A, self._B, self._T)

        attn = segs > 0

        batch = Batch(input_ids=inps, labels=labs, attention_mask=attn, segment_ids=segs)
        if self._device_put:
            import jax  # imported lazily to keep iterator usable in non-JAX contexts

            batch = jax.device_put(batch)
        return batch

    # -------- checkpoint hooks --------

    def get_state(self) -> dict[str, Any]:
        """Capture current iterator state for checkpointing.

        :return dict[str, Any]: State dict with text stream and packer state.
        """
        return {
            "text": self._text_stream.get_state(),
            "packer": self._packer.get_state(),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore iterator state from a checkpoint.

        :param dict[str, Any] state: State dict from get_state().
        """
        if "text" in state:
            self._text_stream.set_state(state["text"])
        if "packer" in state:
            self._packer.set_state(state["packer"])

    def get_stats(self) -> dict[str, int]:
        """Return packer-level document stats if available.

        :return dict[str, int]: Stats like docs_seen/docs_truncated.
        """
        if hasattr(self._packer, "get_stats"):
            return dict(self._packer.get_stats())
        return {}


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


def build_eval_iterator(cfg: Config, *, tokens: list[list[int]], tokenizer: Tokenizer) -> Any:
    """Build a one-pass evaluation iterator from tokenized docs.

    :param Config cfg: Training configuration.
    :param list[list[int]] tokens: Tokenized evaluation documents.
    :param Tokenizer tokenizer: Tokenizer instance.
    :return Any: Iterator yielding fixed-shape Batch objects.
    """
    text_stream = ListTokenStream(tokens=tokens, repeat=False)
    return TrainBatchIterator(cfg, tokenizer=tokenizer, text_stream=text_stream)
