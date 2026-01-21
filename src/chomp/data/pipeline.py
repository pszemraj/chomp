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

from chomp.config import Config
from chomp.types import Batch

from .hf import HFStreamingTextStream, HFStreamSpec, LocalTextStream
from .pack import BinPacker, TokenPacker


class Tokenizer(Protocol):
    """Protocol for tokenizers that convert text to token ids."""

    def encode(self, text: str) -> list[int]:
        """Encode text string to a list of token ids."""
        ...

    def __len__(self) -> int: ...


logger = logging.getLogger(__name__)
_IGNORE_INDEX = -100


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

    if not model_updates:
        return cfg

    return replace(cfg, model=replace(cfg.model, **model_updates))


def prepare_tokenizer_and_config(cfg: Config) -> tuple[Config, Tokenizer]:
    """Build tokenizer and return an updated config with tokenizer-derived fields.

    :param Config cfg: Input configuration.
    :return tuple: (updated_config, tokenizer) tuple.
    """

    tok = build_tokenizer(cfg)
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


def data_fingerprint(cfg: Config) -> dict[str, Any]:
    """A small, stable fingerprint that we store in checkpoint meta.

    :param Config cfg: Training configuration.
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

    packing = {
        "mode": d.packing_mode,
        "buffer_docs": d.packing_buffer_docs,
        "max_docs_per_bin": d.packing_max_docs_per_bin,
        "mask_boundary_loss": d.mask_boundary_loss,
        "train_on_eos": d.train_on_eos,
        "grain_prefetch": d.grain_prefetch,
    }

    return {
        "source": src,
        "tokenizer": tok,
        "packing": packing,
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

    def __init__(self, cfg: Config, *, tokenizer: Tokenizer):
        """Initialize the training batch iterator.

        :param Config cfg: Training configuration.
        :param Tokenizer tokenizer: Tokenizer for encoding text.
        :raises ValueError: If data.backend is unknown.
        """
        self._cfg = cfg
        self._tok = tokenizer

        # Text stream
        if cfg.data.backend == "hf":
            spec = HFStreamSpec(
                dataset=cfg.data.hf_dataset,
                name=cfg.data.hf_name,
                split=cfg.data.hf_split,
                text_key=cfg.data.text_key,
                shuffle=cfg.data.shuffle,
                shuffle_buffer_size=cfg.data.shuffle_buffer_size,
                seed=cfg.data.seed,
                repeat=cfg.data.repeat,
                max_retries=cfg.data.max_retries,
                retry_delay_sec=cfg.data.retry_delay_sec,
                state_update_interval=cfg.data.state_update_interval,
            )
            self._text_stream = HFStreamingTextStream(spec)
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
        self._packing_mode = str(cfg.data.packing_mode)
        self._last_stats: dict[str, float | int | str] = {}

    def __iter__(self) -> TrainBatchIterator:
        return self

    def __next__(self) -> Batch:
        seqs = []
        need = self._A * self._B

        while len(seqs) < need:
            # Ensure packer has enough tokens
            while not self._packer.can_pop():
                text = next(self._text_stream)
                ids = self._tok.encode(text)
                self._packer.add_document(ids)

            seq, segs = self._packer.pop_seq_plus_one_with_segments()  # [T+1]
            # Convert to input/labels [T]
            inp = seq[:-1]
            lab = seq[1:]
            seg = segs[:-1]
            if self._mask_boundary_loss:
                same = (segs[1:] == segs[:-1]) & (segs[1:] > 0) & (segs[:-1] > 0)
                lab = np.where(same, lab, _IGNORE_INDEX).astype(np.int32)
            if not self._train_on_eos:
                lab = np.where(lab == self._eos_id, _IGNORE_INDEX, lab).astype(np.int32)
            seqs.append((inp, lab, seg))

        # Stack -> [A*B, T]
        inps = np.stack([x[0] for x in seqs], axis=0).astype(np.int32)
        labs = np.stack([x[1] for x in seqs], axis=0).astype(np.int32)
        segs = np.stack([x[2] for x in seqs], axis=0).astype(np.int32)

        # Reshape -> [A, B, T]
        inps = inps.reshape(self._A, self._B, self._T)
        labs = labs.reshape(self._A, self._B, self._T)
        segs = segs.reshape(self._A, self._B, self._T)

        attn = segs > 0

        tokens_used = int(np.count_nonzero(attn))
        capacity = int(attn.size)
        utilization = float(tokens_used / capacity) if capacity > 0 else 0.0
        self._last_stats = {
            "packing_mode": self._packing_mode,
            "packing_tokens": tokens_used,
            "packing_capacity": capacity,
            "packing_utilization": utilization,
        }

        batch = Batch(input_ids=inps, labels=labs, attention_mask=attn, segment_ids=segs)
        if self._device_put:
            import jax  # imported lazily to keep iterator usable in non-JAX contexts

            batch = jax.device_put(batch)
        return batch

    def get_stats(self) -> dict[str, float | int | str]:
        """Return latest packing stats from the iterator.

        :return dict[str, float | int | str]: Stats dict with utilization fields.
        """
        return dict(self._last_stats)

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


def build_train_iterator(cfg: Config, *, tokenizer: Tokenizer | None = None) -> Any:
    """Build the training batch iterator.

    :param Config cfg: Training configuration.
    :param tokenizer: Optional pre-built tokenizer; built from config if None.
    :return Any: Iterator yielding fixed-shape Batch objects.
    """
    if tokenizer is None:
        cfg, tokenizer = prepare_tokenizer_and_config(cfg)
    from chomp.data.grain import build_grain_iterator

    return build_grain_iterator(cfg, tokenizer=tokenizer)
