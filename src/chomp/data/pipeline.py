"""Minimal data pipeline for chomp (Phases 0–4-ish).

This is intentionally *not* a framework.

Goal for v0:
- Consume Zyphra/Zyda-2 (streaming) by default
- Tokenize + pack into fixed-shape microbatches [A, B, T]
- Provide get_state/set_state hooks so checkpoint+resume is real

We are **not** using Grain yet in this initial draft. That will come in Phases 5–6.
For now, this pipeline is a single iterator object.

Why remove synthetic batches?
Because synthetic batches turn into a crutch: people think the trainer works
when it doesn't survive contact with real streaming data.

This pipeline keeps debug sources (local_text) but *still* exercises tokenize+pack.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from chomp.config import Config
from chomp.types import Batch

from .hf import HFStreamingTextStream, HFStreamSpec, LocalTextStream
from .pack import TokenPacker


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def __len__(self) -> int: ...


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
        b = text.encode("utf-8", errors="replace")
        off = int(self.byte_offset)
        return [off + int(x) for x in b]

    def __len__(self) -> int:
        return int(self.byte_offset) + 256


class HFTokenizer:
    """Hugging Face tokenizer wrapper.

    Requires `transformers` (install chomp[hf]).
    """

    def __init__(self, name_or_path: str, *, use_fast: bool, trust_remote_code: bool):
        try:
            from transformers import AutoTokenizer
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "HFTokenizer requires transformers. Install extras: pip install -e .[hf]"
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
        out = self._tok(text, add_special_tokens=False)
        ids = out.get("input_ids")
        if ids is None:
            raise RuntimeError("Tokenizer did not return input_ids")
        return list(ids)

    def __len__(self) -> int:
        return int(len(self._tok))

    @property
    def bos_token_id(self) -> int | None:
        return self._tok.bos_token_id

    @property
    def eos_token_id(self) -> int | None:
        return self._tok.eos_token_id

    @property
    def pad_token_id(self) -> int | None:
        return self._tok.pad_token_id


def build_tokenizer(cfg: Config) -> Tokenizer:
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


def validate_tokenizer_compat(cfg: Config, tok: Tokenizer) -> None:
    """Fail fast if the tokenizer and model config are incompatible."""

    if cfg.data.tokenizer.kind != "hf":
        return

    try:
        vocab_size = int(len(tok))
    except Exception as exc:
        raise RuntimeError("HF tokenizer must expose vocab size via __len__") from exc

    if vocab_size <= 0:
        raise ValueError(f"HF tokenizer vocab size must be positive, got {vocab_size}")

    if cfg.model.vocab_size != vocab_size:
        raise ValueError(
            f"model.vocab_size ({cfg.model.vocab_size}) must match HF tokenizer vocab size "
            f"({vocab_size})"
        )

    tok_bos = getattr(tok, "bos_token_id", None)
    tok_eos = getattr(tok, "eos_token_id", None)
    tok_pad = getattr(tok, "pad_token_id", None)

    if cfg.data.tokenizer.add_bos and tok_bos is None:
        raise ValueError("HF tokenizer has no bos_token_id but data.tokenizer.add_bos=true")
    if cfg.data.tokenizer.add_eos and tok_eos is None:
        raise ValueError("HF tokenizer has no eos_token_id but data.tokenizer.add_eos=true")

    if tok_bos is not None and cfg.model.bos_token_id != tok_bos:
        raise ValueError(
            f"model.bos_token_id ({cfg.model.bos_token_id}) must match HF tokenizer bos_token_id "
            f"({tok_bos})"
        )
    if tok_eos is not None and cfg.model.eos_token_id != tok_eos:
        raise ValueError(
            f"model.eos_token_id ({cfg.model.eos_token_id}) must match HF tokenizer eos_token_id "
            f"({tok_eos})"
        )
    if tok_pad is not None and cfg.model.pad_token_id != tok_pad:
        raise ValueError(
            f"model.pad_token_id ({cfg.model.pad_token_id}) must match HF tokenizer pad_token_id "
            f"({tok_pad})"
        )


def data_fingerprint(cfg: Config) -> dict[str, Any]:
    """A small, stable fingerprint that we store in checkpoint meta."""

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
        "byte_offset": t.byte_offset,
        "add_bos": t.add_bos,
        "add_eos": t.add_eos,
        "max_doc_tokens": t.max_doc_tokens,
    }

    return {
        "source": src,
        "tokenizer": tok,
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

    def __iter__(self) -> "TrainBatchIterator":
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

            seq = self._packer.pop_seq_plus_one()  # [T+1]
            # Convert to input/labels [T]
            inp = seq[:-1]
            lab = seq[1:]
            seqs.append((inp, lab))

        # Stack -> [A*B, T]
        inps = np.stack([x[0] for x in seqs], axis=0).astype(np.int32)
        labs = np.stack([x[1] for x in seqs], axis=0).astype(np.int32)

        # Reshape -> [A, B, T]
        inps = inps.reshape(self._A, self._B, self._T)
        labs = labs.reshape(self._A, self._B, self._T)

        attn = np.ones((self._A, self._B, self._T), dtype=np.bool_)

        batch = Batch(input_ids=inps, labels=labs, attention_mask=attn)
        if self._device_put:
            import jax  # imported lazily to keep iterator usable in non-JAX contexts

            batch = jax.device_put(batch)
        return batch

    # -------- checkpoint hooks --------

    def get_state(self) -> dict[str, Any]:
        return {
            "text": self._text_stream.get_state(),
            "packer": self._packer.get_state(),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if "text" in state:
            self._text_stream.set_state(state["text"])
        if "packer" in state:
            self._packer.set_state(state["packer"])


def build_train_iterator(cfg: Config) -> TrainBatchIterator:
    tok = build_tokenizer(cfg)
    validate_tokenizer_compat(cfg, tok)
    return TrainBatchIterator(cfg, tokenizer=tok)
