"""Token packing for causal LM pretraining.

We need fixed-length sequences for JAX compile stability.
Streaming datasets yield variable-length documents, so we *pack* them into a
continuous token stream and slice into fixed windows.

Core contract:
- Packer consumes documents as token id sequences (list[int] / np.ndarray)
- Packer yields fixed arrays of length (seq_len + 1)
  (we need +1 to produce input_ids and labels via shifting)

Senior dev notes:
- Do not build packers with repeated `np.concatenate` in the hot path. You'll
  accidentally write an O(n^2) implementation that collapses at scale.
- We implement a small chunked buffer that supports O(n) total copies.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Iterable, List

import numpy as np


class _ChunkedTokenBuffer:
    """A chunked 1D int32 buffer with efficient take()."""

    def __init__(self):
        self._chunks: Deque[np.ndarray] = deque()
        self._offset: int = 0
        self._size: int = 0  # tokens available

    @property
    def size(self) -> int:
        return self._size

    def append(self, tokens: np.ndarray) -> None:
        if tokens.size == 0:
            return
        if tokens.dtype != np.int32:
            tokens = tokens.astype(np.int32)
        self._chunks.append(tokens)
        self._size += int(tokens.size)

    def take(self, n: int) -> np.ndarray:
        """Remove and return exactly n tokens."""

        if n < 0:
            raise ValueError(f"n must be >=0, got {n}")
        if n == 0:
            return np.empty((0,), dtype=np.int32)
        if self._size < n:
            raise ValueError(f"buffer underflow: need {n}, have {self._size}")

        out = np.empty((n,), dtype=np.int32)
        pos = 0
        need = n

        while need > 0:
            chunk = self._chunks[0]
            avail = int(chunk.size) - self._offset
            take_n = avail if avail < need else need

            out[pos : pos + take_n] = chunk[self._offset : self._offset + take_n]
            pos += take_n
            need -= take_n

            self._offset += take_n
            self._size -= take_n

            if self._offset >= int(chunk.size):
                self._chunks.popleft()
                self._offset = 0

        return out

    def dump_remaining(self) -> List[int]:
        """Return remaining tokens as a python list (for small checkpoint state)."""

        if self._size == 0:
            return []

        out: List[int] = []
        first = True
        for c in self._chunks:
            if first:
                out.extend(c[self._offset :].tolist())
                first = False
            else:
                out.extend(c.tolist())
        return out

    def load_remaining(self, tokens: Iterable[int]) -> None:
        self._chunks.clear()
        self._offset = 0
        arr = np.asarray(list(tokens), dtype=np.int32)
        self._size = int(arr.size)
        if self._size > 0:
            self._chunks.append(arr)


@dataclass(frozen=True)
class PackerState:
    """JSON-serializable packer state."""

    remaining_tokens: List[int]

    def to_dict(self) -> dict[str, Any]:
        return {"remaining_tokens": self.remaining_tokens}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "PackerState":
        toks = d.get("remaining_tokens") or []
        return PackerState(remaining_tokens=list(toks))


class TokenPacker:
    """Pack variable-length tokenized documents into fixed-length sequences."""

    def __init__(
        self,
        *,
        seq_len: int,
        add_bos: bool,
        add_eos: bool,
        bos_id: int,
        eos_id: int,
        max_doc_tokens: int | None,
    ):
        if seq_len < 8:
            raise ValueError(f"seq_len must be >=8, got {seq_len}")
        self.seq_len = int(seq_len)
        self.add_bos = bool(add_bos)
        self.add_eos = bool(add_eos)
        self.bos_id = int(bos_id)
        self.eos_id = int(eos_id)
        self.max_doc_tokens = None if max_doc_tokens is None else int(max_doc_tokens)

        self._buf = _ChunkedTokenBuffer()

    def add_document(self, tokens: Iterable[int]) -> None:
        arr = np.asarray(list(tokens), dtype=np.int32)
        if self.max_doc_tokens is not None and arr.size > self.max_doc_tokens:
            arr = arr[: self.max_doc_tokens]

        pieces = []
        if self.add_bos:
            pieces.append(np.asarray([self.bos_id], dtype=np.int32))
        if arr.size:
            pieces.append(arr)
        if self.add_eos:
            pieces.append(np.asarray([self.eos_id], dtype=np.int32))

        if not pieces:
            return
        self._buf.append(np.concatenate(pieces, axis=0))

    def can_pop(self) -> bool:
        return self._buf.size >= (self.seq_len + 1)

    def pop_seq_plus_one(self) -> np.ndarray:
        """Return [seq_len+1] tokens."""

        return self._buf.take(self.seq_len + 1)

    def get_state(self) -> dict[str, Any]:
        # NOTE: Remaining tokens are at most seq_len in steady state.
        st = PackerState(remaining_tokens=self._buf.dump_remaining())
        return st.to_dict()

    def set_state(self, state: dict[str, Any]) -> None:
        st = PackerState.from_dict(state)
        self._buf.load_remaining(st.remaining_tokens)
