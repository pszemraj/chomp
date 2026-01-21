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
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np


class _ChunkedIntBuffer:
    """A chunked 1D int32 buffer with efficient take()."""

    def __init__(self) -> None:
        """Initialize an empty chunked buffer."""
        self._chunks: deque[np.ndarray] = deque()
        self._offset: int = 0
        self._size: int = 0  # tokens available

    @property
    def size(self) -> int:
        """Number of tokens currently in the buffer.

        :return int: Token count.
        """
        return self._size

    def append(self, tokens: np.ndarray) -> None:
        """Append tokens to the buffer.

        :param np.ndarray tokens: Array of tokens to append.
        """
        if tokens.size == 0:
            return
        if tokens.dtype != np.int32:
            tokens = tokens.astype(np.int32)
        self._chunks.append(tokens)
        self._size += int(tokens.size)

    def take(self, n: int) -> np.ndarray:
        """Remove and return exactly n tokens.

        :param int n: Number of tokens to take.
        :raises ValueError: If n < 0 or buffer has fewer than n tokens.
        :return np.ndarray: Array of n tokens (int32).
        """

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

    def dump_remaining(self) -> list[int]:
        """Return remaining tokens as a python list (for small checkpoint state).

        :return List[int]: All remaining tokens in the buffer.
        """

        if self._size == 0:
            return []

        out: list[int] = []
        first = True
        for c in self._chunks:
            if first:
                out.extend(c[self._offset :].tolist())
                first = False
            else:
                out.extend(c.tolist())
        return out

    def load_remaining(self, tokens: Iterable[int]) -> None:
        """Replace buffer contents with the given tokens.

        :param tokens: Iterable of tokens to load.
        """
        self._chunks.clear()
        self._offset = 0
        arr = np.asarray(list(tokens), dtype=np.int32)
        self._size = int(arr.size)
        if self._size > 0:
            self._chunks.append(arr)


@dataclass(frozen=True)
class PackerState:
    """JSON-serializable packer state."""

    remaining_tokens: list[int]
    remaining_segments: list[int]
    next_segment_id: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary.

        :return dict[str, Any]: State as a dict.
        """
        return {
            "remaining_tokens": self.remaining_tokens,
            "remaining_segments": self.remaining_segments,
            "next_segment_id": int(self.next_segment_id),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> PackerState:
        """Construct PackerState from a dictionary.

        :param dict[str, Any] d: State dict from to_dict().
        :raises ValueError: If segments/tokens lengths don't match.
        :return PackerState: Reconstructed state.
        """
        toks = d.get("remaining_tokens") or []
        segs = d.get("remaining_segments")
        if segs is None:
            segs_list = [1 for _ in range(len(toks))]
        else:
            segs_list = list(segs)
        if len(segs_list) != len(toks):
            raise ValueError(
                f"remaining_segments length ({len(segs_list)}) must match remaining_tokens ({len(toks)})"
            )
        next_id = d.get("next_segment_id")
        if next_id is None:
            max_seg = max(segs_list) if segs_list else 0
            next_id = max_seg + 1
        return PackerState(
            remaining_tokens=list(toks),
            remaining_segments=segs_list,
            next_segment_id=int(next_id),
        )


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
        """Initialize the token packer.

        :param int seq_len: Fixed sequence length for output.
        :param bool add_bos: Whether to prepend BOS token to each document.
        :param bool add_eos: Whether to append EOS token to each document.
        :param int bos_id: BOS token ID.
        :param int eos_id: EOS token ID.
        :param max_doc_tokens: Optional max tokens per document before truncation.
        :raises ValueError: If seq_len < 8.
        """
        if seq_len < 8:
            raise ValueError(f"seq_len must be >=8, got {seq_len}")
        self.seq_len = int(seq_len)
        self.add_bos = bool(add_bos)
        self.add_eos = bool(add_eos)
        self.bos_id = int(bos_id)
        self.eos_id = int(eos_id)
        self.max_doc_tokens = None if max_doc_tokens is None else int(max_doc_tokens)

        self._token_buf = _ChunkedIntBuffer()
        self._segment_buf = _ChunkedIntBuffer()
        self._next_segment_id = 1

    def add_document(self, tokens: Iterable[int]) -> None:
        """Add a tokenized document to the packer buffer.

        :param tokens: Iterable of token IDs for the document.
        """
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
        segment_id = int(self._next_segment_id)
        seg_pieces = [np.full((p.size,), segment_id, dtype=np.int32) for p in pieces]
        self._token_buf.append(np.concatenate(pieces, axis=0))
        self._segment_buf.append(np.concatenate(seg_pieces, axis=0))
        self._next_segment_id += 1

    def can_pop(self) -> bool:
        """Check if buffer has enough tokens for one sequence.

        :raises RuntimeError: If token/segment buffers are misaligned.
        :return bool: True if at least seq_len+1 tokens are available.
        """
        if self._token_buf.size != self._segment_buf.size:
            raise RuntimeError("token/segment buffers are misaligned")
        return self._token_buf.size >= (self.seq_len + 1)

    def pop_seq_plus_one(self) -> np.ndarray:
        """Return [seq_len+1] tokens.

        :return np.ndarray: Array of shape [seq_len+1] containing token IDs.
        """
        tokens, _segs = self.pop_seq_plus_one_with_segments()
        return tokens

    def pop_seq_plus_one_with_segments(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ([seq_len+1] tokens, [seq_len+1] segment_ids).

        :raises RuntimeError: If token/segment buffers are misaligned.
        :return tuple: (tokens, segment_ids) arrays of shape [seq_len+1].
        """

        if self._token_buf.size != self._segment_buf.size:
            raise RuntimeError("token/segment buffers are misaligned")
        tokens = self._token_buf.take(self.seq_len + 1)
        segs = self._segment_buf.take(self.seq_len + 1)
        return tokens, segs

    def get_state(self) -> dict[str, Any]:
        """Capture packer state for checkpointing.

        :return dict[str, Any]: Serializable state dict.
        """
        # NOTE: Remaining tokens are at most seq_len in steady state.
        st = PackerState(
            remaining_tokens=self._token_buf.dump_remaining(),
            remaining_segments=self._segment_buf.dump_remaining(),
            next_segment_id=int(self._next_segment_id),
        )
        return st.to_dict()

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore packer state from a checkpoint.

        :param dict[str, Any] state: State dict from get_state().
        """
        st = PackerState.from_dict(state)
        self._token_buf.load_remaining(st.remaining_tokens)
        self._segment_buf.load_remaining(st.remaining_segments)
        self._next_segment_id = int(st.next_segment_id)
