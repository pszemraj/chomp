"""Token packing for causal LM pretraining.

We need fixed-length sequences for JAX compile stability.
Streaming datasets yield variable-length documents, so we *pack* them into a
continuous token stream and slice into fixed windows.

Core contract:
- Packer consumes documents as token id sequences (list[int] / np.ndarray)
- Packer yields fixed arrays of length (seq_len)
  (input_ids and labels are aligned; model shifts internally)

Senior dev notes:
- Do not build packers with repeated `np.concatenate` in the hot path. You'll
  accidentally write an O(n^2) implementation that collapses at scale.
- We implement a small chunked buffer that supports O(n) total copies.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _prepare_doc_tokens(
    tokens: Iterable[int],
    *,
    add_bos: bool,
    add_eos: bool,
    bos_id: int,
    eos_id: int,
    max_doc_tokens: int | None,
) -> tuple[np.ndarray, bool]:
    """Build a document token array with optional BOS/EOS and truncation.

    :param tokens: Iterable of token IDs for the document.
    :param bool add_bos: Whether to prepend the BOS token.
    :param bool add_eos: Whether to append the EOS token.
    :param int bos_id: BOS token ID.
    :param int eos_id: EOS token ID.
    :param max_doc_tokens: Optional cap on document length before special tokens.
    :return tuple[np.ndarray, bool]: (Token array (int32), truncated flag).
    """
    if isinstance(tokens, np.ndarray):
        arr = tokens.astype(np.int32, copy=False)
    elif isinstance(tokens, (list, tuple)):
        arr = np.asarray(tokens, dtype=np.int32)
    else:
        arr = np.fromiter(tokens, dtype=np.int32)

    truncated = False
    if max_doc_tokens is not None and arr.size > max_doc_tokens:
        arr = arr[:max_doc_tokens]
        truncated = True

    pieces = []
    if add_bos:
        pieces.append(np.asarray([bos_id], dtype=np.int32))
    if arr.size:
        pieces.append(arr)
    if add_eos:
        pieces.append(np.asarray([eos_id], dtype=np.int32))

    if not pieces:
        return np.empty((0,), dtype=np.int32), truncated
    if len(pieces) == 1:
        return pieces[0], truncated
    return np.concatenate(pieces, axis=0), truncated


def _positions_from_segments(segs: np.ndarray) -> np.ndarray:
    """Derive per-segment position IDs from segment IDs (vectorized).

    Positions restart at 0 at every contiguous segment-ID run and are 0 on
    padding (segment id <= 0). Positions are always a pure function of the
    emitted segment IDs, so they are computed here in exactly one place
    rather than stored or re-derived by consumers.

    :param np.ndarray segs: Segment IDs of length T.
    :return np.ndarray: Position IDs of length T (int32).
    """
    n = int(segs.size)
    idx = np.arange(n, dtype=np.int64)
    boundary = np.empty(n, dtype=bool)
    boundary[0] = True
    boundary[1:] = segs[1:] != segs[:-1]
    run_starts = idx[boundary]
    pos = (idx - run_starts[np.cumsum(boundary) - 1]).astype(np.int32)
    pos[segs <= 0] = 0
    return pos


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
    docs_seen: int
    docs_truncated: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary.

        :return dict[str, Any]: State as a dict.
        """
        return {
            "remaining_tokens": self.remaining_tokens,
            "remaining_segments": self.remaining_segments,
            "next_segment_id": int(self.next_segment_id),
            "docs_seen": int(self.docs_seen),
            "docs_truncated": int(self.docs_truncated),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> PackerState:
        """Construct PackerState from a dictionary.

        :param dict[str, Any] d: State dict from to_dict().
        :raises KeyError: If a required field is missing (corrupt/foreign state).
        :raises ValueError: If segments/tokens lengths don't match.
        :return PackerState: Reconstructed state.
        """
        # Deliberately strict: synthesizing missing segment metadata (as older
        # loaders did) silently merges all buffered documents into one segment
        # on resume — wrong boundary masking and, under strict multipack,
        # wrong isolation. Corrupt/foreign state must fail loud.
        toks = list(d["remaining_tokens"])
        segs = list(d["remaining_segments"])
        if len(segs) != len(toks):
            raise ValueError(
                f"remaining_segments length ({len(segs)}) must match remaining_tokens ({len(toks)})"
            )
        if any(int(segment_id) not in (1, 2) for segment_id in segs):
            raise ValueError("remaining_segments must contain only current segment IDs 1 or 2")
        next_segment_id = int(d["next_segment_id"])
        if next_segment_id not in (1, 2):
            raise ValueError(f"next_segment_id must be 1 or 2, got {next_segment_id}")
        docs_seen = int(d["docs_seen"])
        docs_truncated = int(d["docs_truncated"])
        if docs_seen < 0 or docs_truncated < 0 or docs_truncated > docs_seen:
            raise ValueError(
                "invalid document counters: expected 0 <= docs_truncated <= docs_seen, "
                f"got docs_truncated={docs_truncated}, docs_seen={docs_seen}"
            )
        return PackerState(
            remaining_tokens=toks,
            remaining_segments=segs,
            next_segment_id=next_segment_id,
            docs_seen=docs_seen,
            docs_truncated=docs_truncated,
        )


class _PackerBase:
    """Shared document preparation and diagnostics for all packers."""

    def __init__(
        self,
        *,
        seq_len: int,
        add_bos: bool,
        add_eos: bool,
        bos_id: int,
        eos_id: int,
        max_doc_tokens: int | None,
    ) -> None:
        """Initialize common packer settings.

        :param int seq_len: Fixed output sequence length.
        :param bool add_bos: Whether to prepend BOS to each document.
        :param bool add_eos: Whether to append EOS to each document.
        :param int bos_id: BOS token ID.
        :param int eos_id: EOS token ID.
        :param max_doc_tokens: Optional document truncation limit.
        :raises ValueError: If seq_len is below the supported minimum.
        """
        if seq_len < 8:
            raise ValueError(f"seq_len must be >=8, got {seq_len}")
        self.seq_len = int(seq_len)
        self.add_bos = bool(add_bos)
        self.add_eos = bool(add_eos)
        self.bos_id = int(bos_id)
        self.eos_id = int(eos_id)
        self.max_doc_tokens = None if max_doc_tokens is None else int(max_doc_tokens)
        self._docs_seen = 0
        self._docs_truncated = 0

    def _prepare_document(self, tokens: Iterable[int]) -> np.ndarray:
        """Prepare one document and update shared counters.

        :param tokens: Iterable of input token IDs.
        :return np.ndarray: Prepared int32 document tokens.
        """
        document, truncated = _prepare_doc_tokens(
            tokens,
            add_bos=self.add_bos,
            add_eos=self.add_eos,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            max_doc_tokens=self.max_doc_tokens,
        )
        self._docs_seen += 1
        if truncated:
            self._docs_truncated += 1
        return document

    def get_stats(self) -> dict[str, int]:
        """Return common document counters.

        :return dict[str, int]: docs_seen and docs_truncated counts.
        """
        return {
            "docs_seen": int(self._docs_seen),
            "docs_truncated": int(self._docs_truncated),
        }


class TokenPacker(_PackerBase):
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
        super().__init__(
            seq_len=seq_len,
            add_bos=add_bos,
            add_eos=add_eos,
            bos_id=bos_id,
            eos_id=eos_id,
            max_doc_tokens=max_doc_tokens,
        )

        self._token_buf = _ChunkedIntBuffer()
        self._segment_buf = _ChunkedIntBuffer()
        self._next_segment_id = 1

    @staticmethod
    def _flip_segment_id(segment_id: int) -> int:
        """Flip between the two internal segment IDs used for streaming state.

        :param int segment_id: Current segment ID.
        :return int: The opposite internal segment ID.
        """
        return 2 if int(segment_id) == 1 else 1

    @staticmethod
    def _reindex_popped_segments(segs: np.ndarray) -> np.ndarray:
        """Reindex one output window to compact positive segment IDs.

        Segment IDs only need to be unique within the emitted sequence window.
        This keeps output deterministic and avoids exposing internal ID choices.

        :param np.ndarray segs: Internal segment IDs for one sequence window.
        :return np.ndarray: Compacted segment IDs, starting at 1.
        """
        boundary = np.empty(segs.size, dtype=np.int64)
        boundary[0] = 1
        boundary[1:] = segs[1:] != segs[:-1]
        return np.cumsum(boundary).astype(np.int32)

    def add_document(self, tokens: Iterable[int]) -> None:
        """Add a tokenized document to the packer buffer.

        :param tokens: Iterable of token IDs for the document.
        """
        doc = self._prepare_document(tokens)
        if doc.size == 0:
            return
        segment_id = int(self._next_segment_id)
        self._token_buf.append(doc)
        self._segment_buf.append(np.full((doc.size,), segment_id, dtype=np.int32))
        self._next_segment_id = self._flip_segment_id(segment_id)

    def can_pop(self) -> bool:
        """Check if buffer has enough tokens for one sequence.

        :raises RuntimeError: If token/segment buffers are misaligned.
        :return bool: True if at least seq_len tokens are available.
        """
        if self._token_buf.size != self._segment_buf.size:
            raise RuntimeError("token/segment buffers are misaligned")
        return self._token_buf.size >= self.seq_len

    def finish(self) -> None:
        """Mark the upstream document stream exhausted (no-op here).

        Sequential windows are unpadded slices of the continuous token
        stream, so a partial tail window (< seq_len tokens) cannot be emitted
        under the fixed-shape contract and is dropped by design. The FFD
        packers implement this hook as a real flush of pending documents.
        """

    def pop_seq_with_metadata(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ([seq_len] tokens, [seq_len] segment_ids, [seq_len] position_ids).

        Position IDs restart at each window: a document spanning multiple
        windows restarts at 0 in every window. They are informational for
        stream-semantics modes (never consumed by the model).

        :raises RuntimeError: If token/segment buffers are misaligned.
        :return tuple: (tokens, segment_ids, position_ids) arrays of shape [seq_len].
        """

        tokens, segs = self.pop_seq_with_segments()
        return tokens, segs, _positions_from_segments(segs)

    def pop_seq_with_segments(self) -> tuple[np.ndarray, np.ndarray]:
        """Return tokens and segment IDs without materializing position IDs.

        :raises RuntimeError: If token/segment buffers are misaligned.
        :return tuple[np.ndarray, np.ndarray]: Fixed-length token and segment arrays.
        """
        if self._token_buf.size != self._segment_buf.size:
            raise RuntimeError("token/segment buffers are misaligned")
        tokens = self._token_buf.take(self.seq_len)
        segs = self._reindex_popped_segments(self._segment_buf.take(self.seq_len))
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
            docs_seen=int(self._docs_seen),
            docs_truncated=int(self._docs_truncated),
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
        self._docs_seen = int(st.docs_seen)
        self._docs_truncated = int(st.docs_truncated)


@dataclass(frozen=True)
class FFDPackerState:
    """JSON-serializable state shared by the FFD packers (bin, multipack).

    Position IDs are a pure function of segment IDs and are derived at pop
    time, so they are deliberately not part of the state.
    """

    pending_docs: list[list[int]]
    ready_tokens: list[list[int]]
    ready_segments: list[list[int]]
    docs_seen: int
    docs_truncated: int
    exhausted: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary.

        :return dict[str, Any]: State as a dict.
        """
        return {
            "pending_docs": self.pending_docs,
            "ready_tokens": self.ready_tokens,
            "ready_segments": self.ready_segments,
            "docs_seen": int(self.docs_seen),
            "docs_truncated": int(self.docs_truncated),
            "exhausted": bool(self.exhausted),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> FFDPackerState:
        """Construct FFDPackerState from a dictionary.

        :param dict[str, Any] d: State dict from to_dict().
        :raises KeyError: If a required field is missing (corrupt/foreign state).
        :raises ValueError: If any structural invariant fails (queue/row length
            pairing, counter signs).
        :return FFDPackerState: Reconstructed state.
        """
        # Deliberately strict, mirroring PackerState: defaulting missing
        # pending/ready queues to [] (as the old `d.get(...) or []` did)
        # silently resumes with an empty buffer instead of failing loud on
        # corrupt/foreign state. Capacity-dependent invariants (chunk sizes,
        # fixed ready-row length) live in set_state, which knows seq_len.
        pending = list(d["pending_docs"])
        ready_tokens = list(d["ready_tokens"])
        ready_segments = list(d["ready_segments"])
        if len(ready_tokens) != len(ready_segments):
            raise ValueError(
                "ready_tokens and ready_segments must have the same length "
                f"({len(ready_tokens)} != {len(ready_segments)})"
            )
        for i, (row_t, row_s) in enumerate(zip(ready_tokens, ready_segments, strict=True)):
            if len(row_t) != len(row_s):
                raise ValueError(
                    f"ready_tokens[{i}] and ready_segments[{i}] lengths differ "
                    f"({len(row_t)} != {len(row_s)})"
                )
        docs_seen = int(d["docs_seen"])
        docs_truncated = int(d["docs_truncated"])
        if docs_seen < 0 or docs_truncated < 0 or docs_truncated > docs_seen:
            raise ValueError(
                f"invalid document counters (docs_seen={docs_seen}, "
                f"docs_truncated={docs_truncated})"
            )
        return FFDPackerState(
            pending_docs=[list(x) for x in pending],
            ready_tokens=[list(x) for x in ready_tokens],
            ready_segments=[list(x) for x in ready_segments],
            docs_seen=docs_seen,
            docs_truncated=docs_truncated,
            exhausted=bool(d["exhausted"]),
        )


@dataclass
class _Bin:
    """A single bin used during FFD packing."""

    capacity: int
    max_docs: int | None
    segments: list[np.ndarray] = field(default_factory=list)
    remaining: int = field(init=False)

    def __post_init__(self) -> None:
        self.remaining = int(self.capacity)

    def can_fit(self, seg: np.ndarray) -> bool:
        """Return True if the segment can fit in this bin.

        :param np.ndarray seg: Token segment to place in the bin.
        :return bool: True if segment fits.
        """
        if seg.size > self.remaining:
            return False
        if self.max_docs is not None:
            return len(self.segments) < self.max_docs
        return True

    def add(self, seg: np.ndarray) -> None:
        """Add a segment to the bin.

        :param np.ndarray seg: Token segment to add.
        :raises ValueError: If the segment does not fit.
        """
        if not self.can_fit(seg):
            raise ValueError("segment does not fit in bin")
        self.segments.append(seg)
        self.remaining -= int(seg.size)


def _chunk_to_capacity(doc: np.ndarray, capacity: int) -> list[np.ndarray]:
    """Split a document into capacity-sized chunks (last chunk may be short).

    :param np.ndarray doc: Document token array.
    :param int capacity: Maximum chunk length.
    :return list[np.ndarray]: Non-empty chunks in document order.
    """
    if doc.size <= capacity:
        return [doc]
    return [doc[start : start + capacity] for start in range(0, int(doc.size), capacity)]


def _place_first_fit(bins: list[_Bin], seg: np.ndarray) -> bool:
    """Place a segment into the first compatible bin.

    :param list[_Bin] bins: Candidate bins in search order.
    :param np.ndarray seg: Segment to place.
    :return bool: True when a bin accepted the segment.
    """
    for b in bins:
        if b.can_fit(seg):
            b.add(seg)
            return True
    return False


def _ffd_pack(
    candidates: list[np.ndarray],
    *,
    bins_per_pack: int,
    capacity: int,
    max_docs: int | None,
) -> tuple[list[_Bin], list[np.ndarray]]:
    """First-fit-decreasing pack candidates into bins_per_pack bins.

    Seeds each bin with one of the oldest candidates to guarantee FIFO
    progress, then considers the remaining candidates by size (descending,
    stable) for first-fit placement. Requires len(candidates) >=
    bins_per_pack and every candidate <= capacity.

    :param list[np.ndarray] candidates: Candidate segments to pack.
    :param int bins_per_pack: Number of bins to produce.
    :param int capacity: Bin capacity in tokens.
    :param max_docs: Optional cap on segments per bin.
    :return tuple[list[_Bin], list[np.ndarray]]: Filled bins and leftover
        segments in original candidate (arrival) order, so callers can requeue
        them without scrambling stream locality.
    """
    bins = [_Bin(capacity=capacity, max_docs=max_docs) for _ in range(bins_per_pack)]
    for slot in range(bins_per_pack):
        bins[slot].add(candidates[slot])

    leftover_idx: list[int] = []
    fill_order = sorted(
        range(bins_per_pack, len(candidates)),
        key=lambda i: int(candidates[i].size),
        reverse=True,
    )
    for idx in fill_order:
        if not _place_first_fit(bins, candidates[idx]):
            leftover_idx.append(idx)
    leftover = [candidates[i] for i in sorted(leftover_idx)]
    return bins, leftover


def _ffd_pack_all(
    candidates: list[np.ndarray],
    *,
    capacity: int,
    max_docs: int | None,
) -> list[_Bin]:
    """First-fit-decreasing pack all candidates, opening bins as needed.

    Used to flush a partially filled pending queue at stream exhaustion:
    unlike :func:`_ffd_pack` there is no fixed bin count and no leftover —
    every candidate is placed (each fits an empty bin because documents are
    pre-chunked to capacity).

    :param list[np.ndarray] candidates: Candidate segments to pack.
    :param int capacity: Bin capacity in tokens.
    :param max_docs: Optional cap on segments per bin.
    :return list[_Bin]: Filled bins in creation order.
    """
    order = sorted(range(len(candidates)), key=lambda i: int(candidates[i].size), reverse=True)
    bins: list[_Bin] = []
    for idx in order:
        seg = candidates[idx]
        if not _place_first_fit(bins, seg):
            b = _Bin(capacity=capacity, max_docs=max_docs)
            b.add(seg)
            bins.append(b)
    return bins


def _render_bin(b: _Bin, *, capacity: int, pad_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Render a bin into (tokens, segment_ids) arrays of length capacity.

    :param _Bin b: Bin with packed segments.
    :param int capacity: Output sequence length.
    :param int pad_id: Padding token ID for the tail.
    :return tuple[np.ndarray, np.ndarray]: Token and segment arrays.
    """
    tokens = np.full((capacity,), pad_id, dtype=np.int32)
    segs = np.zeros((capacity,), dtype=np.int32)

    pos = 0
    seg_id = 1
    for seg in b.segments:
        end = pos + int(seg.size)
        tokens[pos:end] = seg
        segs[pos:end] = seg_id
        pos = end
        seg_id += 1

    return tokens, segs


class _FFDPackerBase(_PackerBase):
    """Shared machinery for the FFD-based packers (bin, multipack).

    Subclasses select whether each cycle consumes the entire pending pool
    (bin) or one bounded FIFO group (multipack). Every cycle seeds bins with
    the oldest candidates before using FFD for fill, so no pending chunk can
    starve behind a continuing stream of larger arrivals. Once `finish()`
    marks the stream exhausted, sub-threshold pending documents are flushed
    into padded bins instead of being dropped.
    """

    _mode_name = "ffd"

    def __init__(
        self,
        *,
        seq_len: int,
        add_bos: bool,
        add_eos: bool,
        bos_id: int,
        eos_id: int,
        max_doc_tokens: int | None,
        bins_per_pack: int,
        lookahead_docs: int,
        bounded_group: bool,
        max_docs_per_bin: int | None,
        pad_id: int,
    ):
        """Initialize shared FFD packer state.

        :param int seq_len: Fixed sequence length (T) for output.
        :param bool add_bos: Whether to prepend BOS token to each document.
        :param bool add_eos: Whether to append EOS token to each document.
        :param int bos_id: BOS token ID.
        :param int eos_id: EOS token ID.
        :param max_doc_tokens: Optional max tokens per document before truncation.
        :param int bins_per_pack: Number of sequences to pack per cycle.
        :param int lookahead_docs: Pending-document threshold for packing.
        :param bool bounded_group: Whether one cycle consumes at most the threshold.
        :param max_docs_per_bin: Optional cap on docs per bin.
        :param int pad_id: Padding token ID.
        :raises ValueError: If an input argument is invalid.
        """
        super().__init__(
            seq_len=seq_len,
            add_bos=add_bos,
            add_eos=add_eos,
            bos_id=bos_id,
            eos_id=eos_id,
            max_doc_tokens=max_doc_tokens,
        )
        if bins_per_pack <= 0:
            raise ValueError(f"bins_per_pack must be positive, got {bins_per_pack}")
        if max_docs_per_bin is not None and max_docs_per_bin <= 0:
            raise ValueError(f"max_docs_per_bin must be positive when set, got {max_docs_per_bin}")

        self._bins_per_pack = int(bins_per_pack)
        self._lookahead_docs = int(lookahead_docs)
        self._bounded_group = bool(bounded_group)
        self._max_docs_per_bin = None if max_docs_per_bin is None else int(max_docs_per_bin)
        self._pad_id = int(pad_id)

        self._pending_docs: deque[np.ndarray] = deque()
        self._ready: deque[tuple[np.ndarray, np.ndarray]] = deque()
        self._exhausted = False

    def _pack_threshold(self) -> int:
        """Return the pending-doc count that triggers a pack cycle.

        :return int: Minimum pending documents before packing.
        """
        return max(self._bins_per_pack, self._lookahead_docs)

    def _pack(self) -> None:
        """FFD-pack candidates selected by the configured queue policy."""
        candidate_count = len(self._pending_docs)
        if self._bounded_group:
            candidate_count = min(candidate_count, self._pack_threshold())
        candidates = [self._pending_docs.popleft() for _ in range(candidate_count)]
        bins, leftovers = _ffd_pack(
            candidates,
            bins_per_pack=self._bins_per_pack,
            capacity=self.seq_len,
            max_docs=self._max_docs_per_bin,
        )
        for segment in reversed(leftovers):
            self._pending_docs.appendleft(segment)
        for bin_ in bins:
            self._ready.append(_render_bin(bin_, capacity=self.seq_len, pad_id=self._pad_id))

    def _flush(self) -> None:
        """FFD-pack every pending document into as many bins as needed.

        Runs only after :meth:`finish`, when the pending queue can no longer
        reach the pack threshold; without it every sub-threshold tail document
        would be silently dropped at stream end.
        """
        candidates = list(self._pending_docs)
        self._pending_docs.clear()
        bins = _ffd_pack_all(candidates, capacity=self.seq_len, max_docs=self._max_docs_per_bin)
        for b in bins:
            self._ready.append(_render_bin(b, capacity=self.seq_len, pad_id=self._pad_id))

    def finish(self) -> None:
        """Mark the upstream document stream exhausted.

        Later `can_pop` calls flush partially filled buffers instead of
        waiting forever for the pack threshold. Idempotent; part of the
        checkpointed state.
        """
        self._exhausted = True

    def add_document(self, tokens: Iterable[int]) -> None:
        """Add a tokenized document to the pending queue.

        Oversized documents are split into capacity-sized chunks first.

        :param tokens: Iterable of token IDs for the document.
        """
        doc = self._prepare_document(tokens)
        if doc.size == 0:
            return
        self._pending_docs.extend(_chunk_to_capacity(doc, self.seq_len))

    def can_pop(self) -> bool:
        """Check if we can pop a packed sequence.

        :return bool: True if a sequence is ready.
        """
        if self._ready:
            return True
        if len(self._pending_docs) >= self._pack_threshold():
            self._pack()
        elif self._exhausted and self._pending_docs:
            self._flush()
        return bool(self._ready)

    def pop_seq_with_metadata(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ([seq_len] tokens, [seq_len] segment_ids, [seq_len] position_ids).

        :raises RuntimeError: If called before any sequences are ready.
        :return tuple: (tokens, segment_ids, position_ids) arrays of shape [seq_len].
        """
        tokens, segs = self.pop_seq_with_segments()
        return tokens, segs, _positions_from_segments(segs)

    def pop_seq_with_segments(self) -> tuple[np.ndarray, np.ndarray]:
        """Return tokens and segment IDs without materializing position IDs.

        :raises RuntimeError: If called before any sequences are ready.
        :return tuple[np.ndarray, np.ndarray]: Fixed-length token and segment arrays.
        """
        if not self.can_pop():
            raise RuntimeError(f"{self._mode_name} packer has no ready sequences")
        return self._ready.popleft()

    def get_state(self) -> dict[str, Any]:
        """Capture packer state for checkpointing.

        :return dict[str, Any]: Serializable state dict.
        """
        st = FFDPackerState(
            pending_docs=[x.tolist() for x in self._pending_docs],
            ready_tokens=[x.tolist() for x, _ in self._ready],
            ready_segments=[x.tolist() for _, x in self._ready],
            docs_seen=int(self._docs_seen),
            docs_truncated=int(self._docs_truncated),
            exhausted=bool(self._exhausted),
        )
        return st.to_dict()

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore packer state from a checkpoint.

        :param dict[str, Any] state: State dict from get_state().
        :raises ValueError: If a queue entry violates a capacity invariant
            (corrupt/foreign state).
        """
        st = FFDPackerState.from_dict(state)
        # Capacity invariants from_dict cannot check: pending entries are
        # pre-split chunks in 1..capacity, ready rows are padded to exactly
        # seq_len with nonnegative segment ids (0 = padding).
        for i, doc in enumerate(st.pending_docs):
            if not 0 < len(doc) <= self.seq_len:
                raise ValueError(
                    f"pending_docs[{i}] has {len(doc)} tokens; expected 1..{self.seq_len}"
                )
        for i, (tokens, segs) in enumerate(zip(st.ready_tokens, st.ready_segments, strict=True)):
            if len(tokens) != self.seq_len:
                raise ValueError(
                    f"ready sequence {i} has {len(tokens)} tokens; expected "
                    f"exactly seq_len={self.seq_len}"
                )
            if any(s < 0 for s in segs):
                raise ValueError(f"ready sequence {i} has negative segment ids")
        self._pending_docs = deque(np.asarray(x, dtype=np.int32) for x in st.pending_docs)
        self._ready = deque(
            (
                np.asarray(tokens, dtype=np.int32),
                np.asarray(segs, dtype=np.int32),
            )
            for tokens, segs in zip(st.ready_tokens, st.ready_segments, strict=True)
        )
        self._docs_seen = int(st.docs_seen)
        self._docs_truncated = int(st.docs_truncated)
        self._exhausted = bool(st.exhausted)


class BinPacker(_FFDPackerBase):
    """Bin-pack documents into fixed-length sequences (FFD heuristic).

    Buffers at least `buffer_docs` documents, then FFD-packs the entire
    pending pool each cycle; leftovers stay pending for the next cycle.
    """

    _mode_name = "bin"

    def __init__(
        self,
        *,
        seq_len: int,
        add_bos: bool,
        add_eos: bool,
        bos_id: int,
        eos_id: int,
        max_doc_tokens: int | None,
        bins_per_pack: int,
        buffer_docs: int,
        max_docs_per_bin: int | None,
        pad_id: int,
    ):
        """Initialize the bin packer.

        :param int seq_len: Fixed sequence length for output.
        :param bool add_bos: Whether to prepend BOS to each document.
        :param bool add_eos: Whether to append EOS to each document.
        :param int bos_id: BOS token ID.
        :param int eos_id: EOS token ID.
        :param max_doc_tokens: Optional max tokens per document before truncation.
        :param int bins_per_pack: Number of bins emitted per pack cycle.
        :param int buffer_docs: Minimum docs to buffer before packing.
        :param max_docs_per_bin: Optional cap on documents per bin.
        :param int pad_id: Padding token ID.
        :raises ValueError: If an input argument is invalid.
        """
        if buffer_docs <= 0:
            raise ValueError(f"buffer_docs must be positive, got {buffer_docs}")
        super().__init__(
            seq_len=seq_len,
            add_bos=add_bos,
            add_eos=add_eos,
            bos_id=bos_id,
            eos_id=eos_id,
            max_doc_tokens=max_doc_tokens,
            bins_per_pack=bins_per_pack,
            lookahead_docs=buffer_docs,
            bounded_group=False,
            max_docs_per_bin=max_docs_per_bin,
            pad_id=pad_id,
        )


class MultipackPacker(_FFDPackerBase):
    """Groupwise FFD sample packer with segment-local position IDs.

    FFD-packs a bounded FIFO group of `group_docs` candidates per cycle;
    leftovers return to the front of the queue to preserve stream locality.
    """

    _mode_name = "multipack"

    def __init__(
        self,
        *,
        seq_len: int,
        add_bos: bool,
        add_eos: bool,
        bos_id: int,
        eos_id: int,
        max_doc_tokens: int | None,
        bins_per_pack: int,
        group_docs: int,
        max_docs_per_bin: int | None,
        pad_id: int,
    ):
        """Initialize the multipack packer.

        :param int seq_len: Fixed sequence length for output.
        :param bool add_bos: Whether to prepend BOS to each document.
        :param bool add_eos: Whether to append EOS to each document.
        :param int bos_id: BOS token ID.
        :param int eos_id: EOS token ID.
        :param max_doc_tokens: Optional max tokens per document before truncation.
        :param int bins_per_pack: Number of bins emitted per pack cycle.
        :param int group_docs: Number of candidate documents to consider per cycle.
        :param max_docs_per_bin: Optional cap on packed segments per sequence.
        :param int pad_id: Padding token ID.
        :raises ValueError: If an input argument is invalid.
        """
        if group_docs <= 0:
            raise ValueError(f"group_docs must be positive, got {group_docs}")
        super().__init__(
            seq_len=seq_len,
            add_bos=add_bos,
            add_eos=add_eos,
            bos_id=bos_id,
            eos_id=eos_id,
            max_doc_tokens=max_doc_tokens,
            bins_per_pack=bins_per_pack,
            lookahead_docs=group_docs,
            bounded_group=True,
            max_docs_per_bin=max_docs_per_bin,
            pad_id=pad_id,
        )
