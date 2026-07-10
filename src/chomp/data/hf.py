"""Hugging Face streaming dataset wrapper.

This module adds the state that pretraining needs around HF streaming:
- deterministic, resume-safe fixed-window document shuffling
- resumability via the source dataset's `state_dict()` / `load_state_dict()`
- network hiccup resistance (retry + rebuild-from-last-state)

We start with Zyphra/Zyda-2's `sample-100BT` config because it has a common schema
(`nemo_id`, `text`) and is pre-weighted.

We intentionally keep this wrapper minimal and 'dumb'. It yields raw text strings.
Tokenization + packing happen elsewhere.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import datasets

logger = logging.getLogger(__name__)

_UINT64_MASK = 2**64 - 1
_SPLITMIX_INCREMENT = 0x9E3779B97F4A7C15
CONTENT_HOLDOUT_SCHEMA_VERSION = 1
ContentPartition = Literal["all", "train", "eval"]


def is_eval_holdout(text: str, *, fraction: float) -> bool:
    """Return the stable content-hash holdout assignment for one document.

    Identical content always lands on the same side, so duplicated documents
    cannot leak between train and eval even when source row IDs are absent.

    :param str text: Complete source document text.
    :param float fraction: Eval share in the open interval (0, 1).
    :return bool: True when the document belongs exclusively to evaluation.
    """
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8, person=b"chomp-eval-v1").digest()
    bucket = int.from_bytes(digest, "big")
    return bucket < int(float(fraction) * (2**64))


def _splitmix64(value: int) -> int:
    """Mix one unsigned 64-bit value deterministically.

    :param int value: Integer state to mix.
    :return int: Mixed unsigned 64-bit value.
    """
    value = (int(value) + _SPLITMIX_INCREMENT) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return (value ^ (value >> 31)) & _UINT64_MASK


def _shuffle_window(items: list[str], *, seed: int, epoch: int, window_index: int) -> None:
    """Shuffle a document window with an implementation-owned permutation.

    :param list[str] items: Text documents to shuffle in place.
    :param int seed: Configured base shuffle seed.
    :param int epoch: Source epoch number.
    :param int window_index: Zero-based window number within the epoch.
    """
    state = _splitmix64(seed)
    state ^= _splitmix64(epoch)
    state ^= _splitmix64(window_index)
    for index in range(len(items) - 1, 0, -1):
        state = _splitmix64(state)
        swap_index = state % (index + 1)
        items[index], items[swap_index] = items[swap_index], items[index]


@dataclass(frozen=True)
class HFStreamSpec:
    """Specification for a HuggingFace streaming dataset."""

    dataset: str
    name: str
    split: str
    text_key: str
    revision: str | None
    shuffle: bool
    shuffle_buffer_size: int
    shuffle_buffer_bytes: int
    seed: int
    repeat: bool
    content_partition: ContentPartition
    eval_holdout_fraction: float

    max_retries: int
    retry_delay_sec: float
    state_update_interval: int


class HFStreamingTextStream:
    """Resumable streaming text stream from Hugging Face `datasets`.

    Implements:
      - `__next__` yielding `str`
      - `get_state()` / `set_state()` for checkpointing

    HF's own ``IterableDataset.shuffle()`` does not serialize documents still
    resident in its read-ahead buffer. This wrapper therefore keeps the HF
    source unshuffled and owns fixed-window document mixing. A shuffled
    checkpoint stores the source state at the beginning of the current window,
    its deterministic window index, and the output cursor. Restore replays the
    source window and reconstructs the same permutation without storing the
    documents themselves.

    Correct source restore ordering is rebuild dataset, load state, then create
    its iterator.

    We have an explicit test for this ordering (see the HF state roundtrip
    test in tests/test_data_pipeline.py).
    """

    def __init__(self, spec: HFStreamSpec):
        """Initialize the streaming text stream.

        :param HFStreamSpec spec: Dataset specification.
        """
        self._spec = spec
        self._closed = False
        self._epoch = 0
        self._ds: datasets.IterableDataset
        self._it: Iterator[dict[str, Any]]
        self._source_started = False
        self._n_since_state = 0
        self._last_state: dict[str, Any] | None = None
        self._window: list[str] = []
        self._window_cursor = 0
        self._window_index = 0
        self._window_parent_state: dict[str, Any] | None = None
        self._window_bytes = 0
        self._peak_window_docs = 0
        self._peak_window_bytes = 0
        self._replayed_window_docs = 0
        self._replayed_window_bytes = 0
        self._build()

    def _load_ds_for_epoch(self, epoch: int) -> datasets.IterableDataset:
        """Load and configure the dataset for a given epoch.

        :param int epoch: Epoch number (used to seed shuffle).
        :return datasets.IterableDataset: Configured streaming dataset.
        """
        import datasets

        ds = datasets.load_dataset(
            self._spec.dataset,
            name=self._spec.name,
            split=self._spec.split,
            streaming=True,
            revision=self._spec.revision,
        )

        # Keep only the text column (smaller item dicts, less accidental schema drift).
        ds = ds.select_columns([self._spec.text_key])

        return ds

    def _build(self) -> None:
        """Build or rebuild the dataset iterator for the current epoch."""
        self._close_source_iterator()
        self._ds = self._load_ds_for_epoch(self._epoch)
        self._it = iter(self._ds)
        self._closed = False
        self._source_started = False
        self._n_since_state = 0
        self._last_state = None
        self._window = []
        self._window_cursor = 0
        self._window_index = 0
        self._window_parent_state = None
        self._window_bytes = 0
        # Retry recovery starts exact even when the first source read fails.
        self._last_state = self.get_state()

    def _close_source_iterator(self) -> None:
        """Close the active Hugging Face generator when one exists."""
        iterator = getattr(self, "_it", None)
        close = getattr(iterator, "close", None)
        if callable(close):
            close()

    def close(self) -> None:
        """Release the active Hugging Face streaming iterator."""
        if self._closed:
            return
        ex_iterable = getattr(getattr(self, "_ds", None), "_ex_iterable", None)
        wait_for_arrow = self._source_started and bool(
            getattr(ex_iterable, "sleep_on_threads_shutdown", False)
        )
        self._close_source_iterator()
        self._window = []
        self._window_cursor = 0
        self._window_parent_state = None
        self._window_bytes = 0
        if wait_for_arrow:
            # Datasets marks remote Parquet builders that need this grace for
            # Apache Arrow #45214. Its multi-source iterators apply the delay
            # themselves, but a partially consumed single-source iterator can
            # otherwise still call into CPython after interpreter finalization.
            from datasets import config

            time.sleep(config.SLEEP_TIME_ON_THREADS_SHUTDOWN)
        self._source_started = False
        self._closed = True

    def __iter__(self) -> HFStreamingTextStream:
        return self

    def _record_state(self) -> None:
        """Cache compact stream state periodically for retry recovery."""
        self._n_since_state += 1
        if self._n_since_state < self._spec.state_update_interval:
            return
        try:
            state = self.get_state()
        except Exception:
            logger.warning(
                "HF stream state_dict() failed during periodic caching; "
                "retry recovery will restart from the last good state (if any).",
                exc_info=True,
            )
            # Keep both the last known-good state and the exact number of
            # documents yielded since it. Discarding either makes recovery
            # silently replay or skip records.
            return
        self._last_state = state
        self._n_since_state = 0

    def _recover_iterator(self) -> None:
        """Rebuild and fast-forward exactly from the last cached state.

        :raises RuntimeError: If no known-good state exists or reconstruction
            cannot reach the exact pre-failure logical position.
        """
        if self._last_state is None:
            raise RuntimeError(
                "HF retry recovery has no known-good iterator state; refusing an inexact retry."
            )
        state = self._last_state
        yielded_since_state = int(self._n_since_state)
        try:
            self.set_state(state)
            for _ in range(yielded_since_state):
                if self._spec.shuffle:
                    self._next_shuffled_text()
                else:
                    self._read_text()
            # Promote the reconstructed current position to the next
            # known-good recovery point. A failure here must terminate rather
            # than retry from partially reconstructed state.
            self._last_state = self.get_state()
            self._n_since_state = 0
        except Exception as exc:
            raise RuntimeError(
                "HF stream could not reconstruct its exact pre-failure position; "
                "stop and resume from the last Chomp checkpoint."
            ) from exc
        logger.info(
            "HF stream rebuilt and fast-forwarded %d document(s) after failure.",
            yielded_since_state,
        )

    def _source_state(self) -> dict[str, Any]:
        """Capture the unshuffled HF source state.

        :raises RuntimeError: If the source cannot produce checkpoint state.
        :return dict[str, Any]: Source iterator state.
        """
        try:
            return self._ds.state_dict()  # type: ignore[attr-defined]
        except Exception as exc:
            raise RuntimeError(
                "HF streaming dataset failed to produce state_dict(); refusing "
                "to write a checkpoint whose data stream cannot resume exactly."
            ) from exc

    def _read_text(self) -> str:
        """Read and validate one document from the HF source.

        :return str: Text payload from the dataset item.
        """
        while True:
            self._source_started = True
            item = next(self._it)
            if self._spec.text_key not in item:
                raise KeyError(
                    f"HF item missing text key {self._spec.text_key!r}. Keys: {sorted(item.keys())}"
                )
            value = item[self._spec.text_key]
            text = value if isinstance(value, str) else str(value)
            if self._spec.content_partition == "all":
                return text
            held_out = is_eval_holdout(text, fraction=self._spec.eval_holdout_fraction)
            if (self._spec.content_partition == "eval") == held_out:
                return text

    def _shuffle_checkpoint(self, *, parent_state: dict[str, Any]) -> dict[str, Any]:
        """Build compact state for the current shuffled document window.

        :param dict[str, Any] parent_state: Source state at the window start.
        :return dict[str, Any]: Resume state without buffered documents.
        """
        return {
            "epoch": int(self._epoch),
            "shuffle_state": {
                "window_index": int(self._window_index),
                "cursor": int(self._window_cursor),
                "parent_state": parent_state,
            },
        }

    def _fill_shuffle_window(self, *, replay: bool = False) -> None:
        """Read and deterministically permute the next source document window.

        A window stops after reaching either the document-count cap or the
        UTF-8 payload-byte budget. The document that reaches the byte budget
        remains in the window, avoiding an uncheckpointed read-ahead item.

        :param bool replay: Whether this fill reconstructs a checkpointed window.
        """
        parent_state = self._source_state()
        self._window_parent_state = parent_state
        self._window = []
        self._window_cursor = 0
        self._window_bytes = 0

        # A network failure can occur only while the source window is being
        # filled. Cache its start unconditionally so retry reconstruction
        # cannot lose a partially read buffer or replay emitted documents.
        self._last_state = self._shuffle_checkpoint(parent_state=parent_state)
        self._n_since_state = 0

        for _ in range(int(self._spec.shuffle_buffer_size)):
            try:
                text = self._read_text()
            except StopIteration:
                break
            self._window.append(text)
            self._window_bytes += len(text.encode("utf-8"))
            if self._window_bytes >= int(self._spec.shuffle_buffer_bytes):
                break
        if not self._window:
            raise StopIteration
        self._peak_window_docs = max(self._peak_window_docs, len(self._window))
        self._peak_window_bytes = max(self._peak_window_bytes, self._window_bytes)
        if replay:
            self._replayed_window_docs += len(self._window)
            self._replayed_window_bytes += self._window_bytes
        _shuffle_window(
            self._window,
            seed=int(self._spec.seed),
            epoch=int(self._epoch),
            window_index=int(self._window_index),
        )

    def _next_shuffled_text(self) -> str:
        """Return the next document from the owned shuffle window.

        :return str: Shuffled text document.
        """
        if self._window and self._window_cursor >= len(self._window):
            self._window_index += 1
            self._window = []
            self._window_cursor = 0
            self._window_parent_state = None
        if not self._window:
            self._fill_shuffle_window()
        text = self._window[self._window_cursor]
        self._window_cursor += 1
        return text

    def _next_item(self) -> str:
        """Fetch and validate the next text item.

        :return str: Text payload from the dataset item.
        """
        text = self._next_shuffled_text() if self._spec.shuffle else self._read_text()
        self._record_state()
        return text

    def __next__(self) -> str:
        if self._closed:
            raise ValueError("Attempting to use a closed HF streaming iterator.")
        # Epoch rollover is ordinary stream control flow, not a retry. Keep it
        # outside transient-failure accounting so max_retries=0 can still
        # repeat and the first read of every epoch gets the full retry budget.
        attempt = 0
        rolled_epoch = False
        while True:
            try:
                return self._next_item()

            except StopIteration:
                if not self._spec.repeat:
                    raise
                if rolled_epoch:
                    raise RuntimeError(
                        "HF repeated stream produced no documents in a complete epoch. "
                        "Check the selected split and content partition."
                    ) from None
                self._epoch += 1
                self._build()
                attempt = 0
                rolled_epoch = True
                continue

            except Exception:
                if attempt >= self._spec.max_retries:
                    raise

                delay = self._spec.retry_delay_sec * (2**attempt)
                logger.warning(
                    "HF stream next() failed (attempt %d/%d); retrying in %.1fs.",
                    attempt + 1,
                    self._spec.max_retries,
                    delay,
                    exc_info=True,
                )
                # Exact recovery: rebuild and discard precisely the records
                # already yielded since the last compact state.
                self._recover_iterator()

                time.sleep(delay)
                attempt += 1

    def get_state(self) -> dict[str, Any]:
        """Capture stream state for checkpointing.

        :return dict[str, Any]: State dict with epoch and HF iterator state.
        :raises RuntimeError: If the dataset cannot produce a ``state_dict()``.
            Better to fail the save than write a checkpoint that silently
            cannot resume exactly.
        """
        if not self._spec.shuffle:
            return {"epoch": int(self._epoch), "hf_state": self._source_state()}

        parent_state = self._window_parent_state
        if parent_state is None:
            parent_state = self._source_state()
        return self._shuffle_checkpoint(parent_state=parent_state)

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore stream state from a checkpoint.

        :param dict[str, Any] state: State dict from get_state().
        :raises RuntimeError: If required source or shuffle replay state is
            missing or invalid.
        :raises Exception: If load_state_dict fails (better to crash than silently reset).
        """
        self._close_source_iterator()
        epoch = int(state["epoch"])
        self._epoch = epoch
        self._ds = self._load_ds_for_epoch(self._epoch)

        if not self._spec.shuffle:
            hf_state = state.get("hf_state")
            if hf_state is None:
                raise RuntimeError(
                    "Checkpoint is missing hf_state for the HF streaming dataset; "
                    "refusing to approximate resume by rebuilding from epoch/seed."
                )
            self._ds.load_state_dict(hf_state)  # type: ignore[attr-defined]
            self._it = iter(self._ds)
            self._closed = False
            self._source_started = False
            self._window = []
            self._window_cursor = 0
            self._window_index = 0
            self._window_parent_state = None
            self._window_bytes = 0
        else:
            shuffle_state = state.get("shuffle_state")
            if not isinstance(shuffle_state, dict):
                raise RuntimeError(
                    "Checkpoint is missing shuffle_state for the resume-safe HF "
                    "document shuffle; exact resume is impossible."
                )
            parent_state = shuffle_state.get("parent_state")
            if not isinstance(parent_state, dict):
                raise RuntimeError("HF shuffle_state is missing its parent_state.")
            window_index = int(shuffle_state.get("window_index", -1))
            cursor = int(shuffle_state.get("cursor", -1))
            if window_index < 0 or cursor < 0:
                raise RuntimeError(
                    "HF shuffle_state has a negative window_index or cursor; checkpoint is corrupt."
                )
            self._ds.load_state_dict(parent_state)  # type: ignore[attr-defined]
            self._it = iter(self._ds)
            self._closed = False
            self._source_started = False
            self._window_index = window_index
            self._window = []
            self._window_cursor = 0
            self._window_parent_state = parent_state
            self._fill_shuffle_window(replay=True)
            if cursor > len(self._window):
                raise RuntimeError(
                    "HF shuffle_state cursor exceeds the reconstructed window length "
                    f"({cursor} > {len(self._window)}); checkpoint or source is incompatible."
                )
            self._window_cursor = cursor

        self._n_since_state = 0
        self._last_state = self.get_state()

    def get_stats(self) -> dict[str, int]:
        """Return owned document-shuffle memory and replay diagnostics.

        :return dict[str, int]: Current/peak window sizes and replay totals.
        """
        if not self._spec.shuffle:
            return {}
        return {
            "shuffle_window_docs": len(self._window),
            "shuffle_window_bytes": int(self._window_bytes),
            "shuffle_peak_window_docs": int(self._peak_window_docs),
            "shuffle_peak_window_bytes": int(self._peak_window_bytes),
            "shuffle_replayed_window_docs": int(self._replayed_window_docs),
            "shuffle_replayed_window_bytes": int(self._replayed_window_bytes),
        }


class LocalTextStream:
    """Deterministic local text stream.

    This exists for:
    - offline tests
    - smoke configs that shouldn't hit the network

    It still exercises the real tokenize+pack path, so it isn't a "synthetic batch" crutch.
    """

    def __init__(self, *, text: str, repeat: bool = True):
        """Initialize the local text stream.

        :param str text: Text string to yield.
        :param bool repeat: Whether to repeat indefinitely (default True).
        """
        self._text = text
        self._repeat = bool(repeat)
        self._i = 0

    def __iter__(self) -> LocalTextStream:
        return self

    def __next__(self) -> str:
        if not self._repeat and self._i > 0:
            raise StopIteration
        self._i += 1
        return self._text

    def get_state(self) -> dict[str, Any]:
        """Capture stream state for checkpointing.

        :return dict[str, Any]: State dict with iteration count.
        """
        return {"i": int(self._i)}

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore stream state from a checkpoint.

        :param dict[str, Any] state: State dict from get_state().
        """
        self._i = int(state["i"])

    def close(self) -> None:
        """Close the in-memory stream (a no-op)."""
