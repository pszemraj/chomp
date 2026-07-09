"""Hugging Face streaming dataset wrapper.

This module exists because HF streaming is *almost* a perfect fit for pretraining,
but you need a little engineering around it:
- deterministic(ish) shuffling via `.shuffle(buffer_size, seed)`
- resumability via `state_dict()` / `load_state_dict()`
- network hiccup resistance (retry + rebuild-from-last-state)

We start with Zyphra/Zyda-2's `sample-100BT` config because it has a common schema
(`nemo_id`, `text`) and is pre-weighted.

We intentionally keep this wrapper minimal and 'dumb'. It yields raw text strings.
Tokenization + packing happen elsewhere.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import datasets

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HFStreamSpec:
    """Specification for a HuggingFace streaming dataset."""

    dataset: str
    name: str
    split: str
    text_key: str
    shuffle: bool
    shuffle_buffer_size: int
    seed: int
    repeat: bool

    max_retries: int
    retry_delay_sec: float
    state_update_interval: int


class HFStreamingTextStream:
    """Resumable streaming text stream from Hugging Face `datasets`.

    Implements:
      - `__next__` yielding `str`
      - `get_state()` / `set_state()` for checkpointing

    **Correct restore ordering** (important):
      1) rebuild dataset for epoch
      2) load_state_dict
      3) create iterator (`iter(ds)`)

    We have an explicit test for this ordering (see the HF state roundtrip
    test in tests/test_data_pipeline.py).
    """

    def __init__(self, spec: HFStreamSpec):
        """Initialize the streaming text stream.

        :param HFStreamSpec spec: Dataset specification.
        """
        self._spec = spec
        self._epoch = 0
        self._ds: datasets.IterableDataset
        self._it: Iterator[dict[str, Any]]
        self._n_since_state = 0
        self._last_state: dict[str, Any] | None = None
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
        )

        # Keep only the text column (smaller item dicts, less accidental schema drift).
        ds = ds.select_columns([self._spec.text_key])

        if self._spec.shuffle:
            ds = ds.shuffle(
                seed=int(self._spec.seed) + int(epoch),
                buffer_size=int(self._spec.shuffle_buffer_size),
            )
        return ds

    def _build(self) -> None:
        """Build or rebuild the dataset iterator for the current epoch."""
        self._ds = self._load_ds_for_epoch(self._epoch)
        self._it = iter(self._ds)
        self._n_since_state = 0
        self._last_state = None

    def __iter__(self) -> HFStreamingTextStream:
        return self

    def _record_state(self) -> None:
        """Cache state_dict periodically for retry recovery."""
        self._n_since_state += 1
        if self._n_since_state < self._spec.state_update_interval:
            return
        self._n_since_state = 0
        try:
            self._last_state = self._ds.state_dict()  # type: ignore[attr-defined]
        except Exception:
            logger.warning(
                "HF stream state_dict() failed during periodic caching; "
                "retry recovery will restart from the last good state (if any).",
                exc_info=True,
            )
            self._last_state = None

    def _recover_iterator(self) -> None:
        """Best-effort rebuild from last cached state."""
        if self._last_state is None:
            return
        try:
            self._ds = self._load_ds_for_epoch(self._epoch)
            self._ds.load_state_dict(self._last_state)  # type: ignore[attr-defined]
            self._it = iter(self._ds)
            logger.info("HF stream rebuilt from last cached state after failure.")
        except Exception:
            # Fall back to sleeping and retrying next() on current iterator.
            logger.warning(
                "HF stream rebuild from cached state failed; retrying on the "
                "current iterator instead.",
                exc_info=True,
            )
            return

    def _next_item(self) -> str:
        """Fetch and validate the next text item.

        :return str: Text payload from the dataset item.
        """
        item = next(self._it)
        if self._spec.text_key not in item:
            raise KeyError(
                f"HF item missing text key {self._spec.text_key!r}. Keys: {sorted(item.keys())}"
            )
        text = item[self._spec.text_key]
        if not isinstance(text, str):
            text = str(text)
        self._record_state()
        return text

    def __next__(self) -> str:
        # Retry loop for transient failures
        for attempt in range(self._spec.max_retries + 1):
            try:
                return self._next_item()

            except StopIteration:
                if not self._spec.repeat:
                    raise
                # Repeat => advance epoch and rebuild
                self._epoch += 1
                self._build()
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
                # Best-effort recovery: rebuild ds from last known state if available.
                self._recover_iterator()

                time.sleep(delay)

        # Should not reach
        raise RuntimeError("HFStreamingTextStream retry loop fell through")

    def get_state(self) -> dict[str, Any]:
        """Capture stream state for checkpointing.

        :return dict[str, Any]: State dict with epoch and HF iterator state.
        :raises RuntimeError: If the dataset cannot produce a ``state_dict()``.
            Better to fail the save than write a checkpoint that silently
            cannot resume exactly.
        """
        try:
            hf_state = self._ds.state_dict()  # type: ignore[attr-defined]
        except Exception as e:
            raise RuntimeError(
                "HF streaming dataset failed to produce state_dict(); refusing "
                "to write a checkpoint whose data stream cannot resume exactly."
            ) from e

        return {
            "epoch": int(self._epoch),
            "hf_state": hf_state,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore stream state from a checkpoint.

        :param dict[str, Any] state: State dict from get_state().
        :raises RuntimeError: If ``hf_state`` is missing. Every checkpoint
            written by this module has a non-None ``hf_state`` (get_state()
            fails loud if it can't capture one); a missing value means the
            checkpoint predates exact HF stream capture or is corrupt, and
            only an approximate epoch/seed rebuild is possible — refuse
            rather than resume silently wrong.
        :raises Exception: If load_state_dict fails (better to crash than silently reset).
        """
        epoch = int(state["epoch"])
        hf_state = state.get("hf_state")
        self._epoch = epoch

        if hf_state is None:
            raise RuntimeError(
                "Checkpoint is missing hf_state for the HF streaming dataset; "
                "refusing to approximate resume by rebuilding from epoch/seed. "
                "This checkpoint predates exact HF stream capture (or is "
                "corrupt) and exact resume is impossible."
            )

        # Correct ordering:
        # 1) rebuild dataset
        # 2) load_state_dict (crash on failure — never silently restart from zero)
        # 3) iter(ds)
        self._ds = self._load_ds_for_epoch(self._epoch)
        self._ds.load_state_dict(hf_state)  # type: ignore[attr-defined]
        self._it = iter(self._ds)
        self._n_since_state = 0
        self._last_state = hf_state


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
