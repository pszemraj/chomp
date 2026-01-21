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

import contextlib
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import datasets


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

    We have an explicit test for this ordering (see tests/test_hf_state_roundtrip.py).
    """

    def __init__(self, spec: HFStreamSpec, *, epoch0: int = 0):
        """Initialize the streaming text stream.

        :param HFStreamSpec spec: Dataset specification.
        :param int epoch0: Starting epoch number (default 0).
        """
        self._spec = spec
        self._epoch = int(epoch0)
        self._ds = None
        self._it: Iterator[dict[str, Any]] | None = None
        self._n_since_state = 0
        self._last_state: dict[str, Any] | None = None
        self._build()

    @property
    def epoch(self) -> int:
        """Current epoch number.

        :return int: The current epoch index.
        """
        return self._epoch

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

        # Keep only the text column (smaller item dicts, less accidental schema drift)
        # Some older datasets versions may not support select_columns in streaming.
        with contextlib.suppress(Exception):
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

    def __next__(self) -> str:
        if self._it is None or self._ds is None:
            self._build()

        # Retry loop for transient failures
        for attempt in range(self._spec.max_retries + 1):
            try:
                item = next(self._it)
                if self._spec.text_key not in item:
                    raise KeyError(
                        f"HF item missing text key {self._spec.text_key!r}. Keys: {sorted(item.keys())}"
                    )
                text = item[self._spec.text_key]
                if not isinstance(text, str):
                    text = str(text)

                # Periodically cache state for retry recovery.
                self._n_since_state += 1
                if self._n_since_state >= self._spec.state_update_interval:
                    self._n_since_state = 0
                    # Best-effort: only works on streaming datasets.
                    try:
                        self._last_state = self._ds.state_dict()  # type: ignore[attr-defined]
                    except Exception:
                        self._last_state = None

                return text

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

                # Best-effort recovery: rebuild ds from last known state if available.
                if self._last_state is not None:
                    try:
                        self._ds = self._load_ds_for_epoch(self._epoch)
                        self._ds.load_state_dict(self._last_state)  # type: ignore[attr-defined]
                        self._it = iter(self._ds)
                    except Exception:
                        # Fall back to sleeping and retrying next() on current iterator.
                        pass

                time.sleep(self._spec.retry_delay_sec * (2**attempt))

        # Should not reach
        raise RuntimeError("HFStreamingTextStream retry loop fell through")

    def get_state(self) -> dict[str, Any]:
        """Capture stream state for checkpointing.

        :return dict[str, Any]: State dict with epoch and HF iterator state.
        """
        if self._ds is None:
            self._build()
        hf_state = None
        try:
            hf_state = self._ds.state_dict()  # type: ignore[attr-defined]
        except Exception:
            # If state_dict isn't available, resumability is broken.
            hf_state = None

        return {
            "epoch": int(self._epoch),
            "hf_state": hf_state,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore stream state from a checkpoint.

        :param dict[str, Any] state: State dict from get_state().
        :raises Exception: If load_state_dict fails (better to crash than silently reset).
        """
        epoch = int(state.get("epoch", 0))
        hf_state = state.get("hf_state")
        self._epoch = epoch

        # Correct ordering:
        # 1) rebuild dataset
        # 2) load_state_dict
        # 3) iter(ds)
        self._ds = self._load_ds_for_epoch(self._epoch)
        if hf_state is not None:
            try:
                self._ds.load_state_dict(hf_state)  # type: ignore[attr-defined]
            except Exception:
                # If this fails, better to crash than silently restart from beginning.
                raise
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
        self._i = int(state.get("i", 0))


class ListTextStream:
    """Stream over a fixed list of texts.

    Useful for validation sets derived from streaming datasets.
    """

    def __init__(self, *, texts: list[str], repeat: bool = True):
        """Initialize the list-backed text stream.

        :param list[str] texts: Ordered list of text examples.
        :param bool repeat: Whether to loop when reaching the end.
        """
        self._texts = list(texts)
        self._repeat = bool(repeat)
        self._i = 0

    def __iter__(self) -> ListTextStream:
        return self

    def __next__(self) -> str:
        if not self._texts:
            raise StopIteration
        if self._i >= len(self._texts):
            if not self._repeat:
                raise StopIteration
            self._i = 0
        text = self._texts[self._i]
        self._i += 1
        return text

    def get_state(self) -> dict[str, Any]:
        """Capture stream state for checkpointing.

        :return dict[str, Any]: State dict with current index.
        """
        return {"i": int(self._i)}

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore stream state from a checkpoint.

        :param dict[str, Any] state: State dict from get_state().
        """
        self._i = int(state.get("i", 0))


class ListTokenStream:
    """Stream over a fixed list of tokenized documents."""

    def __init__(self, *, tokens: list[list[int]], repeat: bool = True):
        """Initialize the list-backed token stream.

        :param list[list[int]] tokens: Ordered list of tokenized documents.
        :param bool repeat: Whether to loop when reaching the end.
        """
        self._tokens = [list(x) for x in tokens]
        self._repeat = bool(repeat)
        self._i = 0

    def __iter__(self) -> ListTokenStream:
        return self

    def __next__(self) -> list[int]:
        if not self._tokens:
            raise StopIteration
        if self._i >= len(self._tokens):
            if not self._repeat:
                raise StopIteration
            self._i = 0
        item = self._tokens[self._i]
        self._i += 1
        return item

    def get_state(self) -> dict[str, Any]:
        """Capture stream state for checkpointing.

        :return dict[str, Any]: State dict with current index.
        """
        return {"i": int(self._i)}

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore stream state from a checkpoint.

        :param dict[str, Any] state: State dict from get_state().
        """
        self._i = int(state.get("i", 0))
