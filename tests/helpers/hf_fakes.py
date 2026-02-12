"""Reusable fake Hugging Face streaming iterables for tests."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any


@dataclass
class FakeHFIterable:
    """Mock iterable dataset with optional shuffle hook."""

    items: list[dict[str, Any]]
    index: int = 0
    on_shuffle: Callable[[int, int], None] | None = None

    def select_columns(self, _columns: list[str]) -> FakeHFIterable:
        """Return self (columns not used in tests)."""
        return self

    def shuffle(self, *, seed: int, buffer_size: int) -> FakeHFIterable:
        """Return self and optionally record shuffle parameters."""
        if self.on_shuffle is not None:
            self.on_shuffle(int(seed), int(buffer_size))
        return self

    def state_dict(self) -> dict[str, Any]:
        """Return iterator state."""
        return {"index": int(self.index)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore iterator state."""
        self.index = int(state.get("index", 0))

    def __iter__(self) -> FakeHFIterator:
        return FakeHFIterator(self)


class FakeHFIterator:
    """Iterator companion for ``FakeHFIterable``."""

    def __init__(self, ds: FakeHFIterable) -> None:
        """Initialize iterator from dataset state."""
        self._ds = ds
        self._i = int(ds.index)

    def __iter__(self) -> FakeHFIterator:
        return self

    def __next__(self) -> dict[str, Any]:
        if self._i >= len(self._ds.items):
            raise StopIteration
        item = self._ds.items[self._i]
        self._i += 1
        self._ds.index = self._i
        return item


@dataclass
class FakeHFStateIterable:
    """State-aware fake iterable with optional transient failure injection."""

    items: list[dict[str, Any]]
    index: int = 0
    fail_at: int | None = None
    record: dict[str, Any] | None = None

    def select_columns(self, _columns: list[str]) -> FakeHFStateIterable:
        """Return self (columns not used in tests)."""
        return self

    def shuffle(self, *, seed: int, buffer_size: int) -> FakeHFStateIterable:
        """Return self (shuffle not used in these tests)."""
        _ = (seed, buffer_size)
        return self

    def state_dict(self) -> dict[str, Any]:
        """Return iterator state."""
        return {"index": int(self.index)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state and capture load calls when requested."""
        self.index = int(state["index"])
        if self.record is not None:
            self.record["load_calls"] = self.record.get("load_calls", 0) + 1
            self.record["last_loaded"] = dict(state)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return FakeHFStateIterator(self)


class FakeHFStateIterator:
    """Iterator that can fail once at a configured index."""

    def __init__(self, ds: FakeHFStateIterable) -> None:
        """Initialize iterator from dataset state."""
        self._ds = ds
        self._i = int(ds.index)

    def __iter__(self) -> FakeHFStateIterator:
        return self

    def __next__(self) -> dict[str, Any]:
        if self._ds.fail_at is not None and self._i == self._ds.fail_at:
            rec = self._ds.record
            if rec is None or not rec.get("fail_consumed", False):
                if rec is not None:
                    rec["fail_consumed"] = True
                raise RuntimeError("transient failure")
        if self._i >= len(self._ds.items):
            raise StopIteration
        item = self._ds.items[self._i]
        self._i += 1
        self._ds.index = self._i
        return item
