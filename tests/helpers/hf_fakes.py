"""Reusable fake Hugging Face streaming iterables for tests.

This intentionally small fake covers ordinary unit tests. Resume-safe document
shuffle coverage uses a real local ``datasets.IterableDataset`` so the source
state implementation cannot silently drift away from this stand-in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FakeHFIterable:
    """In-memory stand-in for a streaming IterableDataset.

    Optional ``fail_at``/``record`` hooks inject a single transient failure at an item index
    (``record`` tracks load_state_dict calls and failure consumption).
    """

    items: list[dict[str, Any]]
    index: int = 0
    fail_at: int | None = None
    record: dict[str, Any] | None = None

    def select_columns(self, _columns: list[str]) -> FakeHFIterable:
        """Return self (columns not used in tests)."""
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

    def __iter__(self) -> FakeHFIterator:
        return FakeHFIterator(self)


class FakeHFIterator:
    """Iterator companion for ``FakeHFIterable``; can fail once at ``fail_at``."""

    def __init__(self, ds: FakeHFIterable) -> None:
        """Initialize iterator from dataset state."""
        self._ds = ds
        self._i = int(ds.index)

    def __iter__(self) -> FakeHFIterator:
        return self

    def __next__(self) -> dict[str, Any]:
        ds = self._ds
        if ds.fail_at is not None and self._i == ds.fail_at:
            rec = ds.record
            if rec is None or not rec.get("fail_consumed", False):
                if rec is not None:
                    rec["fail_consumed"] = True
                raise RuntimeError("transient failure")
        if self._i >= len(ds.items):
            raise StopIteration
        item = ds.items[self._i]
        self._i += 1
        ds.index = self._i
        return item
