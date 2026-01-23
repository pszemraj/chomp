"""HF streaming state roundtrip should resume deterministically."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest

from chomp.data.hf import HFStreamingTextStream, HFStreamSpec


@dataclass
class _FakeHFIterable:
    """Mock HF iterable dataset with optional failure injection."""

    items: list[dict[str, Any]]
    index: int = 0
    fail_at: int | None = None
    record: dict[str, Any] | None = None

    def select_columns(self, _columns: list[str]) -> _FakeHFIterable:
        """Return self (columns not used in tests).

        :param list[str] _columns: Column names to select.
        :return _FakeHFIterable: Self for chaining.
        """
        return self

    def shuffle(self, *, seed: int, buffer_size: int) -> _FakeHFIterable:
        """Return self (shuffle not used in tests).

        :param int seed: Shuffle seed.
        :param int buffer_size: Shuffle buffer size.
        :return _FakeHFIterable: Self for chaining.
        """
        _ = (seed, buffer_size)
        return self

    def state_dict(self) -> dict[str, Any]:
        """Return iterator state.

        :return dict[str, Any]: State dictionary.
        """
        return {"index": int(self.index)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore iterator state and record load calls."""
        self.index = int(state["index"])
        if self.record is not None:
            self.record["load_calls"] = self.record.get("load_calls", 0) + 1
            self.record["last_loaded"] = dict(state)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return _FakeHFIterator(self)


class _FakeHFIterator:
    """Mock HF iterator with optional failure injection."""

    def __init__(self, ds: _FakeHFIterable) -> None:
        """Initialize iterator from dataset."""
        self._ds = ds
        self._i = int(ds.index)

    def __iter__(self) -> _FakeHFIterator:
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


def test_hf_state_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    """HF stream should resume to same position after state roundtrip."""
    items = [{"text": "alpha"}, {"text": "bravo"}, {"text": "charlie"}]

    def _load_dataset(dataset: str, *, name: str, split: str, streaming: bool) -> _FakeHFIterable:
        """Mock load_dataset returning fake iterable.

        :param str dataset: Dataset name.
        :param str name: Config name.
        :param str split: Split name.
        :param bool streaming: Streaming flag.
        :return _FakeHFIterable: Fake dataset iterable.
        """
        _ = (dataset, name, split, streaming)
        return _FakeHFIterable(items=items)

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", _load_dataset)

    spec = HFStreamSpec(
        dataset="dummy",
        name="dummy",
        split="train",
        text_key="text",
        shuffle=False,
        shuffle_buffer_size=8,
        seed=0,
        repeat=False,
        max_retries=0,
        retry_delay_sec=0.0,
        state_update_interval=2,
    )

    stream = HFStreamingTextStream(spec)
    _ = next(stream)
    _ = next(stream)
    state = stream.get_state()
    expected = next(stream)

    resumed = HFStreamingTextStream(spec)
    resumed.set_state(state)
    assert next(resumed) == expected


def test_hf_retry_rebuild_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    """HF stream should recover from transient failure via state restore."""
    items = [{"text": "alpha"}, {"text": "bravo"}, {"text": "charlie"}]
    record: dict[str, Any] = {"builds": 0, "fail_consumed": False}

    def _load_dataset(dataset: str, *, name: str, split: str, streaming: bool) -> _FakeHFIterable:
        """Mock load_dataset with failure injection.

        :param str dataset: Dataset name.
        :param str name: Config name.
        :param str split: Split name.
        :param bool streaming: Streaming flag.
        :return _FakeHFIterable: Fake dataset iterable.
        """
        _ = (dataset, name, split, streaming)
        record["builds"] += 1
        return _FakeHFIterable(items=items, fail_at=1, record=record)

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", _load_dataset)

    spec = HFStreamSpec(
        dataset="dummy",
        name="dummy",
        split="train",
        text_key="text",
        shuffle=False,
        shuffle_buffer_size=8,
        seed=0,
        repeat=False,
        max_retries=1,
        retry_delay_sec=0.0,
        state_update_interval=1,
    )

    stream = HFStreamingTextStream(spec)
    assert next(stream) == "alpha"
    assert next(stream) == "bravo"

    assert record["builds"] >= 2
    assert record.get("load_calls", 0) >= 1
    assert record.get("last_loaded") == {"index": 1}
