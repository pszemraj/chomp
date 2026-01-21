"""HF streaming state roundtrip should resume deterministically."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

from chomp.data.hf import HFStreamSpec, HFStreamingTextStream


@dataclass
class _FakeHFIterable:
    items: list[dict[str, Any]]
    index: int = 0

    def select_columns(self, _columns: list[str]):
        return self

    def shuffle(self, *, seed: int, buffer_size: int):
        _ = (seed, buffer_size)
        return self

    def state_dict(self) -> dict[str, Any]:
        return {"index": int(self.index)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.index = int(state["index"])

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return _FakeHFIterator(self)


class _FakeHFIterator:
    def __init__(self, ds: _FakeHFIterable):
        self._ds = ds
        self._i = int(ds.index)

    def __iter__(self) -> "_FakeHFIterator":
        return self

    def __next__(self) -> dict[str, Any]:
        if self._i >= len(self._ds.items):
            raise StopIteration
        item = self._ds.items[self._i]
        self._i += 1
        self._ds.index = self._i
        return item


def test_hf_state_roundtrip(monkeypatch):
    items = [{"text": "alpha"}, {"text": "bravo"}, {"text": "charlie"}]

    def _load_dataset(dataset: str, *, name: str, split: str, streaming: bool):
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
