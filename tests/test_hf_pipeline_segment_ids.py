"""HF streaming pipeline should emit segment IDs and boundary-masked labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from chomp.config import Config, DataConfig, ModelConfig, TokenizerConfig, TrainConfig
from chomp.data.pipeline import build_train_iterator


@dataclass
class _FakeHFIterable:
    """Mock HF iterable dataset for testing."""

    items: list[dict[str, Any]]
    index: int = 0

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
        """Restore iterator state."""
        self.index = int(state["index"])

    def __iter__(self) -> _FakeHFIterable:
        return self

    def __next__(self) -> dict[str, Any]:
        if self.index >= len(self.items):
            raise StopIteration
        item = self.items[self.index]
        self.index += 1
        return item


def test_hf_pipeline_segment_ids_and_label_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HF pipeline should emit segment IDs and mask labels at boundaries."""
    items = [
        {"text": "hi"},
        {"text": "ok"},
        {"text": "yo"},
        {"text": "sup"},
    ]

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

    cfg = Config(
        model=ModelConfig(backend="dummy", vocab_size=256, d_model=32, dropout=0.0),
        data=DataConfig(
            backend="hf",
            hf_dataset="dummy",
            hf_name="dummy",
            hf_split="train",
            text_key="text",
            shuffle=False,
            shuffle_buffer_size=8,
            seed=0,
            repeat=False,
            mask_boundary_loss=True,
            train_on_eos=True,
            tokenizer=TokenizerConfig(kind="byte", byte_offset=0, add_bos=True, add_eos=True),
        ),
        train=TrainConfig(
            steps=1,
            batch_size=1,
            seq_len=8,
            grad_accum=1,
            jit=False,
            deterministic=True,
            allow_cpu=True,
        ),
    )

    it = build_train_iterator(cfg)
    batch = next(it)

    segs = batch.segment_ids[0, 0]
    unique = np.unique(segs)
    assert unique.size >= 2
    assert np.all(unique > 0)

    boundary = segs[1:] != segs[:-1]
    assert boundary.any()
    masked_labels = batch.labels[0, 0][1:][boundary]
    assert np.all(masked_labels == -100)
