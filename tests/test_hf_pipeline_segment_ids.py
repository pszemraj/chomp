"""HF streaming pipeline should emit segment IDs and boundary-masked labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from chomp.config import Config, DataConfig, ModelConfig, TokenizerConfig, TrainConfig
from chomp.data.pipeline import build_train_iterator


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

    def __iter__(self) -> _FakeHFIterable:
        return self

    def __next__(self) -> dict[str, Any]:
        if self.index >= len(self.items):
            raise StopIteration
        item = self.items[self.index]
        self.index += 1
        return item


def test_hf_pipeline_segment_ids_and_label_mask(monkeypatch):
    items = [
        {"text": "hi"},
        {"text": "ok"},
        {"text": "yo"},
        {"text": "sup"},
    ]

    def _load_dataset(dataset: str, *, name: str, split: str, streaming: bool):
        _ = (dataset, name, split, streaming)
        return _FakeHFIterable(items=items)

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", _load_dataset)

    cfg = Config(
        model=ModelConfig(
            backend="dummy", vocab_size=256, d_model=32, dropout=0.0, segment_masking=True
        ),
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
    masked_labels = batch.labels[0, 0][:-1][boundary]
    assert np.all(masked_labels == -100)
