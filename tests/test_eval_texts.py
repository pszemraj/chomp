"""Eval text caching should prefer validation split and fall back to train."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from chomp.config import Config, DataConfig, ModelConfig, TokenizerConfig, TrainConfig
from chomp.data import load_or_create_eval_texts


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
        self.index = int(state.get("index", 0))

    def __iter__(self):
        return _FakeHFIterator(self)


class _FakeHFIterator:
    def __init__(self, ds: _FakeHFIterable):
        self._ds = ds
        self._i = int(ds.index)

    def __iter__(self) -> _FakeHFIterator:
        return self

    def __next__(self) -> dict[str, Any]:
        if self._i >= len(self._ds.items):
            raise StopIteration
        item = self._ds.items[self._i]
        self._i += 1
        self._ds.index = self._i
        return item


def _base_cfg() -> Config:
    return Config(
        model=ModelConfig(backend="dummy", vocab_size=256, d_model=16, dropout=0.0),
        data=DataConfig(
            backend="hf",
            hf_dataset="dummy",
            hf_name="dummy",
            hf_split="train",
            hf_eval_split="validation",
            text_key="text",
            shuffle=False,
            repeat=False,
            max_eval_samples=2,
            tokenizer=TokenizerConfig(kind="byte", byte_offset=0, add_bos=False, add_eos=False),
        ),
        train=TrainConfig(
            steps=1,
            batch_size=1,
            seq_len=8,
            grad_accum=1,
            jit=False,
            deterministic=True,
            allow_cpu=True,
            log_every=1000,
            eval_every=0,
        ),
    )


def test_eval_prefers_validation_split(monkeypatch, tmp_path: Path):
    val_items = [{"text": "val-a"}, {"text": "val-b"}]
    train_items = [{"text": "train-a"}, {"text": "train-b"}]

    def _load_dataset(dataset: str, *, name: str, split: str, streaming: bool):
        _ = (dataset, name, streaming)
        if split == "validation":
            return _FakeHFIterable(items=val_items)
        if split == "train":
            return _FakeHFIterable(items=train_items)
        raise ValueError("unknown split")

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", _load_dataset)

    cfg = _base_cfg()
    texts = load_or_create_eval_texts(cfg, run_dir=tmp_path, allow_existing=False)
    assert texts == ["val-a", "val-b"]


def test_eval_falls_back_to_train_split(monkeypatch, tmp_path: Path):
    train_items = [{"text": "train-a"}, {"text": "train-b"}]

    def _load_dataset(dataset: str, *, name: str, split: str, streaming: bool):
        _ = (dataset, name, streaming)
        if split == "validation":
            raise FileNotFoundError("no validation split")
        if split == "train":
            return _FakeHFIterable(items=train_items)
        raise ValueError("unknown split")

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", _load_dataset)

    cfg = _base_cfg()
    texts = load_or_create_eval_texts(cfg, run_dir=tmp_path, allow_existing=False)
    assert texts == ["train-a", "train-b"]


def test_eval_cache_required_for_resume(tmp_path: Path):
    cfg = _base_cfg()
    with pytest.raises(RuntimeError, match="Eval texts missing"):
        load_or_create_eval_texts(cfg, run_dir=tmp_path, allow_existing=True)
