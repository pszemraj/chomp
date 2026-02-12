"""Evaluation tests consolidated by module."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from chomp.config import (
    CheckpointConfig,
    Config,
    DataConfig,
    DebugConfig,
    LoggingConfig,
    ModelConfig,
    OptimConfig,
    TokenizerConfig,
    TrainConfig,
)
from chomp.data import build_tokenizer, load_or_create_eval_texts
from chomp.train import run
from tests.helpers.hf_fakes import FakeHFIterable


def test_eval_logging_writes_metrics(tmp_path: Path) -> None:
    """Eval should write eval_loss to metrics file."""
    run_dir = tmp_path / "run"
    cfg = Config(
        model=ModelConfig(backend="dummy", vocab_size=256, d_model=32, dropout=0.0),
        data=DataConfig(
            backend="local_text",
            repeat=True,
            local_text="abcdefghijklmnopqrstuvwxyz" * 4,
            max_eval_samples=3,
            tokenizer=TokenizerConfig(kind="byte", byte_offset=0, add_bos=False, add_eos=False),
        ),
        train=TrainConfig(
            seed=0,
            steps=2,
            batch_size=1,
            seq_len=8,
            grad_accum=1,
            jit=False,
            deterministic=True,
            allow_cpu=True,
            log_every=1000,
            eval_every=1,
        ),
        optim=OptimConfig(lr=1e-3, weight_decay=0.0, grad_clip_norm=0.0, warmup_steps=0),
        checkpoint=CheckpointConfig(enabled=False),
        debug=DebugConfig(nan_check=True, check_device_every=0),
        logging=LoggingConfig(project="chomp", run_dir=str(run_dir)),
    )

    run(cfg, config_path=None, resume="none")

    metrics_path = run_dir / cfg.logging.metrics_file
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert any(row.get("eval_loss") not in (None, "") for row in rows)
    assert any("step" in row for row in rows)
    for row in rows:
        for key in (
            "eval_tokens",
            "wall_time_s",
            "packing_tokens",
            "packing_capacity",
            "device_memory_gb",
        ):
            assert key not in row


def _base_cfg() -> Config:
    """Create a base config for eval text selection tests."""
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


def test_eval_split_selection_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Eval text loading should prefer eval split, then fall back to train as configured."""
    cases: list[
        tuple[str | None, dict[str, list[dict[str, Any]]], dict[str, bool], list[str], list[str]]
    ] = [
        (
            "validation",
            {
                "validation": [{"text": "val-a"}, {"text": "val-b"}],
                "train": [{"text": "train-a"}, {"text": "train-b"}],
            },
            {"validation": False},
            ["validation"],
            ["val-a", "val-b"],
        ),
        (
            "validation",
            {"train": [{"text": "train-a"}, {"text": "train-b"}]},
            {"validation": True},
            ["validation", "train"],
            ["train-a", "train-b"],
        ),
        (
            None,
            {"train": [{"text": "train-a"}, {"text": "train-b"}]},
            {},
            ["train"],
            ["train-a", "train-b"],
        ),
    ]

    for hf_eval_split, datasets, missing, expected_splits, expected_texts in cases:
        requested_splits: list[str] = []

        def _load_dataset(
            dataset: str,
            *,
            name: str,
            split: str,
            streaming: bool,
            _requested_splits: list[str] = requested_splits,
            _missing: dict[str, bool] = missing,
            _datasets: dict[str, list[dict[str, Any]]] = datasets,
        ) -> FakeHFIterable:
            _ = (dataset, name, streaming)
            _requested_splits.append(split)
            if _missing.get(split, False):
                raise FileNotFoundError(f"missing split: {split}")
            if split not in _datasets:
                raise ValueError(f"unknown split: {split}")
            return FakeHFIterable(items=_datasets[split])

        import datasets as hf_datasets

        monkeypatch.setattr(hf_datasets, "load_dataset", _load_dataset)

        cfg = _base_cfg()
        cfg = replace(cfg, data=replace(cfg.data, hf_eval_split=hf_eval_split))
        tok = build_tokenizer(cfg)
        tokens = load_or_create_eval_texts(cfg, tokenizer=tok)

        assert tokens == [tok.encode(text) for text in expected_texts]
        assert requested_splits == expected_splits


def test_eval_empty_when_disabled() -> None:
    """Eval should return empty list when max_eval_samples=0."""
    cfg = _base_cfg()
    cfg = replace(cfg, data=replace(cfg.data, max_eval_samples=0))
    tok = build_tokenizer(cfg)
    assert load_or_create_eval_texts(cfg, tokenizer=tok) == []


def test_eval_train_fallback_uses_train_seed_when_data_seed_is_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Train fallback should shuffle with train.seed when data.seed=0."""
    seen: dict[str, int] = {}
    train_items = [{"text": "train-a"}, {"text": "train-b"}]

    def _capture_shuffle(seed: int, buffer_size: int) -> None:
        seen["seed"] = int(seed)
        seen["buffer_size"] = int(buffer_size)

    def _load_dataset(dataset: str, *, name: str, split: str, streaming: bool) -> FakeHFIterable:
        _ = (dataset, name, streaming)
        if split == "train":
            return FakeHFIterable(items=train_items, on_shuffle=_capture_shuffle)
        raise FileNotFoundError("no validation split")

    import datasets as hf_datasets

    monkeypatch.setattr(hf_datasets, "load_dataset", _load_dataset)

    cfg = _base_cfg()
    cfg = replace(
        cfg,
        data=replace(cfg.data, hf_eval_split=None, shuffle=True, seed=0),
        train=replace(cfg.train, seed=69),
    )
    tok = build_tokenizer(cfg)
    tokens = load_or_create_eval_texts(cfg, tokenizer=tok)

    assert tokens == [tok.encode("train-a"), tok.encode("train-b")]
    assert seen["seed"] == 69
    assert seen["buffer_size"] == cfg.data.shuffle_buffer_size
