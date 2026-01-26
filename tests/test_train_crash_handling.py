"""Crash handling tests for training runs."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax.numpy as jnp
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
    WandbConfig,
)
from chomp.train import run
from chomp.types import Batch


class DummyWandbRun:
    """Minimal W&B stub to capture finish calls and logs."""

    def __init__(self) -> None:
        self.finish_calls: list[int] = []
        self.logged: list[tuple[int | None, dict[str, Any]]] = []
        self.summary: dict[str, Any] = {}

    def log(self, row: dict[str, Any], *, step: int | None = None) -> None:
        self.logged.append((step, row))

    def finish(self, *, exit_code: int = 0) -> None:
        self.finish_calls.append(exit_code)


class DummyIter:
    """Single-batch iterator for crash tests."""

    def __init__(self) -> None:
        self._done = False

    def __iter__(self) -> DummyIter:
        return self

    def __next__(self) -> Batch:
        if self._done:
            raise StopIteration
        self._done = True
        zeros = jnp.zeros((1, 1, 8), dtype=jnp.int32)
        attn = jnp.ones((1, 1, 8), dtype=bool)
        return Batch(input_ids=zeros, labels=zeros, attention_mask=attn, segment_ids=zeros)

    def get_stats(self) -> dict[str, Any]:
        return {}

    def get_state(self) -> dict[str, Any]:
        return {}

    def set_state(self, state: dict[str, Any]) -> None:
        _ = state


def test_training_crash_marks_wandb_failed_and_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Crashes should write a metrics row and finish W&B with exit_code=1.

    :param Path tmp_path: Temporary directory for run output.
    :param pytest.MonkeyPatch monkeypatch: Pytest monkeypatch fixture.
    """
    run_dir = tmp_path / "run"
    dummy_wandb = DummyWandbRun()

    def boom_make_train_step(*args: Any, **kwargs: Any):
        def boom(state: Any, batch: Any):
            raise RuntimeError("kaboom")

        return boom

    monkeypatch.setattr("chomp.train.make_train_step", boom_make_train_step)
    monkeypatch.setattr("chomp.train.build_train_iterator", lambda *args, **kwargs: DummyIter())
    monkeypatch.setattr("chomp.train._maybe_init_wandb", lambda *args, **kwargs: dummy_wandb)

    cfg = Config(
        model=ModelConfig(backend="dummy", vocab_size=32, d_model=8, dropout=0.0),
        data=DataConfig(
            backend="local_text",
            local_text="boom",
            repeat=True,
            max_eval_samples=0,
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
            log_every=1,
            eval_every=0,
            generate_every=0,
        ),
        optim=OptimConfig(warmup_steps=0),
        checkpoint=CheckpointConfig(enabled=False),
        logging=LoggingConfig(
            run_dir=str(run_dir),
            wandb=replace(WandbConfig(), enabled=True),
        ),
        debug=DebugConfig(nan_check=False, check_device_every=0),
    )

    with pytest.raises(RuntimeError, match="kaboom"):
        run(cfg, config_path=None, resume="none", dry_run=False, max_steps=1)

    assert dummy_wandb.finish_calls == [1]

    metrics_path = run_dir / cfg.logging.metrics_file
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    assert any(row.get("crash") for row in rows)

    log_text = (run_dir / cfg.logging.log_file).read_text()
    assert "Training crashed" in log_text
