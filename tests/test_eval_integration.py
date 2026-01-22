"""Integration test for eval caching + eval logging."""

from __future__ import annotations

import json
from pathlib import Path

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
from chomp.train import run


def test_eval_logging_writes_metrics(tmp_path: Path):
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
        optim=OptimConfig(
            lr=1e-3, weight_decay=0.0, grad_clip_norm=0.0, warmup_steps=0, total_steps=2
        ),
        checkpoint=CheckpointConfig(enabled=False),
        debug=DebugConfig(nan_check=True, check_device_every=0),
        logging=LoggingConfig(project="chomp", run_dir=str(run_dir)),
    )

    run(cfg, config_path=None, resume="none")

    eval_cache = run_dir / "eval_texts.json"
    payload = json.loads(eval_cache.read_text(encoding="utf-8"))
    assert payload["max_eval_samples"] == 3
    assert len(payload["texts"]) == 3
    assert len(payload.get("tokens", [])) == 3

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
