"""Training should exit cleanly when repeat=False data is exhausted."""

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


def test_train_repeat_false_exits_cleanly(tmp_path: Path):
    cfg = Config(
        model=ModelConfig(backend="dummy", vocab_size=256, d_model=32, dropout=0.0),
        data=DataConfig(
            backend="local_text",
            repeat=False,
            local_text="short local text to exhaust\n",
            max_eval_samples=0,
            tokenizer=TokenizerConfig(kind="byte", byte_offset=0, add_bos=False, add_eos=False),
        ),
        train=TrainConfig(
            seed=0,
            steps=5,
            batch_size=1,
            seq_len=8,
            grad_accum=1,
            jit=False,
            deterministic=True,
            allow_cpu=True,
            log_every=1,
            eval_every=0,
        ),
        optim=OptimConfig(
            lr=1e-3, weight_decay=0.0, grad_clip_norm=0.0, warmup_steps=0, total_steps=5
        ),
        checkpoint=CheckpointConfig(enabled=False),
        debug=DebugConfig(nan_check=True, check_device_every=0),
        logging=LoggingConfig(project="chomp", run_dir=str(tmp_path / "run")),
    )

    run(cfg, config_path=None, resume="none")

    metrics_path = Path(cfg.logging.run_dir) / cfg.logging.metrics_file
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert any(row.get("data_exhausted") for row in rows)
