"""Resume should reject config mismatches like changed seq_len."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

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
from chomp.train import run


def test_resume_rejects_seq_len_mismatch(tmp_path: Path) -> None:
    """Resuming with different seq_len should raise RuntimeError."""
    base = Config(
        model=ModelConfig(backend="dummy", vocab_size=256, d_model=32, dropout=0.0),
        data=DataConfig(
            backend="local_text",
            repeat=True,
            local_text="Deterministic local text for resume mismatch test.\n",
            tokenizer=TokenizerConfig(kind="byte", byte_offset=0, add_bos=False, add_eos=False),
        ),
        train=TrainConfig(
            seed=0,
            steps=0,
            batch_size=2,
            seq_len=16,
            grad_accum=2,
            jit=False,
            deterministic=True,
            allow_cpu=True,
            log_every=1000,
        ),
        optim=OptimConfig(lr=1e-3, weight_decay=0.0, grad_clip_norm=0.0, warmup_steps=0),
        checkpoint=CheckpointConfig(enabled=True, save_every=1, max_to_keep=2, async_save=False),
        debug=DebugConfig(nan_check=True, check_device_every=0),
        logging=LoggingConfig(
            project="chomp", run_dir=None, metrics_file="metrics.jsonl", level="INFO"
        ),
    )

    run_dir = tmp_path / "run"
    cfg_a = replace(
        base,
        logging=replace(base.logging, run_dir=str(run_dir)),
        train=replace(base.train, steps=2),
    )
    run(cfg_a, config_path=None, resume="none")

    cfg_b = replace(
        base,
        logging=replace(base.logging, run_dir=str(run_dir)),
        train=replace(base.train, steps=3, seq_len=32),
    )
    with pytest.raises(RuntimeError, match="Resume config mismatch"):
        run(cfg_b, config_path=None, resume="latest")
