"""Dry-run sanity check compiles a single step and exits early."""

from __future__ import annotations

from pathlib import Path

from chomp.config import (
    CheckpointConfig,
    Config,
    DataConfig,
    LoggingConfig,
    ModelConfig,
    TokenizerConfig,
    TrainConfig,
)
from chomp.train import run


def test_dry_run_compiles_single_step(tmp_path: Path) -> None:
    run_dir = tmp_path / "dry_run"
    cfg = Config(
        model=ModelConfig(backend="dummy", vocab_size=128, d_model=32, dropout=0.0),
        data=DataConfig(
            backend="local_text",
            repeat=True,
            local_text="dry run text\n" * 8,
            max_eval_samples=4,
            tokenizer=TokenizerConfig(kind="byte", byte_offset=0, add_bos=False, add_eos=False),
        ),
        train=TrainConfig(
            steps=5,
            batch_size=1,
            seq_len=8,
            grad_accum=1,
            jit=True,
            deterministic=True,
            allow_cpu=True,
            log_every=1000,
            eval_every=0,
        ),
        checkpoint=CheckpointConfig(enabled=False),
        logging=LoggingConfig(
            project="chomp", run_dir=str(run_dir), metrics_file="metrics.jsonl", wandb_enabled=True
        ),
    )

    run(cfg, config_path=None, resume="none", dry_run=True)

    assert (run_dir / "config_resolved.json").exists()
    assert not (run_dir / cfg.logging.metrics_file).exists()
