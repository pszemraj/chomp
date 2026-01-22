from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from chomp.config import (
    CheckpointConfig,
    Config,
    DataConfig,
    LoggingConfig,
    ModelConfig,
    OptimConfig,
    TokenizerConfig,
    TrainConfig,
)
from chomp.train import run


def test_tokenizer_snapshot_saved(tmp_path: Path):
    base = Config(
        model=ModelConfig(backend="dummy", vocab_size=256, d_model=32, dropout=0.0),
        data=DataConfig(
            backend="local_text",
            repeat=True,
            local_text="Tokenizer snapshot test.\n",
            tokenizer=TokenizerConfig(kind="byte", byte_offset=0, add_bos=False, add_eos=False),
        ),
        train=TrainConfig(
            seed=0,
            steps=1,
            batch_size=1,
            seq_len=16,
            grad_accum=1,
            jit=False,
            deterministic=True,
            allow_cpu=True,
            log_every=1000,
        ),
        optim=OptimConfig(warmup_steps=0),
        checkpoint=CheckpointConfig(enabled=False),
        logging=LoggingConfig(
            project="chomp", run_dir=None, metrics_file="metrics.jsonl", level="INFO"
        ),
    )

    run_dir = tmp_path / "run"
    cfg = replace(base, logging=replace(base.logging, run_dir=str(run_dir)))
    run(cfg, config_path=None, resume="none")

    tok_file = run_dir / "tokenizer" / "tokenizer.json"
    assert tok_file.exists()

    data = json.loads(tok_file.read_text(encoding="utf-8"))
    assert data["kind"] == "byte"
