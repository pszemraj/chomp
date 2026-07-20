"""Shared config builders for integration-style tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from chomp.config import (
    Config,
    DataConfig,
    ModelConfig,
    OptimConfig,
    TokenizerConfig,
    TrainConfig,
)

DEFAULT_SMALL_RUN_TEXT = "hello from chomp"


def make_tiny_megalodon_model(**overrides: Any) -> ModelConfig:
    """Build the shared smoke-sized Megalodon model configuration.

    :param overrides: ModelConfig fields specific to the test.
    :return ModelConfig: Tiny valid Megalodon model configuration.
    """
    fields: dict[str, Any] = {
        "backend": "megalodon",
        "vocab_size": 128,
        "model_dim": 32,
        "num_layers": 1,
        "num_heads": 1,
        "z_dim": 16,
        "value_dim": 32,
        "ffn_hidden_dim": 64,
        "cema_ndim": 4,
        "chunk_size": 8,
        "norm_num_groups": 4,
        "dropout": 0.0,
    }
    fields.update(overrides)
    return ModelConfig(**fields)


def make_pipeline_cfg(
    *,
    batch_size: int = 1,
    seq_len: int = 8,
    vocab_size: int = 512,
    **data_kwargs: Any,
) -> Config:
    """Tiny data-pipeline Config: dummy model, byte(offset=4)+BOS/EOS tokenizer.

    ``data_kwargs`` override/extend the local-text DataConfig defaults; pass
    ``backend="hf"`` plus hf_* fields for HF-backed tests. Call sites should
    state only the knobs the test is about.

    :param int batch_size: Rows per micro-batch.
    :param int seq_len: Packed sequence length.
    :param int vocab_size: Dummy model vocab size.
    :param data_kwargs: DataConfig field overrides.
    :return Config: Validated-shape test configuration.
    """
    data: dict[str, Any] = {
        "backend": "local_text",
        "repeat": True,
        "local_text": "hi",
        # Tiny tests opt into packed-window shuffling only when they exercise it.
        "window_shuffle_tokens": 0,
        "tokenizer": TokenizerConfig(kind="byte", byte_offset=4, add_bos=True, add_eos=True),
    }
    data.update(data_kwargs)
    if data.get("backend") == "hf" and "hf_revision" not in data:
        data["hf_revision"] = "0" * 40
    return Config(
        model=ModelConfig(backend="dummy", vocab_size=vocab_size, d_model=32, dropout=0.0),
        data=DataConfig(**data),
        train=TrainConfig(
            steps=1,
            batch_size=batch_size,
            seq_len=seq_len,
            grad_accum=1,
            jit=False,
            deterministic=True,
            allow_cpu=True,
        ),
        optim=OptimConfig(warmup_steps=0),
    )


def make_small_run_cfg(
    tmp_path: Path,
    *,
    run_subdir: str = "run",
    local_text: str = DEFAULT_SMALL_RUN_TEXT,
    decay_steps: int | None = None,
) -> Config:
    """Build a tiny local-text config for fast train/checkpoint tests.

    :param Path tmp_path: Temporary directory provided by pytest.
    :param str run_subdir: Name of the run subdirectory under tmp_path.
    :param str local_text: Local text corpus for the dataset backend.
    :param int | None decay_steps: Optional optimizer decay horizon override.
    :return Config: Smoke-sized training configuration.
    """
    cfg = make_pipeline_cfg(
        seq_len=16,
        vocab_size=256,
        packing_mode="sequential",
        packing_buffer_docs=4,
        grain_prefetch=0,
        local_text=local_text,
        tokenizer=TokenizerConfig(kind="byte", byte_offset=0, add_bos=False, add_eos=False),
    )

    optim = replace(cfg.optim, warmup_steps=0)
    if decay_steps is not None:
        optim = replace(optim, decay_steps=int(decay_steps))

    cfg = replace(
        cfg,
        train=replace(
            cfg.train,
            steps=2,
            batch_size=1,
            seq_len=16,
            grad_accum=1,
            jit=False,
            deterministic=True,
            allow_cpu=True,
            log_every=1,
            eval_every=0,
            generate_every=0,
        ),
        checkpoint=replace(
            cfg.checkpoint,
            enabled=True,
            save_every=1,
            max_to_keep=2,
            async_save=False,
        ),
        optim=optim,
        logging=replace(
            cfg.logging,
            run_dir=str(tmp_path / run_subdir),
            console_use_rich=False,
        ),
        debug=replace(
            cfg.debug,
            nan_check=True,
        ),
    )
    return cfg
