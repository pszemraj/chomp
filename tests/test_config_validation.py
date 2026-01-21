"""Config validation guards for Megalodon-specific constraints."""

from __future__ import annotations

from dataclasses import replace

import pytest

from chomp.config import (
    Config,
    DataConfig,
    ModelConfig,
    TokenizerConfig,
    TrainConfig,
    validate_config,
)


def _base_cfg() -> Config:
    return Config(
        model=ModelConfig(backend="megalodon", model_dim=128, num_heads=8, chunk_size=16),
        data=DataConfig(
            backend="local_text",
            local_text="config validation text\n",
            tokenizer=TokenizerConfig(kind="byte", byte_offset=0, add_bos=False, add_eos=False),
        ),
        train=TrainConfig(steps=1, batch_size=2, seq_len=16, grad_accum=1, allow_cpu=True),
    )


def test_chunk_size_must_not_exceed_seq_len():
    cfg = _base_cfg()
    bad = replace(cfg, model=replace(cfg.model, chunk_size=32))
    with pytest.raises(ValueError, match="chunk_size"):
        validate_config(bad)


def test_chunk_size_must_divide_seq_len():
    cfg = _base_cfg()
    bad = replace(cfg, model=replace(cfg.model, chunk_size=10))
    with pytest.raises(ValueError, match="divisible"):
        validate_config(bad)


def test_model_dim_divisible_by_num_heads():
    cfg = _base_cfg()
    bad = replace(cfg, model=replace(cfg.model, model_dim=130, num_heads=8))
    with pytest.raises(ValueError, match="model_dim"):
        validate_config(bad)


def test_bin_packing_buffer_requires_min_docs():
    cfg = _base_cfg()
    bad_data = replace(cfg.data, packing_mode="bin", packing_buffer_docs=1)
    bad = replace(cfg, data=bad_data)
    with pytest.raises(ValueError, match="packing_buffer_docs"):
        validate_config(bad)


def test_invalid_packing_mode():
    cfg = _base_cfg()
    bad_data = replace(cfg.data, packing_mode="unknown")
    bad = replace(cfg, data=bad_data)
    with pytest.raises(ValueError, match="packing_mode"):
        validate_config(bad)
