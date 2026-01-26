"""Config validation guards for Megalodon-specific constraints."""

from __future__ import annotations

from dataclasses import replace

import pytest

from chomp.config import (
    Config,
    DataConfig,
    ModelConfig,
    OptimConfig,
    TokenizerConfig,
    TrainConfig,
    validate_config,
)


def _base_cfg() -> Config:
    """Create a base config for validation tests.

    :return Config: Base configuration for validation tests.
    """
    return Config(
        model=ModelConfig(backend="megalodon", model_dim=128, num_heads=8, chunk_size=16),
        data=DataConfig(
            backend="local_text",
            local_text="config validation text\n",
            tokenizer=TokenizerConfig(kind="byte", byte_offset=0, add_bos=False, add_eos=False),
        ),
        train=TrainConfig(steps=1, batch_size=2, seq_len=16, grad_accum=1, allow_cpu=True),
        optim=OptimConfig(warmup_steps=0),
    )


def test_chunk_size_must_not_exceed_seq_len() -> None:
    """Chunk size exceeding seq_len must raise ValueError."""
    cfg = _base_cfg()
    bad = replace(cfg, model=replace(cfg.model, chunk_size=32))
    with pytest.raises(ValueError, match="chunk_size"):
        validate_config(bad)


def test_chunk_size_must_divide_seq_len() -> None:
    """Chunk size not dividing seq_len must raise ValueError."""
    cfg = _base_cfg()
    bad = replace(cfg, model=replace(cfg.model, chunk_size=10))
    with pytest.raises(ValueError, match="divisible"):
        validate_config(bad)


def test_model_dim_divisible_by_num_heads() -> None:
    """Model dim not divisible by num_heads must raise ValueError."""
    cfg = _base_cfg()
    bad = replace(cfg, model=replace(cfg.model, model_dim=130, num_heads=8))
    with pytest.raises(ValueError, match="model_dim"):
        validate_config(bad)


def test_bin_packing_buffer_requires_min_docs() -> None:
    """Bin packing with buffer_docs < 2 must raise ValueError."""
    cfg = _base_cfg()
    bad_data = replace(cfg.data, packing_mode="bin", packing_buffer_docs=1)
    bad = replace(cfg, data=bad_data)
    with pytest.raises(ValueError, match="packing_buffer_docs"):
        validate_config(bad)


def test_invalid_packing_mode() -> None:
    """Unknown packing_mode must raise ValueError."""
    cfg = _base_cfg()
    bad_data = replace(cfg.data, packing_mode="unknown")
    bad = replace(cfg, data=bad_data)
    with pytest.raises(ValueError, match="packing_mode"):
        validate_config(bad)


def test_max_eval_samples_must_be_non_negative() -> None:
    """Negative max_eval_samples must raise ValueError."""
    cfg = _base_cfg()
    bad_data = replace(cfg.data, max_eval_samples=-1)
    bad = replace(cfg, data=bad_data)
    with pytest.raises(ValueError, match="max_eval_samples"):
        validate_config(bad)


def test_eval_every_must_be_non_negative() -> None:
    """Negative eval_every must raise ValueError."""
    cfg = _base_cfg()
    bad_train = replace(cfg.train, eval_every=-1)
    bad = replace(cfg, train=bad_train)
    with pytest.raises(ValueError, match="eval_every"):
        validate_config(bad)


def test_generate_every_must_be_non_negative() -> None:
    """Negative generate_every must raise ValueError."""
    cfg = _base_cfg()
    bad_train = replace(cfg.train, generate_every=-1)
    bad = replace(cfg, train=bad_train)
    with pytest.raises(ValueError, match="generate_every"):
        validate_config(bad)


def test_generate_input_len_must_be_valid() -> None:
    """generate_input_len must be positive and <= seq_len."""
    cfg = _base_cfg()
    bad_small = replace(cfg.train, generate_input_len=0)
    with pytest.raises(ValueError, match="generate_input_len"):
        validate_config(replace(cfg, train=bad_small))
    bad_large = replace(cfg.train, generate_input_len=1024)
    with pytest.raises(ValueError, match="generate_input_len"):
        validate_config(replace(cfg, train=bad_large))


def test_generate_top_p_must_be_in_range() -> None:
    """generate_top_p outside (0,1] must raise ValueError."""
    cfg = _base_cfg()
    bad_train = replace(cfg.train, generate_top_p=1.5)
    with pytest.raises(ValueError, match="generate_top_p"):
        validate_config(replace(cfg, train=bad_train))


def test_wandb_mode_must_be_valid() -> None:
    """Invalid wandb mode must raise ValueError."""
    cfg = _base_cfg()
    bad_logging = replace(cfg.logging, wandb=replace(cfg.logging.wandb, mode="bogus"))
    bad = replace(cfg, logging=bad_logging)
    with pytest.raises(ValueError, match="wandb.mode"):
        validate_config(bad)


def test_min_lr_ratio_must_be_in_range() -> None:
    """min_lr_ratio outside [0, 1] must raise ValueError."""
    cfg = _base_cfg()
    bad_optim = replace(cfg.optim, min_lr_ratio=1.5)
    bad = replace(cfg, optim=bad_optim)
    with pytest.raises(ValueError, match="min_lr_ratio"):
        validate_config(bad)


def test_muon_lr_scale_must_be_positive() -> None:
    """muon_lr_scale <= 0 must raise ValueError."""
    cfg = _base_cfg()
    bad_optim = replace(cfg.optim, muon_lr_scale=0.0)
    bad = replace(cfg, optim=bad_optim)
    with pytest.raises(ValueError, match="muon_lr_scale"):
        validate_config(bad)


def test_muon_weight_decay_mult_must_be_non_negative() -> None:
    """muon_weight_decay_mult < 0 must raise ValueError."""
    cfg = _base_cfg()
    bad_optim = replace(cfg.optim, muon_weight_decay_mult=-1.0)
    bad = replace(cfg, optim=bad_optim)
    with pytest.raises(ValueError, match="muon_weight_decay_mult"):
        validate_config(bad)


def test_optim_name_must_be_valid() -> None:
    """Unknown optim.name must raise ValueError."""
    cfg = _base_cfg()
    bad_optim = replace(cfg.optim, name="sgd")
    bad = replace(cfg, optim=bad_optim)
    with pytest.raises(ValueError, match="optim.name"):
        validate_config(bad)


def test_muon_momentum_must_be_in_range() -> None:
    """muon_momentum outside (0, 1) must raise ValueError."""
    cfg = _base_cfg()
    bad_optim = replace(cfg.optim, muon_momentum=1.5)
    bad = replace(cfg, optim=bad_optim)
    with pytest.raises(ValueError, match="muon_momentum"):
        validate_config(bad)


def test_muon_ns_steps_must_be_positive() -> None:
    """muon_ns_steps <= 0 must raise ValueError."""
    cfg = _base_cfg()
    bad_optim = replace(cfg.optim, muon_ns_steps=0)
    bad = replace(cfg, optim=bad_optim)
    with pytest.raises(ValueError, match="muon_ns_steps"):
        validate_config(bad)


def test_muon_consistent_rms_must_be_non_negative() -> None:
    """muon_consistent_rms < 0 must raise ValueError."""
    cfg = _base_cfg()
    bad_optim = replace(cfg.optim, muon_consistent_rms=-0.1)
    bad = replace(cfg, optim=bad_optim)
    with pytest.raises(ValueError, match="muon_consistent_rms"):
        validate_config(bad)


def test_log_file_must_be_non_empty() -> None:
    """Empty or whitespace log_file must raise ValueError."""
    cfg = _base_cfg()
    bad_logging = replace(cfg.logging, log_file=" ")
    bad = replace(cfg, logging=bad_logging)
    with pytest.raises(ValueError, match="log_file"):
        validate_config(bad)


def test_default_init_mode_is_he() -> None:
    """Default init_mode should be 'he'."""
    cfg = Config()
    assert cfg.model.init_mode == "he"


def test_pad_token_id_equal_to_eos_warns() -> None:
    """pad_token_id equal to eos_token_id should warn but still validate."""
    cfg = _base_cfg()
    bad_model = replace(cfg.model, pad_token_id=2, eos_token_id=2)
    bad = replace(cfg, model=bad_model)
    with pytest.warns(UserWarning, match="pad_token_id equals model.eos_token_id"):
        validate_config(bad)


def test_warmup_steps_must_not_exceed_train_steps() -> None:
    """warmup_steps exceeding train.steps must raise ValueError."""
    cfg = _base_cfg()
    bad_optim = replace(cfg.optim, warmup_steps=10)
    bad = replace(cfg, optim=bad_optim, train=replace(cfg.train, steps=5))
    with pytest.raises(ValueError, match="warmup_steps"):
        validate_config(bad)
