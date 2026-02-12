"""Config tests consolidated by module."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from chomp.config import (
    Config,
    DataConfig,
    ModelConfig,
    OptimConfig,
    TokenizerConfig,
    TrainConfig,
    load_config,
    validate_config,
)
from chomp.data.pipeline import build_tokenizer, resolve_tokenizer_config
from chomp.utils.checkpoints import load_config_for_checkpoint


def _base_cfg() -> Config:
    """Create a base config for validation tests."""
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


def test_model_and_train_validation_rejects_invalid_values() -> None:
    """Model/train validation should fail with actionable errors."""
    cases: list[tuple[Callable[[Config], Config], str]] = [
        (lambda cfg: replace(cfg, model=replace(cfg.model, chunk_size=32)), "chunk_size"),
        (lambda cfg: replace(cfg, model=replace(cfg.model, chunk_size=10)), "divisible"),
        (
            lambda cfg: replace(cfg, model=replace(cfg.model, model_dim=130, num_heads=8)),
            "model_dim",
        ),
        (lambda cfg: replace(cfg, train=replace(cfg.train, eval_every=-1)), "eval_every"),
        (lambda cfg: replace(cfg, train=replace(cfg.train, generate_every=-1)), "generate_every"),
        (
            lambda cfg: replace(cfg, train=replace(cfg.train, generate_input_len=0)),
            "generate_input_len",
        ),
        (
            lambda cfg: replace(cfg, train=replace(cfg.train, generate_input_len=1024)),
            "generate_input_len",
        ),
        (
            lambda cfg: replace(cfg, train=replace(cfg.train, generate_top_p=1.5)),
            "generate_top_p",
        ),
        (
            lambda cfg: replace(
                cfg,
                optim=replace(cfg.optim, warmup_steps=10),
                train=replace(cfg.train, steps=5),
            ),
            "warmup_steps",
        ),
    ]

    for mutate, match in cases:
        with pytest.raises(ValueError, match=match):
            validate_config(mutate(_base_cfg()))


def test_optim_validation_rejects_invalid_values() -> None:
    """Optimizer validation should fail for out-of-range settings."""
    cases: list[tuple[Callable[[Config], Config], str]] = [
        (lambda cfg: replace(cfg, optim=replace(cfg.optim, min_lr_ratio=1.5)), "min_lr_ratio"),
        (
            lambda cfg: replace(
                cfg, optim=replace(cfg.optim, muon=replace(cfg.optim.muon, lr_scale=0.0))
            ),
            "optim.muon.lr_scale",
        ),
        (
            lambda cfg: replace(
                cfg,
                optim=replace(cfg.optim, muon=replace(cfg.optim.muon, weight_decay_mult=-1.0)),
            ),
            "optim.muon.weight_decay_mult",
        ),
        (lambda cfg: replace(cfg, optim=replace(cfg.optim, name="sgd")), "optim.name"),
        (
            lambda cfg: replace(
                cfg,
                optim=replace(cfg.optim, muon=replace(cfg.optim.muon, momentum=1.5)),
            ),
            "optim.muon.momentum",
        ),
        (
            lambda cfg: replace(
                cfg,
                optim=replace(cfg.optim, muon=replace(cfg.optim.muon, ns_steps=0)),
            ),
            "optim.muon.ns_steps",
        ),
        (
            lambda cfg: replace(
                cfg,
                optim=replace(cfg.optim, muon=replace(cfg.optim.muon, consistent_rms=-0.1)),
            ),
            "optim.muon.consistent_rms",
        ),
        (
            lambda cfg: replace(
                cfg, optim=replace(cfg.optim, adam=replace(cfg.optim.adam, b1=1.1))
            ),
            "optim.adam.b1",
        ),
        (
            lambda cfg: replace(
                cfg, optim=replace(cfg.optim, adam=replace(cfg.optim.adam, b2=0.0))
            ),
            "optim.adam.b2",
        ),
        (
            lambda cfg: replace(
                cfg, optim=replace(cfg.optim, adam=replace(cfg.optim.adam, eps=0.0))
            ),
            "optim.adam.eps",
        ),
    ]

    for mutate, match in cases:
        with pytest.raises(ValueError, match=match):
            validate_config(mutate(_base_cfg()))


def test_data_and_logging_validation_rejects_invalid_values() -> None:
    """Data/logging validation should reject invalid configuration values."""
    cases: list[tuple[Callable[[Config], Config], str]] = [
        (
            lambda cfg: replace(
                cfg,
                data=replace(cfg.data, packing_mode="bin", packing_buffer_docs=1),
            ),
            "packing_buffer_docs",
        ),
        (lambda cfg: replace(cfg, data=replace(cfg.data, packing_mode="unknown")), "packing_mode"),
        (
            lambda cfg: replace(cfg, data=replace(cfg.data, max_eval_samples=-1)),
            "max_eval_samples",
        ),
        (
            lambda cfg: replace(
                cfg,
                logging=replace(cfg.logging, wandb=replace(cfg.logging.wandb, mode="bogus")),
            ),
            "wandb.mode",
        ),
        (lambda cfg: replace(cfg, logging=replace(cfg.logging, log_file=" ")), "log_file"),
    ]

    for mutate, match in cases:
        with pytest.raises(ValueError, match=match):
            validate_config(mutate(_base_cfg()))


def test_hf_eval_split_allows_null() -> None:
    """hf_eval_split=None should validate and imply train-split eval fallback."""
    cfg = _base_cfg()
    hf_data = DataConfig(
        backend="hf",
        hf_dataset="dummy",
        hf_name="dummy",
        hf_split="train",
        hf_eval_split=None,
        text_key="text",
        shuffle=False,
        repeat=True,
        tokenizer=TokenizerConfig(kind="byte", byte_offset=0, add_bos=False, add_eos=False),
    )
    validate_config(replace(cfg, data=hf_data))


def test_hf_eval_split_default_is_null() -> None:
    """DataConfig should default hf_eval_split to None for train-only corpora."""
    assert DataConfig().hf_eval_split is None


def test_hf_eval_split_rejects_non_string_types() -> None:
    """hf_eval_split must be either None or a non-empty string."""
    for bad_split in [False, 0, 1.5, [], {}]:
        cfg = _base_cfg()
        hf_data = DataConfig(
            backend="hf",
            hf_dataset="dummy",
            hf_name="dummy",
            hf_split="train",
            hf_eval_split=bad_split,  # type: ignore[arg-type]
            text_key="text",
            shuffle=False,
            repeat=True,
            tokenizer=TokenizerConfig(kind="byte", byte_offset=0, add_bos=False, add_eos=False),
        )
        with pytest.raises(ValueError, match="hf_eval_split"):
            validate_config(replace(cfg, data=hf_data))


def test_muon_defaults_reflect_sweep_results() -> None:
    """Muon defaults should match the best 1k-step sweep settings."""
    cfg = _base_cfg()
    assert cfg.optim.muon.lr_scale == pytest.approx(100.0)
    assert cfg.optim.muon.consistent_rms is None


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


class _DummyTokenizer:
    """Mock tokenizer with configurable special tokens."""

    def __init__(self, size: int, *, bos: int | None, eos: int | None, pad: int | None) -> None:
        self._size = int(size)
        self._bos = bos
        self._eos = eos
        self._pad = pad

    def __len__(self) -> int:
        return self._size

    @property
    def bos_token_id(self) -> int | None:
        return self._bos

    @property
    def eos_token_id(self) -> int | None:
        return self._eos

    @property
    def pad_token_id(self) -> int | None:
        return self._pad


def test_vocab_size_rounds_up_to_multiple() -> None:
    """Vocab size should round up to configured multiple."""
    cfg = Config(
        model=ModelConfig(backend="dummy", vocab_size=300, d_model=32),
        data=DataConfig(
            backend="local_text",
            local_text="tokenizer config text\n",
            tokenizer=TokenizerConfig(kind="byte", vocab_size_multiple=128),
        ),
        train=TrainConfig(steps=1, batch_size=1, seq_len=8, grad_accum=1, allow_cpu=True),
        optim=OptimConfig(warmup_steps=0),
    )
    tok = _DummyTokenizer(size=256, bos=None, eos=None, pad=None)
    updated = resolve_tokenizer_config(cfg, tok)
    assert updated.model.vocab_size == 384


def test_auto_sets_special_token_ids() -> None:
    """auto_set_special_tokens should copy IDs from tokenizer to config."""
    cfg = Config(
        model=ModelConfig(
            backend="dummy",
            vocab_size=512,
            d_model=32,
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=2,
        ),
        data=DataConfig(
            backend="local_text",
            local_text="tokenizer config text\n",
            tokenizer=TokenizerConfig(
                kind="hf",
                hf_name_or_path="dummy",
                auto_set_special_tokens=True,
                add_bos=False,
                add_eos=False,
            ),
        ),
        train=TrainConfig(steps=1, batch_size=1, seq_len=8, grad_accum=1, allow_cpu=True),
        optim=OptimConfig(warmup_steps=0),
    )
    tok = _DummyTokenizer(size=512, bos=10, eos=11, pad=12)
    updated = resolve_tokenizer_config(cfg, tok)
    assert updated.model.bos_token_id == 10
    assert updated.model.eos_token_id == 11
    assert updated.model.pad_token_id == 12


def test_tokenizer_pad_equals_eos_warns() -> None:
    """Tokenizer with pad==eos should warn but still resolve."""
    cfg = Config(
        model=ModelConfig(
            backend="dummy",
            vocab_size=512,
            d_model=32,
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=2,
        ),
        data=DataConfig(
            backend="local_text",
            local_text="tokenizer config text\n",
            tokenizer=TokenizerConfig(
                kind="hf",
                hf_name_or_path="dummy",
                auto_set_special_tokens=True,
                add_bos=False,
                add_eos=False,
            ),
        ),
        train=TrainConfig(steps=1, batch_size=1, seq_len=8, grad_accum=1, allow_cpu=True),
        optim=OptimConfig(warmup_steps=0),
    )
    tok = _DummyTokenizer(size=512, bos=0, eos=0, pad=0)
    with pytest.warns(UserWarning, match="pad_token_id equals model.eos_token_id"):
        updated = resolve_tokenizer_config(cfg, tok)
    assert updated.model.pad_token_id == 0
    assert updated.model.eos_token_id == 0


def test_default_max_doc_tokens_inferred() -> None:
    """max_doc_tokens should default to 4 * seq_len when unset."""
    cfg = Config(
        model=ModelConfig(backend="dummy", vocab_size=512, d_model=32),
        data=DataConfig(
            backend="local_text",
            local_text="tokenizer config text\n",
            tokenizer=TokenizerConfig(kind="byte", vocab_size_multiple=128, max_doc_tokens=None),
        ),
        train=TrainConfig(steps=10, batch_size=1, seq_len=16, grad_accum=1, allow_cpu=True),
        optim=OptimConfig(warmup_steps=0),
    )
    tok = _DummyTokenizer(size=256, bos=None, eos=None, pad=None)
    updated = resolve_tokenizer_config(cfg, tok)
    assert updated.data.tokenizer.max_doc_tokens == 64


def test_zero_max_doc_tokens_disables_truncation() -> None:
    """max_doc_tokens=0 should resolve to None (no truncation)."""
    cfg = Config(
        model=ModelConfig(backend="dummy", vocab_size=512, d_model=32),
        data=DataConfig(
            backend="local_text",
            local_text="tokenizer config text\n",
            tokenizer=TokenizerConfig(kind="byte", vocab_size_multiple=128, max_doc_tokens=0),
        ),
        train=TrainConfig(steps=10, batch_size=1, seq_len=16, grad_accum=1, allow_cpu=True),
        optim=OptimConfig(warmup_steps=0),
    )
    tok = _DummyTokenizer(size=256, bos=None, eos=None, pad=None)
    updated = resolve_tokenizer_config(cfg, tok)
    assert updated.data.tokenizer.max_doc_tokens is None


def test_load_config_for_checkpoint_resolves_variables(tmp_path: Path) -> None:
    """Variable placeholders in override configs should resolve before validation."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
variables:
  seq_len: 64

model:
  backend: dummy
  vocab_size: 256
  d_model: 32
  dropout: 0.0

data:
  backend: local_text
  repeat: true
  local_text: "hello"
  packing_mode: sequential
  tokenizer:
    kind: byte
    byte_offset: 0
    add_bos: false
    add_eos: false

train:
  steps: 10
  batch_size: 1
  seq_len: $variables.seq_len
  grad_accum: 1
  jit: false
  deterministic: true
  allow_cpu: true
  log_every: 1
  eval_every: 0

optim:
  lr: 3.0e-4
  warmup_steps: 0

checkpoint:
  enabled: true
  save_every: 5
  max_to_keep: 1
  async_save: false

logging:
  project: chomp
  run_dir: null
  metrics_file: metrics.jsonl
  level: INFO
  console_use_rich: false
  log_file: null
  wandb:
    enabled: false

debug:
  nan_check: true
  check_device_every: 1
""".lstrip()
    )

    cfg = load_config_for_checkpoint(
        step_dir=tmp_path, run_dir=None, config_override=str(config_path)
    )

    assert cfg.train.seq_len == 64


def test_generate_config_applies_tokenizer_derived_fields(tmp_path: Path) -> None:
    """Generate config loading should apply tokenizer-derived fields."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  backend: dummy
  vocab_size: 300
  d_model: 32
  dropout: 0.0

data:
  backend: local_text
  repeat: true
  local_text: "hello"
  packing_mode: sequential
  tokenizer:
    kind: byte
    byte_offset: 0
    add_bos: false
    add_eos: false
    vocab_size_multiple: 128

train:
  steps: 10
  batch_size: 1
  seq_len: 32
  grad_accum: 1
  jit: false
  deterministic: true
  allow_cpu: true
  log_every: 1
  eval_every: 0

optim:
  lr: 3.0e-4
  warmup_steps: 0

checkpoint:
  enabled: true
  save_every: 5
  max_to_keep: 1
  async_save: false

logging:
  project: chomp
  run_dir: null
  wandb:
    enabled: false

debug:
  nan_check: true
  check_device_every: 1
""".lstrip()
    )

    cfg = load_config_for_checkpoint(
        step_dir=tmp_path, run_dir=None, config_override=str(config_path)
    )

    assert cfg.model.vocab_size == 300

    tokenizer = build_tokenizer(cfg)
    cfg_resolved = resolve_tokenizer_config(cfg, tokenizer)
    assert cfg_resolved.model.vocab_size == 384


def test_override_casts_float_when_default_none(tmp_path: Path) -> None:
    """Optional float overrides should parse into float values."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  backend: dummy
  vocab_size: 256
  d_model: 32
  dropout: 0.0

data:
  backend: local_text
  repeat: true
  local_text: "hello"
  packing_mode: sequential
  tokenizer:
    kind: byte
    byte_offset: 0
    add_bos: false
    add_eos: false

train:
  steps: 20
  batch_size: 1
  seq_len: 8
  grad_accum: 1
  jit: false
  deterministic: true
  allow_cpu: true
  log_every: 1
  eval_every: 0

logging:
  wandb:
    enabled: false

optim:
  warmup_steps: 0
""".lstrip()
    )

    cfg = load_config(config_path, overrides=["optim.muon.consistent_rms=0.2"])

    assert isinstance(cfg.optim.muon.consistent_rms, float)
    assert cfg.optim.muon.consistent_rms == pytest.approx(0.2)
