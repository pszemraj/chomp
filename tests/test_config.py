"""Config tests consolidated by module."""

from __future__ import annotations

import json
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
from chomp.utils.checkpoints import load_config_for_checkpoint, resolve_checkpoint_path


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


def test_muon_defaults_reflect_sweep_results() -> None:
    """Muon defaults should match the best 1k-step sweep settings."""
    cfg = _base_cfg()
    assert cfg.optim.muon_lr_scale == pytest.approx(100.0)
    assert cfg.optim.muon_consistent_rms is None


def test_adam_b1_must_be_in_range() -> None:
    """adam_b1 outside (0, 1) must raise ValueError."""
    cfg = _base_cfg()
    bad_optim = replace(cfg.optim, adam_b1=1.1)
    bad = replace(cfg, optim=bad_optim)
    with pytest.raises(ValueError, match="adam_b1"):
        validate_config(bad)


def test_adam_b2_must_be_in_range() -> None:
    """adam_b2 outside (0, 1) must raise ValueError."""
    cfg = _base_cfg()
    bad_optim = replace(cfg.optim, adam_b2=0.0)
    bad = replace(cfg, optim=bad_optim)
    with pytest.raises(ValueError, match="adam_b2"):
        validate_config(bad)


def test_adam_eps_must_be_positive() -> None:
    """adam_eps <= 0 must raise ValueError."""
    cfg = _base_cfg()
    bad_optim = replace(cfg.optim, adam_eps=0.0)
    bad = replace(cfg, optim=bad_optim)
    with pytest.raises(ValueError, match="adam_eps"):
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


class _DummyTokenizer:
    """Mock tokenizer with configurable special tokens."""

    def __init__(self, size: int, *, bos: int | None, eos: int | None, pad: int | None) -> None:
        """Initialize mock tokenizer.

        :param int size: Tokenizer vocab size.
        :param int | None bos: BOS token id.
        :param int | None eos: EOS token id.
        :param int | None pad: PAD token id.
        """
        self._size = int(size)
        self._bos = bos
        self._eos = eos
        self._pad = pad

    def __len__(self) -> int:
        return self._size

    @property
    def bos_token_id(self) -> int | None:
        """Return BOS token ID.

        :return int | None: BOS token id.
        """
        return self._bos

    @property
    def eos_token_id(self) -> int | None:
        """Return EOS token ID.

        :return int | None: EOS token id.
        """
        return self._eos

    @property
    def pad_token_id(self) -> int | None:
        """Return pad token ID.

        :return int | None: PAD token id.
        """
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


def test_resolve_checkpoint_root_dir_relative_to_run(tmp_path: Path) -> None:
    """Relative checkpoint.root_dir should resolve against the run directory."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    cfg = Config()
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, root_dir="relative_ckpts"))
    (run_dir / "config_resolved.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    step_dir = run_dir / "relative_ckpts" / "1"
    (step_dir / "train_state").mkdir(parents=True)

    found_step, found_run = resolve_checkpoint_path(str(run_dir))

    assert found_run == run_dir
    assert found_step == step_dir


def test_resolve_checkpoint_ignores_cwd_shadow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Relative root_dir should resolve against run_dir even if CWD has same-named dir.

    :param Path tmp_path: Temporary directory for test artifacts.
    :param pytest.MonkeyPatch monkeypatch: Pytest monkeypatch fixture.
    """
    # Create run directory with checkpoints
    run_dir = tmp_path / "runs" / "my_run"
    run_dir.mkdir(parents=True)

    cfg = Config()
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, root_dir="ckpts"))
    (run_dir / "config_resolved.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    correct_step_dir = run_dir / "ckpts" / "100"
    (correct_step_dir / "train_state").mkdir(parents=True)

    # Create a shadow "ckpts" directory in a different location (simulating CWD)
    shadow_dir = tmp_path / "ckpts" / "999"
    shadow_dir.mkdir(parents=True)

    # Change CWD to tmp_path where shadow exists
    monkeypatch.chdir(tmp_path)

    # Should find run's checkpoint, not the CWD shadow
    found_step, found_run = resolve_checkpoint_path(str(run_dir))

    assert found_run == run_dir
    assert found_step == correct_step_dir
    assert "999" not in str(found_step)  # Ensure we didn't pick up shadow


def test_resolve_step_dir_external_root_uses_meta(tmp_path: Path) -> None:
    """Step directories under external roots should resolve config via metadata."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)

    cfg = Config()
    cfg = replace(cfg, logging=replace(cfg.logging, run_dir=str(run_dir)))

    step_dir = tmp_path / "external_ckpts" / "100"
    (step_dir / "train_state").mkdir(parents=True)
    meta_dir = step_dir / "meta"
    meta_dir.mkdir(parents=True)
    (meta_dir / "metadata").write_text(json.dumps({"config": cfg.to_dict()}, indent=2))

    found_step, found_run = resolve_checkpoint_path(str(step_dir))

    assert found_step == step_dir
    assert found_run == run_dir

    loaded = load_config_for_checkpoint(step_dir=step_dir, run_dir=None, config_override=None)
    assert loaded.logging.run_dir == str(run_dir)


def test_run_dir_uses_resolved_config_for_ckpt_root(tmp_path: Path) -> None:
    """Run dir resolution should use config_resolved.json for checkpoint root."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)

    ckpt_root = tmp_path / "external_ckpts"
    step_dir = ckpt_root / "5" / "train_state"
    step_dir.mkdir(parents=True)

    cfg = Config()
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, root_dir=str(ckpt_root)))
    (run_dir / "config_resolved.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    override_path = tmp_path / "override.yaml"
    override_path.write_text(
        """
model:
  backend: dummy
""".lstrip()
    )

    found_step, found_run = resolve_checkpoint_path(
        str(run_dir), config_override=str(override_path)
    )

    assert found_run == run_dir
    assert found_step.parent == ckpt_root
    assert found_step.name == "5"


def test_generate_config_applies_tokenizer_derived_fields(tmp_path: Path) -> None:
    """Generate config loading should round vocab_size via resolve_tokenizer_config.

    This test verifies that when loading a config for generation, the tokenizer-
    derived fields (vocab_size rounding, special tokens) are applied. Without
    calling resolve_tokenizer_config, model shapes would mismatch the checkpoint.
    """
    config_path = tmp_path / "config.yaml"
    # Create config with vocab_size=300 which should round up to 384 (multiple of 128)
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

    # Load config (simulating what generate CLI does)
    cfg = load_config_for_checkpoint(
        step_dir=tmp_path, run_dir=None, config_override=str(config_path)
    )

    # Before tokenizer resolution, vocab_size is 300
    assert cfg.model.vocab_size == 300

    # After building tokenizer and resolving, vocab_size should be 384
    tokenizer = build_tokenizer(cfg)
    cfg_resolved = resolve_tokenizer_config(cfg, tokenizer)

    # Byte tokenizer has 256 tokens, rounded up to 384 (next multiple of 128)
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

    cfg = load_config(config_path, overrides=["optim.muon_consistent_rms=0.2"])

    assert isinstance(cfg.optim.muon_consistent_rms, float)
    assert cfg.optim.muon_consistent_rms == pytest.approx(0.2)
