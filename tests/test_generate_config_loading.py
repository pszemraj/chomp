"""Tests for config loading behavior in generate CLI helpers."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from chomp.config import Config, load_config
from chomp.data.pipeline import build_tokenizer, resolve_tokenizer_config
from chomp.utils.checkpoints import load_config_for_checkpoint, resolve_checkpoint_path


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
