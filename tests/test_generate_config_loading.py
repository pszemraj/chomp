"""Tests for config loading behavior in generate CLI helpers."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from chomp.cli.generate import _find_checkpoint_dir, _load_config_from_checkpoint
from chomp.config import Config


def test_load_config_from_checkpoint_resolves_variables(tmp_path: Path) -> None:
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

    cfg = _load_config_from_checkpoint(tmp_path, str(config_path))

    assert cfg.train.seq_len == 64


def test_find_checkpoint_dir_resolves_root_dir_relative_to_run(tmp_path: Path) -> None:
    """Relative checkpoint.root_dir should resolve against the run directory."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    cfg = Config()
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, root_dir="relative_ckpts"))
    (run_dir / "config_resolved.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    step_dir = run_dir / "relative_ckpts" / "1"
    step_dir.mkdir(parents=True)

    found_step, found_run = _find_checkpoint_dir(str(run_dir))

    assert found_run == run_dir
    assert found_step == step_dir
