"""Tests for config variable interpolation."""

from __future__ import annotations

from pathlib import Path

import pytest

from chomp.config import load_config


def _write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_variables_resolve_and_interpolate(tmp_path: Path) -> None:
    """Variables should resolve to typed values and interpolate into strings."""
    cfg_path = _write_config(
        tmp_path,
        """
variables:
  chunk_size: 1024
  seq_len: 2048
model:
  backend: megalodon
  vocab_size: 256
  model_dim: 128
  num_layers: 2
  num_heads: 1
  z_dim: 64
  value_dim: 128
  ffn_hidden_dim: 256
  cema_ndim: 16
  chunk_size: $variables.chunk_size
  dropout: 0.0
  attention_dropout: 0.0
  hidden_dropout: 0.0
  pad_token_id: 0
  bos_token_id: 1
  eos_token_id: 2
  init_mode: he
  param_dtype: float32
  compute_dtype: bfloat16
  accum_dtype: float32
  softmax_dtype: float32
data:
  backend: local_text
  local_text: "hello"
  repeat: true
  packing_mode: sequential
  packing_buffer_docs: 4
  grain_prefetch: 0
  tokenizer:
    kind: byte
    add_bos: false
    add_eos: false
train:
  steps: 2
  batch_size: 1
  seq_len: $variables.seq_len
  grad_accum: 1
  jit: false
  deterministic: true
  allow_cpu: true
  log_every: 1
  eval_every: 0
optim:
  warmup_steps: 0
logging:
  wandb:
    enabled: false
    tags: ["seq{$variables.seq_len}", "chunk{$variables.chunk_size}"]
""",
    )
    cfg = load_config(cfg_path)
    assert cfg.model.chunk_size == 1024
    assert cfg.train.seq_len == 2048
    tags = list(cfg.logging.wandb.tags)
    assert tags == ["seq2048", "chunk1024"]


def test_variables_missing_reference_raises(tmp_path: Path) -> None:
    """Missing variable references should raise a ValueError."""
    cfg_path = _write_config(
        tmp_path,
        """
variables:
  chunk_size: 1024
model:
  backend: megalodon
  vocab_size: 256
  model_dim: 128
  num_layers: 2
  num_heads: 1
  z_dim: 64
  value_dim: 128
  ffn_hidden_dim: 256
  cema_ndim: 16
  chunk_size: $variables.missing
data:
  backend: local_text
  local_text: "hello"
  repeat: true
  packing_mode: sequential
  packing_buffer_docs: 4
  grain_prefetch: 0
  tokenizer:
    kind: byte
    add_bos: false
    add_eos: false
train:
  steps: 2
  batch_size: 1
  seq_len: 16
  grad_accum: 1
  jit: false
  deterministic: true
  allow_cpu: true
  log_every: 1
  eval_every: 0
""",
    )
    with pytest.raises(ValueError, match="Unknown variable reference"):
        load_config(cfg_path)
