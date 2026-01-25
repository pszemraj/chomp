"""Tests for config variable interpolation."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from chomp.config import load_config


def _write_config(tmp_path: Path, text: str) -> Path:
    """Write a temporary config file.

    :param Path tmp_path: Temporary directory for test files.
    :param str text: Config contents to write.
    :return Path: Path to the written config file.
    """
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


def test_variables_circular_reference_raises(tmp_path: Path) -> None:
    """Circular variable references should raise a ValueError."""
    cfg_path = _write_config(
        tmp_path,
        """
variables:
  a: $variables.b
  b: $variables.a
model:
  backend: megalodon
  vocab_size: 256
  model_dim: $variables.a
  num_layers: 2
  num_heads: 1
  z_dim: 64
  value_dim: 128
  ffn_hidden_dim: 256
  cema_ndim: 16
  chunk_size: 16
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
    with pytest.raises(ValueError, match="Circular variable reference"):
        load_config(cfg_path)


def test_variables_nested_depth_two(tmp_path: Path) -> None:
    """Nested variables at depth 2+ should resolve correctly."""
    cfg_path = _write_config(
        tmp_path,
        """
variables:
  dims:
    model: 256
    ffn: 512
model:
  backend: megalodon
  vocab_size: 256
  model_dim: $variables.dims.model
  num_layers: 2
  num_heads: 1
  z_dim: 64
  value_dim: 128
  ffn_hidden_dim: $variables.dims.ffn
  cema_ndim: 16
  chunk_size: 16
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
  seq_len: 16
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
""",
    )
    cfg = load_config(cfg_path)
    assert cfg.model.model_dim == 256
    assert cfg.model.ffn_hidden_dim == 512


def test_variables_multiple_in_single_string(tmp_path: Path) -> None:
    """Multiple variables in a single string should all be interpolated."""
    cfg_path = _write_config(
        tmp_path,
        """
variables:
  batch_size: 4
  seq_len: 128
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
  chunk_size: 16
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
  batch_size: $variables.batch_size
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
    tags: ["b{$variables.batch_size}_s{$variables.seq_len}"]
""",
    )
    cfg = load_config(cfg_path)
    assert cfg.train.batch_size == 4
    assert cfg.train.seq_len == 128
    tags = list(cfg.logging.wandb.tags)
    assert tags == ["b4_s128"]


def test_variables_suspicious_pattern_warns(tmp_path: Path) -> None:
    """Bare $variables.x in a string (not entire value) should warn."""
    cfg_path = _write_config(
        tmp_path,
        """
variables:
  batch_size: 4
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
  chunk_size: 16
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
  seq_len: 16
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
    tags: ["run_$variables.batch_size"]
""",
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = load_config(cfg_path)
        # Tag should be unchanged (not interpolated)
        tags = list(cfg.logging.wandb.tags)
        assert tags == ["run_$variables.batch_size"]
        # Should have warned
        assert len(w) == 1
        assert "unresolved variable-like patterns" in str(w[0].message)
