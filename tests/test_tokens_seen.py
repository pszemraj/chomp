"""Tests for token counter helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from chomp.train import _init_tokens_seen


def test_init_tokens_seen_dtype() -> None:
    """Token counter should use the widest available integer dtype."""
    counter = _init_tokens_seen(0)
    expected = jnp.int64 if jax.config.read("jax_enable_x64") else jnp.int32
    assert counter.dtype == expected
