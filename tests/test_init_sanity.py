"""Initialization sanity checks for model parameters."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from chomp.config import Config, ModelConfig
from chomp.model import build_model


def test_dummy_init_stats_are_sane() -> None:
    cfg = Config(model=ModelConfig(backend="dummy", vocab_size=128, d_model=32, dropout=0.0))
    key = jax.random.PRNGKey(0)
    params, _static = build_model(cfg, key=key)

    leaves = [x for x in jax.tree_util.tree_leaves(params) if hasattr(x, "shape")]
    assert leaves, "Expected parameter leaves for dummy model."

    samples = leaves[: min(10, len(leaves))]
    for leaf in samples:
        arr = jnp.asarray(leaf, dtype=jnp.float32)
        std = float(jnp.std(arr))
        max_abs = float(jnp.max(jnp.abs(arr)))
        assert bool(jnp.all(jnp.isfinite(arr)))
        assert std > 0.0
        assert max_abs > 0.0
