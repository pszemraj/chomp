"""Training loss must not accept cache arguments."""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import pytest

from chomp.config import Config, ModelConfig
from chomp.model import build_model, training_loss
from chomp.types import Batch


def test_training_loss_rejects_cache_kwarg() -> None:
    """Training_loss signature must not include cache arguments."""
    sig = inspect.signature(training_loss)
    assert "cache" not in sig.parameters
    assert "return_cache" not in sig.parameters

    cfg = Config(model=ModelConfig(backend="dummy", vocab_size=64, d_model=16, dropout=0.0))
    key = jax.random.PRNGKey(0)
    params, static = build_model(cfg, key=key)

    input_ids = jnp.zeros((1, 8), dtype=jnp.int32)
    labels = jnp.zeros((1, 8), dtype=jnp.int32)
    attention = jnp.ones((1, 8), dtype=jnp.bool_)
    segment_ids = jnp.ones((1, 8), dtype=jnp.int32)
    batch = Batch(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention,
        segment_ids=segment_ids,
    )

    with pytest.raises(TypeError):
        training_loss(
            params,
            static,
            batch=batch,
            deterministic=True,
            key=None,
            use_segment_ids=True,
            cache=None,
        )
