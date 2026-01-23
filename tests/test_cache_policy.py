"""Training loss must not accept cache arguments."""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import pytest

import chomp.model as model_mod
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
            cache=None,
        )


def test_training_loss_default_omits_extra_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default training_loss should not pass unexpected kwargs."""
    cfg = Config(model=ModelConfig(backend="dummy", vocab_size=64, d_model=16, dropout=0.0))
    key = jax.random.PRNGKey(0)
    params, static = build_model(cfg, key=key)

    input_ids = jnp.zeros((1, 4), dtype=jnp.int32)
    labels = jnp.zeros((1, 4), dtype=jnp.int32)
    attention = jnp.ones((1, 4), dtype=jnp.bool_)
    segment_ids = jnp.ones((1, 4), dtype=jnp.int32)
    batch = Batch(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention,
        segment_ids=segment_ids,
    )

    seen: dict[str, dict[str, object]] = {}

    def _spy(
        self: model_mod.DummyLM,
        input_ids: jax.Array,
        labels: jax.Array,
        attention_mask: jax.Array | None = None,
        *,
        ignore_index: int = -100,
        deterministic: bool = True,
        key: jax.Array | None = None,
        **kwargs: object,
    ) -> jax.Array:
        _ = (input_ids, labels, attention_mask, ignore_index, deterministic, key)
        seen["kwargs"] = dict(kwargs)
        return jnp.zeros((), dtype=jnp.float32)

    monkeypatch.setattr(model_mod.DummyLM, "compute_loss", _spy, raising=True)

    training_loss(params, static, batch=batch, deterministic=True, key=None)

    assert seen["kwargs"] == {}
