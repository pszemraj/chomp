# SPDX-License-Identifier: Apache-2.0

"""Training loss must not accept cache arguments.

Guards the architectural invariant that training never touches cache
(cache is inference-only). Without this, a convenience ``**kwargs``
passthrough on ``training_loss`` could silently reopen the cache path.
"""

from __future__ import annotations

import inspect
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import pytest

import chomp.model as model_mod
from chomp.config import Config, ModelConfig
from chomp.model import build_model, training_loss
from chomp.types import Batch


def _dummy_batch(seq_len: int) -> Batch:
    """Build a minimal [B=1, T] batch for DummyLM loss calls."""
    return Batch(
        input_ids=jnp.zeros((1, seq_len), dtype=jnp.int32),
        labels=jnp.zeros((1, seq_len), dtype=jnp.int32),
        segment_ids=jnp.ones((1, seq_len), dtype=jnp.int32),
    )


def test_training_loss_rejects_cache_kwarg() -> None:
    """training_loss signature must not include cache arguments."""
    sig = inspect.signature(training_loss)
    assert "cache" not in sig.parameters
    assert "return_cache" not in sig.parameters

    cfg = Config(model=ModelConfig(backend="dummy", vocab_size=64, d_model=16, dropout=0.0))
    params, static = build_model(cfg, key=jax.random.PRNGKey(0))

    with pytest.raises(TypeError):
        training_loss(  # type: ignore[call-arg]
            params,
            static,
            batch=_dummy_batch(8),
            deterministic=True,
            key=None,
            cache=None,
        )


def test_training_loss_default_omits_extra_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default training_loss must pass no kwargs beyond the documented set."""
    cfg = Config(model=ModelConfig(backend="dummy", vocab_size=64, d_model=16, dropout=0.0))
    params, static = build_model(cfg, key=jax.random.PRNGKey(0))

    seen: dict[str, object] = {}

    def _spy(
        self: model_mod.DummyLM,
        input_ids: jax.Array,
        labels: jax.Array,
        attention_mask: jax.Array | None = None,
        segment_ids: jax.Array | None = None,
        *,
        ignore_index: int = -100,
        deterministic: bool = True,
        key: jax.Array | None = None,
        **kwargs: object,
    ) -> jax.Array:
        """Capture the kwargs compute_loss receives from training_loss."""
        _ = (self, input_ids, labels, attention_mask, ignore_index, deterministic, key)
        seen["segment_ids"] = segment_ids
        seen["kwargs"] = dict(kwargs)
        return jnp.zeros((), dtype=jnp.float32)

    monkeypatch.setattr(model_mod.DummyLM, "compute_loss", _spy, raising=True)

    training_loss(params, static, batch=_dummy_batch(4), deterministic=True, key=None)

    assert seen["kwargs"] == {}
    # Packed segments are opt-in; the default path must not forward them.
    assert seen["segment_ids"] is None
