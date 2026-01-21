"""Segment IDs should induce block-diagonal attention within a chunk."""

from __future__ import annotations

import jax.numpy as jnp

from chomp.patches.megalodon_segment_ids import apply_segment_ids_patch


def test_segment_ids_block_attention():
    apply_segment_ids_patch()

    from megalodon_jax.layers.attention import attention_single_chunk

    q = jnp.zeros((1, 4, 1, 1), dtype=jnp.float32)
    k = jnp.zeros_like(q)
    v = jnp.arange(1, 5, dtype=jnp.float32).reshape(1, 4, 1, 1)
    segment_ids = jnp.array([[1, 1, 2, 2]], dtype=jnp.int32)

    out = attention_single_chunk(q, k, v, segment_ids=segment_ids, deterministic=True)
    values = out[:, :, 0, 0]

    expected = jnp.array([[1.0, 1.5, 3.0, 3.5]], dtype=jnp.float32)
    assert jnp.allclose(values, expected)
