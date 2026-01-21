"""Megalodon integration: segment masking should change loss outputs."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from chomp.config import Config, DataConfig, ModelConfig, TokenizerConfig, TrainConfig
from chomp.model import build_model, training_loss
from chomp.patches.megalodon_segment_ids import apply_segment_ids_patch
from chomp.types import Batch


def test_megalodon_segment_mask_changes_loss():
    pytest.importorskip("megalodon_jax")
    assert apply_segment_ids_patch()

    cfg = Config(
        model=ModelConfig(
            backend="megalodon",
            vocab_size=32,
            model_dim=16,
            num_layers=1,
            num_heads=2,
            z_dim=8,
            value_dim=8,
            ffn_hidden_dim=32,
            cema_ndim=4,
            chunk_size=4,
            norm_num_groups=1,
            dropout=0.0,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            segment_masking=True,
        ),
        data=DataConfig(
            backend="local_text",
            local_text="segment mask integration\n",
            tokenizer=TokenizerConfig(kind="byte", byte_offset=0, add_bos=False, add_eos=False),
        ),
        train=TrainConfig(
            steps=1,
            batch_size=1,
            seq_len=4,
            grad_accum=1,
            jit=False,
            deterministic=True,
            allow_cpu=True,
        ),
    )

    key = jax.random.PRNGKey(0)
    params, static = build_model(cfg, key=key)

    batch = Batch(
        input_ids=jnp.array([[1, 2, 3, 4]], dtype=jnp.int32),
        labels=jnp.array([[1, 2, 3, 4]], dtype=jnp.int32),
        attention_mask=jnp.ones((1, 4), dtype=jnp.bool_),
        segment_ids=jnp.array([[1, 1, 2, 2]], dtype=jnp.int32),
    )

    loss_no_seg = training_loss(
        params,
        static,
        batch=batch,
        deterministic=True,
        key=None,
        use_segment_ids=False,
    )
    loss_seg = training_loss(
        params,
        static,
        batch=batch,
        deterministic=True,
        key=None,
        use_segment_ids=True,
    )

    delta = jnp.abs(loss_seg - loss_no_seg)
    assert float(delta) > 1e-6
