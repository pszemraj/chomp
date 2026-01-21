"""Train-step integration for segment masking with checkpointing and GA."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from chomp.config import Config, DataConfig, ModelConfig, TokenizerConfig, TrainConfig
from chomp.model import build_model
from chomp.train import build_optimizer, init_train_state, make_train_step
from chomp.types import Batch


def test_train_step_segment_masking_with_checkpointing():
    pytest.importorskip("megalodon_jax")

    cfg = Config(
        model=ModelConfig(
            backend="megalodon",
            vocab_size=32,
            model_dim=8,
            num_layers=1,
            num_heads=2,
            z_dim=4,
            value_dim=4,
            ffn_hidden_dim=16,
            cema_ndim=2,
            chunk_size=2,
            norm_num_groups=1,
            dropout=0.0,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            param_dtype="float32",
            compute_dtype="float32",
            accum_dtype="float32",
            softmax_dtype="float32",
            use_checkpoint=True,
            segment_masking=True,
        ),
        data=DataConfig(
            backend="local_text",
            local_text="segment mask train step\n",
            tokenizer=TokenizerConfig(kind="byte", byte_offset=0, add_bos=False, add_eos=False),
        ),
        train=TrainConfig(
            steps=1,
            batch_size=2,
            seq_len=4,
            grad_accum=2,
            jit=False,
            deterministic=False,
            allow_cpu=True,
        ),
    )

    key = jax.random.PRNGKey(0)
    params, static = build_model(cfg, key=key)
    tx, sched = build_optimizer(cfg, params)
    state = init_train_state(cfg, params=params, tx=tx, key=key)
    train_step = make_train_step(cfg, static=static, tx=tx, lr_schedule=sched)

    input_ids = jnp.array(
        [
            [[1, 2, 3, 4], [4, 3, 2, 1]],
            [[1, 1, 2, 2], [3, 3, 4, 4]],
        ],
        dtype=jnp.int32,
    )
    labels = input_ids
    attention_mask = jnp.ones_like(input_ids, dtype=jnp.bool_)
    segment_ids = jnp.array(
        [
            [[1, 2, 1, 2], [1, 1, 2, 2]],
            [[1, 2, 1, 2], [2, 2, 3, 3]],
        ],
        dtype=jnp.int32,
    )
    batch = Batch(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        segment_ids=segment_ids,
    )

    state2, metrics = train_step(state, batch)
    metrics_host = jax.device_get(metrics)

    assert int(jax.device_get(state2.step)) == 1
    assert jnp.isfinite(metrics_host["loss"])
    assert jnp.isfinite(metrics_host["grad_norm"])
