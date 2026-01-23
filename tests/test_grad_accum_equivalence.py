"""Gradient accumulation correctness test.

This checks a subtle but critical invariant:

- `grad_accum=A` inside `lax.scan` should produce the same *single* optimizer update
  as explicitly averaging per-microbatch gradients and applying **one** Optax update.

This is NOT the same as running A sequential optimizer steps (especially for Adam).

We intentionally generate the batch through the real tokenize+pack pipeline (local_text)
so this test doesn't rely on a synthetic-batch helper.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from chomp.config import Config, DataConfig, ModelConfig, OptimConfig, TokenizerConfig, TrainConfig
from chomp.data.pipeline import build_train_iterator
from chomp.model import build_model, training_loss
from chomp.train import build_optimizer, init_train_state, make_train_step
from chomp.types import Batch
from chomp.utils.tree import tree_allclose


def test_grad_accum_equivalence_dummy_local_text() -> None:
    """Scan-based grad accum should match manual averaging + single update."""
    cfg = Config(
        model=ModelConfig(backend="dummy", vocab_size=256, d_model=32, dropout=0.0),
        data=DataConfig(
            backend="local_text",
            repeat=True,
            local_text="The quick brown fox jumps over the lazy dog.\n",
            tokenizer=TokenizerConfig(kind="byte", byte_offset=0, add_bos=False, add_eos=False),
        ),
        train=TrainConfig(
            seed=0,
            steps=1,
            batch_size=2,
            seq_len=16,
            grad_accum=4,
            jit=False,
            allow_cpu=True,
            deterministic=True,
        ),
        optim=OptimConfig(lr=1e-3, weight_decay=0.0, grad_clip_norm=0.0, warmup_steps=0),
    )

    key = jax.random.PRNGKey(cfg.train.seed)
    key, k_model = jax.random.split(key, 2)

    params, static = build_model(cfg, key=k_model)
    tx, sched = build_optimizer(cfg, params)
    state0 = init_train_state(cfg, params=params, tx=tx, key=key)

    # Build one batch via the real pipeline
    it = build_train_iterator(cfg)
    batch = next(it)
    batch = jax.device_put(batch)

    # --- Implementation under test ---
    train_step = make_train_step(cfg, static=static, tx=tx, lr_schedule=sched)
    state1, metrics = train_step(state0, batch)

    # --- Reference: average microbatch grads + one update ---
    deterministic = True

    def micro_loss(
        p: jax.Array,
        in_ids: jax.Array,
        labs: jax.Array,
        attn: jax.Array,
        segs: jax.Array,
        k: jax.Array,
        token_count: jax.Array,
    ) -> jax.Array:
        """Compute loss for a single microbatch scaled by token count."""
        micro = Batch(input_ids=in_ids, labels=labs, attention_mask=attn, segment_ids=segs)
        loss = training_loss(
            p,
            static,
            batch=micro,
            deterministic=deterministic,
            key=k,
            use_segment_ids=cfg.model.segment_masking,
        )
        return loss * token_count

    loss_and_grad = eqx.filter_value_and_grad(micro_loss)

    grads_sum = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), state0.params)
    loss_sum = jnp.zeros((), dtype=jnp.float32)
    token_sum = jnp.zeros((), dtype=jnp.float32)

    # Same micro-keys generation as train_step (split once)
    rng, step_key = jax.random.split(state0.rng)
    micro_keys = jax.random.split(step_key, cfg.train.grad_accum)

    for i in range(cfg.train.grad_accum):
        shift_labels = batch.labels[i][:, 1:]
        valid = shift_labels != -100
        valid = valid & batch.attention_mask[i][:, 1:].astype(bool)
        token_count = jnp.sum(valid, dtype=jnp.int32).astype(jnp.float32)
        loss_i, grads_i = loss_and_grad(
            state0.params,
            batch.input_ids[i],
            batch.labels[i],
            batch.attention_mask[i],
            batch.segment_ids[i],
            micro_keys[i],
            token_count,
        )
        loss_sum = loss_sum + loss_i.astype(jnp.float32)
        grads_sum = jax.tree_util.tree_map(lambda a, b: a + b, grads_sum, grads_i)
        token_sum = token_sum + token_count

    loss_ref = loss_sum / token_sum
    grads_ref = jax.tree_util.tree_map(lambda g: g / token_sum, grads_sum)

    updates_ref, opt_state_ref = tx.update(grads_ref, state0.opt_state, state0.params)
    params_ref = optax.apply_updates(state0.params, updates_ref)

    plat = jax.devices()[0].platform
    if plat == "cpu":
        rtol, atol = 0.0, 1e-8
    else:
        rtol, atol = 1e-5, 1e-5

    assert tree_allclose(state1.params, params_ref, rtol=rtol, atol=atol)
    assert tree_allclose(state1.opt_state, opt_state_ref, rtol=rtol, atol=atol)
    assert jnp.allclose(metrics["loss"], loss_ref)
