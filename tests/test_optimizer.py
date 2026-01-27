"""Optimizer and gradient accumulation tests consolidated by module."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from megalodon_jax.config import MegalodonConfig
from megalodon_jax.model import MegalodonForCausalLM
from optax.contrib import MuonDimensionNumbers

from chomp.config import Config, DataConfig, ModelConfig, OptimConfig, TokenizerConfig, TrainConfig
from chomp.data.pipeline import build_train_iterator
from chomp.model import build_model, training_loss
from chomp.train import (
    _muon_lr_from_adam,
    _muon_weight_dim_numbers,
    _path_to_str,
    _weight_decay_mask,
    build_optimizer,
    init_train_state,
    make_train_step,
)
from chomp.types import Batch
from chomp.utils.tree import tree_allclose


def _megalodon_params() -> Any:
    """Build a small Megalodon parameter pytree for tests.

    :return Any: Filtered parameter pytree.
    """
    cfg = MegalodonConfig(
        vocab_size=128,
        model_dim=32,
        num_layers=2,
        num_heads=1,
        chunk_size=16,
    )
    key = jax.random.PRNGKey(0)
    model = MegalodonForCausalLM(cfg, key=key)
    return eqx.filter(model, eqx.is_array)


def _label_map(dim_nums: Any) -> dict[str, bool]:
    """Create a mapping from parameter path to muon eligibility.

    :param Any dim_nums: Muon dimension numbers pytree.
    :return dict[str, bool]: Map of path string to muon eligibility.
    """
    return {path: dim is not None for path, dim in _dim_map(dim_nums).items()}


def _dim_map(dim_nums: Any) -> dict[str, MuonDimensionNumbers | None]:
    """Create a mapping from parameter path to Muon dimension numbers.

    :param Any dim_nums: Muon dimension numbers pytree.
    :return dict[str, MuonDimensionNumbers | None]: Map of path string to dim spec.
    """

    def _is_leaf(node: Any) -> bool:
        """Return True when a node should be treated as a leaf in the dim tree."""
        return node is None or isinstance(node, MuonDimensionNumbers)

    flat_dims, _ = jax.tree_util.tree_flatten_with_path(dim_nums, is_leaf=_is_leaf)
    return {_path_to_str(path): dim for path, dim in flat_dims}


def _leaf_map(tree: Any) -> dict[str, Any]:
    """Create a mapping from parameter path to a leaf value.

    :param Any tree: Pytree to flatten.
    :return dict[str, Any]: Map of path string to leaf value.
    """
    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {_path_to_str(path): leaf for path, leaf in flat}


def test_muon_param_labels_whitelist_excludes_embed() -> None:
    """Muon labels should include projection weights but exclude embeddings."""
    params = _megalodon_params()
    dim_nums = _muon_weight_dim_numbers(params, allow_all_2d=False)
    mapping = _label_map(dim_nums)

    assert mapping["model.embed.weight"] is False
    assert mapping["model.layers.[0].attn.wz.weight"] is True
    assert mapping["model.layers.[0].ffn.fc1.weight"] is True
    assert mapping["model.layers.[0].attn.gamma"] is False
    assert mapping["model.layers.[0].attn.timenorm.weight"] is False
    assert mapping["model.layers.[0].ffn.norm.weight"] is False


def test_muon_param_labels_allow_all_2d() -> None:
    """allow_all_2d should label every 2D tensor as muon."""
    params = _megalodon_params()
    dim_nums = _muon_weight_dim_numbers(params, allow_all_2d=True)
    mapping = _label_map(dim_nums)

    assert mapping["model.embed.weight"] is True


def test_muon_param_labels_allow_tied_embed() -> None:
    """allow_embed should include the tied embedding matrix."""
    params = _megalodon_params()
    dim_nums = _muon_weight_dim_numbers(params, allow_all_2d=False, allow_embed=True)
    mapping = _label_map(dim_nums)

    assert mapping["model.embed.weight"] is True


def test_muon_dim_numbers_match_eqx_orientation() -> None:
    """Muon dimension numbers should treat eqx Linear weights as (out, in)."""
    params = _megalodon_params()
    dim_nums = _muon_weight_dim_numbers(params, allow_all_2d=False)
    dims = _dim_map(dim_nums)

    spec = dims["model.layers.[0].attn.wz.weight"]
    assert isinstance(spec, MuonDimensionNumbers)
    assert spec.reduction_axis == (1,)
    assert spec.output_axis == (0,)


def test_muon_update_handles_none_leaves() -> None:
    """Muon optimizer should tolerate None leaves in the update tree."""
    params = _megalodon_params()
    cfg = Config()
    cfg = replace(cfg, optim=replace(cfg.optim, name="muon"))
    tx, _ = build_optimizer(cfg, params)
    opt_state = tx.init(params)
    grads = jax.tree_util.tree_map(
        lambda x: None if x is None else jnp.ones_like(x),
        params,
        is_leaf=lambda x: x is None,
    )
    updates, _ = tx.update(grads, opt_state, params)
    leaves = jax.tree_util.tree_leaves(updates, is_leaf=lambda x: x is None)
    assert any(leaf is None for leaf in leaves)
    assert any(leaf is not None for leaf in leaves)


def test_muon_lr_scale_matches_schedule() -> None:
    """Muon LR should be a scaled copy of the Adam schedule."""
    params = _megalodon_params()
    cfg = Config()
    cfg = replace(cfg, train=replace(cfg.train, steps=10))
    cfg = replace(
        cfg,
        optim=replace(
            cfg.optim, name="muon", lr=1e-3, warmup_steps=2, decay_steps=8, muon_lr_scale=10.0
        ),
    )
    _, schedule = build_optimizer(cfg, params)
    for step in (0, 1, 2, 5, 9):
        lr_adam = schedule(jnp.array(step))
        lr_muon = _muon_lr_from_adam(lr_adam, cfg)
        assert jnp.allclose(lr_muon, lr_adam * cfg.optim.muon_lr_scale)


def test_muon_allow_all_2d_warns(caplog: Any) -> None:
    """Allowing all 2D params should warn for Megalodon backends."""
    params = _megalodon_params()
    cfg = Config()
    cfg = replace(cfg, optim=replace(cfg.optim, name="muon", muon_allow_all_2d=True))
    build_optimizer(cfg, params)
    assert any("muon_allow_all_2d" in rec.message for rec in caplog.records)


def test_muon_non_muon_params_use_plain_adamw() -> None:
    """Non-Muon params should use AdamW even when Muon uses Nesterov."""
    params = _megalodon_params()
    cfg = Config()
    cfg = replace(
        cfg,
        optim=replace(
            cfg.optim,
            name="muon",
            muon_nesterov=True,
            adam_nesterov=False,
            muon_consistent_rms=None,
        ),
    )
    tx, schedule = build_optimizer(cfg, params)
    opt_state = tx.init(params)
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    updates_muon, _ = tx.update(grads, opt_state, params)

    adam_tx = optax.adamw(
        learning_rate=schedule,
        b1=cfg.optim.adam_b1,
        b2=cfg.optim.adam_b2,
        eps=cfg.optim.adam_eps,
        weight_decay=cfg.optim.weight_decay,
        mask=_weight_decay_mask,
        nesterov=cfg.optim.adam_nesterov,
    )
    adam_state = adam_tx.init(params)
    updates_adam, _ = adam_tx.update(grads, adam_state, params)

    muon_map = _leaf_map(updates_muon)
    adam_map = _leaf_map(updates_adam)
    path = "model.layers.[0].attn.gamma"
    assert jnp.allclose(muon_map[path], adam_map[path])


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
        """Compute loss for a single microbatch scaled by token count.

        :param jax.Array p: Model parameters.
        :param jax.Array in_ids: Input token ids.
        :param jax.Array labs: Label token ids.
        :param jax.Array attn: Attention mask.
        :param jax.Array segs: Segment ids.
        :param jax.Array k: PRNG key.
        :param jax.Array token_count: Token count for scaling.
        :return jax.Array: Scaled microbatch loss.
        """
        micro = Batch(input_ids=in_ids, labels=labs, attention_mask=attn, segment_ids=segs)
        loss = training_loss(
            p,
            static,
            batch=micro,
            deterministic=deterministic,
            key=k,
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
