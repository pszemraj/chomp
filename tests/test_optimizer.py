"""Optimizer and gradient accumulation tests consolidated by module."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest
from optax.contrib import MuonDimensionNumbers

from chomp.config import Config, DataConfig, ModelConfig, OptimConfig, TokenizerConfig, TrainConfig
from chomp.data.pipeline import build_train_iterator
from chomp.model import (
    build_model,
    build_parameter_manifest,
    classify_model_array,
    parameter_decay_mask,
    parameter_optimizer_groups,
    training_loss,
)
from chomp.train import (
    _muon_lr_from_adam,
    _muon_weight_dim_numbers,
    build_optimizer,
    init_train_state,
    make_train_step,
)
from chomp.types import Batch
from chomp.utils.tree import path_to_str
from tests.helpers.assertions import tree_allclose


@pytest.fixture(scope="module")
def megalodon_parts() -> tuple[Config, Any, Any]:
    """Small classified Megalodon model, built once per module.

    Consumers only read it (JAX arrays are immutable; optimizer calls do not
    mutate params), so module scope is safe and avoids 8 model builds.

    :return tuple[Config, Any, Any]: Config, trainable params, and fixed partition.
    """
    cfg = Config(
        model=ModelConfig(
            backend="megalodon",
            vocab_size=128,
            model_dim=32,
            num_layers=2,
            num_heads=1,
            z_dim=16,
            value_dim=32,
            ffn_hidden_dim=64,
            cema_ndim=4,
            chunk_size=16,
            norm_num_groups=4,
            share_emb=False,
        )
    )
    params, static = build_model(cfg, key=jax.random.PRNGKey(0))
    return cfg, params, static


@pytest.fixture(scope="module")
def megalodon_params(megalodon_parts: tuple[Config, Any, Any]) -> Any:
    """Return the trainable partition from the shared Megalodon fixture."""
    return megalodon_parts[1]


def _dim_map(dim_nums: Any) -> dict[str, MuonDimensionNumbers | None]:
    """Create a mapping from parameter path to Muon dimension numbers.

    :param Any dim_nums: Muon dimension numbers pytree.
    :return dict[str, MuonDimensionNumbers | None]: Map of path string to dim spec.
    """

    def _is_leaf(node: Any) -> bool:
        """Return True when a node should be treated as a leaf in the dim tree."""
        return node is None or isinstance(node, MuonDimensionNumbers)

    flat_dims, _ = jax.tree_util.tree_flatten_with_path(dim_nums, is_leaf=_is_leaf)
    return {path_to_str(path): dim for path, dim in flat_dims}


def _leaf_map(tree: Any) -> dict[str, Any]:
    """Create a mapping from parameter path to a leaf value.

    :param Any tree: Pytree to flatten.
    :return dict[str, Any]: Map of path string to leaf value.
    """
    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {path_to_str(path): leaf for path, leaf in flat}


def test_parameter_contract_excludes_derived_rope_arrays(
    megalodon_parts: tuple[Config, Any, Any],
) -> None:
    """Derived RoPE state stays outside the array tree and optimizer state."""
    cfg, params, static = megalodon_parts
    manifest = build_parameter_manifest(cfg, params, static)
    assert not any("rotary" in entry["path"] for entry in manifest["arrays"])
    assert all(entry["trainable"] for entry in manifest["arrays"])

    cfg = replace(cfg, optim=replace(cfg.optim, lr=1e-3, weight_decay=0.1, warmup_steps=0))
    tx, _ = build_optimizer(cfg, params)
    opt_state = tx.init(params)
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    updates, _ = tx.update(zeros, opt_state, params)
    updated = optax.apply_updates(params, updates)
    before = eqx.combine(params, static)
    after = eqx.combine(updated, static)
    assert (
        before.model.layers[0].attn.inner.rotary.dim == after.model.layers[0].attn.inner.rotary.dim
    )
    assert (
        before.model.layers[0].attn.inner.rotary.base
        == after.model.layers[0].attn.inner.rotary.base
    )
    assert not jnp.array_equal(before.model.embed.weight, after.model.embed.weight)


def test_bounded_megalodon_loss_matches_full_logits(
    megalodon_parts: tuple[Config, Any, Any],
) -> None:
    """The Chomp adapter should preserve loss when chunking the vocabulary head."""
    _, params, static = megalodon_parts
    input_ids = jnp.arange(16, dtype=jnp.int32)[None, :]
    batch = Batch(
        input_ids=input_ids,
        labels=input_ids,
        segment_ids=jnp.ones_like(input_ids),
    )
    kwargs = {
        "batch": batch,
        "deterministic": True,
        "key": None,
        "use_packed_segments": True,
    }
    full = training_loss(params, static, **kwargs)
    bounded = training_loss(params, static, loss_chunk_size=3, **kwargs)
    assert jnp.allclose(full, bounded, rtol=1e-6, atol=1e-6)


def test_parameter_decay_policy_is_model_aware(
    megalodon_parts: tuple[Config, Any, Any],
) -> None:
    """Only embeddings and dense projections receive decoupled weight decay."""
    cfg, params, _ = megalodon_parts
    decay = _leaf_map(parameter_decay_mask(cfg, params))
    assert decay["model.embed.weight"] is True
    assert decay["model.layers.[0].attn.wz.weight"] is True
    assert decay["model.layers.[0].ffn.fc1.weight"] is True
    assert decay["model.layers.[0].attn.gamma"] is False
    assert decay["model.layers.[0].attn.cema.gamma_real"] is False
    assert decay["model.layers.[0].attn.timenorm.weight"] is False


def test_parameter_contract_fails_closed_on_unknown_array() -> None:
    """A changed dependency layout must be classified before training proceeds."""
    with pytest.raises(RuntimeError, match="Unclassified megalodon model array"):
        classify_model_array(Config(), "model.layers.[0].future_array")


@pytest.mark.parametrize(
    "model_updates",
    [
        pytest.param({"swiglu": True}, id="swiglu"),
        pytest.param({"norm_affine": False}, id="no-norm-affine"),
        pytest.param({"share_emb": False, "output_size": 96}, id="custom-output"),
        pytest.param({"share_emb": True}, id="tied-embedding"),
    ],
)
def test_parameter_contract_covers_supported_model_variants(
    model_updates: dict[str, Any],
) -> None:
    """Every array in each supported Megalodon layout must classify explicitly."""
    base = ModelConfig(
        backend="megalodon",
        vocab_size=128,
        model_dim=32,
        num_layers=1,
        num_heads=1,
        z_dim=16,
        value_dim=32,
        ffn_hidden_dim=64,
        cema_ndim=4,
        chunk_size=16,
        norm_num_groups=4,
        share_emb=False,
    )
    cfg = Config(model=replace(base, **model_updates))
    params, static = build_model(cfg, key=jax.random.PRNGKey(1))
    manifest = build_parameter_manifest(cfg, params, static)
    assert all(entry["trainable"] for entry in manifest["arrays"])
    if model_updates.get("swiglu"):
        assert any(entry["path"].endswith("ffn.fc3.weight") for entry in manifest["arrays"])
    if model_updates.get("output_size"):
        assert any(entry["path"] == "lm_head.weight" for entry in manifest["arrays"])
    if model_updates.get("share_emb"):
        assert not any(entry["path"] == "lm_head.weight" for entry in manifest["arrays"])


def test_muon_param_labels_whitelist_excludes_embed(
    megalodon_parts: tuple[Config, Any, Any],
) -> None:
    """Muon labels should include projection weights but exclude embeddings."""
    cfg, params, _ = megalodon_parts
    cfg = replace(cfg, optim=replace(cfg.optim, name="muon"))
    mapping = _leaf_map(parameter_optimizer_groups(cfg, params))

    assert mapping["model.embed.weight"] == "adam"
    assert mapping["model.layers.[0].attn.wz.weight"] == "muon"
    assert mapping["model.layers.[0].ffn.fc1.weight"] == "muon"
    assert mapping["model.layers.[0].attn.gamma"] == "adam"
    assert mapping["model.layers.[0].attn.timenorm.weight"] == "adam"
    assert mapping["model.layers.[0].ffn.norm.weight"] == "adam"


def test_muon_param_labels_allow_all_2d(megalodon_parts: tuple[Config, Any, Any]) -> None:
    """allow_all_2d should label every 2D tensor as muon."""
    cfg, params, _ = megalodon_parts
    muon = replace(cfg.optim.muon, allow_all_2d=True)
    cfg = replace(cfg, optim=replace(cfg.optim, name="muon", muon=muon))
    mapping = _leaf_map(parameter_optimizer_groups(cfg, params))

    assert mapping["model.embed.weight"] == "muon"


def test_muon_param_labels_allow_tied_embed(megalodon_parts: tuple[Config, Any, Any]) -> None:
    """allow_tied_embed should affect only an actually tied embedding matrix."""
    cfg, params, _ = megalodon_parts
    muon = replace(cfg.optim.muon, allow_tied_embed=True)
    cfg = replace(cfg, optim=replace(cfg.optim, name="muon", muon=muon))
    mapping = _leaf_map(parameter_optimizer_groups(cfg, params))
    assert mapping["model.embed.weight"] == "adam"

    cfg = replace(cfg, model=replace(cfg.model, share_emb=True))
    params, _ = build_model(cfg, key=jax.random.PRNGKey(2))
    mapping = _leaf_map(parameter_optimizer_groups(cfg, params))
    assert mapping["model.embed.weight"] == "muon"


def test_muon_dim_numbers_match_eqx_orientation(megalodon_params: Any) -> None:
    """Muon dimension numbers should treat eqx Linear weights as (out, in)."""
    params = megalodon_params
    dim_nums = _muon_weight_dim_numbers(params)
    dims = _dim_map(dim_nums)

    spec = dims["model.layers.[0].attn.wz.weight"]
    assert isinstance(spec, MuonDimensionNumbers)
    assert spec.reduction_axis == (1,)
    assert spec.output_axis == (0,)


def test_muon_update_handles_none_leaves(megalodon_params: Any) -> None:
    """Muon optimizer should tolerate None leaves in the update tree."""
    params = megalodon_params
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


def test_muon_lr_scale_matches_schedule(megalodon_params: Any) -> None:
    """Muon LR should be a scaled copy of the Adam schedule."""
    params = megalodon_params
    cfg = Config()
    cfg = replace(cfg, train=replace(cfg.train, steps=10))
    muon_cfg = replace(cfg.optim.muon, lr_scale=10.0)
    cfg = replace(
        cfg,
        optim=replace(
            cfg.optim, name="muon", lr=1e-3, warmup_steps=2, decay_steps=8, muon=muon_cfg
        ),
    )
    _, schedule = build_optimizer(cfg, params)
    for step in (0, 1, 2, 5, 9):
        lr_adam = schedule(jnp.array(step))
        lr_muon = _muon_lr_from_adam(lr_adam, cfg)
        assert jnp.allclose(lr_muon, lr_adam * cfg.optim.muon.lr_scale)


def test_muon_allow_all_2d_warns(megalodon_params: Any, caplog: Any) -> None:
    """Allowing all 2D params should warn for Megalodon backends."""
    params = megalodon_params
    cfg = Config()
    muon_cfg = replace(cfg.optim.muon, allow_all_2d=True)
    cfg = replace(cfg, optim=replace(cfg.optim, name="muon", muon=muon_cfg))
    build_optimizer(cfg, params)
    assert any("muon.allow_all_2d" in rec.message for rec in caplog.records)


def test_muon_non_muon_params_use_plain_adamw(megalodon_params: Any) -> None:
    """Non-Muon params should use AdamW even when Muon uses Nesterov."""
    params = megalodon_params
    cfg = Config()
    muon_cfg = replace(cfg.optim.muon, nesterov=True, consistent_rms=None)
    adam_cfg = replace(cfg.optim.adam, nesterov=False)
    cfg = replace(
        cfg,
        optim=replace(cfg.optim, name="muon", muon=muon_cfg, adam=adam_cfg),
    )
    tx, schedule = build_optimizer(cfg, params)
    opt_state = tx.init(params)
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    updates_muon, _ = tx.update(grads, opt_state, params)

    adam_tx = optax.adamw(
        learning_rate=schedule,
        b1=cfg.optim.adam.b1,
        b2=cfg.optim.adam.b2,
        eps=cfg.optim.adam.eps,
        weight_decay=cfg.optim.weight_decay,
        mask=lambda tree: parameter_decay_mask(cfg, tree),
        nesterov=cfg.optim.adam.nesterov,
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
    state0 = init_train_state(params=params, tx=tx, key=key)

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
        segs: jax.Array,
        k: jax.Array,
        token_count: jax.Array,
    ) -> jax.Array:
        """Compute loss for a single microbatch scaled by token count.

        :param jax.Array p: Model parameters.
        :param jax.Array in_ids: Input token ids.
        :param jax.Array labs: Label token ids.
        :param jax.Array segs: Segment ids.
        :param jax.Array k: PRNG key.
        :param jax.Array token_count: Token count for scaling.
        :return jax.Array: Scaled microbatch loss.
        """
        micro = Batch(
            input_ids=in_ids,
            labels=labs,
            segment_ids=segs,
        )
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
        valid = valid & (batch.segment_ids[i][:, 1:] > 0)
        token_count = jnp.sum(valid, dtype=jnp.int32).astype(jnp.float32)
        loss_i, grads_i = loss_and_grad(
            state0.params,
            batch.input_ids[i],
            batch.labels[i],
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


def test_bf16_params_accumulate_grads_in_fp32() -> None:
    """Gradient accumulation must always run in fp32.

    With bf16 params, zeros_like-initialized accumulators would sum
    micro-gradients in bf16 across the scan, silently dropping low-order
    bits. Floating scan carries (loss and gradient tree) must therefore be
    fp32 even when every param leaf is bf16. The exact token counter remains
    int32 and is cast only for normalization. model.accum_dtype governs
    model-internal accumulation and must not leak into optimizer math.
    """
    cfg = Config(
        model=ModelConfig(
            backend="dummy",
            vocab_size=256,
            d_model=32,
            dropout=0.0,
            accum_dtype="float32",
        ),
        data=DataConfig(
            backend="local_text",
            repeat=True,
            local_text="The quick brown fox jumps over the lazy dog.\n",
            tokenizer=TokenizerConfig(kind="byte", byte_offset=0, add_bos=False, add_eos=False),
        ),
        train=TrainConfig(
            seed=0,
            steps=1,
            batch_size=1,
            seq_len=16,
            grad_accum=2,
            jit=False,
            allow_cpu=True,
            deterministic=True,
        ),
        optim=OptimConfig(lr=1e-3, weight_decay=0.0, grad_clip_norm=1.0, warmup_steps=0),
    )

    key = jax.random.PRNGKey(cfg.train.seed)
    key, k_model = jax.random.split(key, 2)
    params, static = build_model(cfg, key=k_model)
    params = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.bfloat16) if jnp.issubdtype(x.dtype, jnp.floating) else x,
        params,
    )
    tx, sched = build_optimizer(cfg, params)
    state0 = init_train_state(params=params, tx=tx, key=key)

    batch = jax.device_put(next(build_train_iterator(cfg)))
    train_step = make_train_step(cfg, static=static, tx=tx, lr_schedule=sched)

    jaxpr = jax.make_jaxpr(train_step)(state0, batch)
    scan_eqns = [e for e in jaxpr.eqns if e.primitive.name == "scan"]
    assert scan_eqns, "grad accumulation scan not found in train_step jaxpr"
    for eqn in scan_eqns:
        num_carry = int(eqn.params["num_carry"])
        carry_dtypes = {v.aval.dtype for v in eqn.outvars[:num_carry]}
        assert carry_dtypes == {jnp.dtype(jnp.float32), jnp.dtype(jnp.int32)}

    state1, metrics = train_step(state0, batch)
    assert jnp.isfinite(metrics["loss"])
    assert jnp.isfinite(metrics["grad_norm"])
    for leaf0, leaf1 in zip(
        jax.tree_util.tree_leaves(state0.params),
        jax.tree_util.tree_leaves(state1.params),
        strict=True,
    ):
        assert leaf0.dtype == leaf1.dtype  # params keep their dtype through the update
