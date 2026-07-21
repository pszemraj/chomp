"""Optimizer and gradient accumulation tests consolidated by module."""

from __future__ import annotations

import logging
from collections.abc import Callable
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
    classify_model_array,
    parameter_decay_mask,
    parameter_family_counts,
    parameter_optimizer_groups,
    training_loss,
)
from chomp.train import (
    _muon_weight_dim_numbers,
    build_optimizer,
    init_train_state,
    make_train_step,
)
from chomp.types import Batch
from chomp.utils.tree import path_to_str
from tests.helpers.config_factories import make_tiny_megalodon_model


@pytest.fixture(scope="module")
def megalodon_parts() -> tuple[Config, Any, Any]:
    """Small classified Megalodon model, built once per module.

    Consumers only read it (JAX arrays are immutable; optimizer calls do not
    mutate params), so module scope is safe and avoids 8 model builds.

    :return tuple[Config, Any, Any]: Config, trainable params, and fixed partition.
    """
    cfg = Config(model=make_tiny_megalodon_model(num_layers=2, chunk_size=16, share_emb=False))
    params, static = build_model(cfg, key=jax.random.PRNGKey(0))
    return cfg, params, static


@pytest.fixture(scope="module")
def megalodon_params(megalodon_parts: tuple[Config, Any, Any]) -> Any:
    """Return the trainable partition from the shared Megalodon fixture."""
    return megalodon_parts[1]


def _leaf_map(
    tree: Any,
    *,
    is_leaf: Callable[[Any], bool] | None = None,
) -> dict[str, Any]:
    """Create a mapping from parameter path to a leaf value.

    :param Any tree: Pytree to flatten.
    :param Callable[[Any], bool] | None is_leaf: Optional custom leaf predicate, defaults to None.
    :return dict[str, Any]: Map of path string to leaf value.
    """
    flat, _ = jax.tree_util.tree_flatten_with_path(tree, is_leaf=is_leaf)
    return {path_to_str(path): leaf for path, leaf in flat}


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
    """Only learned embeddings and dense projections receive weight decay."""
    cfg, params, _ = megalodon_parts
    assert not any("rotary" in path for path in _leaf_map(params))

    decay = _leaf_map(parameter_decay_mask(cfg, params))
    assert decay["model.embed.weight"] is True
    assert decay["model.layers.[0].attn.wz.weight"] is True
    assert decay["model.layers.[0].ffn.fc1.weight"] is True
    assert decay["model.layers.[0].attn.gamma"] is False
    assert decay["model.layers.[0].attn.cema.gamma_real"] is False
    assert decay["model.layers.[0].attn.timenorm.weight"] is False


def test_megalodon_classification_covers_every_array(
    megalodon_parts: tuple[Config, Any, Any],
) -> None:
    """No trainable array in a real build falls through to the 'other' family.

    A megalodon-jax update that adds parameters classify_model_array does not
    know would silently train them under Adam without decay; this is the
    tripline that catches the version bump.
    """
    cfg, params, _ = megalodon_parts
    counts = parameter_family_counts(cfg, params)
    assert counts.get("other", 0) == 0, counts


def test_build_optimizer_warns_on_unclassified_arrays(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Arrays that fall through classification are surfaced, not silent."""
    cfg = Config()
    params = {"model": {"future_array": jnp.ones((2, 2), dtype=jnp.float32)}}
    with caplog.at_level(logging.WARNING, logger="chomp.train"):
        build_optimizer(cfg, params)
    assert any("'other' family" in rec.getMessage() for rec in caplog.records)


def test_unknown_model_array_uses_adam_without_decay() -> None:
    """Unrecognized model arrays should follow the ordinary conservative policy."""
    cfg = Config()
    cfg = replace(cfg, optim=replace(cfg.optim, name="muon"))
    classification = classify_model_array(cfg, "model.layers.[0].future_array")

    assert classification.family == "other"
    assert classification.decay is False


@pytest.mark.parametrize(
    "model_updates",
    [
        pytest.param({"swiglu": True}, id="swiglu"),
        pytest.param({"rescale_nffn": True}, id="rescale-nffn"),
        pytest.param({"norm_affine": False}, id="no-norm-affine"),
        pytest.param({"share_emb": False, "output_size": 96}, id="custom-output"),
        pytest.param({"share_emb": True}, id="tied-embedding"),
    ],
)
def test_parameter_contract_covers_supported_model_variants(
    model_updates: dict[str, Any],
) -> None:
    """Supported Megalodon variants should build with the expected parameter paths."""
    base = make_tiny_megalodon_model(chunk_size=16, share_emb=False)
    cfg = Config(model=replace(base, **model_updates))
    params, _ = build_model(cfg, key=jax.random.PRNGKey(1))
    leaves = _leaf_map(params)
    assert parameter_family_counts(cfg, params).get("other", 0) == 0
    if model_updates.get("swiglu"):
        assert any(path.endswith("ffn.fc3.weight") for path in leaves)
    if model_updates.get("rescale_nffn"):
        path = next(path for path in leaves if path.endswith("ffn.alpha"))
        classification = classify_model_array(cfg, path)
        assert classification.family == "ffn_residual_scale"
        assert classification.decay is False
        assert _leaf_map(parameter_optimizer_groups(cfg, params))[path] == "adam"
    if model_updates.get("output_size"):
        assert "lm_head.weight" in leaves
    if model_updates.get("share_emb"):
        assert "lm_head.weight" not in leaves


def test_ffn_residual_scale_is_trainable_without_weight_decay(
    megalodon_parts: tuple[Config, Any, Any],
) -> None:
    """Normalized-FFN residual scales must receive Adam updates without decay.

    :param tuple[Config, Any, Any] megalodon_parts: Shared base model parts.
    """
    cfg, _, _ = megalodon_parts
    cfg = replace(
        cfg,
        model=replace(cfg.model, rescale_nffn=True),
        optim=replace(cfg.optim, name="muon", lr=1e-3, weight_decay=1.0, warmup_steps=0),
    )
    params, _ = build_model(cfg, key=jax.random.PRNGKey(2))
    path = "model.layers.[0].ffn.alpha"
    before = _leaf_map(params)[path]

    assert _leaf_map(parameter_optimizer_groups(cfg, params))[path] == "adam"
    assert _leaf_map(parameter_decay_mask(cfg, params))[path] is False

    tx, _ = build_optimizer(cfg, params)
    opt_state = tx.init(params)
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    updates, _ = tx.update(grads, opt_state, params)
    after = _leaf_map(optax.apply_updates(params, updates))[path]
    assert not jnp.array_equal(after, before)


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
    dims = _leaf_map(
        dim_nums,
        is_leaf=lambda node: node is None or isinstance(node, MuonDimensionNumbers),
    )

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


def test_grad_accum_matches_equivalent_large_batch() -> None:
    """Accumulated microbatches should match one equivalent physical batch."""
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
            batch_size=1,
            seq_len=16,
            grad_accum=2,
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

    accumulated = jax.device_put(next(build_train_iterator(cfg)))
    # Give the two microbatches different valid-token counts; equal averaging
    # would then diverge from the equivalent large-batch result.
    accumulated = Batch(
        input_ids=accumulated.input_ids,
        labels=accumulated.labels.at[0, 0, -4:].set(-100),
        segment_ids=accumulated.segment_ids,
    )
    batched = Batch(
        input_ids=jnp.swapaxes(accumulated.input_ids, 0, 1),
        labels=jnp.swapaxes(accumulated.labels, 0, 1),
        segment_ids=jnp.swapaxes(accumulated.segment_ids, 0, 1),
    )
    large_batch_cfg = replace(
        cfg,
        train=replace(cfg.train, batch_size=2, grad_accum=1),
    )

    accumulated_step = make_train_step(cfg, static=static, tx=tx, lr_schedule=sched)
    batched_step = make_train_step(large_batch_cfg, static=static, tx=tx, lr_schedule=sched)
    accumulated_state, accumulated_metrics = accumulated_step(state0, accumulated)
    batched_state, batched_metrics = batched_step(state0, batched)

    plat = jax.devices()[0].platform
    if plat == "cpu":
        rtol, atol = 1e-6, 1e-7
    else:
        rtol, atol = 1e-5, 1e-5

    assert eqx.tree_equal(accumulated_state, batched_state, rtol=rtol, atol=atol)
    assert jnp.allclose(accumulated_metrics["loss"], batched_metrics["loss"], rtol=rtol, atol=atol)
    assert accumulated_metrics["token_sum"] == batched_metrics["token_sum"]


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
