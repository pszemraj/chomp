"""Tests for Muon optimizer parameter labeling."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
from megalodon_jax.config import MegalodonConfig
from megalodon_jax.model import MegalodonForCausalLM
from optax.contrib import MuonDimensionNumbers

from chomp.train import _muon_weight_dim_numbers, _path_to_str


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

    def _is_leaf(node: Any) -> bool:
        return node is None or isinstance(node, MuonDimensionNumbers)

    flat_dims, _ = jax.tree_util.tree_flatten_with_path(dim_nums, is_leaf=_is_leaf)
    return {_path_to_str(path): dim is not None for path, dim in flat_dims}


def test_muon_param_labels_whitelist_excludes_embed() -> None:
    """Muon labels should include projection weights but exclude embeddings."""
    params = _megalodon_params()
    dim_nums = _muon_weight_dim_numbers(params, allow_all_2d=False)
    mapping = _label_map(dim_nums)

    assert mapping["model.embed.weight"] is False
    assert mapping["model.layers.[0].attn.wz.weight"] is True
    assert mapping["model.layers.[0].ffn.fc1.weight"] is True
    assert mapping["model.layers.[0].attn.gamma"] is False


def test_muon_param_labels_allow_all_2d() -> None:
    """allow_all_2d should label every 2D tensor as muon."""
    params = _megalodon_params()
    dim_nums = _muon_weight_dim_numbers(params, allow_all_2d=True)
    mapping = _label_map(dim_nums)

    assert mapping["model.embed.weight"] is True
