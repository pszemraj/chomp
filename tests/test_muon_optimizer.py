"""Tests for Muon optimizer parameter labeling."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
from megalodon_jax.config import MegalodonConfig
from megalodon_jax.model import MegalodonForCausalLM

from chomp.train import _muon_param_labels, _path_to_str


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


def _label_map(labels: Any, params: Any) -> dict[str, str]:
    """Create a mapping from parameter path to optimizer label.

    :param Any labels: Label pytree.
    :param Any params: Parameter pytree.
    :return dict[str, str]: Map of path string to label.
    """
    flat_params, _ = jax.tree_util.tree_flatten_with_path(params)
    flat_labels, _ = jax.tree_util.tree_flatten(labels)
    return {
        _path_to_str(path): label for (path, _), label in zip(flat_params, flat_labels, strict=True)
    }


def test_muon_param_labels_whitelist_excludes_embed() -> None:
    """Muon labels should include projection weights but exclude embeddings."""
    params = _megalodon_params()
    labels = _muon_param_labels(params, allow_all_2d=False)
    mapping = _label_map(labels, params)

    assert mapping["model.embed.weight"] == "adam"
    assert mapping["model.layers.[0].attn.wz.weight"] == "muon"
    assert mapping["model.layers.[0].ffn.fc1.weight"] == "muon"
    assert mapping["model.layers.[0].attn.gamma"] == "adam"


def test_muon_param_labels_allow_all_2d() -> None:
    """allow_all_2d should label every 2D tensor as muon."""
    params = _megalodon_params()
    labels = _muon_param_labels(params, allow_all_2d=True)
    mapping = _label_map(labels, params)

    assert mapping["model.embed.weight"] == "muon"
