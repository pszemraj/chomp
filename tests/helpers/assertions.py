"""Shared assertions for structured JAX values."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def tree_allclose(a: Any, b: Any, *, rtol: float = 1e-6, atol: float = 1e-6) -> bool:
    """Return whether two pytrees have equal structure and close leaves.

    :param Any a: First pytree.
    :param Any b: Second pytree.
    :param float rtol: Relative tolerance.
    :param float atol: Absolute tolerance.
    :return bool: Whether structure, leaf metadata, and values match.
    """
    if jax.tree_util.tree_structure(a) != jax.tree_util.tree_structure(b):
        return False
    leaves = zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True)
    for left, right in leaves:
        if hasattr(left, "shape") and hasattr(right, "shape"):
            if left.shape != right.shape or left.dtype != right.dtype:
                return False
            if not bool(jnp.allclose(left, right, rtol=rtol, atol=atol)):
                return False
        elif left != right:
            return False
    return True
