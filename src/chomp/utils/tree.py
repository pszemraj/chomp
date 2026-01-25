"""Pytree helper functions.

This is where you put the small utilities that you end up rewriting in every
JAX training repo:
- parameter counts
- tree equality / closeness checks
- shape/dtype summaries

Keep it minimal: this is not a generic library.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def param_count(params: Any) -> int:
    """Count total number of scalar parameters in a params pytree.

    :param Any params: Parameter pytree.
    :return int: Total number of scalar parameters.
    """

    leaves = jax.tree_util.tree_leaves(params)
    total = 0
    for x in leaves:
        if hasattr(x, "size"):
            total += int(x.size)
    return total


def tree_allclose(a: Any, b: Any, *, rtol: float = 1e-6, atol: float = 1e-6) -> bool:
    """Tree-wise allclose for arrays.

    :param Any a: First pytree.
    :param Any b: Second pytree.
    :param float rtol: Relative tolerance.
    :param float atol: Absolute tolerance.
    :return bool: True if all arrays are element-wise close.
    """

    la = jax.tree_util.tree_leaves(a)
    lb = jax.tree_util.tree_leaves(b)
    if len(la) != len(lb):
        return False
    for xa, xb in zip(la, lb, strict=True):
        if hasattr(xa, "shape") and hasattr(xb, "shape"):
            if xa.shape != xb.shape:
                return False
            if xa.dtype != xb.dtype:
                return False
            if not jnp.allclose(xa, xb, rtol=rtol, atol=atol):
                return False
        else:
            if xa != xb:
                return False
    return True


def abstractify_tree(tree: Any) -> Any:
    """Convert a pytree of arrays to ShapeDtypeStruct leaves.

    :param Any tree: Pytree of JAX arrays.
    :return Any: Pytree of ShapeDtypeStruct with matching structure.
    """

    def to_struct(x: jax.Array) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=getattr(x, "sharding", None))

    return jax.tree_util.tree_map(to_struct, tree)
