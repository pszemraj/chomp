"""Runtime pytree helpers for model sizing and checkpoint restoration."""

from __future__ import annotations

from typing import Any

import jax


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


def abstractify_tree(tree: Any) -> Any:
    """Convert a pytree of arrays to ShapeDtypeStruct leaves.

    :param Any tree: Pytree of JAX arrays.
    :return Any: Pytree of ShapeDtypeStruct with matching structure.
    """

    def to_struct(x: jax.Array) -> jax.ShapeDtypeStruct:
        """Convert a leaf array to a ShapeDtypeStruct.

        :param jax.Array x: Leaf array.
        :return jax.ShapeDtypeStruct: Shape/dtype/sharding descriptor.
        """
        return jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=getattr(x, "sharding", None))

    return jax.tree_util.tree_map(to_struct, tree)
