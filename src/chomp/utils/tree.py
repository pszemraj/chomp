"""Pytree helper functions.

This is where you put the small utilities that you end up rewriting in every
JAX training repo:
- parameter counts
- tree equality / closeness checks
- shape/dtype summaries

Keep it minimal: this is not a generic library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import jax
import jax.numpy as jnp


def param_count(params: Any) -> int:
    """Count total number of scalar parameters in a params pytree."""

    leaves = jax.tree_util.tree_leaves(params)
    total = 0
    for x in leaves:
        if hasattr(x, "size"):
            total += int(x.size)
    return total


def tree_allclose(a: Any, b: Any, *, rtol: float = 1e-6, atol: float = 1e-6) -> bool:
    """Tree-wise allclose for arrays."""

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


def tree_equal(a: Any, b: Any) -> bool:
    """Tree-wise exact equality for arrays."""

    la = jax.tree_util.tree_leaves(a)
    lb = jax.tree_util.tree_leaves(b)
    if len(la) != len(lb):
        return False
    for xa, xb in zip(la, lb, strict=True):
        if hasattr(xa, "shape") and hasattr(xb, "shape"):
            if xa.shape != xb.shape or xa.dtype != xb.dtype:
                return False
            if not jnp.array_equal(xa, xb):
                return False
        else:
            if xa != xb:
                return False
    return True


@dataclass(frozen=True)
class TensorStats:
    shape: tuple[int, ...]
    dtype: str
    mean: float
    std: float
    min: float
    max: float


def sample_tensor_stats(params: Any, *, max_tensors: int = 8) -> list[TensorStats]:
    """Sample a few tensors from the params tree and compute simple stats.

    This catches obvious broken initialization (all-zeros, NaNs, infs) early.
    """

    leaves = [x for x in jax.tree_util.tree_leaves(params) if hasattr(x, "shape")]
    out: list[TensorStats] = []
    for x in leaves[:max_tensors]:
        xf = x.astype(jnp.float32)
        out.append(
            TensorStats(
                shape=tuple(x.shape),
                dtype=str(x.dtype),
                mean=float(jnp.mean(xf)),
                std=float(jnp.std(xf)),
                min=float(jnp.min(xf)),
                max=float(jnp.max(xf)),
            )
        )
    return out
