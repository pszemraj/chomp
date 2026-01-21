"""Profiler glue.

This is intentionally tiny:
- if profiling is enabled, we start/stop a JAX trace
- we annotate training steps for easier trace navigation

If JAX profiling APIs change, this is the only file you should need to touch.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path

import jax


def start_trace(log_dir: str | Path) -> None:
    """Start a JAX trace (best-effort)."""

    try:
        jax.profiler.start_trace(str(log_dir))
    except Exception as e:  # pragma: no cover
        # We intentionally don't hard fail. Profiling is optional.
        print(f"[chomp] Warning: could not start JAX trace: {e}")


def stop_trace() -> None:
    """Stop a JAX trace (best-effort)."""
    with suppress(Exception):
        jax.profiler.stop_trace()


@contextmanager
def step_annotation(name: str) -> Iterator[None]:
    """Annotate a region as a step (best-effort).

    :param str name: Name for the step annotation.
    :return Iterator[None]: Context manager that yields None.
    """

    try:
        ctx = jax.profiler.StepTraceAnnotation(name)
    except Exception:
        ctx = None

    if ctx is None:
        yield
        return

    with ctx:
        yield
