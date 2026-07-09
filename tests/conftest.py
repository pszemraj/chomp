"""Test session configuration."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import pytest

from chomp.utils import xla
from tests.helpers.hf_fakes import FakeHFIterable


def _pin_deterministic_gpu_ops() -> None:
    """Pin deterministic GPU kernels for bit-exact resume assertions."""
    if not xla._query_nvidia_gpu_names() or xla.deterministic_gpu_ops_setting() is not None:
        return
    flag = "--xla_gpu_deterministic_ops=true"
    os.environ["XLA_FLAGS"] = f"{os.environ.get('XLA_FLAGS', '')} {flag}".strip()


# Ensure XLA env quirks are applied before any JAX imports in tests.
# Deterministic GPU ops are pinned HERE ONLY (production does not set them —
# fast kernels are the default): the exact-resume tests assert atol=0 state
# equality to catch harness replay bugs, and without the flag XLA kernel
# choices depend on prior GPU state, adding low-order optimizer-state noise
# under full-suite ordering that has nothing to do with the harness.
xla.configure_blackwell_xla_env()
_pin_deterministic_gpu_ops()

# Tests should not rely on users exporting preallocation flags.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


@pytest.fixture
def patch_hf_load_dataset(monkeypatch: pytest.MonkeyPatch) -> Callable[..., dict[str, int]]:
    """Patch datasets.load_dataset to serve in-memory items via FakeHFIterable.

    The returned function accepts either a list of items (served for every
    split) or a mapping of split -> items; extra kwargs are forwarded to
    FakeHFIterable (on_shuffle, fail_at, record). It returns a dict whose
    "builds" key counts loader invocations.
    """
    import datasets

    def _patch(
        items: list[dict[str, Any]] | dict[str, list[dict[str, Any]]],
        **fake_kwargs: Any,
    ) -> dict[str, int]:
        calls = {"builds": 0}

        def _load_dataset(
            dataset: str, *, name: str, split: str, streaming: bool
        ) -> FakeHFIterable:
            _ = (dataset, name, streaming)
            calls["builds"] += 1
            data = items[split] if isinstance(items, dict) else items
            return FakeHFIterable(items=data, **fake_kwargs)

        monkeypatch.setattr(datasets, "load_dataset", _load_dataset)
        return calls

    return _patch
