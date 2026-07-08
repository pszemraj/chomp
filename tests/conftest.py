"""Test session configuration."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import pytest

from chomp.utils.xla import configure_blackwell_xla_env
from tests.helpers.hf_fakes import FakeHFIterable

# Ensure XLA env quirks are applied before any JAX imports in tests.
configure_blackwell_xla_env()

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
