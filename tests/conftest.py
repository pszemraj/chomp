"""Test session configuration."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest

from chomp.config import Config
from chomp.utils.xla import configure_blackwell_xla_env
from tests.helpers.config_factories import make_small_run_cfg
from tests.helpers.hf_fakes import FakeHFIterable, FakeHFStateIterable

# Ensure XLA env quirks are applied before any JAX imports in tests.
configure_blackwell_xla_env()

# Tests should not rely on users exporting preallocation flags.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


@pytest.fixture
def small_run_cfg_factory() -> Callable[[Path], tuple[Config, Path]]:
    """Expose the shared small-run config factory."""
    return make_small_run_cfg


@pytest.fixture
def small_run_cfg(tmp_path: Path) -> tuple[Config, Path]:
    """Provide a smoke-sized run config tuple for tests."""
    return make_small_run_cfg(tmp_path)


@pytest.fixture
def hf_iterable_cls() -> type[FakeHFIterable]:
    """Expose reusable HF iterable fakes."""
    return FakeHFIterable


@pytest.fixture
def hf_state_iterable_cls() -> type[FakeHFStateIterable]:
    """Expose stateful HF iterable fakes."""
    return FakeHFStateIterable
