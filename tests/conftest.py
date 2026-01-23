"""Test session configuration."""

from __future__ import annotations

from chomp.utils.xla import configure_blackwell_xla_env

# Ensure XLA env quirks are applied before any JAX imports in tests.
configure_blackwell_xla_env()
