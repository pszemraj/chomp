"""XLA env helper tests."""

from __future__ import annotations

import logging
import os

from chomp.utils import xla


def test_configure_blackwell_sets_flags_and_warns(monkeypatch, caplog) -> None:
    """RTX 50xx detection should set XLA_FLAGS and warn on preallocate."""
    monkeypatch.setattr(xla, "_query_nvidia_gpu_names", lambda: ["NVIDIA GeForce RTX 5090"])
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=true --foo=bar")
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)

    caplog.set_level(logging.INFO)
    changed = xla.configure_blackwell_xla_env()

    assert changed is True
    flags = os.environ.get("XLA_FLAGS", "")
    assert "--xla_gpu_enable_triton_gemm=false" in flags
    assert "--xla_gpu_enable_triton_gemm=true" not in flags
    assert "--foo=bar" in flags
    assert any(
        rec.levelno >= logging.WARNING and "XLA_PYTHON_CLIENT_PREALLOCATE" in rec.message
        for rec in caplog.records
    )


def test_configure_blackwell_skips_non_blackwell(monkeypatch, caplog) -> None:
    """Non-50xx GPUs should not modify XLA_FLAGS."""
    monkeypatch.setattr(xla, "_query_nvidia_gpu_names", lambda: ["NVIDIA GeForce RTX 4090"])
    monkeypatch.setenv("XLA_FLAGS", "--keep")

    caplog.set_level(logging.DEBUG)
    changed = xla.configure_blackwell_xla_env()

    assert changed is False
    assert os.environ.get("XLA_FLAGS") == "--keep"
