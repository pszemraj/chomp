"""Utility tests consolidated by module.

Note: global XLA/device environment setup happens in tests/conftest.py via
configure_blackwell_xla_env() and setting XLA_PYTHON_CLIENT_PREALLOCATE.
This file focuses on functional behavior rather than session bootstrapping.
"""

from __future__ import annotations

import logging
import os

import jax
import jax.numpy as jnp
import pytest
from _pytest.logging import LogCaptureFixture

from chomp.config import Config, ModelConfig, TrainConfig
from chomp.model import build_model
from chomp.train import _check_finite_metrics, _estimate_tokens_seen_increment, _init_tokens_seen
from chomp.types import Batch
from chomp.utils import devices, xla
from chomp.utils.devices import device_platform, validate_default_device
from chomp.utils.tree import param_count


def test_cpu_fails_when_disallowed() -> None:
    """Running on CPU with allow_cpu=False must raise RuntimeError."""
    # This test is only meaningful on CPU-only environments.
    if jax.devices()[0].platform != "cpu":
        pytest.skip("Not running on CPU")

    with pytest.raises(RuntimeError):
        validate_default_device(allow_cpu=False)


def test_cpu_allowed_when_configured() -> None:
    """Running on CPU with allow_cpu=True should succeed."""
    validate_default_device(allow_cpu=True)


def test_device_platform_detects_array() -> None:
    """device_platform should detect platform from JAX array."""
    arr = jax.numpy.zeros((1,))
    plat = device_platform(arr)
    assert isinstance(plat, str) and plat


def test_device_platform_handles_device_property() -> None:
    """device_platform should handle arrays with .device property."""

    class _Dev:
        """Mock device with platform attribute."""

        def __init__(self, platform: str) -> None:
            """Initialize mock device."""
            self.platform = platform

    class _Arr:
        """Mock array with device property."""

        def __init__(self) -> None:
            """Initialize mock array."""
            self.device = _Dev("gpu")

    assert device_platform(_Arr()) == "gpu"  # type: ignore[arg-type]


def test_device_platform_handles_device_method() -> None:
    """device_platform should handle arrays with .device() method."""

    class _Dev:
        """Mock device with platform attribute."""

        def __init__(self, platform: str) -> None:
            """Initialize mock device."""
            self.platform = platform

    class _Arr:
        """Mock array with device method."""

        def device(self) -> _Dev:
            """Return mock device.

            :return _Dev: Mock device.
            """
            return _Dev("cpu")

    assert device_platform(_Arr()) == "cpu"  # type: ignore[arg-type]


def test_device_platform_handles_device_buffer() -> None:
    """device_platform should handle arrays with device_buffer attribute."""

    class _Dev:
        """Mock device with platform attribute."""

        def __init__(self, platform: str) -> None:
            """Initialize mock device."""
            self.platform = platform

    class _Buffer:
        """Mock buffer with device method."""

        def device(self) -> _Dev:
            """Return mock device.

            :return _Dev: Mock device.
            """
            return _Dev("gpu")

    class _Arr:
        """Mock array with device_buffer attribute."""

        device_buffer = _Buffer()

    assert device_platform(_Arr()) == "gpu"  # type: ignore[arg-type]


def test_device_platform_returns_none_when_unknown() -> None:
    """device_platform should return None for unknown array types."""

    class _Arr:
        """Mock array without device info."""

    assert device_platform(_Arr()) is None  # type: ignore[arg-type]


def _make_batch() -> Batch:
    """Create a minimal Batch for testing.

    :return Batch: Batch with minimal shapes.
    """
    arr = jax.numpy.zeros((1, 1, 1), dtype=jax.numpy.int32)
    return Batch(
        input_ids=arr,
        labels=arr,
        attention_mask=arr.astype(bool),
        segment_ids=arr,
    )


def test_assert_batch_on_device_accepts_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """assert_batch_on_device should accept GPU batches."""
    batch = _make_batch()
    monkeypatch.setattr(devices, "device_platform", lambda _: "gpu")
    devices.assert_batch_on_device(batch, allow_cpu=False)


def test_assert_batch_on_device_rejects_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """assert_batch_on_device should reject CPU batches when disallowed."""
    batch = _make_batch()
    monkeypatch.setattr(devices, "device_platform", lambda _: "cpu")
    with pytest.raises(RuntimeError):
        devices.assert_batch_on_device(batch, allow_cpu=False)


def test_assert_batch_on_device_allows_unknown_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """assert_batch_on_device should allow unknown platform when allow_cpu=True."""
    batch = _make_batch()
    monkeypatch.setattr(devices, "device_platform", lambda _: None)
    devices.assert_batch_on_device(batch, allow_cpu=True)


def test_assert_batch_on_device_rejects_unknown_when_disallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """assert_batch_on_device should reject unknown platform when allow_cpu=False."""
    batch = _make_batch()
    monkeypatch.setattr(devices, "device_platform", lambda _: None)
    with pytest.raises(RuntimeError):
        devices.assert_batch_on_device(batch, allow_cpu=False)


def test_dummy_init_stats_are_sane() -> None:
    """Model parameters should be finite with positive variance."""
    cfg = Config(model=ModelConfig(backend="dummy", vocab_size=128, d_model=32, dropout=0.0))
    key = jax.random.PRNGKey(0)
    params, _static = build_model(cfg, key=key)

    leaves = [x for x in jax.tree_util.tree_leaves(params) if hasattr(x, "shape")]
    assert leaves, "Expected parameter leaves for dummy model."

    samples = leaves[: min(10, len(leaves))]
    for leaf in samples:
        arr = jnp.asarray(leaf, dtype=jnp.float32)
        std = float(jnp.std(arr))
        max_abs = float(jnp.max(jnp.abs(arr)))
        assert bool(jnp.all(jnp.isfinite(arr)))
        assert std > 0.0
        assert max_abs > 0.0


def test_dummy_param_count() -> None:
    """Dummy model param count should match expected formula."""
    cfg = Config(
        model=ModelConfig(backend="dummy", vocab_size=128, d_model=64, dropout=0.0),
        train=TrainConfig(allow_cpu=True),
    )
    key = jax.random.PRNGKey(0)
    params, static = build_model(cfg, key=key)
    n = param_count(params)
    expected = 2 * cfg.model.vocab_size * cfg.model.d_model
    assert n == expected


def test_init_tokens_seen_host_int() -> None:
    """Token counter should be a host-side Python int."""
    counter = _init_tokens_seen(123)
    assert isinstance(counter, int)
    assert counter == 123


def test_estimate_tokens_seen_increment_prefers_packing_stats() -> None:
    """Token estimates should use packing stats when available."""
    cfg = Config(train=TrainConfig(batch_size=2, grad_accum=3, seq_len=8, allow_cpu=True))
    sequences = int(cfg.train.batch_size) * int(cfg.train.grad_accum)
    packing_tokens = sequences * int(cfg.train.seq_len)

    inc_stats = _estimate_tokens_seen_increment(cfg, {"packing_tokens": packing_tokens})
    inc_default = _estimate_tokens_seen_increment(cfg, None)

    assert inc_stats == inc_default


def test_finite_check_rejects_nan_loss() -> None:
    """NaN loss should raise RuntimeError."""
    with pytest.raises(RuntimeError, match="loss"):
        _check_finite_metrics({"loss": float("nan"), "grad_norm": 1.0}, step=3)


def test_finite_check_rejects_inf_grad_norm() -> None:
    """Inf grad_norm should raise RuntimeError."""
    with pytest.raises(RuntimeError, match="grad_norm"):
        _check_finite_metrics({"loss": 1.0, "grad_norm": float("inf")}, step=3)


def test_configure_blackwell_sets_flags_and_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    """RTX 50xx detection should set XLA_FLAGS and warn on preallocate.

    :param pytest.MonkeyPatch monkeypatch: Pytest monkeypatch fixture.
    :param LogCaptureFixture caplog: Log capture fixture.
    """
    monkeypatch.setattr(xla, "_query_nvidia_gpu_names", lambda: ["NVIDIA GeForce RTX 5090"])
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=true --foo=bar")
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)

    caplog.set_level(logging.INFO)
    changed = xla.configure_blackwell_xla_env(force=True)

    assert changed is True
    flags = os.environ.get("XLA_FLAGS", "")
    assert "--xla_gpu_enable_triton_gemm=false" in flags
    assert "--xla_gpu_enable_triton_gemm=true" not in flags
    assert "--foo=bar" in flags
    assert any(
        rec.levelno >= logging.WARNING and "XLA_PYTHON_CLIENT_PREALLOCATE" in rec.message
        for rec in caplog.records
    )


def test_configure_blackwell_skips_non_blackwell(
    monkeypatch: pytest.MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    """Non-50xx GPUs should not modify XLA_FLAGS.

    :param pytest.MonkeyPatch monkeypatch: Pytest monkeypatch fixture.
    :param LogCaptureFixture caplog: Log capture fixture.
    """
    monkeypatch.setattr(xla, "_query_nvidia_gpu_names", lambda: ["NVIDIA GeForce RTX 4090"])
    monkeypatch.setenv("XLA_FLAGS", "--keep")

    caplog.set_level(logging.DEBUG)
    changed = xla.configure_blackwell_xla_env(force=True)

    assert changed is False
    assert os.environ.get("XLA_FLAGS") == "--keep"
