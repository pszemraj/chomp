"""Device placement and platform detection tests."""

import jax
import pytest

from chomp.types import Batch
from chomp.utils import devices
from chomp.utils.devices import device_platform, validate_default_device


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
