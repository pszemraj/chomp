import jax
import pytest

from chomp.types import Batch
from chomp.utils import devices
from chomp.utils.devices import device_platform, validate_default_device


def test_cpu_fails_when_disallowed():
    # This test is only meaningful on CPU-only environments.
    if jax.devices()[0].platform != "cpu":
        pytest.skip("Not running on CPU")

    with pytest.raises(RuntimeError):
        validate_default_device(allow_cpu=False)


def test_cpu_allowed_when_configured():
    validate_default_device(allow_cpu=True)


def test_device_platform_detects_array():
    arr = jax.numpy.zeros((1,))
    plat = device_platform(arr)
    assert isinstance(plat, str) and plat


def test_device_platform_handles_device_property():
    class _Dev:
        def __init__(self, platform: str) -> None:
            self.platform = platform

    class _Arr:
        def __init__(self) -> None:
            self.device = _Dev("gpu")

    assert device_platform(_Arr()) == "gpu"  # type: ignore[arg-type]


def test_device_platform_handles_device_method():
    class _Dev:
        def __init__(self, platform: str) -> None:
            self.platform = platform

    class _Arr:
        def device(self) -> _Dev:
            return _Dev("cpu")

    assert device_platform(_Arr()) == "cpu"  # type: ignore[arg-type]


def test_device_platform_handles_device_buffer():
    class _Dev:
        def __init__(self, platform: str) -> None:
            self.platform = platform

    class _Buffer:
        def device(self) -> _Dev:
            return _Dev("gpu")

    class _Arr:
        device_buffer = _Buffer()

    assert device_platform(_Arr()) == "gpu"  # type: ignore[arg-type]


def test_device_platform_returns_none_when_unknown():
    class _Arr:
        pass

    assert device_platform(_Arr()) is None  # type: ignore[arg-type]


def _make_batch() -> Batch:
    arr = jax.numpy.zeros((1, 1, 1), dtype=jax.numpy.int32)
    return Batch(
        input_ids=arr,
        labels=arr,
        attention_mask=arr.astype(bool),
        segment_ids=arr,
    )


def test_assert_batch_on_device_accepts_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    batch = _make_batch()
    monkeypatch.setattr(devices, "device_platform", lambda _: "gpu")
    devices.assert_batch_on_device(batch, allow_cpu=False)


def test_assert_batch_on_device_rejects_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    batch = _make_batch()
    monkeypatch.setattr(devices, "device_platform", lambda _: "cpu")
    with pytest.raises(RuntimeError):
        devices.assert_batch_on_device(batch, allow_cpu=False)


def test_assert_batch_on_device_allows_unknown_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _make_batch()
    monkeypatch.setattr(devices, "device_platform", lambda _: None)
    devices.assert_batch_on_device(batch, allow_cpu=True)


def test_assert_batch_on_device_rejects_unknown_when_disallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _make_batch()
    monkeypatch.setattr(devices, "device_platform", lambda _: None)
    with pytest.raises(RuntimeError):
        devices.assert_batch_on_device(batch, allow_cpu=False)
