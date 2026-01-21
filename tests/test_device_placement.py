import jax
import pytest

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
