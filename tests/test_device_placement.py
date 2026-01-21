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
