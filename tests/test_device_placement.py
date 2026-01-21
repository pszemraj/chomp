import pytest

import jax

from chomp.utils.devices import validate_default_device


def test_cpu_fails_when_disallowed():
    # This test is only meaningful on CPU-only environments.
    if jax.devices()[0].platform != "cpu":
        pytest.skip("Not running on CPU")

    with pytest.raises(RuntimeError):
        validate_default_device(allow_cpu=False)


def test_cpu_allowed_when_configured():
    validate_default_device(allow_cpu=True)
