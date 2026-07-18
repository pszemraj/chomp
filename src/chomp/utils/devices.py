"""Device validation utilities.

Silent CPU fallback is one of the most expensive failures in JAX training:
- you think you're benchmarking the GPU
- but you're actually running on CPU and burning hours

So we fail fast unless explicitly allowed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax

if TYPE_CHECKING:
    from chomp.types import Batch


def validate_default_device(*, allow_cpu: bool) -> None:
    """Require JAX's default backend to be CUDA unless CPU debugging is allowed.

    :param bool allow_cpu: Whether a non-GPU backend is permitted for debugging.
    :raises RuntimeError: If JAX has no devices or the default device is not a GPU.
    """

    devs = jax.devices()
    if not devs:
        raise RuntimeError("JAX reports no devices. JAX installation is broken.")

    platform = devs[0].platform
    if platform != "gpu" and not allow_cpu:
        raise RuntimeError(
            f"JAX is using {platform!r}, not the required CUDA GPU backend, while "
            "train.allow_cpu=false. Ensure the CUDA device is visible. "
            "Set train.allow_cpu=true only for debugging."
        )


def device_platform(x: jax.Array) -> str | None:
    """Best-effort: return the device platform for an array.

    :param jax.Array x: JAX array to check.
    :return str | None: Platform name (e.g., "cpu", "gpu") or None if unknown.
    """
    # .device can raise on multi-device sharded arrays; callers handle None.
    try:
        return x.device.platform
    except Exception:
        return None


def assert_batch_on_device(batch: Batch, *, allow_cpu: bool) -> None:
    """Assert that a Batch's arrays are on GPU unless CPU allowed.

    :param Batch batch: Batch object to check.
    :param bool allow_cpu: If True, don't raise on CPU placement.
    :raises RuntimeError: If batch is on CPU and allow_cpu=False.
    """

    plat = device_platform(batch.input_ids)
    if plat is None:
        # Can't determine -> don't hard fail; but warn by raising if strict desired
        if not allow_cpu:
            raise RuntimeError(
                "Could not determine array device platform; refusing to proceed with allow_cpu=false."
            )
        return

    if plat != "gpu" and not allow_cpu:
        raise RuntimeError(
            f"Batch is on {plat!r}, not the required CUDA GPU backend, while train.allow_cpu=false."
        )
