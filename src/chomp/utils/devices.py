"""Device validation utilities.

Silent CPU fallback is one of the most expensive failures in JAX training:
- you think you're benchmarking the GPU
- but you're actually running on CPU and burning hours

So we fail fast unless explicitly allowed.
"""

from __future__ import annotations

import jax


def validate_default_device(*, allow_cpu: bool) -> None:
    """Fail fast if JAX is running on CPU (unless explicitly allowed)."""

    devs = jax.devices()
    if not devs:
        raise RuntimeError("JAX reports no devices. JAX installation is broken.")

    platform = devs[0].platform
    if platform == "cpu" and not allow_cpu:
        raise RuntimeError(
            "JAX is using CPU backend but train.allow_cpu=false. "
            "Install a CUDA-enabled jaxlib and ensure CUDA is visible. "
            "Set train.allow_cpu=true only for debugging."
        )


def device_platform(x) -> str | None:
    """Best-effort: return the device platform for an array."""

    # Newer JAX: x.device() -> Device
    try:
        return x.device().platform  # type: ignore[attr-defined]
    except Exception:
        pass

    # Older JAX: x.device_buffer.device()
    try:
        return x.device_buffer.device().platform  # type: ignore[attr-defined]
    except Exception:
        return None


def assert_batch_on_device(batch, *, allow_cpu: bool) -> None:
    """Assert that a Batch's arrays are on GPU unless CPU allowed."""

    plat = device_platform(batch.input_ids)
    if plat is None:
        # Can't determine -> don't hard fail; but warn by raising if strict desired
        if not allow_cpu:
            raise RuntimeError(
                "Could not determine array device platform; refusing to proceed with allow_cpu=false."
            )
        return

    if plat == "cpu" and not allow_cpu:
        raise RuntimeError(
            "Batch appears to be on CPU but train.allow_cpu=false. "
            "This usually means you don't have CUDA-enabled jaxlib installed."
        )
