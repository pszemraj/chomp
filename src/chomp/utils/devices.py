"""Device validation utilities.

Silent CPU fallback is one of the most expensive failures in JAX training:
- you think you're benchmarking the GPU
- but you're actually running on CPU and burning hours

So we fail fast unless explicitly allowed.
"""

from __future__ import annotations

import jax


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
