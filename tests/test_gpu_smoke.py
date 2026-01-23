"""GPU-specific smoke tests requiring a real GPU."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import pytest

from chomp.config import Config, validate_config
from chomp.train import run
from chomp.utils.devices import device_platform, validate_default_device


def _has_gpu() -> bool:
    """Check if a GPU device is available.

    :return bool: True if a GPU is present.
    """
    return any(dev.platform == "gpu" for dev in jax.devices())


@pytest.mark.skipif(not _has_gpu(), reason="GPU not available")
def test_device_platform_reports_gpu() -> None:
    """device_platform should report 'gpu' for GPU arrays."""
    arr = jax.device_put(jax.numpy.zeros((1,)))
    assert device_platform(arr) == "gpu"


@pytest.mark.skipif(not _has_gpu(), reason="GPU not available")
@pytest.mark.parametrize("device_put", [False, True])
def test_gpu_train_smoke(tmp_path: Path, device_put: bool) -> None:
    """Single training step should succeed on GPU.

    :param Path tmp_path: Temporary directory for run output.
    :param bool device_put: Whether iterator device_put is enabled.
    """
    cfg = Config()
    cfg = replace(
        cfg,
        model=replace(
            cfg.model,
            backend="dummy",
            vocab_size=256,
            d_model=32,
            dropout=0.0,
        ),
        data=replace(
            cfg.data,
            backend="local_text",
            local_text="hello from gpu",
            repeat=True,
            max_eval_samples=4,
            packing_mode="sequential",
            device_put=device_put,
        ),
        train=replace(
            cfg.train,
            steps=1,
            batch_size=1,
            seq_len=32,
            grad_accum=1,
            allow_cpu=False,
            log_every=1,
            eval_every=0,
            jit=False,
            deterministic=True,
        ),
        optim=replace(
            cfg.optim,
            lr=1e-3,
            warmup_steps=0,
            min_lr_ratio=0.0,
        ),
        checkpoint=replace(cfg.checkpoint, enabled=False),
        logging=replace(
            cfg.logging,
            run_dir=str(tmp_path / f"run_{int(device_put)}"),
            wandb=replace(cfg.logging.wandb, enabled=False),
        ),
        debug=replace(cfg.debug, check_device_every=1),
    )
    validate_config(cfg)
    validate_default_device(allow_cpu=cfg.train.allow_cpu)

    run_dir = run(cfg)
    metrics_path = run_dir / cfg.logging.metrics_file
    assert metrics_path.exists()
    assert metrics_path.read_text().strip()
