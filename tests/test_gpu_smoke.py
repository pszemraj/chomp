from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import pytest

from chomp.config import Config, validate_config
from chomp.train import run
from chomp.utils.devices import device_platform, validate_default_device


def _has_gpu() -> bool:
    return any(dev.platform == "gpu" for dev in jax.devices())


@pytest.mark.skipif(not _has_gpu(), reason="GPU not available")
def test_device_platform_reports_gpu() -> None:
    arr = jax.device_put(jax.numpy.zeros((1,)))
    assert device_platform(arr) == "gpu"


@pytest.mark.skipif(not _has_gpu(), reason="GPU not available")
@pytest.mark.parametrize("device_put", [False, True])
def test_gpu_train_smoke(tmp_path: Path, device_put: bool) -> None:
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
            total_steps=1,
        ),
        checkpoint=replace(cfg.checkpoint, enabled=False),
        logging=replace(
            cfg.logging,
            run_dir=str(tmp_path / f"run_{int(device_put)}"),
            wandb_enabled=False,
        ),
        debug=replace(cfg.debug, check_device_every=1),
    )
    validate_config(cfg)
    validate_default_device(allow_cpu=cfg.train.allow_cpu)

    run_dir = run(cfg)
    metrics_path = run_dir / cfg.logging.metrics_file
    assert metrics_path.exists()
    assert metrics_path.read_text().strip()
