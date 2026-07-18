"""GPU-specific smoke tests requiring a real GPU.

These run chomp in a SUBPROCESS with an explicit environment: other test
modules pin JAX_PLATFORMS=cpu at import time and JAX reads that variable
exactly once at backend init, so an in-process probe would report CPU even
on a GPU host (and these tests would silently never run). The gate is
nvidia-smi (no JAX import): if the host has a visible NVIDIA GPU, the tests
run and genuinely fail when JAX cannot use it — that is the coverage they
exist to provide.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from chomp.utils.xla import _query_nvidia_gpu_names

_REPO_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    not _query_nvidia_gpu_names(), reason="No NVIDIA GPU visible to nvidia-smi"
)


def _run_on_gpu(code: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run Python code in a subprocess whose JAX backend selection is untouched.

    :param str code: Python source to execute.
    :param Path cwd: Working directory for the subprocess.
    :return subprocess.CompletedProcess[str]: Completed process with captured output.
    """
    env = os.environ.copy()
    env.pop("JAX_PLATFORMS", None)
    env["PYTHONPATH"] = str(_REPO_ROOT / "src")
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_device_platform_reports_gpu(tmp_path: Path) -> None:
    """device_platform should report 'gpu' for arrays on a GPU backend."""
    code = (
        "import jax\n"
        "from chomp.utils.devices import device_platform\n"
        "arr = jax.device_put(jax.numpy.zeros((1,)))\n"
        "assert device_platform(arr) == 'gpu', device_platform(arr)\n"
    )
    p = _run_on_gpu(code, cwd=tmp_path)
    assert p.returncode == 0, f"stdout:\n{p.stdout}\nstderr:\n{p.stderr}"


@pytest.mark.parametrize("device_put", [False, True])
def test_gpu_train_smoke(tmp_path: Path, device_put: bool) -> None:
    """Single training step should succeed on GPU with device placement asserted.

    :param Path tmp_path: Temporary directory for run output.
    :param bool device_put: Whether iterator device_put is enabled.
    """
    run_dir = tmp_path / f"run_{int(device_put)}"
    code = f"""
from dataclasses import replace

from chomp.config import Config, validate_config
from chomp.train import run
from chomp.utils.devices import validate_default_device

cfg = Config()
cfg = replace(
    cfg,
    model=replace(cfg.model, backend="dummy", vocab_size=256, d_model=32, dropout=0.0),
    data=replace(
        cfg.data,
        backend="local_text",
        local_text="hello from gpu",
        repeat=True,
        max_eval_samples=4,
        packing_mode="sequential",
        device_put={device_put!r},
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
    optim=replace(cfg.optim, lr=1e-3, warmup_steps=0, min_lr_ratio=0.0),
    checkpoint=replace(cfg.checkpoint, enabled=False),
    logging=replace(
        cfg.logging,
        run_dir={str(run_dir)!r},
        wandb=replace(cfg.logging.wandb, enabled=False),
    ),
    debug=replace(cfg.debug, check_device_every=1),
)
validate_config(cfg)
validate_default_device(allow_cpu=False)
out_dir = run(cfg)
metrics_path = out_dir / cfg.logging.metrics_file
assert metrics_path.exists() and metrics_path.read_text().strip()
print("GPU_SMOKE_OK")
"""
    p = _run_on_gpu(code, cwd=tmp_path)
    assert p.returncode == 0, f"stdout:\n{p.stdout}\nstderr:\n{p.stderr}"
    assert "GPU_SMOKE_OK" in p.stdout


def test_megalodon_gpu_train_smoke(tmp_path: Path) -> None:
    """Published Megalodon should train through the packed BF16/JIT path on GPU."""
    run_dir = tmp_path / "megalodon_run"
    code = f"""
from dataclasses import replace

from chomp.config import Config, validate_config
from chomp.train import run
from chomp.utils.devices import validate_default_device

cfg = Config()
cfg = replace(
    cfg,
    model=replace(
        cfg.model,
        backend="megalodon",
        vocab_size=256,
        model_dim=32,
        num_layers=1,
        num_heads=1,
        z_dim=16,
        value_dim=32,
        ffn_hidden_dim=64,
        cema_ndim=4,
        chunk_size=8,
        norm_num_groups=4,
        dropout=0.0,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        use_checkpoint=True,
        compute_dtype="bfloat16",
        loss_chunk_size=7,
    ),
    data=replace(
        cfg.data,
        backend="local_text",
        local_text="real megalodon gpu smoke path with packed documents",
        repeat=True,
        max_eval_samples=0,
        packing_mode="bin",
        packing_buffer_docs=4,
        packing_strict_segments=True,
        mask_boundary_loss=True,
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
        jit=True,
        deterministic=False,
    ),
    optim=replace(cfg.optim, lr=1e-3, warmup_steps=0, min_lr_ratio=0.0),
    checkpoint=replace(cfg.checkpoint, enabled=False),
    logging=replace(
        cfg.logging,
        run_dir={str(run_dir)!r},
        wandb=replace(cfg.logging.wandb, enabled=False),
    ),
    debug=replace(cfg.debug, check_device_every=1),
)
validate_config(cfg)
validate_default_device(allow_cpu=False)
out_dir = run(cfg)
metrics_path = out_dir / cfg.logging.metrics_file
assert metrics_path.exists() and metrics_path.read_text().strip()
print("MEGALODON_GPU_SMOKE_OK")
"""
    p = _run_on_gpu(code, cwd=tmp_path)
    assert p.returncode == 0, f"stdout:\n{p.stdout}\nstderr:\n{p.stderr}"
    assert "MEGALODON_GPU_SMOKE_OK" in p.stdout
