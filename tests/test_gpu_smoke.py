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

from tests.helpers.gpu import query_nvidia_gpu_names

_REPO_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    not query_nvidia_gpu_names(), reason="No NVIDIA GPU visible to nvidia-smi"
)


def _run_on_gpu(code: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run Python code in a subprocess whose JAX backend selection is untouched.

    :param str code: Python source to execute.
    :param Path cwd: Working directory for the subprocess.
    :return subprocess.CompletedProcess[str]: Completed process with captured output.
    """
    env = os.environ.copy()
    env.pop("JAX_PLATFORMS", None)
    env["PYTHONPATH"] = os.pathsep.join((str(_REPO_ROOT), str(_REPO_ROOT / "src")))
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_gpu_train_smoke(tmp_path: Path) -> None:
    """Single dummy-model training step should succeed on GPU."""
    run_dir = tmp_path / "run"
    code = (
        "from tests.helpers.gpu import run_training_smoke; "
        f"run_training_smoke({str(run_dir)!r}, backend='dummy')"
    )
    p = _run_on_gpu(code, cwd=tmp_path)
    assert p.returncode == 0, f"stdout:\n{p.stdout}\nstderr:\n{p.stderr}"
    assert "GPU_SMOKE_OK" in p.stdout


def test_megalodon_gpu_train_smoke(tmp_path: Path) -> None:
    """Published Megalodon should train through the packed BF16/JIT path on GPU."""
    run_dir = tmp_path / "megalodon_run"
    code = (
        "from tests.helpers.gpu import run_training_smoke; "
        f"run_training_smoke({str(run_dir)!r}, backend='megalodon')"
    )
    p = _run_on_gpu(code, cwd=tmp_path)
    assert p.returncode == 0, f"stdout:\n{p.stdout}\nstderr:\n{p.stderr}"
    assert "MEGALODON_GPU_SMOKE_OK" in p.stdout


def test_megalodon_gpu_export_keeps_parameters_off_the_device(tmp_path: Path) -> None:
    """Exporting must not allocate a second copy of the weights on the GPU."""
    run_dir = tmp_path / "export_run"
    code = (
        "from tests.helpers.gpu import run_export_placement_smoke; "
        f"run_export_placement_smoke({str(run_dir)!r})"
    )
    p = _run_on_gpu(code, cwd=tmp_path)
    assert p.returncode == 0, f"stdout:\n{p.stdout}\nstderr:\n{p.stderr}"
    assert "MEGALODON_GPU_EXPORT_OK" in p.stdout


def test_megalodon_gpu_policy_dtype_export_is_inference_equivalent(tmp_path: Path) -> None:
    """The policy-dtype export must compute the identical thing on real hardware."""
    run_dir = tmp_path / "policy_dtype_run"
    code = (
        "from tests.helpers.gpu import run_policy_dtype_equivalence_smoke; "
        f"run_policy_dtype_equivalence_smoke({str(run_dir)!r})"
    )
    p = _run_on_gpu(code, cwd=tmp_path)
    assert p.returncode == 0, f"stdout:\n{p.stdout}\nstderr:\n{p.stderr}"
    assert "MEGALODON_GPU_POLICY_DTYPE_OK" in p.stdout
