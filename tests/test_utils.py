"""Utility tests consolidated by module."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from chomp.config import Config, ModelConfig, TrainConfig
from chomp.model import build_model
from chomp.train import _check_finite_metrics
from chomp.utils.devices import validate_default_device
from chomp.utils.io import RunDirectoryLock, create_run_dir
from chomp.utils.tree import param_count


def test_cpu_fails_when_disallowed() -> None:
    """Running on CPU with allow_cpu=False must raise RuntimeError."""
    # This test is only meaningful on CPU-only environments.
    if jax.devices()[0].platform != "cpu":
        pytest.skip("Not running on CPU")

    with pytest.raises(RuntimeError):
        validate_default_device(allow_cpu=False)


def test_cpu_allowed_when_configured() -> None:
    """Running on CPU with allow_cpu=True should succeed."""
    validate_default_device(allow_cpu=True)


def test_non_gpu_accelerator_is_not_accepted_as_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """The production device gate must require a CUDA-backed GPU."""
    monkeypatch.setattr(jax, "devices", lambda: [SimpleNamespace(platform="tpu")])

    with pytest.raises(RuntimeError, match="required CUDA GPU backend"):
        validate_default_device(allow_cpu=False)


def test_resume_requires_an_existing_run_directory(tmp_path: Path) -> None:
    """Resume setup must not create a missing run directory."""
    run_dir = tmp_path / "missing-run"
    cfg = Config(logging=replace(Config().logging, run_dir=str(run_dir)))

    with pytest.raises(RuntimeError, match="does not exist"):
        create_run_dir(cfg, config_path=None, allow_existing=True)

    assert not run_dir.exists()


def test_run_directory_lock_rejects_concurrent_owner_and_releases(tmp_path: Path) -> None:
    """Only one process handle may own a run, and close should permit reacquisition."""
    run_dir = tmp_path / "run"
    first = RunDirectoryLock(run_dir)
    second = RunDirectoryLock(run_dir)

    with first, pytest.raises(RuntimeError, match="already active"):
        second.acquire()

    with second:
        assert second.path.exists()


def test_run_directory_lock_canonicalizes_symlinked_run(tmp_path: Path) -> None:
    """A symlink alias and its target must contend on the same run lock."""
    run_dir = tmp_path / "runs" / "job-1"
    run_dir.mkdir(parents=True)
    alias = run_dir.parent / "latest"
    alias.symlink_to(run_dir, target_is_directory=True)
    direct = RunDirectoryLock(run_dir)
    via_alias = RunDirectoryLock(alias)

    assert direct.path == via_alias.path
    with direct, pytest.raises(RuntimeError, match="already active"):
        via_alias.acquire()


def test_dummy_init_stats_are_sane() -> None:
    """Model parameters should be finite with positive variance."""
    cfg = Config(model=ModelConfig(backend="dummy", vocab_size=128, d_model=32, dropout=0.0))
    key = jax.random.PRNGKey(0)
    params, _static = build_model(cfg, key=key)

    leaves = [x for x in jax.tree_util.tree_leaves(params) if hasattr(x, "shape")]
    assert leaves, "Expected parameter leaves for dummy model."

    samples = leaves[: min(10, len(leaves))]
    for leaf in samples:
        arr = jnp.asarray(leaf, dtype=jnp.float32)
        std = float(jnp.std(arr))
        max_abs = float(jnp.max(jnp.abs(arr)))
        assert bool(jnp.all(jnp.isfinite(arr)))
        assert std > 0.0
        assert max_abs > 0.0


def test_dummy_param_count() -> None:
    """Dummy model param count should match expected formula."""
    cfg = Config(
        model=ModelConfig(backend="dummy", vocab_size=128, d_model=64, dropout=0.0),
        train=TrainConfig(allow_cpu=True),
    )
    key = jax.random.PRNGKey(0)
    params, static = build_model(cfg, key=key)
    n = param_count(params)
    expected = 2 * cfg.model.vocab_size * cfg.model.d_model
    assert n == expected


@pytest.mark.parametrize(
    ("metrics", "match"),
    [
        ({"loss": float("nan"), "grad_norm": 1.0, "lr": 1e-3}, "loss"),
        ({"loss": 1.0, "grad_norm": float("inf"), "lr": 1e-3}, "grad_norm"),
        ({"loss": 1.0, "grad_norm": 1.0, "lr": float("nan")}, "lr"),
    ],
)
def test_finite_check_rejects_nonfinite_metrics(metrics: dict[str, float], match: str) -> None:
    """Non-finite metrics should raise RuntimeError with the metric name."""
    with pytest.raises(RuntimeError, match=match):
        _check_finite_metrics(metrics, step=3)
