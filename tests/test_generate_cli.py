"""Tests for the generate CLI subcommand."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import pytest
from click.testing import CliRunner

from chomp.ckpt import default_ckpt_dir
from chomp.cli.generate import _find_checkpoint_dir, _restore_params
from chomp.config import Config, load_config
from chomp.model import build_model
from chomp.train import run


def _small_cfg(tmp_path: Path) -> tuple[Config, Path]:
    """Return a tiny local_text config for fast tests.

    :param Path tmp_path: Temporary directory provided by pytest.
    :return tuple[Config, Path]: (config, config_path) for the smoke run.
    """
    config_src = Path(__file__).resolve().parents[1] / "configs" / "debug_smoke.yaml"
    cfg = load_config(str(config_src))

    cfg = replace(
        cfg,
        train=replace(
            cfg.train,
            steps=2,
            batch_size=1,
            seq_len=16,
            grad_accum=1,
            jit=False,
            deterministic=True,
            allow_cpu=True,
            log_every=1,
            eval_every=0,
            generate_every=0,
        ),
        data=replace(
            cfg.data,
            backend="local_text",
            repeat=True,
            packing_mode="sequential",
            packing_buffer_docs=4,
            grain_prefetch=0,
            local_text="hello from chomp test generate",
        ),
        checkpoint=replace(
            cfg.checkpoint,
            enabled=True,
            save_every=1,
            max_to_keep=2,
            async_save=False,
        ),
        optim=replace(
            cfg.optim,
            warmup_steps=0,
        ),
        logging=replace(
            cfg.logging,
            run_dir=str(tmp_path / "run"),
            console_use_rich=False,
        ),
        debug=replace(
            cfg.debug,
            nan_check=True,
            check_device_every=0,
        ),
    )
    return cfg, config_src


def _abstractify_tree(tree: Any) -> Any:
    """Convert array leaves to ShapeDtypeStruct for Orbax restores.

    :param Any tree: Pytree of JAX arrays.
    :return Any: Pytree of ShapeDtypeStruct with matching structure.
    """
    return jax.tree_util.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), tree)


def test_find_checkpoint_dir_with_run_dir(tmp_path: Path) -> None:
    """_find_checkpoint_dir finds latest checkpoint from run directory."""
    cfg, config_src = _small_cfg(tmp_path)
    run_dir = run(cfg, config_path=str(config_src), resume="none", dry_run=False)

    step_dir, found_run_dir = _find_checkpoint_dir(str(run_dir))

    assert found_run_dir == run_dir
    assert step_dir.name == "2"  # latest step
    assert (step_dir / "train_state").exists()


def test_find_checkpoint_dir_with_root_dir(tmp_path: Path) -> None:
    """_find_checkpoint_dir respects checkpoint.root_dir when given run_dir."""
    cfg, config_src = _small_cfg(tmp_path)
    ckpt_root = tmp_path / "ckpt_root"
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, root_dir=str(ckpt_root)))
    run_dir = run(cfg, config_path=str(config_src), resume="none", dry_run=False)

    step_dir, found_run_dir = _find_checkpoint_dir(str(run_dir))

    assert found_run_dir == run_dir
    assert step_dir.parent == ckpt_root
    assert step_dir.name == "2"
    assert (step_dir / "train_state").exists()


def test_find_checkpoint_dir_with_step_dir(tmp_path: Path) -> None:
    """_find_checkpoint_dir accepts direct step directory."""
    cfg, config_src = _small_cfg(tmp_path)
    run_dir = run(cfg, config_path=str(config_src), resume="none", dry_run=False)

    ckpt_dir = default_ckpt_dir(run_dir)
    step_dir_input = ckpt_dir / "1"

    step_dir, found_run_dir = _find_checkpoint_dir(str(step_dir_input))

    assert found_run_dir == run_dir
    assert step_dir == step_dir_input


def test_restore_params_partial_restore(tmp_path: Path) -> None:
    """_restore_params loads only params from a full TrainState checkpoint.

    Regression test: ensure partial restore works when checkpoint contains
    step, params, opt_state, and rng but we only need params.
    """
    cfg, config_src = _small_cfg(tmp_path)
    run_dir = run(cfg, config_path=str(config_src), resume="none", dry_run=False)

    # Build model to get abstract params shape
    params, _static = build_model(cfg, key=jax.random.PRNGKey(0))
    abstract_params = _abstractify_tree(params)

    # Find checkpoint
    ckpt_dir = default_ckpt_dir(run_dir)
    step_dir = ckpt_dir / "2"
    assert step_dir.exists()

    # Restore params only (the bug was here - structure mismatch with partial_restore)
    restored_params = _restore_params(step_dir, abstract_params)

    # Verify structure matches
    params_leaves = jax.tree_util.tree_leaves(params)
    restored_leaves = jax.tree_util.tree_leaves(restored_params)
    assert len(params_leaves) == len(restored_leaves)

    # Verify shapes match
    for orig, restored in zip(params_leaves, restored_leaves, strict=True):
        assert orig.shape == restored.shape
        assert orig.dtype == restored.dtype


def test_restore_params_values_differ_from_init(tmp_path: Path) -> None:
    """Restored params should differ from freshly initialized params.

    Ensures we're actually loading trained weights, not just matching shapes.
    """
    cfg, config_src = _small_cfg(tmp_path)
    run_dir = run(cfg, config_path=str(config_src), resume="none", dry_run=False)

    # Build fresh model
    params_fresh, _static = build_model(cfg, key=jax.random.PRNGKey(0))
    abstract_params = _abstractify_tree(params_fresh)

    # Restore from checkpoint
    ckpt_dir = default_ckpt_dir(run_dir)
    step_dir = ckpt_dir / "2"
    restored_params = _restore_params(step_dir, abstract_params)

    # At least some params should differ after training
    fresh_leaves = jax.tree_util.tree_leaves(params_fresh)
    restored_leaves = jax.tree_util.tree_leaves(restored_params)

    any_differ = False
    for fresh, restored in zip(fresh_leaves, restored_leaves, strict=True):
        if not jax.numpy.allclose(fresh, restored):
            any_differ = True
            break

    assert any_differ, "Restored params should differ from fresh init after training"


@pytest.mark.skipif(
    os.environ.get("JAX_PLATFORMS") == "cpu",
    reason="Generate requires megalodon_jax.generate which needs full model",
)
def test_generate_cli_produces_output(tmp_path: Path) -> None:
    """End-to-end test of the generate CLI command."""
    from chomp.cli import cli

    cfg, config_src = _small_cfg(tmp_path)
    run_dir = run(cfg, config_path=str(config_src), resume="none", dry_run=False)

    # Write config_resolved.json (normally done by train.run)
    config_resolved = run_dir / "config_resolved.json"
    if not config_resolved.exists():
        with open(config_resolved, "w") as f:
            json.dump(cfg.to_dict(), f)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "generate",
            str(run_dir),
            "--prompt",
            "hello",
            "--max-tokens",
            "5",
            "--temperature",
            "0",
            "--seed",
            "42",
        ],
    )

    # Should complete without error (exit code 0)
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert "Generated:" in result.output
