"""Checkpoint path and config resolution tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from chomp.ckpt import default_ckpt_dir
from chomp.config import Config
from chomp.train import run
from chomp.utils.checkpoints import load_config_for_checkpoint, resolve_checkpoint_path


def test_resolve_checkpoint_with_root_dir(
    tmp_path: Path,
    small_run_cfg_factory: Callable[..., tuple[Config, Path]],
) -> None:
    """resolve_checkpoint_path should respect checkpoint.root_dir from run config."""
    cfg, config_src = small_run_cfg_factory(tmp_path)
    ckpt_root = tmp_path / "ckpt_root"
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, root_dir=str(ckpt_root)))

    run_dir = run(cfg, config_path=str(config_src), resume="none", dry_run=False)
    step_dir, found_run_dir = resolve_checkpoint_path(str(run_dir))

    assert found_run_dir == run_dir
    assert step_dir.parent == ckpt_root
    assert step_dir.name == "2"
    assert (step_dir / "train_state").exists()


def test_resolve_checkpoint_with_step_dir(
    tmp_path: Path,
    small_run_cfg_factory: Callable[..., tuple[Config, Path]],
) -> None:
    """resolve_checkpoint_path should accept direct step-directory input."""
    cfg, config_src = small_run_cfg_factory(tmp_path)
    run_dir = run(cfg, config_path=str(config_src), resume="none", dry_run=False)

    step_dir_input = default_ckpt_dir(run_dir) / "1"
    step_dir, found_run_dir = resolve_checkpoint_path(str(step_dir_input))

    assert found_run_dir == run_dir
    assert step_dir == step_dir_input


def test_resolve_checkpoint_root_dir_relative_to_run(tmp_path: Path) -> None:
    """Relative checkpoint.root_dir should resolve under the run directory."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    cfg = Config()
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, root_dir="relative_ckpts"))
    (run_dir / "config_resolved.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    step_dir = run_dir / "relative_ckpts" / "1"
    (step_dir / "train_state").mkdir(parents=True)

    found_step, found_run = resolve_checkpoint_path(str(run_dir))

    assert found_run == run_dir
    assert found_step == step_dir


def test_resolve_checkpoint_ignores_cwd_shadow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolver should ignore similarly named checkpoint dirs in CWD."""
    run_dir = tmp_path / "runs" / "my_run"
    run_dir.mkdir(parents=True)

    cfg = Config()
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, root_dir="ckpts"))
    (run_dir / "config_resolved.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    correct_step_dir = run_dir / "ckpts" / "100"
    (correct_step_dir / "train_state").mkdir(parents=True)

    shadow_dir = tmp_path / "ckpts" / "999"
    shadow_dir.mkdir(parents=True)

    monkeypatch.chdir(tmp_path)

    found_step, found_run = resolve_checkpoint_path(str(run_dir))

    assert found_run == run_dir
    assert found_step == correct_step_dir
    assert "999" not in str(found_step)


def test_resolve_step_dir_external_root_uses_meta(tmp_path: Path) -> None:
    """Step directories under external roots should infer run_dir from metadata."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)

    cfg = Config()
    cfg = replace(cfg, logging=replace(cfg.logging, run_dir=str(run_dir)))

    step_dir = tmp_path / "external_ckpts" / "100"
    (step_dir / "train_state").mkdir(parents=True)
    meta_dir = step_dir / "meta"
    meta_dir.mkdir(parents=True)
    (meta_dir / "metadata").write_text(json.dumps({"config": cfg.to_dict()}, indent=2))

    found_step, found_run = resolve_checkpoint_path(str(step_dir))

    assert found_step == step_dir
    assert found_run == run_dir

    loaded = load_config_for_checkpoint(step_dir=step_dir, run_dir=None, config_override=None)
    assert loaded.logging.run_dir == str(run_dir)


def test_run_dir_uses_resolved_config_for_ckpt_root(tmp_path: Path) -> None:
    """Run-dir resolution should use config_resolved.json checkpoint root, not override file."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)

    ckpt_root = tmp_path / "external_ckpts"
    (ckpt_root / "5" / "train_state").mkdir(parents=True)

    cfg = Config()
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, root_dir=str(ckpt_root)))
    (run_dir / "config_resolved.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    override_path = tmp_path / "override.yaml"
    override_path.write_text(
        """
model:
  backend: dummy
""".lstrip()
    )

    found_step, found_run = resolve_checkpoint_path(
        str(run_dir), config_override=str(override_path)
    )

    assert found_run == run_dir
    assert found_step.parent == ckpt_root
    assert found_step.name == "5"
