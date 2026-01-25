"""Tests for checkpoint path resolution."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from chomp.config import Config
from chomp.train import _build_checkpoint_manager


def test_checkpoint_root_dir_resolves_relative_to_run_dir(tmp_path: Path) -> None:
    """Relative checkpoint.root_dir should resolve against run_dir."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    cfg = Config()
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, root_dir="ckpts"))

    manager = _build_checkpoint_manager(cfg, run_dir)

    assert manager is not None
    assert Path(manager.directory) == (run_dir / "ckpts").resolve()
