"""Checkpoint path and config resolution tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from chomp.config import Config
from chomp.utils.ckpt_paths import load_config_for_checkpoint, resolve_checkpoint_path


def test_resolve_checkpoint_with_run_dir(tmp_path: Path) -> None:
    """resolve_checkpoint_path should find the latest step beneath a run."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config_resolved.json").write_text(json.dumps(Config().to_dict(), indent=2))
    (run_dir / "checkpoints" / "2" / "train_state").mkdir(parents=True)

    step_dir, found_run_dir = resolve_checkpoint_path(str(run_dir))

    assert found_run_dir == run_dir
    assert step_dir.parent == run_dir / "checkpoints"
    assert step_dir.name == "2"
    assert (step_dir / "train_state").exists()


def test_resolve_checkpoint_with_step_dir(tmp_path: Path) -> None:
    """resolve_checkpoint_path should accept direct step-directory input."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config_resolved.json").write_text(json.dumps(Config().to_dict(), indent=2))
    step_dir_input = run_dir / "checkpoints" / "1"
    (step_dir_input / "train_state").mkdir(parents=True)

    step_dir, found_run_dir = resolve_checkpoint_path(str(step_dir_input))

    assert found_run_dir == run_dir
    assert step_dir == step_dir_input


def test_resolve_checkpoint_ignores_cwd_shadow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolver should ignore similarly named checkpoint dirs in CWD."""
    run_dir = tmp_path / "runs" / "my_run"
    run_dir.mkdir(parents=True)

    (run_dir / "config_resolved.json").write_text(json.dumps(Config().to_dict(), indent=2))

    correct_step_dir = run_dir / "checkpoints" / "100"
    (correct_step_dir / "train_state").mkdir(parents=True)

    shadow_dir = tmp_path / "checkpoints" / "999"
    shadow_dir.mkdir(parents=True)

    monkeypatch.chdir(tmp_path)

    found_step, found_run = resolve_checkpoint_path(str(run_dir))

    assert found_run == run_dir
    assert found_step == correct_step_dir
    assert "999" not in str(found_step)


def test_standalone_step_recovers_metadata_run_dir(tmp_path: Path) -> None:
    """A detached step should retain access to its run-pinned artifacts."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)

    run_cfg = Config()
    run_cfg = replace(run_cfg, model=replace(run_cfg.model, rope_base=10_000.0))
    (run_dir / "config_resolved.json").write_text(json.dumps(run_cfg.to_dict(), indent=2))

    checkpoint_cfg = replace(
        run_cfg,
        model=replace(run_cfg.model, rope_base=100_000.0),
        logging=replace(run_cfg.logging, run_dir=str(run_dir)),
    )

    step_dir = tmp_path / "external_ckpts" / "100"
    (step_dir / "train_state").mkdir(parents=True)
    meta_dir = step_dir / "meta"
    meta_dir.mkdir(parents=True)
    (meta_dir / "metadata").write_text(json.dumps({"config": checkpoint_cfg.to_dict()}, indent=2))

    found_step, found_run = resolve_checkpoint_path(str(step_dir))

    assert found_step == step_dir
    assert found_run == run_dir

    loaded = load_config_for_checkpoint(step_dir=step_dir, config_override=None)
    assert loaded.logging.run_dir == str(run_dir)
    assert loaded.model.rope_base == 100_000.0


def test_checkpoint_metadata_beats_run_start_config(tmp_path: Path) -> None:
    """Selected checkpoint semantics must override the immutable run config."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_cfg = replace(Config(), model=replace(Config().model, attention_window=64))
    checkpoint_cfg = replace(run_cfg, model=replace(run_cfg.model, attention_window=128))
    (run_dir / "config_resolved.json").write_text(json.dumps(run_cfg.to_dict()))

    step_dir = run_dir / "checkpoints" / "1"
    (step_dir / "train_state").mkdir(parents=True)
    (step_dir / "meta").mkdir()
    (step_dir / "meta" / "metadata").write_text(json.dumps({"config": checkpoint_cfg.to_dict()}))

    loaded = load_config_for_checkpoint(step_dir=step_dir, config_override=None)

    assert loaded.model.attention_window == 128


def test_metadata_free_checkpoint_refuses_the_run_start_config(tmp_path: Path) -> None:
    """A checkpoint without its own config must not borrow the run's starting one.

    ``config_resolved.json`` records what the run began with, which a later
    checkpoint may have long since diverged from. Silently substituting it
    would hand back a model whose static semantics disagree with the arrays.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_cfg = replace(Config(), model=replace(Config().model, rope_base=20_000.0))
    (run_dir / "config_resolved.json").write_text(json.dumps(run_cfg.to_dict()))
    step_dir = run_dir / "checkpoints" / "1"
    (step_dir / "train_state").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="no config metadata"):
        load_config_for_checkpoint(step_dir=step_dir, config_override=None)


def test_explicit_checkpoint_config_override_has_highest_priority(tmp_path: Path) -> None:
    """An explicit generation config should override checkpoint metadata."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_cfg = replace(Config(), model=replace(Config().model, rope_base=10_000.0))
    checkpoint_cfg = replace(run_cfg, model=replace(run_cfg.model, rope_base=20_000.0))
    (run_dir / "config_resolved.json").write_text(json.dumps(run_cfg.to_dict()))
    step_dir = run_dir / "checkpoints" / "1"
    (step_dir / "train_state").mkdir(parents=True)
    (step_dir / "meta").mkdir()
    (step_dir / "meta" / "metadata").write_text(json.dumps({"config": checkpoint_cfg.to_dict()}))
    override_path = tmp_path / "override.yaml"
    override_path.write_text("model:\n  rope_base: 30000.0\n")

    loaded = load_config_for_checkpoint(step_dir=step_dir, config_override=str(override_path))

    assert loaded.model.rope_base == 30_000.0
