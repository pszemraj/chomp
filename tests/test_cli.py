"""CLI tests consolidated by module."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import click
import jax
import pytest
from click.testing import CliRunner

from chomp.ckpt import default_ckpt_dir
from chomp.cli import cli
from chomp.cli.generate import _restore_params
from chomp.cli.main import BANNER, parse_resume, print_banner
from chomp.config import Config
from chomp.model import build_model
from chomp.train import run
from chomp.utils.checkpoints import resolve_checkpoint_path
from chomp.utils.tree import abstractify_tree
from tests.helpers.config_factories import make_small_run_cfg


@pytest.fixture(scope="module")
def trained_small_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Config, Path]:
    """Train one shared tiny run for CLI restore-path tests."""
    tmp_path = tmp_path_factory.mktemp("cli_small_run")
    cfg, config_src = make_small_run_cfg(
        tmp_path,
        local_text="hello from chomp test generate",
    )
    run_dir = run(cfg, config_path=str(config_src), resume="none", dry_run=False)
    return cfg, run_dir


def test_print_banner_outputs_expected_text(capsys: object) -> None:
    """print_banner emits the banner once with a trailing newline."""
    print_banner()
    captured = capsys.readouterr()

    assert captured.out.endswith("\n")
    assert captured.out.rstrip("\n") == BANNER


def test_parse_resume_accepts_valid_variants() -> None:
    """parse_resume should normalize valid alias and numeric variants."""
    for raw, expected in [
        ("none", "none"),
        ("no", "none"),
        ("false", "none"),
        ("0", "none"),
        ("  NONE  ", "none"),
        ("latest", "latest"),
        ("last", "latest"),
        ("  LATEST  ", "latest"),
        ("100", 100),
        ("5000", 5000),
        ("  42  ", 42),
    ]:
        assert parse_resume(raw) == expected


def test_parse_resume_rejects_negative_step() -> None:
    """parse_resume should reject negative step numbers."""
    for raw in ["-1", "-100"]:
        with pytest.raises(click.BadParameter, match="non-negative"):
            parse_resume(raw)


def test_parse_resume_rejects_invalid_strings() -> None:
    """parse_resume should reject unparseable resume values."""
    for raw in ["invalid", "step100"]:
        with pytest.raises(click.BadParameter, match="Invalid resume value"):
            parse_resume(raw)


def test_resolve_checkpoint_with_run_dir(trained_small_run: tuple[Config, Path]) -> None:
    """CLI integration smoke: run-dir input should resolve latest checkpoint."""
    _, run_dir = trained_small_run

    step_dir, found_run_dir = resolve_checkpoint_path(str(run_dir))

    assert found_run_dir == run_dir
    assert step_dir.name == "2"
    assert (step_dir / "train_state").exists()


def test_generate_rejects_non_megalodon_backend(tmp_path: Path) -> None:
    """generate should fail fast when model.backend is not megalodon."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    cfg = Config()
    cfg = replace(cfg, model=replace(cfg.model, backend="dummy"))
    (run_dir / "config_resolved.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    step_dir = run_dir / "checkpoints" / "1" / "train_state"
    step_dir.mkdir(parents=True)

    runner = CliRunner()
    result = runner.invoke(cli, ["generate", str(run_dir), "--prompt", "hello"])

    assert result.exit_code != 0
    assert "model.backend" in result.output


def test_restore_params_partial_restore(trained_small_run: tuple[Config, Path]) -> None:
    """_restore_params should load only params from a full TrainState checkpoint."""
    cfg, run_dir = trained_small_run

    params, _static = build_model(cfg, key=jax.random.PRNGKey(0))
    abstract_params = abstractify_tree(params)

    step_dir = default_ckpt_dir(run_dir) / "2"
    assert step_dir.exists()

    restored_params = _restore_params(step_dir, abstract_params)

    params_leaves = jax.tree_util.tree_leaves(params)
    restored_leaves = jax.tree_util.tree_leaves(restored_params)
    assert len(params_leaves) == len(restored_leaves)

    for orig, restored in zip(params_leaves, restored_leaves, strict=True):
        assert orig.shape == restored.shape
        assert orig.dtype == restored.dtype


def test_restore_params_values_differ_from_init(trained_small_run: tuple[Config, Path]) -> None:
    """Restored params should differ from fresh init after training."""
    cfg, run_dir = trained_small_run

    params_fresh, _static = build_model(cfg, key=jax.random.PRNGKey(0))
    abstract_params = abstractify_tree(params_fresh)

    step_dir = default_ckpt_dir(run_dir) / "2"
    restored_params = _restore_params(step_dir, abstract_params)

    fresh_leaves = jax.tree_util.tree_leaves(params_fresh)
    restored_leaves = jax.tree_util.tree_leaves(restored_params)

    assert any(
        not jax.numpy.allclose(fresh, restored)
        for fresh, restored in zip(fresh_leaves, restored_leaves, strict=True)
    ), "Restored params should differ from fresh init after training"


@pytest.mark.skipif(
    os.environ.get("JAX_PLATFORMS") == "cpu",
    reason="Generate requires megalodon_jax.generate which needs full model",
)
def test_generate_cli_produces_output(tmp_path: Path) -> None:
    """End-to-end test of the generate CLI command."""
    import orbax.checkpoint as ocp

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

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
            cema_ndim=16,
            chunk_size=8,
            dropout=0.0,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            param_dtype="float32",
            compute_dtype="float32",
            accum_dtype="float32",
            softmax_dtype="float32",
        ),
        data=replace(
            cfg.data,
            backend="local_text",
            local_text="hello from generate test",
            tokenizer=replace(
                cfg.data.tokenizer,
                kind="byte",
                add_bos=False,
                add_eos=False,
            ),
        ),
        train=replace(
            cfg.train,
            seq_len=16,
            batch_size=1,
            grad_accum=1,
            allow_cpu=False,
        ),
        logging=replace(
            cfg.logging,
            run_dir=str(run_dir),
            console_use_rich=False,
            wandb=replace(cfg.logging.wandb, enabled=False),
        ),
        checkpoint=replace(
            cfg.checkpoint,
            enabled=True,
            save_every=1,
            max_to_keep=1,
            async_save=False,
        ),
    )

    config_resolved = run_dir / "config_resolved.json"
    config_resolved.write_text(json.dumps(cfg.to_dict(), indent=2))

    params, _static = build_model(cfg, key=jax.random.PRNGKey(0))
    ckpt_dir = run_dir / "checkpoints" / "1" / "train_state"
    ckpt_dir.parent.mkdir(parents=True, exist_ok=True)
    ckptr = ocp.PyTreeCheckpointer()
    ckptr.save(ckpt_dir, {"params": params}, force=True)

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

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert "Generated:" in result.output
