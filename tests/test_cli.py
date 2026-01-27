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
from chomp.config import Config, load_config
from chomp.model import build_model
from chomp.train import run
from chomp.utils.checkpoints import resolve_checkpoint_path
from chomp.utils.tree import abstractify_tree


def test_print_banner_outputs_expected_text(capsys: object) -> None:
    """print_banner emits the banner once with a trailing newline."""
    print_banner()
    captured = capsys.readouterr()

    assert captured.out.endswith("\n")
    assert captured.out.rstrip("\n") == BANNER


def test_parse_resume_returns_none_for_none_variants() -> None:
    """parse_resume should return 'none' for none/no/false/0."""
    assert parse_resume("none") == "none"
    assert parse_resume("no") == "none"
    assert parse_resume("false") == "none"
    assert parse_resume("0") == "none"
    assert parse_resume("  NONE  ") == "none"


def test_parse_resume_returns_latest_for_latest_variants() -> None:
    """parse_resume should return 'latest' for latest/last."""
    assert parse_resume("latest") == "latest"
    assert parse_resume("last") == "latest"
    assert parse_resume("  LATEST  ") == "latest"


def test_parse_resume_returns_int_for_valid_step() -> None:
    """parse_resume should return an int for valid positive step numbers."""
    assert parse_resume("100") == 100
    assert parse_resume("5000") == 5000
    assert parse_resume("  42  ") == 42


def test_parse_resume_rejects_negative_step() -> None:
    """parse_resume should raise BadParameter for negative step numbers."""
    with pytest.raises(click.BadParameter, match="non-negative"):
        parse_resume("-1")
    with pytest.raises(click.BadParameter, match="non-negative"):
        parse_resume("-100")


def test_parse_resume_rejects_invalid_string() -> None:
    """parse_resume should raise BadParameter for invalid strings."""
    with pytest.raises(click.BadParameter, match="Invalid resume value"):
        parse_resume("invalid")
    with pytest.raises(click.BadParameter, match="Invalid resume value"):
        parse_resume("step100")


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


def test_resolve_checkpoint_with_run_dir(tmp_path: Path) -> None:
    """resolve_checkpoint_path finds latest checkpoint from run directory."""
    cfg, config_src = _small_cfg(tmp_path)
    run_dir = run(cfg, config_path=str(config_src), resume="none", dry_run=False)

    step_dir, found_run_dir = resolve_checkpoint_path(str(run_dir))

    assert found_run_dir == run_dir
    assert step_dir.name == "2"  # latest step
    assert (step_dir / "train_state").exists()


def test_resolve_checkpoint_with_root_dir(tmp_path: Path) -> None:
    """resolve_checkpoint_path respects checkpoint.root_dir when given run_dir."""
    cfg, config_src = _small_cfg(tmp_path)
    ckpt_root = tmp_path / "ckpt_root"
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, root_dir=str(ckpt_root)))
    run_dir = run(cfg, config_path=str(config_src), resume="none", dry_run=False)

    step_dir, found_run_dir = resolve_checkpoint_path(str(run_dir))

    assert found_run_dir == run_dir
    assert step_dir.parent == ckpt_root
    assert step_dir.name == "2"
    assert (step_dir / "train_state").exists()


def test_resolve_checkpoint_with_step_dir(tmp_path: Path) -> None:
    """resolve_checkpoint_path accepts direct step directory."""
    cfg, config_src = _small_cfg(tmp_path)
    run_dir = run(cfg, config_path=str(config_src), resume="none", dry_run=False)

    ckpt_dir = default_ckpt_dir(run_dir)
    step_dir_input = ckpt_dir / "1"

    step_dir, found_run_dir = resolve_checkpoint_path(str(step_dir_input))

    assert found_run_dir == run_dir
    assert step_dir == step_dir_input


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


def test_restore_params_partial_restore(tmp_path: Path) -> None:
    """_restore_params loads only params from a full TrainState checkpoint.

    Regression test: ensure partial restore works when checkpoint contains
    step, params, opt_state, and rng but we only need params.
    """
    cfg, config_src = _small_cfg(tmp_path)
    run_dir = run(cfg, config_path=str(config_src), resume="none", dry_run=False)

    # Build model to get abstract params shape
    params, _static = build_model(cfg, key=jax.random.PRNGKey(0))
    abstract_params = abstractify_tree(params)

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
    abstract_params = abstractify_tree(params_fresh)

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

    # Write config_resolved.json for CLI lookup
    config_resolved = run_dir / "config_resolved.json"
    config_resolved.write_text(json.dumps(cfg.to_dict(), indent=2))

    # Create a minimal checkpoint with just params.
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

    # Should complete without error (exit code 0)
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert "Generated:" in result.output
