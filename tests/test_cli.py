"""CLI tests consolidated by module."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import click
import jax
import pytest
from click.testing import CliRunner

from chomp.ckpt import CHECKPOINT_META_SCHEMA_VERSION
from chomp.cli import cli
from chomp.cli.main import parse_resume
from chomp.config import Config
from chomp.data.pipeline import ByteTokenizer, save_tokenizer_snapshot
from chomp.model import build_model
from tests.helpers.config_factories import make_tiny_megalodon_model


def test_cli_import_does_not_initialize_jax() -> None:
    """Importing command registration must leave XLA configuration mutable."""
    code = "import sys; import chomp.cli.main; assert 'jax' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
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
    ],
)
def test_parse_resume_accepts_valid_variants(raw: str, expected: object) -> None:
    """parse_resume should normalize valid alias and numeric variants."""
    assert parse_resume(raw) == expected


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ("-1", "non-negative"),
        ("-100", "non-negative"),
        ("invalid", "Invalid resume value"),
        ("step100", "Invalid resume value"),
    ],
)
def test_parse_resume_rejects_invalid_values(raw: str, match: str) -> None:
    """parse_resume should reject unparseable resume values."""
    with pytest.raises(click.BadParameter, match=match):
        parse_resume(raw)


def test_train_reports_nested_config_typos_without_a_traceback(tmp_path: Path) -> None:
    """Dataclass construction errors should surface as concise CLI errors."""
    config_path = tmp_path / "typo.yaml"
    config_path.write_text("optim:\n  lrr: 0.001\n")

    result = CliRunner().invoke(cli, ["train", str(config_path)])

    assert result.exit_code != 0
    assert "Error: Invalid config:" in result.output
    assert "unexpected keyword argument 'lrr'" in result.output
    assert "Traceback" not in result.output


def test_train_validates_run_dir_override(tmp_path: Path) -> None:
    """The final CLI run-directory override must satisfy config validation."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n")

    result = CliRunner().invoke(cli, ["train", str(config_path), "--run-dir", ""])

    assert result.exit_code != 0
    assert "logging.run_dir must be a non-empty string" in result.output


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


def test_generate_cli_produces_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generate with checkpoint-bound tokenizer identity and reject drift.

    :param Path tmp_path: Temporary directory for checkpoint artifacts.
    :param pytest.MonkeyPatch monkeypatch: Tokenizer drift injection fixture.
    """
    import orbax.checkpoint as ocp

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    cfg = replace(
        cfg,
        model=make_tiny_megalodon_model(
            vocab_size=256,
            cema_ndim=16,
            param_dtype="float32",
            compute_dtype="float32",
            accum_dtype="float32",
            attention_softmax_dtype="float32",
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
    _tokenizer, tokenizer_identity = save_tokenizer_snapshot(
        run_dir,
        cfg,
        ByteTokenizer(byte_offset=cfg.data.tokenizer.byte_offset),
    )
    params, _static = build_model(cfg, key=jax.random.PRNGKey(0))
    step_dir = run_dir / "checkpoints" / "1"
    ckpt_dir = step_dir / "train_state"
    step_dir.mkdir(parents=True, exist_ok=True)
    ckptr = ocp.PyTreeCheckpointer()
    ckptr.save(ckpt_dir, {"params": params}, force=True)
    meta_dir = step_dir / "meta"
    meta_dir.mkdir()
    (meta_dir / "metadata").write_text(
        json.dumps(
            {
                "schema_version": CHECKPOINT_META_SCHEMA_VERSION,
                "config": cfg.to_dict(),
                "tokenizer_identity": tokenizer_identity,
            }
        )
    )

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

    encode = ByteTokenizer.encode

    def _drifted_encode(tokenizer: ByteTokenizer, text: str) -> list[int]:
        """Change canary outputs without changing the saved manifest.

        :param ByteTokenizer tokenizer: Byte tokenizer instance.
        :param str text: Text to encode.
        :return list[int]: Original token IDs plus one extra ID.
        """
        return [*encode(tokenizer, text), 0]

    monkeypatch.setattr(ByteTokenizer, "encode", _drifted_encode)
    drifted = runner.invoke(
        cli,
        ["generate", str(run_dir), "--prompt", "hello", "--max-tokens", "1"],
    )

    assert drifted.exit_code != 0
    assert "Tokenizer identity does not match the selected checkpoint" in drifted.output
