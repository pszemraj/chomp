"""Portable safetensors export tests."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import pytest
from click.testing import CliRunner

from chomp.ckpt import restore_params_only
from chomp.cli import cli
from chomp.config import Config, build_config
from chomp.data.pipeline import load_tokenizer_snapshot
from chomp.export import (
    EXPORT_SCHEMA_VERSION,
    MANIFEST_FILENAME,
    WEIGHTS_FILENAME,
    _verify_weights,
    export_checkpoint,
    is_export_dir,
    load_export,
    load_export_tokenizer,
    read_export_manifest,
)
from chomp.model import build_model, megalodon_config_from
from chomp.train import run
from chomp.utils.ckpt_paths import resolve_checkpoint_path
from chomp.utils.tree import abstractify_tree, param_count

TRAINED_STEPS = 2
IN_VOCAB_TEXT = "hello world The quick brown fox ."


def _write_bert_tokenizer(path: Path) -> Path:
    """Create the same deterministic local tokenizer the conftest fixture builds.

    Duplicated here rather than reused because ``local_bert_tokenizer`` is
    function-scoped and this module trains once for the whole file.

    :param Path path: Directory to write the tokenizer into.
    :return Path: Local tokenizer source directory.
    """
    from transformers import BertTokenizerFast

    path.mkdir(parents=True)
    vocab = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "The",
        "quick",
        "brown",
        "fox",
        "hello",
        "world",
        ".",
    ]
    vocab_path = path / "vocab.txt"
    vocab_path.write_text("\n".join(vocab) + "\n")
    BertTokenizerFast(
        vocab_file=str(vocab_path),
        do_lower_case=False,
        clean_up_tokenization_spaces=False,
    ).save_pretrained(path)
    return path


def _run_config(*, tokenizer_dir: Path, run_dir: Path) -> Config:
    """Build the tiny megalodon + HF-tokenizer config the exported run trains.

    :param Path tokenizer_dir: Local tokenizer snapshot source.
    :param Path run_dir: Destination run directory.
    :return Config: Validated smoke-sized training configuration.
    """
    return build_config(
        {
            "model": {
                "backend": "megalodon",
                "model_dim": 32,
                "num_layers": 2,
                "num_heads": 1,
                "z_dim": 16,
                "value_dim": 64,
                "ffn_hidden_dim": 48,
                "cema_ndim": 4,
                "chunk_size": 16,
                "norm_num_groups": 4,
                "swiglu": True,
                "share_emb": True,
                "compute_dtype": "float32",
            },
            "data": {
                "backend": "local_text",
                "local_text": (
                    "hello world chomp the quick brown fox . "
                    "hello world chomp the quick brown fox ."
                ),
                "packing_mode": "sequential",
                "window_shuffle_tokens": 0,
                "max_eval_samples": 0,
                "tokenizer": {
                    "kind": "hf",
                    "hf_name_or_path": str(tokenizer_dir),
                    "hf_use_fast": True,
                    # The fixture tokenizer has no eos token; requesting one raises.
                    "add_eos": False,
                },
            },
            "train": {
                "steps": TRAINED_STEPS,
                "batch_size": 2,
                "seq_len": 64,
                "grad_accum": 1,
                "deterministic": True,
                "allow_cpu": True,
                "log_every": 1,
                "eval_every": 0,
                "generate_every": 0,
            },
            "optim": {"warmup_steps": 1},
            "checkpoint": {"save_every": TRAINED_STEPS, "async_save": False},
            "logging": {"run_dir": str(run_dir), "console_use_rich": False},
        }
    )


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train a two-step megalodon run on CPU once for the whole module.

    :param pytest.TempPathFactory tmp_path_factory: Session temp-directory factory.
    :return Path: Run directory holding a checkpoint and a tokenizer snapshot.
    """
    base = tmp_path_factory.mktemp("export")
    tokenizer_dir = _write_bert_tokenizer(base / "bert-tokenizer")
    cfg = _run_config(tokenizer_dir=tokenizer_dir, run_dir=base / "run")
    return run(cfg, config_path=None, resume="none", dry_run=False)


@pytest.fixture(scope="module")
def exported_dir(trained_run: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Export the trained run once, with verification on.

    :param Path trained_run: Trained run directory.
    :param pytest.TempPathFactory tmp_path_factory: Session temp-directory factory.
    :return Path: Export directory.
    """
    destination = tmp_path_factory.mktemp("exported") / "model"
    result = export_checkpoint(trained_run, destination)

    assert result.verified is True
    return result.export_dir


@pytest.fixture
def export_copy(exported_dir: Path, tmp_path: Path) -> Path:
    """Copy the shared export so a test may corrupt it.

    :param Path exported_dir: Module-scoped pristine export.
    :param Path tmp_path: Pytest temporary directory.
    :return Path: Writable copy of the export directory.
    """
    destination = tmp_path / "export"
    shutil.copytree(exported_dir, destination)
    return destination


def _checkpoint_params(run_dir: Path, cfg: Config) -> Any:
    """Restore parameters straight from the Orbax training checkpoint.

    :param Path run_dir: Trained run directory.
    :param Config cfg: Resolved config describing the checkpointed model.
    :return Any: Parameter pytree as the training harness restores it.
    """
    step_dir, _run_dir = resolve_checkpoint_path(str(run_dir))
    skeleton, _static = build_model(cfg, key=jax.random.key(0))
    return restore_params_only(step_dir, abstractify_tree(skeleton))


def _corrupt_weights_payload(weights_path: Path) -> None:
    """Flip bits inside the tensor payload of a safetensors file.

    The 8-byte length prefix and the JSON header are left untouched, so the
    file still parses and still advertises the same tensors; only the raw
    tensor bytes change.

    :param Path weights_path: safetensors file damaged in place.
    """
    raw = bytearray(weights_path.read_bytes())
    payload_start = 8 + int.from_bytes(raw[:8], "little")
    for offset in range(payload_start, payload_start + 64):
        raw[offset] ^= 0xFF
    weights_path.write_bytes(bytes(raw))


def _generated_text(output: str) -> str:
    """Pull the generated continuation out of ``chomp generate`` console output.

    :param str output: Full CLI stdout.
    :return str: Text printed under the "Generated:" banner.
    """
    assert "Generated:" in output, output
    return output.split("Generated:\n", 1)[1].split("=" * 60, 1)[0]


def test_export_round_trips_checkpoint_parameters_bitwise(
    trained_run: Path, exported_dir: Path
) -> None:
    """Every exported parameter must be the checkpoint's parameter, bit for bit."""
    loaded = load_export(exported_dir)
    expected = _checkpoint_params(trained_run, loaded.config)

    assert jax.tree_util.tree_structure(loaded.params) == jax.tree_util.tree_structure(expected)
    exported_leaves = jax.tree_util.tree_leaves(loaded.params)
    expected_leaves = jax.tree_util.tree_leaves(expected)
    assert exported_leaves, "export produced no parameter arrays"
    for index, (after, before) in enumerate(zip(exported_leaves, expected_leaves, strict=True)):
        assert after.shape == before.shape, index
        assert after.dtype == before.dtype, index
        assert bool(jax.numpy.array_equal(after, before)), index


def test_weights_file_alone_reconstructs_the_model(exported_dir: Path) -> None:
    """megalodon_jax must rebuild the model from the safetensors file with no chomp state."""
    from megalodon_jax import load_checkpoint

    manifest = read_export_manifest(exported_dir)
    cfg = build_config(manifest["config"])

    model = load_checkpoint(exported_dir / WEIGHTS_FILENAME, key=jax.random.key(0))

    assert model.config == megalodon_config_from(cfg)


def test_manifest_records_provenance_and_round_trips(trained_run: Path, exported_dir: Path) -> None:
    """The manifest must describe what was exported and rebuild its own config."""
    manifest = read_export_manifest(exported_dir)

    assert manifest["schema_version"] == EXPORT_SCHEMA_VERSION
    assert manifest["chomp_version"]
    assert manifest["megalodon_jax"]["distribution"] == "megalodon-jax"
    assert manifest["weights_file"] == WEIGHTS_FILENAME
    assert manifest["source"]["step"] == TRAINED_STEPS
    assert manifest["source"]["run_dir"] == str(trained_run)

    rebuilt = build_config(manifest["config"])
    assert rebuilt.model.backend == "megalodon"
    assert rebuilt.model.num_layers == 2
    assert rebuilt.model.swiglu is True
    assert rebuilt.data.tokenizer.kind == "hf"
    # The manifest survives a full JSON + dataclass round trip unchanged.
    assert build_config(json.loads(json.dumps(rebuilt.to_dict()))) == rebuilt

    assert manifest["param_count"] == param_count(_checkpoint_params(trained_run, rebuilt))


def test_tokenizer_ships_at_the_export_root(trained_run: Path, exported_dir: Path) -> None:
    """AutoTokenizer.from_pretrained(export_dir) must resolve without a subdirectory."""
    assert not (exported_dir / "tokenizer").exists()
    assert (exported_dir / "tokenizer.json").is_file()
    assert (exported_dir / "tokenizer.json").read_bytes() == (
        trained_run / "tokenizer" / "tokenizer.json"
    ).read_bytes()

    cfg = build_config(read_export_manifest(exported_dir)["config"])
    export_tokenizer = load_export_tokenizer(exported_dir, cfg)
    run_tokenizer = load_tokenizer_snapshot(trained_run, cfg)

    ids = export_tokenizer.encode(IN_VOCAB_TEXT)
    assert ids
    assert ids == run_tokenizer.encode(IN_VOCAB_TEXT)


def test_is_export_dir_distinguishes_exports_from_runs(
    trained_run: Path, exported_dir: Path
) -> None:
    """Only a directory holding both manifest and weights is an export."""
    step_dir, _run_dir = resolve_checkpoint_path(str(trained_run))

    assert is_export_dir(exported_dir) is True
    assert is_export_dir(trained_run) is False
    assert is_export_dir(step_dir) is False


def test_read_export_manifest_rejects_unknown_schema(export_copy: Path, tmp_path: Path) -> None:
    """A manifest from another schema must fail loudly rather than load partially."""
    manifest_path = export_copy / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = 999
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="schema version 999"):
        read_export_manifest(export_copy)

    empty = tmp_path / "not-an-export"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match=MANIFEST_FILENAME):
        read_export_manifest(empty)


def test_load_export_rejects_manifest_that_disagrees_with_weights(export_copy: Path) -> None:
    """A hand-edited manifest must not silently re-describe the weights."""
    manifest_path = export_copy / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    assert manifest["config"]["model"]["num_layers"] == 2
    manifest["config"]["model"]["num_layers"] = 1
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="describes a different model"):
        load_export(export_copy)


def test_export_destination_rules(trained_run: Path, tmp_path: Path) -> None:
    """Export writes into empty directories, replaces exports, and refuses the rest."""
    fresh = tmp_path / "fresh"
    assert export_checkpoint(trained_run, fresh).weights_path.is_file()

    with pytest.raises(FileExistsError, match="already holds an export"):
        export_checkpoint(trained_run, fresh)
    assert export_checkpoint(trained_run, fresh, overwrite=True).weights_path.is_file()

    # A directory chomp did not write is never chomp's to clear, flag or not.
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "stray.txt").write_text("do not clobber me\n")
    for overwrite in (False, True):
        with pytest.raises(FileExistsError, match="does not contain a chomp export"):
            export_checkpoint(trained_run, occupied, overwrite=overwrite)
    assert (occupied / "stray.txt").read_text() == "do not clobber me\n"
    assert not (occupied / WEIGHTS_FILENAME).exists()


def test_overwrite_removes_files_the_previous_manifest_claimed(
    trained_run: Path, tmp_path: Path
) -> None:
    """A stale tokenizer file beside new weights is the mismatch overwrite must prevent."""
    destination = tmp_path / "restamped"
    export_checkpoint(trained_run, destination)

    manifest_path = destination / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["tokenizer_files"] = [*manifest["tokenizer_files"], "merges.txt"]
    manifest_path.write_text(json.dumps(manifest))
    (destination / "merges.txt").write_text("stale tokenizer file\n")
    (destination / "notes.md").write_text("not claimed by the manifest\n")

    export_checkpoint(trained_run, destination, overwrite=True)

    assert not (destination / "merges.txt").exists()
    # Only manifest-claimed files are removed; the directory itself is left alone.
    assert (destination / "notes.md").is_file()
    assert is_export_dir(destination)
    assert "merges.txt" not in read_export_manifest(destination)["tokenizer_files"]


def test_overwrite_refuses_an_export_whose_manifest_is_unreadable(
    trained_run: Path, tmp_path: Path
) -> None:
    """Without a readable manifest chomp cannot tell which files the export owns."""
    destination = tmp_path / "damaged_manifest"
    export_checkpoint(trained_run, destination)
    (destination / MANIFEST_FILENAME).write_text("{ this is not json")

    with pytest.raises(FileExistsError, match="unreadable"):
        export_checkpoint(trained_run, destination, overwrite=True)

    # Half-cleaning is the failure mode being avoided: deleting the weights and
    # leaving an unaccounted-for tokenizer behind is worse than refusing.
    assert (destination / WEIGHTS_FILENAME).is_file()
    assert (destination / "tokenizer.json").is_file()


def test_a_failed_overwrite_leaves_the_previous_export_intact(
    trained_run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Overwrite must not destroy a good export before proving it can replace it."""
    destination = tmp_path / "survivor"
    original = export_checkpoint(trained_run, destination)
    weights_before = (destination / WEIGHTS_FILENAME).read_bytes()
    manifest_before = (destination / MANIFEST_FILENAME).read_text()

    def _fail(*args: Any, **kwargs: Any) -> Any:
        """Fail after the destination check but before anything is written."""
        raise RuntimeError("restore exploded")

    monkeypatch.setattr("chomp.export.restore_params_only", _fail)

    with pytest.raises(RuntimeError, match="restore exploded"):
        export_checkpoint(trained_run, destination, overwrite=True)

    assert (destination / WEIGHTS_FILENAME).read_bytes() == weights_before
    assert (destination / MANIFEST_FILENAME).read_text() == manifest_before
    assert is_export_dir(destination)
    assert read_export_manifest(destination)["param_count"] == original.param_count


def test_verify_catches_a_corrupted_payload_that_loading_does_not(
    trained_run: Path, tmp_path: Path
) -> None:
    """Verification exists because safetensors carries no payload checksum."""
    unverified = export_checkpoint(trained_run, tmp_path / "unverified", verify=False)
    verified = export_checkpoint(trained_run, tmp_path / "verified", verify=True)

    assert unverified.verified is False
    assert verified.verified is True
    assert unverified.weights_bytes == verified.weights_bytes

    loaded = load_export(verified.export_dir)
    params = _checkpoint_params(trained_run, loaded.config)
    # The flag only controls the read-back check, never what gets written: the
    # file exported without verification passes the very same check.
    _verify_weights(verified.weights_path, params)
    _verify_weights(unverified.weights_path, params)

    _corrupt_weights_payload(verified.weights_path)

    # Corruption is invisible to the loader: the header still validates.
    damaged = load_export(verified.export_dir)
    assert not all(
        bool(jax.numpy.array_equal(after, before))
        for after, before in zip(
            jax.tree_util.tree_leaves(damaged.params),
            jax.tree_util.tree_leaves(params),
            strict=True,
        )
    )
    with pytest.raises(RuntimeError, match="corrupt"):
        _verify_weights(verified.weights_path, params)


def test_export_refuses_non_megalodon_backends(tmp_path: Path) -> None:
    """Only megalodon weights have a portable safetensors representation."""
    run_dir = tmp_path / "dummy_run"
    run_dir.mkdir()
    cfg = replace(Config(), model=replace(Config().model, backend="dummy"))
    (run_dir / "config_resolved.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    (run_dir / "checkpoints" / "1" / "train_state").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="megalodon"):
        export_checkpoint(run_dir, tmp_path / "out")


def test_cli_export_then_generate_matches_the_run(trained_run: Path, tmp_path: Path) -> None:
    """An exported model must generate exactly what its source checkpoint generates."""
    runner = CliRunner()
    export_dir = tmp_path / "cli_export"

    exported = runner.invoke(cli, ["export", str(trained_run), "--out", str(export_dir)])

    assert exported.exit_code == 0, exported.output
    assert (export_dir / WEIGHTS_FILENAME).is_file()
    assert (export_dir / MANIFEST_FILENAME).is_file()
    assert (export_dir / "tokenizer.json").is_file()
    assert "(verified)" in exported.output

    args = ["--prompt", IN_VOCAB_TEXT, "--max-tokens", "8", "--temperature", "0", "--seed", "42"]
    from_export = runner.invoke(cli, ["generate", str(export_dir), *args])
    from_run = runner.invoke(cli, ["generate", str(trained_run), *args])

    assert from_export.exit_code == 0, from_export.output
    assert from_run.exit_code == 0, from_run.output
    assert _generated_text(from_export.output).strip()
    assert _generated_text(from_export.output) == _generated_text(from_run.output)

    # The architecture comes from the weights header, so an override could only disagree.
    overridden = runner.invoke(
        cli,
        ["generate", str(export_dir), *args, "--config", str(trained_run / "config_resolved.json")],
    )

    assert overridden.exit_code != 0
    assert "--config does not apply to an export directory" in overridden.output
