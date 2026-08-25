"""Portable safetensors export tests."""

from __future__ import annotations

import json
import logging
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
    CONFIG_FILENAME,
    DTYPE_FLOAT32,
    DTYPE_POLICY,
    EXPORT_SCHEMA_VERSION,
    HF_ARCHITECTURE,
    HF_MODEL_TYPE,
    MANIFEST_FILENAME,
    WEIGHTS_FILENAME,
    _tokenizer_files_to_copy,
    _verify_weights,
    _weights_metadata,
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


def _run_config(*, tokenizer_dir: Path, run_dir: Path, compute_dtype: str = "float32") -> Config:
    """Build the tiny megalodon + HF-tokenizer config the exported run trains.

    :param Path tokenizer_dir: Local tokenizer snapshot source.
    :param Path run_dir: Destination run directory.
    :param str compute_dtype: Forward-pass dtype; bf16 is what --dtype policy requires.
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
                "compute_dtype": compute_dtype,
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


@pytest.fixture(scope="module")
def bf16_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train a run that computes in bf16, which is what ``--dtype policy`` needs.

    Separate from ``trained_run`` on purpose: that one computes in fp32, where
    the policy variant is correctly refused.

    :param pytest.TempPathFactory tmp_path_factory: Session temp-directory factory.
    :return Path: Run directory holding a checkpoint and a tokenizer snapshot.
    """
    base = tmp_path_factory.mktemp("policy")
    tokenizer_dir = _write_bert_tokenizer(base / "bert-tokenizer")
    cfg = _run_config(tokenizer_dir=tokenizer_dir, run_dir=base / "run", compute_dtype="bfloat16")
    return run(cfg, config_path=None, resume="none", dry_run=False)


@pytest.fixture(scope="module")
def dtype_exports(bf16_run: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Export the bf16-compute run both ways, for comparison.

    :param Path bf16_run: Trained run computing in bf16.
    :param pytest.TempPathFactory tmp_path_factory: Session temp-directory factory.
    :return tuple[Path, Path]: The float32 export and the policy-mixed export.
    """
    base = tmp_path_factory.mktemp("dtype_exports")
    canonical = export_checkpoint(bf16_run, base / "float32")
    policy = export_checkpoint(bf16_run, base / "policy", dtype=DTYPE_POLICY)

    assert canonical.weights_dtype == "float32"
    assert policy.weights_dtype == "policy-mixed"
    return canonical.export_dir, policy.export_dir


def _logits(export_dir: Path) -> Any:
    """Run one deterministic forward pass over a fixed prompt from an export.

    :param Path export_dir: Export directory to load.
    :return Any: Logits array.
    """
    import equinox as eqx

    loaded = load_export(export_dir)
    model = eqx.combine(loaded.params, loaded.static)
    ids = jax.numpy.array([[5, 6, 7, 8, 9, 10, 11, 5]], dtype=jax.numpy.int32)
    logits, _cache = model(ids, deterministic=True)
    return logits


def _tensor_dtypes(weights_path: Path) -> dict[str, str]:
    """Read the per-tensor dtypes out of a safetensors header.

    :param Path weights_path: safetensors file to inspect.
    :return dict[str, str]: Tensor name -> dtype name.
    """
    from safetensors import safe_open

    with safe_open(str(weights_path), framework="numpy") as handle:
        # safe_open is not a mapping; keys() is its API, not a dict idiom.
        names = handle.keys()  # noqa: SIM118
        return {name: str(handle.get_slice(name).get_dtype()) for name in names}


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


@pytest.mark.parametrize("reserved", [CONFIG_FILENAME, WEIGHTS_FILENAME, MANIFEST_FILENAME])
def test_a_tokenizer_file_wearing_an_export_name_is_refused_from_the_listing(
    trained_run: Path, tmp_path: Path, reserved: str
) -> None:
    """A collision must be caught from the source, before a copy overwrites anything.

    Exercised against ``_tokenizer_files_to_copy`` rather than through
    ``export_checkpoint``: a snapshot cannot be edited after the fact without
    breaking the tokenizer identity, which refuses earlier and for a different
    reason. Only a tokenizer whose own ``save_pretrained`` wrote one of these
    names reaches this guard, and then the identity matches.
    """
    run_copy = tmp_path / "run"
    shutil.copytree(trained_run, run_copy)
    (run_copy / "tokenizer" / reserved).write_text("tokenizer file wearing an export name\n")

    with pytest.raises(RuntimeError, match=f"contains a {reserved}"):
        _tokenizer_files_to_copy(run_copy)


def test_a_nested_tokenizer_snapshot_is_refused_rather_than_partially_shipped(
    trained_run: Path, tmp_path: Path
) -> None:
    """Copying only the top level would ship a tokenizer its own identity contradicts."""
    run_copy = tmp_path / "run"
    shutil.copytree(trained_run, run_copy)
    nested = run_copy / "tokenizer" / "extra"
    nested.mkdir()
    (nested / "vocab.txt").write_text("a file the flat export layout would drop\n")

    with pytest.raises(RuntimeError, match="contains a subdirectory"):
        _tokenizer_files_to_copy(run_copy)


def test_listing_the_tokenizer_snapshot_writes_nothing(trained_run: Path) -> None:
    """The guard is only safe to run before the write because it is pure listing."""
    tok_dir = trained_run / "tokenizer"
    before = sorted(path.name for path in tok_dir.iterdir())

    listed = _tokenizer_files_to_copy(trained_run)

    assert [path.name for path in listed] == before
    assert sorted(path.name for path in tok_dir.iterdir()) == before
    assert "tokenizer.json" in before


def test_a_failure_after_the_weights_write_leaves_no_loadable_export(
    trained_run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """New weights must never be left paired with the previous export's manifest."""
    destination = tmp_path / "interrupted"
    export_checkpoint(trained_run, destination)

    def _fail(*args: Any, **kwargs: Any) -> Any:
        """Fail after the weights are durable but before the manifest is rewritten."""
        raise RuntimeError("copy exploded")

    monkeypatch.setattr("chomp.export._copy_tokenizer_files", _fail)

    with pytest.raises(RuntimeError, match="copy exploded"):
        export_checkpoint(trained_run, destination, overwrite=True)

    # The stale manifest described the weights that were just replaced, and
    # load_export would have accepted the pair: same architecture, different file.
    assert not (destination / MANIFEST_FILENAME).exists()
    assert (destination / WEIGHTS_FILENAME).is_file()
    assert is_export_dir(destination) is False
    with pytest.raises(FileNotFoundError, match=MANIFEST_FILENAME):
        load_export(destination)


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
    step_dir = run_dir / "checkpoints" / "1"
    (step_dir / "train_state").mkdir(parents=True)
    (step_dir / "meta").mkdir()
    (step_dir / "meta" / "metadata").write_text(json.dumps({"config": cfg.to_dict()}))

    with pytest.raises(RuntimeError, match="megalodon"):
        export_checkpoint(run_dir, tmp_path / "out")


def test_config_json_rebuilds_the_architecture_in_the_weights_file(exported_dir: Path) -> None:
    """config.json must be sufficient on its own to describe what was exported.

    This is the whole point of writing it: a reader with no megalodon-jax and no
    safetensors parser gets the architecture. If it cannot reconstruct the
    config the weights were saved with, it is decoration.
    """
    import jax.numpy as jnp
    from megalodon_jax import load_checkpoint
    from megalodon_jax.config import MegalodonConfig

    config = json.loads((exported_dir / CONFIG_FILENAME).read_text())
    names = {field.name for field in MegalodonConfig.__dataclass_fields__.values()}
    dtypes = {"float32": jnp.float32, "bfloat16": jnp.bfloat16}
    rebuilt = MegalodonConfig(
        **{
            key: dtypes[value] if key.endswith("dtype") else value
            for key, value in config.items()
            if key in names
        }
    )

    model = load_checkpoint(exported_dir / WEIGHTS_FILENAME, key=jax.random.key(0))
    assert rebuilt == model.config


def test_config_json_carries_hugging_face_names_and_the_weight_contract(
    exported_dir: Path,
) -> None:
    """The HF fields are aliases of the native ones, and must not drift from them."""
    config = json.loads((exported_dir / CONFIG_FILENAME).read_text())

    assert config["model_type"] == HF_MODEL_TYPE
    assert config["architectures"] == [HF_ARCHITECTURE]
    assert config["torch_dtype"] == config["param_dtype"]
    assert config["hidden_size"] == config["model_dim"]
    assert config["num_hidden_layers"] == config["num_layers"]
    assert config["num_attention_heads"] == config["num_heads"]
    assert config["intermediate_size"] == config["ffn_hidden_dim"]
    assert config["tie_word_embeddings"] == config["share_emb"]
    # No positional table exists, so no context bound may be implied.
    assert "max_position_embeddings" not in config

    # The layout contract is copied from the header of the file it describes,
    # so a port reading config.json alone knows how to interpret the tensors.
    metadata = _weights_metadata(exported_dir / WEIGHTS_FILENAME)
    contract = config["megalodon_jax"]
    assert contract["weights_file"] == WEIGHTS_FILENAME
    assert contract["config_fingerprint"] == metadata["config_fingerprint"]
    assert contract["rope_layout"] == metadata["rope_layout"]
    assert contract["normalization_storage"] == metadata["normalization_storage"]
    assert contract["bias_schema"] == metadata["bias_schema"]
    assert contract["tying"] == metadata["tying"]


def test_config_json_survives_an_overwrite(trained_run: Path, tmp_path: Path) -> None:
    """Overwrite sweeps files the old manifest claimed; the model config is not one."""
    destination = tmp_path / "reexported"
    export_checkpoint(trained_run, destination)
    result = export_checkpoint(trained_run, destination, overwrite=True)

    assert result.config_path == destination / CONFIG_FILENAME
    assert result.config_path.is_file()
    assert read_export_manifest(destination)["config_file"] == CONFIG_FILENAME


def test_a_finished_run_exports_itself(trained_run: Path) -> None:
    """A clean run leaves a loadable model in its own directory, with no second command."""
    export_dir = trained_run / "export"

    assert is_export_dir(export_dir)
    manifest = read_export_manifest(export_dir)
    step_dir, _run_dir = resolve_checkpoint_path(str(trained_run))
    assert manifest["source"]["step"] == int(step_dir.name)
    assert (export_dir / CONFIG_FILENAME).is_file()
    # Loadable, not merely present.
    assert param_count(load_export(export_dir).params) == manifest["param_count"]


def test_export_on_finish_false_leaves_the_run_directory_alone(tmp_path: Path) -> None:
    """The end-of-run export is a default, not a fixture of training."""
    tokenizer_dir = _write_bert_tokenizer(tmp_path / "tokenizer")
    cfg = _run_config(tokenizer_dir=tokenizer_dir, run_dir=tmp_path / "run")
    cfg = replace(cfg, export=replace(cfg.export, on_finish=False))

    run_dir = run(cfg, config_path=None, resume="none", dry_run=False)

    assert not (run_dir / "export").exists()


def test_a_failed_end_of_run_export_does_not_fail_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Training that succeeded must not be reported as failed over a convenience copy.

    The checkpoint is already durable at this point and ``chomp export``
    reproduces the export in seconds, so the failure is logged rather than
    raised -- but it must be logged, not swallowed.
    """
    tokenizer_dir = _write_bert_tokenizer(tmp_path / "tokenizer")
    cfg = _run_config(tokenizer_dir=tokenizer_dir, run_dir=tmp_path / "run")

    def _fail(*args: Any, **kwargs: Any) -> Any:
        """Fail the way a full disk or a permission error would."""
        raise RuntimeError("no space left on device")

    monkeypatch.setattr("chomp.export.export_checkpoint", _fail)

    with caplog.at_level(logging.ERROR, logger="chomp.train"):
        run_dir = run(cfg, config_path=None, resume="none", dry_run=False)

    assert not (run_dir / "export").exists()
    # The run's actual output is untouched and still resolvable.
    assert resolve_checkpoint_path(str(run_dir))[0].is_dir()

    failures = [record for record in caplog.records if "End-of-run export" in record.getMessage()]
    assert failures, "the export failure was swallowed"
    assert failures[0].levelno == logging.ERROR
    assert failures[0].exc_info is not None


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


def test_policy_export_is_inference_equivalent_to_the_float32_export(
    dtype_exports: tuple[Path, Path],
) -> None:
    """The acceptance test: the derived variant must compute the identical thing.

    Exact, not close. The forward pass already casts these tensors to bf16
    before using them, so the export drops only bits that never reached the
    arithmetic. Any difference at all means the cast set is wrong -- a tensor
    was rounded that the model actually consumes at fp32 -- and a tolerance
    would hide exactly the bug this variant can have.
    """
    canonical, policy = dtype_exports

    assert bool(jax.numpy.array_equal(_logits(canonical), _logits(policy)))


def test_policy_export_generates_byte_identical_greedy_text(
    bf16_run: Path, dtype_exports: tuple[Path, Path]
) -> None:
    """Greedy decoding from the variant must match the fp32 export and the checkpoint."""
    canonical, policy = dtype_exports
    runner = CliRunner()
    args = ["--prompt", IN_VOCAB_TEXT, "--max-tokens", "8", "--temperature", "0", "--seed", "42"]

    from_policy = runner.invoke(cli, ["generate", str(policy), *args])
    from_canonical = runner.invoke(cli, ["generate", str(canonical), *args])
    from_run = runner.invoke(cli, ["generate", str(bf16_run), *args])

    assert from_policy.exit_code == 0, from_policy.output
    assert from_canonical.exit_code == 0, from_canonical.output
    assert from_run.exit_code == 0, from_run.output
    assert _generated_text(from_policy.output).strip()
    assert _generated_text(from_policy.output) == _generated_text(from_canonical.output)
    assert _generated_text(from_policy.output) == _generated_text(from_run.output)


def test_policy_export_keeps_precision_sensitive_tensors_at_fp32(
    dtype_exports: tuple[Path, Path],
) -> None:
    """The file must be genuinely mixed, and mixed exactly where upstream says.

    The first two assertions are the blanket-cast guard: ``tree.map(astype)``
    would produce a uniformly bf16 file that still loads and still generates
    plausible text. The last one cross-checks the fp32 set against upstream's
    ``precision`` module, which enumerates the sensitive parameters
    independently of the model constructor the exporter derives them from.
    """
    from megalodon_jax import load_checkpoint
    from megalodon_jax.precision import _iter_sensitive_params

    _canonical, policy = dtype_exports
    dtypes = _tensor_dtypes(policy / WEIGHTS_FILENAME)
    fp32 = {name for name, dtype in dtypes.items() if dtype == "F32"}
    bf16 = {name for name, dtype in dtypes.items() if dtype == "BF16"}

    assert fp32, "no tensor stayed fp32; the policy was replaced by a blanket cast"
    assert bf16, "no tensor became bf16; nothing was re-encoded"
    assert fp32 | bf16 == set(dtypes), sorted(set(dtypes) - (fp32 | bf16))

    model = load_checkpoint(policy / WEIGHTS_FILENAME, key=jax.random.key(0))
    sensitive = {f"model.{name}" for name, _array in _iter_sensitive_params(model)}
    assert fp32 == sensitive


def test_policy_export_loads_through_megalodon_jax_alone(
    dtype_exports: tuple[Path, Path],
) -> None:
    """A mixed-dtype file must still load by the same three-line path fp32 does."""
    from megalodon_jax import load_checkpoint

    canonical, policy = dtype_exports

    model = load_checkpoint(policy / WEIGHTS_FILENAME, key=jax.random.key(0))
    logits, _cache = model(jax.numpy.array([[5, 6, 7, 8]], dtype=jax.numpy.int32))

    assert model.config.param_dtype == jax.numpy.bfloat16
    assert logits.shape[-1] == model.config.vocab_size
    # Loading it the chomp way must agree with loading it upstream's way.
    assert bool(jax.numpy.array_equal(logits, _logits(policy)[:, :4]))
    assert (policy / WEIGHTS_FILENAME).stat().st_size < (
        canonical / WEIGHTS_FILENAME
    ).stat().st_size


def test_policy_export_manifest_describes_the_mixed_file(
    dtype_exports: tuple[Path, Path],
) -> None:
    """A loader must be able to tell the variant apart without reading the weights."""
    canonical, policy = dtype_exports

    assert read_export_manifest(canonical)["weights_dtype"] == "float32"
    assert "fp32_tensors" not in read_export_manifest(canonical)["dtype_summary"]

    manifest = read_export_manifest(policy)
    summary = manifest["dtype_summary"]
    dtypes = _tensor_dtypes(policy / WEIGHTS_FILENAME)

    assert manifest["weights_dtype"] == "policy-mixed"
    assert set(summary["by_dtype"]) == {"bfloat16", "float32"}
    assert summary["by_dtype"]["float32"]["tensors"] == sum(
        1 for d in dtypes.values() if d == "F32"
    )
    assert summary["by_dtype"]["bfloat16"]["tensors"] == sum(
        1 for d in dtypes.values() if d == "BF16"
    )
    assert set(summary["fp32_tensors"]) == {n for n, d in dtypes.items() if d == "F32"}
    # config.json follows the file, so a reader is told the ordinary-param dtype.
    config = json.loads((policy / CONFIG_FILENAME).read_text())
    assert config["torch_dtype"] == "bfloat16"
    assert config["megalodon_jax"]["dtype_policy"] == "bf16-ordinary-params-fp32-sensitive"


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"compute_dtype": "float32"}, "compute_dtype"),
        ({"scale_emb": True}, "scale_emb"),
    ],
)
def test_policy_export_is_refused_when_it_would_change_outputs(
    bf16_run: Path, tmp_path: Path, override: dict[str, Any], message: str
) -> None:
    """Where the variant is not inference-equivalent, it must not be written at all.

    Both cases are real: with fp32 compute nothing is cast in the forward pass,
    and with scale_emb the embedding is scaled before the cast, so bf16 storage
    rounds at a different point. Neither can be repaired by holding a tensor
    back at fp32, because the file's dtypes must match what upstream builds
    from the config in its header.
    """
    from chomp.utils.ckpt_paths import load_config_for_checkpoint

    step_dir, _run_dir = resolve_checkpoint_path(str(bf16_run))
    cfg = load_config_for_checkpoint(step_dir=step_dir, config_override=None)
    cfg = replace(cfg, model=replace(cfg.model, **override))
    config_path = tmp_path / "override.json"
    config_path.write_text(json.dumps(cfg.to_dict(), indent=2))
    destination = tmp_path / "refused"

    with pytest.raises(RuntimeError, match=message):
        export_checkpoint(
            bf16_run, destination, dtype=DTYPE_POLICY, config_override=str(config_path)
        )

    assert not destination.exists(), "a refused export must leave nothing behind"


def test_export_rejects_an_unknown_dtype(bf16_run: Path, tmp_path: Path) -> None:
    """A typo must not silently fall back to the canonical export."""
    with pytest.raises(ValueError, match="Unknown export dtype"):
        export_checkpoint(bf16_run, tmp_path / "out", dtype="bfloat16")


def test_manifest_rejects_an_unknown_weights_dtype(export_copy: Path) -> None:
    """weights_dtype decides which dtypes the file should hold; it cannot be guessed."""
    manifest_path = export_copy / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    del manifest["weights_dtype"]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="weights_dtype"):
        read_export_manifest(export_copy)


def test_cli_dtype_choices_match_the_exporter() -> None:
    """The CLI spells its choices out to avoid importing JAX; they must not drift."""
    from chomp.cli.export import export as export_command

    option = next(param for param in export_command.params if param.name == "dtype")

    assert set(option.type.choices) == {DTYPE_FLOAT32, DTYPE_POLICY}
    assert option.default == DTYPE_FLOAT32


def test_cli_export_dtype_policy_writes_the_variant(bf16_run: Path, tmp_path: Path) -> None:
    """The flag is the user-facing surface, so drive it end to end."""
    runner = CliRunner()
    destination = tmp_path / "cli_policy"

    result = runner.invoke(
        cli, ["export", str(bf16_run), "--out", str(destination), "--dtype", DTYPE_POLICY]
    )

    assert result.exit_code == 0, result.output
    assert "policy-mixed" in result.output
    assert "(verified)" in result.output
    assert read_export_manifest(destination)["weights_dtype"] == "policy-mixed"
