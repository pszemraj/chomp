# SPDX-License-Identifier: Apache-2.0

"""Portable weight export.

A training checkpoint is not a model. It is an Orbax pytree keyed by chomp's
config, carrying optimizer moments and data-iterator state alongside the
parameters, and only chomp can interpret it. This module writes the model out
on its own so a finished run can be loaded for inference by anything with
megalodon-jax installed.

The serialization itself is upstream's. ``megalodon_jax.save_checkpoint``
writes safetensors with the full ``MegalodonConfig``, a config fingerprint, and
a parameter manifest in the header, so ``megalodon_jax.load_checkpoint``
rebuilds the model from the file alone -- no chomp, no config file, no run
directory:

    from megalodon_jax import load_checkpoint
    model = load_checkpoint("export/model.safetensors", key=jax.random.key(0))

Chomp's job is the two ends upstream cannot do: getting from a run directory to
a ``MegalodonForCausalLM``, and shipping the tokenizer and provenance beside
the weights so the token IDs and the embedding rows stay married.

Export is lossless and does not change dtypes. ``model.param_dtype`` is pinned
to float32, so an export is float32 and roughly four bytes per parameter. A
bf16 variant is deliberately absent: upstream's ``BF16_DTYPE_POLICY`` keeps
some parameters at fp32 and the choice of which is upstream's to make, not
something to guess at here.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax

from chomp import __version__ as CHOMP_VERSION
from chomp.ckpt import megalodon_jax_identity, restore_params_only
from chomp.config import Config, build_config
from chomp.data.pipeline import (
    load_tokenizer_snapshot,
    load_tokenizer_snapshot_for_resume,
    prepare_tokenizer_and_config,
)
from chomp.model import build_model, megalodon_config_from
from chomp.utils.ckpt_paths import (
    load_config_for_checkpoint,
    read_checkpoint_meta,
    resolve_checkpoint_path,
)
from chomp.utils.tree import abstractify_tree, param_count

if TYPE_CHECKING:
    from chomp.data.pipeline import Tokenizer

#: Bumped when the manifest gains or loses a required key. The weights file has
#: its own independent version in the safetensors header, owned by upstream.
EXPORT_SCHEMA_VERSION = 1

WEIGHTS_FILENAME = "model.safetensors"
MANIFEST_FILENAME = "chomp_export.json"

# Tokenizer files are copied to the export root rather than into a
# ``tokenizer/`` subdirectory, so ``AutoTokenizer.from_pretrained(export_dir)``
# resolves without knowing anything about chomp's run layout.


@dataclass(frozen=True)
class ExportResult:
    """Outcome of one export."""

    export_dir: Path
    weights_path: Path
    step: int
    param_count: int
    weights_bytes: int
    tokenizer_files: tuple[str, ...]
    verified: bool


@dataclass(frozen=True)
class LoadedExport:
    """Model and config restored from an export directory."""

    params: Any
    static: Any
    config: Config
    manifest: dict[str, Any]


def is_export_dir(path: str | Path) -> bool:
    """Report whether a path is an export directory rather than a run directory.

    :param str | Path path: Candidate directory.
    :return bool: True when the directory holds an export manifest and weights.
    """
    directory = Path(path)
    return (directory / MANIFEST_FILENAME).is_file() and (directory / WEIGHTS_FILENAME).is_file()


def read_export_manifest(export_dir: str | Path) -> dict[str, Any]:
    """Read and validate an export manifest.

    :param str | Path export_dir: Export directory.
    :raises FileNotFoundError: If the directory holds no manifest.
    :raises ValueError: If the manifest is corrupt or from a newer schema.
    :return dict[str, Any]: Parsed manifest.
    """
    manifest_path = Path(export_dir) / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No {MANIFEST_FILENAME} in {export_dir}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Corrupted export manifest in {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Export manifest must be a JSON object: {manifest_path}")

    version = manifest.get("schema_version")
    if version != EXPORT_SCHEMA_VERSION:
        # Chomp does not translate across schema versions anywhere else, and an
        # export that silently loads under the wrong schema would produce a
        # model whose tokenizer or vocabulary no longer matches its weights.
        raise ValueError(
            f"Export schema version {version!r} is not supported by this chomp "
            f"(expects {EXPORT_SCHEMA_VERSION}). Re-export from the training checkpoint."
        )
    return manifest


def _tokenizer_for_checkpoint(
    *, step_dir: Path, run_dir: Path | None, cfg: Config
) -> tuple[Tokenizer | None, dict[str, Any] | None]:
    """Load the run-pinned tokenizer and prove it matches the checkpoint.

    Mirrors the generation path: token IDs index restored embedding rows
    directly, so a tokenizer that cannot be proven identical is a refusal
    rather than a warning.

    :param Path step_dir: Checkpoint step directory.
    :param Path | None run_dir: Run directory, when one was found.
    :param Config cfg: Config belonging to the checkpoint.
    :raises RuntimeError: If the tokenizer is missing or does not match.
    :return tuple: Tokenizer (None for byte tokenizers) and its checkpoint identity.
    """
    try:
        meta = read_checkpoint_meta(step_dir)
    except FileNotFoundError:
        meta = None

    identity = None if meta is None else meta.get("tokenizer_identity")
    if meta is not None and meta.get("schema_version") in {2, 3} and not isinstance(identity, dict):
        raise RuntimeError(
            "Checkpoint metadata is missing tokenizer_identity; cannot export a "
            "tokenizer that provably matches these weights."
        )

    if isinstance(identity, dict):
        if run_dir is None or not (run_dir / "tokenizer").exists():
            raise RuntimeError("Checkpoint requires its run-pinned tokenizer snapshot for export.")
        tokenizer, observed = load_tokenizer_snapshot_for_resume(run_dir, cfg)
        if observed != identity:
            raise RuntimeError(
                "Tokenizer identity does not match the selected checkpoint; refusing "
                "to export because token IDs may not match its embedding rows."
            )
        return tokenizer, observed

    if cfg.data.tokenizer.kind == "hf" and run_dir is not None and (run_dir / "tokenizer").exists():
        return load_tokenizer_snapshot(run_dir, cfg), None
    return None, None


def _copy_tokenizer_files(run_dir: Path | None, export_dir: Path) -> tuple[str, ...]:
    """Copy a run's tokenizer snapshot into the export root.

    The files are copied rather than re-serialized: the identity manifest
    hashes their exact bytes, so a round-trip through ``save_pretrained`` could
    invalidate the identity that :func:`_tokenizer_for_checkpoint` just proved.

    :param Path | None run_dir: Run directory, when one was found.
    :param Path export_dir: Destination export directory.
    :return tuple[str, ...]: Sorted names of the copied files.
    """
    if run_dir is None:
        return ()
    tok_dir = run_dir / "tokenizer"
    if not tok_dir.is_dir():
        return ()
    copied = []
    for source in sorted(tok_dir.iterdir()):
        if source.is_file():
            shutil.copy2(source, export_dir / source.name)
            copied.append(source.name)
    return tuple(copied)


def _verify_weights(weights_path: Path, params: Any) -> None:
    """Reload exported weights and assert they match what was written.

    safetensors carries no payload checksum, and upstream's manifest hash
    covers names, shapes, and dtypes rather than bytes. Nothing else in the
    pipeline would notice a corrupted tensor until generation produced
    nonsense, so the export re-reads its own output.

    :param Path weights_path: Written safetensors file.
    :param Any params: Parameter pytree that was exported.
    :raises RuntimeError: If any reloaded parameter differs.
    """
    from megalodon_jax import load_checkpoint

    reloaded, _ = eqx.partition(load_checkpoint(weights_path, key=jax.random.key(0)), eqx.is_array)
    written = jax.tree_util.tree_leaves(params)
    read_back = jax.tree_util.tree_leaves(reloaded)
    if len(written) != len(read_back):
        raise RuntimeError(
            f"Exported weights have {len(read_back)} parameter arrays but "
            f"{len(written)} were written to {weights_path}."
        )
    for index, (before, after) in enumerate(zip(written, read_back, strict=True)):
        if before.shape != after.shape or before.dtype != after.dtype:
            raise RuntimeError(
                f"Exported parameter {index} changed shape or dtype: "
                f"{before.shape}/{before.dtype} -> {after.shape}/{after.dtype}"
            )
        if not bool(jax.numpy.array_equal(before, after)):
            raise RuntimeError(
                f"Exported parameter {index} did not survive the round trip through "
                f"{weights_path}; the file is corrupt."
            )


def _check_destination(destination: Path, *, overwrite: bool) -> tuple[str, ...]:
    """Decide whether a destination may be written, without changing anything.

    Deliberately non-destructive. Everything after this call can still fail --
    the tokenizer identity check, the Orbax restore, the write itself -- and an
    overwrite that deleted a good export before proving it could produce a
    replacement would leave the user with neither. The names returned here are
    swept only once new weights are on disk.

    A non-empty directory that is not an export is refused outright: chomp
    cannot know what else lives there, so deleting is not its call to make.

    :param Path destination: Directory the export will be written into.
    :param bool overwrite: Whether a previous export may be replaced.
    :raises FileExistsError: If the destination holds content export will not replace.
    :return tuple[str, ...]: File names the previous export claimed, if any.
    """
    if not destination.exists() or not any(destination.iterdir()):
        return ()

    if not is_export_dir(destination):
        raise FileExistsError(
            f"{destination} is not empty and does not contain a chomp export. "
            "Choose an empty or new directory."
        )
    if not overwrite:
        raise FileExistsError(
            f"{destination} already holds an export. Pass overwrite to replace it."
        )

    try:
        previous = read_export_manifest(destination)
    except (ValueError, FileNotFoundError) as exc:
        # Refusing beats half-cleaning. Without a readable manifest chomp cannot
        # tell which tokenizer files this directory owns, and leaving one behind
        # beside new weights is the exact mismatch overwrite exists to prevent.
        raise FileExistsError(
            f"{destination} holds an export whose {MANIFEST_FILENAME} is unreadable "
            f"({exc}), so chomp cannot tell which files belong to it. Remove the "
            "directory and export again."
        ) from exc
    owned = [*previous.get("tokenizer_files", []), previous.get("weights_file", "")]
    return tuple(name for name in owned if isinstance(name, str) and name)


def _remove_stale_files(
    destination: Path, *, previous: tuple[str, ...], written: tuple[str, ...]
) -> None:
    """Delete files a previous export owned that this one did not rewrite.

    Called only after the new weights are durable. ``AutoTokenizer`` loads
    whichever tokenizer files it finds, so a ``vocab.txt`` left over from a
    different model would silently pair the wrong vocabulary with these
    weights.

    :param Path destination: Export directory.
    :param tuple[str, ...] previous: Names the previous manifest claimed.
    :param tuple[str, ...] written: Names this export just wrote.
    """
    keep = {*written, WEIGHTS_FILENAME, MANIFEST_FILENAME}
    for name in previous:
        if name in keep:
            continue
        candidate = destination / name
        # Reject path traversal from a hand-edited manifest before unlinking.
        if candidate.parent == destination and candidate.is_file():
            candidate.unlink()


def export_checkpoint(
    checkpoint: str | Path,
    export_dir: str | Path,
    *,
    config_override: str | None = None,
    overwrite: bool = False,
    verify: bool = True,
) -> ExportResult:
    """Write one training checkpoint out as a portable safetensors model.

    :param str | Path checkpoint: Run directory, checkpoint root, or step directory.
    :param str | Path export_dir: Destination directory, created if absent.
    :param str | None config_override: Optional config file replacing the checkpoint's.
    :param bool overwrite: Whether to replace an existing export in the destination.
    :param bool verify: Whether to reload the written file and compare parameters.
    :raises FileNotFoundError: If no checkpoint can be resolved.
    :raises RuntimeError: If the backend is unsupported or the tokenizer cannot be proven.
    :raises FileExistsError: If the destination is non-empty and cannot be safely replaced.
    :return ExportResult: Description of what was written.
    """
    from megalodon_jax import save_checkpoint

    step_dir, run_dir = resolve_checkpoint_path(checkpoint)
    cfg = load_config_for_checkpoint(
        step_dir=step_dir, run_dir=run_dir, config_override=config_override
    )
    if cfg.model.backend != "megalodon":
        raise RuntimeError(
            "export only supports model.backend='megalodon'. "
            f"Found {cfg.model.backend!r} in the checkpoint config."
        )

    destination = Path(export_dir)
    previous_files = _check_destination(destination, overwrite=overwrite)

    tokenizer, tokenizer_identity = _tokenizer_for_checkpoint(
        step_dir=step_dir, run_dir=run_dir, cfg=cfg
    )
    # Vocabulary padding and special-token IDs are resolved here exactly as
    # training resolved them, so the config stored beside the weights is the
    # one the restored arrays were actually shaped by.
    cfg, _tokenizer = prepare_tokenizer_and_config(cfg, tokenizer=tokenizer)

    params, static = build_model(cfg, key=jax.random.key(0))
    params = restore_params_only(step_dir, abstractify_tree(params))
    model = eqx.combine(params, static)

    destination.mkdir(parents=True, exist_ok=True)
    weights_path = destination / WEIGHTS_FILENAME
    save_checkpoint(model, weights_path)
    if verify:
        _verify_weights(weights_path, params)

    tokenizer_files = _copy_tokenizer_files(run_dir, destination)
    _remove_stale_files(destination, previous=previous_files, written=tokenizer_files)

    try:
        meta = read_checkpoint_meta(step_dir)
    except FileNotFoundError:
        meta = {}
    step = int(meta.get("step", int(step_dir.name) if step_dir.name.isdigit() else -1))

    manifest = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "chomp_version": CHOMP_VERSION,
        "megalodon_jax": megalodon_jax_identity(),
        "weights_file": WEIGHTS_FILENAME,
        "param_count": param_count(params),
        "source": {
            "run_dir": None if run_dir is None else str(run_dir),
            "step_dir": str(step_dir),
            "step": step,
        },
        "training": {
            "tokens_seen": meta.get("tokens_seen"),
            "eval_status": meta.get("eval_status"),
        },
        "tokenizer_identity": tokenizer_identity,
        "tokenizer_files": list(tokenizer_files),
        "config": cfg.to_dict(),
    }
    (destination / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )

    return ExportResult(
        export_dir=destination,
        weights_path=weights_path,
        step=step,
        param_count=manifest["param_count"],
        weights_bytes=weights_path.stat().st_size,
        tokenizer_files=tokenizer_files,
        verified=verify,
    )


def load_export(export_dir: str | Path, *, key: jax.Array | None = None) -> LoadedExport:
    """Load an exported model into the ``(params, static)`` pair chomp uses.

    The weights file is authoritative for architecture: upstream rebuilds the
    module from the ``MegalodonConfig`` in its header. The manifest supplies
    what the header cannot -- tokenizer settings and the rest of the chomp
    config -- and is checked against the header so a hand-edited manifest
    cannot quietly pair the wrong vocabulary with these weights.

    :param str | Path export_dir: Export directory.
    :param jax.Array | None key: PRNG key for the skeleton build; unused by the result.
    :raises FileNotFoundError: If the directory is not an export.
    :raises RuntimeError: If the manifest disagrees with the weights file.
    :return LoadedExport: Parameters, static module halves, config, and manifest.
    """
    from megalodon_jax import load_checkpoint

    directory = Path(export_dir)
    manifest = read_export_manifest(directory)
    weights_path = directory / manifest.get("weights_file", WEIGHTS_FILENAME)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Export is missing its weights file: {weights_path}")

    model = load_checkpoint(weights_path, key=jax.random.key(0) if key is None else key)
    cfg = build_config(manifest["config"])

    expected = megalodon_config_from(cfg)
    if model.config != expected:
        raise RuntimeError(
            f"{MANIFEST_FILENAME} describes a different model than {weights_path.name} "
            "contains. The export directory is inconsistent; re-export it."
        )

    params, static = eqx.partition(model, eqx.is_array)
    return LoadedExport(params=params, static=static, config=cfg, manifest=manifest)


def load_export_tokenizer(export_dir: str | Path, cfg: Config) -> Tokenizer:
    """Load the tokenizer shipped inside an export directory.

    :param str | Path export_dir: Export directory holding tokenizer files.
    :param Config cfg: Config from the export manifest.
    :raises FileNotFoundError: If a Hugging Face tokenizer was expected but not shipped.
    :return Tokenizer: Tokenizer bound to these weights.
    """
    from chomp.data.pipeline import ByteTokenizer, HFTokenizer

    directory = Path(export_dir)
    if cfg.data.tokenizer.kind == "byte":
        return ByteTokenizer(byte_offset=cfg.data.tokenizer.byte_offset)

    if not (directory / "tokenizer.json").is_file():
        raise FileNotFoundError(
            f"Export {directory} declares a Hugging Face tokenizer but ships no "
            "tokenizer.json. Re-export from a run directory containing tokenizer/."
        )
    return HFTokenizer(
        str(directory),
        use_fast=cfg.data.tokenizer.hf_use_fast,
        trust_remote_code=cfg.data.tokenizer.hf_trust_remote_code,
        local_files_only=True,
    )
