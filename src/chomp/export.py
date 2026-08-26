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

The architecture is also written out as a Hugging Face shaped ``config.json``.
Nothing in chomp reads it -- the safetensors header stays authoritative here --
but it is the file every other ecosystem looks for, and it is what lets a port
to another framework read the architecture without parsing a safetensors
header or installing megalodon-jax.

The default export is lossless and does not change dtypes. ``model.param_dtype``
is pinned to float32, so it is float32 and roughly four bytes per parameter.
``--dtype policy`` writes a derived variant instead, at the per-tensor dtypes
upstream's bf16 policy assigns; see :func:`_policy_dtype_model`.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax

from chomp import __version__ as CHOMP_VERSION
from chomp.ckpt import megalodon_jax_identity, restore_params_only
from chomp.config import Config, build_config
from chomp.data.pipeline import (
    _build_tokenizer_manifest,
    _write_tokenizer_manifest,
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
EXPORT_SCHEMA_VERSION = 2

WEIGHTS_FILENAME = "model.safetensors"
MANIFEST_FILENAME = "chomp_export.json"
CONFIG_FILENAME = "config.json"

#: Names the export writes into its own root. A tokenizer file carrying one of
#: these would either overwrite an export file or be overwritten by it.
_RESERVED_EXPORT_NAMES = frozenset({WEIGHTS_FILENAME, MANIFEST_FILENAME, CONFIG_FILENAME})

#: ``--dtype`` values. ``float32`` is the canonical artifact -- the master
#: weights, exactly as trained. ``policy`` is derived from it.
DTYPE_FLOAT32 = "float32"
DTYPE_POLICY = "policy"

#: What the manifest records for each. These describe the *file* rather than
#: the request, which is why ``policy`` becomes ``policy-mixed``: the variant
#: is not one dtype, and a loader that assumed it was would be wrong about most
#: of the tensors in either direction.
_WEIGHTS_DTYPE_LABELS = {DTYPE_FLOAT32: "float32", DTYPE_POLICY: "policy-mixed"}

# Tokenizer files are copied to the export root rather than into a
# ``tokenizer/`` subdirectory, so ``AutoTokenizer.from_pretrained(export_dir)``
# resolves without knowing anything about chomp's run layout.

#: ``config.json`` values that Hugging Face itself owns.
HF_MODEL_TYPE = "megalodon"
HF_ARCHITECTURE = "MegalodonForCausalLM"

#: Hugging Face config field -> ``MegalodonConfig`` field. Aliases, not
#: translations: only pairs whose meaning is identical are listed, so a reader
#: that knows either vocabulary sees the same architecture. Megalodon fields
#: with no HF counterpart (``z_dim``, ``cema_ndim``, ``chunk_size``, ...) keep
#: their upstream names, which is what ``PretrainedConfig.attribute_map`` is
#: for on the PyTorch side. ``vocab_size``, ``attention_dropout``, and the
#: special-token IDs already spell the same in both.
#:
#: ``max_position_embeddings`` is deliberately absent: Megalodon has no
#: positional table and no architectural context bound, so any value would be
#: a training detail dressed up as a constraint.
_HF_ALIASES = {
    "hidden_size": "model_dim",
    "num_hidden_layers": "num_layers",
    "num_attention_heads": "num_heads",
    "intermediate_size": "ffn_hidden_dim",
    "tie_word_embeddings": "share_emb",
}

#: Weight-layout contract recorded by ``megalodon_jax.save_checkpoint``. These
#: are what a port has to agree with to read the tensors correctly -- how RoPE
#: pairs are laid out, whether normalization scales are stored as ``1 + w``,
#: which projections carry biases -- so they travel in ``config.json`` rather
#: than staying buried in the safetensors header.
_WEIGHT_CONTRACT_KEYS = (
    "format",
    "format_version",
    "config_fingerprint",
    "parameter_manifest_sha256",
    "rope_layout",
    "normalization_storage",
    "bias_schema",
    "initializer_schema",
    "tying",
    "dtype_policy",
)


@dataclass(frozen=True)
class ExportResult:
    """Outcome of one export."""

    export_dir: Path
    weights_path: Path
    config_path: Path
    step: int
    param_count: int
    weights_bytes: int
    weights_dtype: str
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
    :raises ValueError: If the manifest is corrupt, incomplete, or from a newer schema.
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

    # Required, and load-bearing: it decides which parameter dtypes the weights
    # file is expected to hold, so a missing or unknown value would make the
    # manifest/header cross-check below meaningless rather than merely absent.
    known = set(_WEIGHTS_DTYPE_LABELS.values())
    if manifest.get("weights_dtype") not in known:
        raise ValueError(
            f"Export manifest declares weights_dtype {manifest.get('weights_dtype')!r}, "
            f"which this chomp does not know (expects one of {sorted(known)}): {manifest_path}"
        )

    # The chomp config is the half of the export the safetensors header cannot
    # supply -- tokenizer settings, data, optimizer -- so a manifest without it
    # describes nothing loadable.
    if not isinstance(manifest.get("config"), dict):
        raise ValueError(
            f"Export manifest carries no chomp config, so the tokenizer and training "
            f"settings for these weights are unknown: {manifest_path}"
        )
    return manifest


def _tokenizer_for_checkpoint(
    *, step_dir: Path, run_dir: Path | None, cfg: Config
) -> tuple[Tokenizer | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Load the run-pinned tokenizer and prove it matches the checkpoint.

    Mirrors the generation path: token IDs index restored embedding rows
    directly, so a tokenizer that cannot be proven identical is a refusal
    rather than a warning.

    :param Path step_dir: Checkpoint step directory.
    :param Path | None run_dir: Run directory, when one was found.
    :param Config cfg: Config belonging to the checkpoint.
    :raises RuntimeError: If the tokenizer is missing or does not match.
    :return tuple: Tokenizer, checkpoint identity, and validated full manifest.
    """
    try:
        meta = read_checkpoint_meta(step_dir)
    except FileNotFoundError:
        meta = None

    identity = None if meta is None else meta.get("tokenizer_identity")
    if meta is not None and not isinstance(identity, dict):
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
        observed_manifest = _build_tokenizer_manifest(run_dir / "tokenizer", tokenizer)
        return tokenizer, observed, observed_manifest

    if cfg.data.tokenizer.kind == "hf" and run_dir is not None and (run_dir / "tokenizer").exists():
        return load_tokenizer_snapshot(run_dir, cfg), None, None
    return None, None, None


def _tokenizer_files_to_copy(run_dir: Path | None) -> tuple[Path, ...]:
    """List a run's tokenizer snapshot files, refusing a layout export cannot ship.

    Non-destructive, and called before anything is written: both refusals below
    are knowable from the source alone, and discovering either one halfway
    through the copy would mean the damage is already done.

    :param Path | None run_dir: Run directory, when one was found.
    :raises RuntimeError: If the snapshot is nested or collides with an export file.
    :return tuple[Path, ...]: Sorted snapshot files, empty when there is no snapshot.
    """
    if run_dir is None:
        return ()
    tok_dir = run_dir / "tokenizer"
    if not tok_dir.is_dir():
        return ()

    sources = sorted(tok_dir.iterdir())
    for source in sources:
        if source.is_dir():
            # The snapshot is flat by construction (``save_pretrained`` writes
            # into one directory), and the identity manifest hashes the tree
            # with ``rglob``. Copying only the top level would ship a tokenizer
            # missing files that its own identity says it has.
            raise RuntimeError(
                f"The run's tokenizer snapshot contains a subdirectory ({source.name}), "
                "which chomp's flat export layout cannot represent. Export supports "
                "the single-directory snapshot save_pretrained writes."
            )
        if source.name in _RESERVED_EXPORT_NAMES:
            raise RuntimeError(
                f"The run's tokenizer snapshot contains a {source.name}, which collides "
                "with a file chomp writes for this export. One of the two would "
                "silently win; move the tokenizer file aside and export again."
            )
    return tuple(sources)


def _copy_tokenizer_files(sources: tuple[Path, ...], export_dir: Path) -> tuple[str, ...]:
    """Copy a run's tokenizer snapshot into the export root.

    The files are copied rather than re-serialized: the identity manifest
    hashes their exact bytes, so a round-trip through ``save_pretrained`` could
    invalidate the identity that :func:`_tokenizer_for_checkpoint` just proved.

    :param tuple[Path, ...] sources: Snapshot files from :func:`_tokenizer_files_to_copy`.
    :param Path export_dir: Destination export directory.
    :return tuple[str, ...]: Sorted names of the copied files.
    """
    for source in sources:
        shutil.copy2(source, export_dir / source.name)
    return tuple(source.name for source in sources)


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


def _require_policy_equivalence(config: Any) -> None:
    """Refuse a policy-dtype export that would not be inference-equivalent.

    The variant is only worth having because it changes nothing: every tensor
    it stores at bf16 is one the forward pass casts to bf16 before use, so the
    bits it drops never participated in the computation. Two configurations
    break that, and both are visible here rather than in the output:

    ``compute_dtype`` other than bf16 means the forward pass never casts, so
    every dropped bit is one the model would have used.

    ``scale_emb`` multiplies the gathered embedding rows *before* the cast to
    ``compute_dtype`` (upstream ``model.py``), so fp32 storage computes
    ``bf16(w * scale)`` where bf16 storage computes ``bf16(w) * bf16(scale)``.
    Measured on a smoke model those differ in 2047 of 2048 logits.

    Neither is repairable by holding the affected tensor at fp32: the file's
    dtypes must match what upstream builds from the config in its header, or
    ``load_checkpoint`` rejects it.

    :param Any config: ``MegalodonConfig`` the weights will be written under.
    :raises RuntimeError: If a policy-dtype export would change model outputs.
    """
    if config.compute_dtype != jax.numpy.bfloat16:
        raise RuntimeError(
            f"--dtype {DTYPE_POLICY} needs model.compute_dtype='bfloat16'; this checkpoint "
            f"computes in {jax.numpy.dtype(config.compute_dtype).name}. Storing bf16 would "
            "drop bits the forward pass uses, so the export would not match the checkpoint. "
            f"Export it at --dtype {DTYPE_FLOAT32}."
        )
    if config.scale_emb:
        raise RuntimeError(
            f"--dtype {DTYPE_POLICY} is not inference-equivalent when model.scale_emb is set: "
            "the embedding is scaled before the cast to compute_dtype, so bf16 storage rounds "
            f"at a different point and changes the logits. Export it at --dtype {DTYPE_FLOAT32}."
        )


def _policy_dtype_model(model: Any) -> tuple[Any, dict[str, Any]]:
    """Restate a model's parameters at the dtypes upstream's bf16 policy assigns.

    The cast set is not reconstructed here, and no parameter name is matched
    against a pattern. Upstream decides which leaves its bf16 policy keeps at
    fp32 -- normalization, CEMA, and affine parameters -- and encodes that
    decision in its model constructor. So this asks the constructor: it builds
    the same architecture at ``param_dtype=bfloat16`` and reads the dtype of
    every leaf back out. That skeleton *is* the policy applied to this exact
    parameter tree, including the leaves that only exist under some configs.

    ``apply_model_state_dict`` then validates every tensor's shape *and dtype*
    against that skeleton, so a cast set that disagreed with the policy fails
    here rather than becoming a file that loads and is quietly wrong.

    :param Any model: Restored ``MegalodonForCausalLM`` at ``param_dtype`` float32.
    :raises RuntimeError: If the export would not be inference-equivalent.
    :return tuple[Any, dict[str, Any]]: Policy-dtype model and its dtype summary.
    """
    from megalodon_jax.checkpoint import apply_model_state_dict, model_state_dict
    from megalodon_jax.model import MegalodonForCausalLM

    _require_policy_equivalence(model.config)

    policy_config = replace(model.config, param_dtype=jax.numpy.bfloat16)
    skeleton = MegalodonForCausalLM(policy_config, key=jax.random.key(0))
    template = model_state_dict(skeleton)
    source = model_state_dict(model)
    if set(source) != set(template):
        raise RuntimeError(
            "megalodon-jax builds a different parameter set at bf16 storage than at fp32 "
            f"({sorted(set(source) ^ set(template))}); chomp cannot map one onto the other."
        )

    cast = {name: array.astype(template[name].dtype) for name, array in source.items()}
    policy_model, _report = apply_model_state_dict(skeleton, cast)
    return policy_model, _dtype_summary(cast)


def _dtype_summary(tensors: dict[str, Any]) -> dict[str, Any]:
    """Describe the per-tensor dtypes of a weights file for the manifest.

    :param dict[str, Any] tensors: Native name -> array mapping being written.
    :return dict[str, Any]: Counts and bytes per dtype, plus the fp32 exception list.
    """
    by_dtype: dict[str, dict[str, int]] = {}
    for array in tensors.values():
        entry = by_dtype.setdefault(str(array.dtype), {"tensors": 0, "bytes": 0})
        entry["tensors"] += 1
        entry["bytes"] += array.nbytes

    summary: dict[str, Any] = {"by_dtype": dict(sorted(by_dtype.items()))}
    if len(by_dtype) > 1:
        # Only meaningful for a mixed file, where it is the exception list a
        # reader has to honor. For a single-dtype file it would just restate
        # every tensor name.
        summary["fp32_tensors"] = sorted(
            name for name, array in tensors.items() if array.dtype == jax.numpy.float32
        )
    return summary


def _weights_metadata(weights_path: Path) -> dict[str, str]:
    """Read the safetensors header metadata of a written weights file.

    :param Path weights_path: SafeTensors file to inspect.
    :raises RuntimeError: If the file carries no readable metadata.
    :return dict[str, str]: Header metadata map.
    """
    from safetensors import safe_open

    # ``numpy`` reads the header without materializing tensors or pulling in a
    # second array framework.
    with safe_open(str(weights_path), framework="numpy") as handle:
        metadata = handle.metadata()
    if not metadata:
        raise RuntimeError(f"{weights_path} has no safetensors metadata to describe it.")
    return dict(metadata)


def _write_hf_config(weights_path: Path, destination: Path) -> Path:
    """Write ``config.json`` describing the weights file beside it.

    Built from the header of the file it sits next to, never from the chomp
    config that produced it. Upstream stores the complete ``MegalodonConfig``
    in that header, so reading it back is what makes the two impossible to
    disagree: this function cannot describe a model it did not just write.

    The result is Hugging Face shaped -- ``model_type``, ``architectures``,
    ``torch_dtype``, and the standard size fields -- so a PyTorch
    ``PretrainedConfig`` subclass can consume it directly, while every
    Megalodon-specific field keeps its upstream name.

    :param Path weights_path: Weights file this config describes.
    :param Path destination: Export directory to write into.
    :raises RuntimeError: If the header is missing its config or an alias has no source.
    :return Path: The written ``config.json``.
    """
    metadata = _weights_metadata(weights_path)
    payload = metadata.get("config_json")
    if not payload:
        raise RuntimeError(
            f"{weights_path} carries no config_json; megalodon-jax did not write it, "
            "so chomp cannot describe the architecture it contains."
        )
    native = json.loads(payload)

    config: dict[str, Any] = dict(native)
    for hf_name, native_name in _HF_ALIASES.items():
        if native_name not in native:
            raise RuntimeError(
                f"megalodon-jax no longer records {native_name!r}; the Hugging Face "
                f"alias {hf_name!r} in chomp's exporter needs updating."
            )
        config[hf_name] = native[native_name]

    if "param_dtype" not in native:
        raise RuntimeError(
            "megalodon-jax no longer records 'param_dtype'; the 'torch_dtype' field "
            "in chomp's exporter needs updating."
        )

    config["model_type"] = HF_MODEL_TYPE
    config["architectures"] = [HF_ARCHITECTURE]
    # Ordinary parameter storage. Upstream's bf16 policy keeps normalization,
    # CEMA, and affine parameters at fp32 regardless, so this is the dtype of
    # the bulk of the file rather than of every tensor in it.
    config["torch_dtype"] = native["param_dtype"]
    config["megalodon_jax"] = {
        "weights_file": weights_path.name,
        **megalodon_jax_identity(),
        **{key: metadata[key] for key in _WEIGHT_CONTRACT_KEYS if key in metadata},
    }

    # An alias that collided with a native field would silently redefine the
    # architecture for every reader of this file.
    if {key: config[key] for key in native} != native:
        raise RuntimeError(
            "Hugging Face aliases overwrote a native megalodon field in config.json; "
            "the exported architecture description is not trustworthy."
        )

    config_path = destination / CONFIG_FILENAME
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return config_path


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
    if not destination.exists():
        return ()
    if not destination.is_dir():
        raise FileExistsError(
            f"{destination} exists and is not a directory, so chomp cannot export into it. "
            "Choose an empty or new directory."
        )
    if not any(destination.iterdir()):
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
    owned = [
        *previous.get("tokenizer_files", []),
        previous.get("weights_file", ""),
        previous.get("config_file", ""),
    ]
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
    keep = {*written, WEIGHTS_FILENAME, MANIFEST_FILENAME, CONFIG_FILENAME}
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
    dtype: str = DTYPE_FLOAT32,
) -> ExportResult:
    """Write one training checkpoint out as a portable safetensors model.

    :param str | Path checkpoint: Run directory, checkpoint root, or step directory.
    :param str | Path export_dir: Destination directory, created if absent.
    :param str | None config_override: Optional config file replacing the checkpoint's.
    :param bool overwrite: Whether to replace an existing export in the destination.
    :param bool verify: Whether to reload the written file and compare parameters.
    :param str dtype: ``float32`` for the master weights, ``policy`` for the derived variant.
    :raises FileNotFoundError: If no checkpoint can be resolved.
    :raises ValueError: If ``dtype`` is not a known variant.
    :raises RuntimeError: If the backend is unsupported, or the tokenizer cannot be
        proven to match the checkpoint or has a layout the export cannot ship.
    :raises FileExistsError: If the destination is non-empty and cannot be safely replaced.
    :return ExportResult: Description of what was written.
    """
    from megalodon_jax import save_checkpoint
    from megalodon_jax.checkpoint import model_state_dict

    if dtype not in _WEIGHTS_DTYPE_LABELS:
        raise ValueError(
            f"Unknown export dtype {dtype!r}; expected one of {sorted(_WEIGHTS_DTYPE_LABELS)}."
        )

    step_dir, run_dir = resolve_checkpoint_path(checkpoint)
    cfg = load_config_for_checkpoint(step_dir=step_dir, config_override=config_override)
    if cfg.model.backend != "megalodon":
        raise RuntimeError(
            "export only supports model.backend='megalodon'. "
            f"Found {cfg.model.backend!r} in the checkpoint config."
        )

    destination = Path(export_dir)
    previous_files = _check_destination(destination, overwrite=overwrite)

    tokenizer, tokenizer_identity, tokenizer_manifest = _tokenizer_for_checkpoint(
        step_dir=step_dir, run_dir=run_dir, cfg=cfg
    )
    # Listed and checked here so a snapshot export cannot ship is refused while
    # the destination is still untouched.
    tokenizer_sources = _tokenizer_files_to_copy(run_dir)
    # Vocabulary padding and special-token IDs are resolved here exactly as
    # training resolved them, so the config stored beside the weights is the
    # one the restored arrays were actually shaped by.
    cfg, _tokenizer = prepare_tokenizer_and_config(cfg, tokenizer=tokenizer)

    # Export moves bytes; it computes nothing. Doing it on the host keeps a
    # second full copy of the parameters off the accelerator, which matters
    # because the export at the end of a run competes with a memory pool sized
    # for that run's training step and holding its optimizer state.
    with jax.default_device(jax.devices("cpu")[0]):
        params, static = build_model(cfg, key=jax.random.key(0))
        params = restore_params_only(step_dir, abstractify_tree(params))
        model = eqx.combine(params, static)

        if dtype == DTYPE_POLICY:
            # Raises before anything is written if this checkpoint's config
            # would make the variant something other than a re-encoding.
            model, dtype_summary = _policy_dtype_model(model)
            params, _static_unused = eqx.partition(model, eqx.is_array)
        else:
            dtype_summary = _dtype_summary(model_state_dict(model))

        destination.mkdir(parents=True, exist_ok=True)
        weights_path = destination / WEIGHTS_FILENAME

        # The manifest is what makes a directory an export, so it is dropped
        # before the weights it describes are replaced. Everything after the
        # write can still fail -- the tokenizer copy, config.json, the manifest
        # write itself -- and without this the window leaves new weights beside
        # the previous export's config and tokenizer, which is a pairing
        # ``load_export`` accepts whenever the two share an architecture.
        # Failing inside the window now leaves a directory that is loudly not
        # an export instead.
        (destination / MANIFEST_FILENAME).unlink(missing_ok=True)
        save_checkpoint(model, weights_path)
        if verify:
            _verify_weights(weights_path, params)

    tokenizer_files = _copy_tokenizer_files(tokenizer_sources, destination)
    if tokenizer_manifest is not None:
        _write_tokenizer_manifest(destination, tokenizer_manifest)
    config_path = _write_hf_config(weights_path, destination)
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
        "config_file": CONFIG_FILENAME,
        "weights_dtype": _WEIGHTS_DTYPE_LABELS[dtype],
        "dtype_summary": dtype_summary,
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
        config_path=config_path,
        step=step,
        param_count=manifest["param_count"],
        weights_bytes=weights_path.stat().st_size,
        weights_dtype=manifest["weights_dtype"],
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
    if manifest["weights_dtype"] == _WEIGHTS_DTYPE_LABELS[DTYPE_POLICY]:
        # The chomp config records how the model was trained -- fp32 master
        # weights -- and the manifest says this file re-encoded them. Comparing
        # without that adjustment would reject every policy export.
        expected = replace(expected, param_dtype=jax.numpy.bfloat16)
    if model.config != expected:
        raise RuntimeError(
            f"{MANIFEST_FILENAME} describes a different model than {weights_path.name} "
            "contains. The export directory is inconsistent; re-export it."
        )

    params, static = eqx.partition(model, eqx.is_array)
    return LoadedExport(params=params, static=static, config=cfg, manifest=manifest)


def _verify_exported_tokenizer(directory: Path, identity: dict[str, Any]) -> None:
    """Check the shipped tokenizer files against the identity the export recorded.

    Export refuses a tokenizer it cannot prove matches the checkpoint, but
    nothing re-checked that proof on the way back in: ``HFTokenizer`` loads
    whichever files are present. Everything needed is already in the directory
    -- the copied ``identity.json`` records a SHA-256 per file, and hashing it
    reproduces the compact identity stored in the export manifest -- so this
    verifies rather than trusts.

    :param Path directory: Export directory.
    :param dict[str, Any] identity: ``tokenizer_identity`` from the export manifest.
    :raises FileNotFoundError: If the identity manifest was not shipped.
    :raises RuntimeError: If a tokenizer file is missing or does not match.
    """
    from chomp.data.pipeline import (
        TOKENIZER_MANIFEST_FILENAME,
        sha256_file,
        tokenizer_checkpoint_identity,
    )

    manifest_path = directory / TOKENIZER_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Export {directory} records a tokenizer identity but ships no "
            f"{TOKENIZER_MANIFEST_FILENAME}, so its tokenizer files cannot be "
            "checked against the weights. Re-export it."
        )
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Corrupted tokenizer identity in {manifest_path}: {exc}") from exc

    if tokenizer_checkpoint_identity(manifest) != identity:
        raise RuntimeError(
            f"{TOKENIZER_MANIFEST_FILENAME} in {directory} is not the one this export "
            f"recorded in {MANIFEST_FILENAME}. The directory pairs a tokenizer with "
            "weights it was not exported beside; re-export it."
        )

    for record in manifest.get("files", []):
        path = directory / record["path"]
        if not path.is_file():
            raise RuntimeError(
                f"Export {directory} is missing tokenizer file {record['path']!r}, "
                "which its identity says these weights were trained with."
            )
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(
                f"Tokenizer file {record['path']!r} in {directory} does not match the "
                "identity recorded for these weights; its token IDs may not index the "
                "embedding rows they were trained against."
            )


def load_export_tokenizer(export_dir: str | Path, cfg: Config) -> Tokenizer:
    """Load the tokenizer shipped inside an export directory.

    :param str | Path export_dir: Export directory holding tokenizer files.
    :param Config cfg: Config from the export manifest.
    :raises FileNotFoundError: If a checkpoint-bound tokenizer identity was not shipped.
    :raises RuntimeError: If the shipped files do not match the recorded identity.
    :return Tokenizer: Tokenizer bound to these weights.
    """
    from chomp.data.pipeline import ByteTokenizer, HFTokenizer

    directory = Path(export_dir)
    if cfg.data.tokenizer.kind == "byte":
        # Built from the config, not from files, so there is nothing shipped to
        # verify -- and no vocabulary to get wrong.
        return ByteTokenizer(byte_offset=cfg.data.tokenizer.byte_offset)

    identity = read_export_manifest(directory).get("tokenizer_identity")
    if isinstance(identity, dict):
        _verify_exported_tokenizer(directory, identity)

    return HFTokenizer(
        str(directory),
        use_fast=cfg.data.tokenizer.hf_use_fast,
        trust_remote_code=cfg.data.tokenizer.hf_trust_remote_code,
        local_files_only=True,
    )
