"""Checkpointing (Orbax) for chomp.

Senior-engineer stance:
- "Resume" is not a nice-to-have. It's a contract.
- If you can't restore train_state (params+opt_state+rng+step) you don't have
  a training system; you have a demo.

This module is intentionally small and opinionated. It wraps Orbax in a way
that keeps the rest of the codebase boring.

We save three logical things:
- train_state: arrays-only pytree (TrainState)
- data_state: JSON dict (stream position + packer buffer), small but essential
- meta: JSON dict with versions/config fingerprint

Orbax notes:
- We pin orbax-checkpoint in pyproject.toml to avoid API drift.
- We use the newer `args=` API (not deprecated `items=`).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import orbax.checkpoint as ocp

from chomp.config import Config
from chomp.data import data_fingerprint

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CheckpointMeta:
    """Metadata stored alongside checkpoints.

    Keep this JSON-serializable.
    """

    step: int
    timestamp: str

    # Versions for debugging (not for strict gating in v0)
    python: str
    jax: str | None
    orbax: str | None
    chomp: str
    megalodon_jax: str | None

    # Repro snapshot
    config: dict[str, Any]

    # Minimal data fingerprint
    data_fingerprint: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to a JSON-serializable dictionary.

        :return dict[str, Any]: All fields as a nested dict.
        """
        return asdict(self)


def _safe_version(pkg: str) -> str | None:
    """Get package version string, returning None if not installed.

    :param str pkg: Package name to look up.
    :return str | None: Version string or None if unavailable.
    """
    try:
        import importlib.metadata as im

        return im.version(pkg)
    except Exception:
        return None


def build_meta(
    *, step: int, config: dict[str, Any], data_fingerprint: dict[str, Any]
) -> CheckpointMeta:
    """Build checkpoint metadata with version info and config snapshot.

    :param int step: Current training step.
    :param dict[str, Any] config: Full config dict for reproducibility.
    :param dict[str, Any] data_fingerprint: Data pipeline fingerprint.
    :return CheckpointMeta: Populated metadata object.
    """
    import platform

    return CheckpointMeta(
        step=int(step),
        timestamp=datetime.now().isoformat(timespec="seconds"),
        python=platform.python_version(),
        jax=_safe_version("jax"),
        orbax=_safe_version("orbax-checkpoint"),
        chomp=_safe_version("chomp") or "0.0.0",
        megalodon_jax=_safe_version("megalodon-jax"),
        config=config,
        data_fingerprint=data_fingerprint,
    )


def default_ckpt_dir(run_dir: Path) -> Path:
    """Return the default checkpoint directory for a run.

    :param Path run_dir: Run directory path.
    :return Path: Path to checkpoints subdirectory.
    """
    return run_dir / "checkpoints"


def _dir_size_bytes(path: Path) -> int:
    """Calculate total size of all files in a directory recursively.

    :param Path path: Directory to measure.
    :return int: Total size in bytes.
    """
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except FileNotFoundError:
            # async / concurrent cleanup
            pass
    return total


def make_manager(
    ckpt_dir: Path, *, max_to_keep: int, save_every: int, async_save: bool
) -> ocp.CheckpointManager:
    """Create an Orbax CheckpointManager.

    We keep this wrapper here so Orbax API drift is contained.

    :param Path ckpt_dir: Directory for checkpoint storage.
    :param int max_to_keep: Maximum number of checkpoints to retain.
    :param int save_every: Step interval for checkpoint saves.
    :param bool async_save: Whether to enable asynchronous saving.
    :return ocp.CheckpointManager: Configured checkpoint manager.
    """

    import orbax.checkpoint as ocp

    ckpt_dir = Path(ckpt_dir).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_to_keep,
        save_interval_steps=save_every,
        create=True,
        enable_async_checkpointing=async_save,
    )

    # item_names defines what keys are expected in Composite save/restore.
    mgr = ocp.CheckpointManager(
        directory=str(ckpt_dir),
        item_names=("train_state", "data_state", "meta"),
        options=options,
    )
    return mgr


def save(
    manager: ocp.CheckpointManager,
    *,
    step: int,
    train_state: Any,
    data_state: dict[str, Any],
    meta: CheckpointMeta,
    enforce_size_gb: float | None = None,
) -> None:
    """Save a checkpoint.

    - `train_state` is saved via StandardSave (PyTree)
    - `data_state` and `meta` via JsonSave

    `enforce_size_gb` is a guardrail to catch saving static graphs or duplicating tensors.
    It's intentionally a blunt instrument.

    :param ocp.CheckpointManager manager: Orbax checkpoint manager.
    :param int step: Training step number.
    :param Any train_state: TrainState pytree (arrays only).
    :param dict[str, Any] data_state: Data iterator state dict.
    :param CheckpointMeta meta: Checkpoint metadata.
    :param enforce_size_gb: Optional max size in GB; raises if exceeded.
    :raises RuntimeError: If checkpoint size exceeds enforce_size_gb.
    """

    import orbax.checkpoint as ocp

    step = int(step)

    manager.save(
        step,
        args=ocp.args.Composite(
            train_state=ocp.args.StandardSave(train_state),
            data_state=ocp.args.JsonSave(data_state),
            meta=ocp.args.JsonSave(meta.to_dict()),
        ),
    )

    if enforce_size_gb is not None:
        # Best-effort size check after save finishes.
        manager.wait_until_finished()
        ckpt_path = Path(manager.directory) / str(step)
        size_gb = _dir_size_bytes(ckpt_path) / 1e9
        if size_gb > enforce_size_gb:
            raise RuntimeError(
                f"Checkpoint at step {step} is {size_gb:.2f}GB, exceeding {enforce_size_gb:.2f}GB. "
                "This usually means you're saving non-array static state or duplicating tensors."
            )


def restore_latest(
    manager: ocp.CheckpointManager,
    *,
    abstract_train_state: Any,
) -> tuple[int, Any, dict[str, Any] | None, dict[str, Any] | None]:
    """Restore latest checkpoint.

    Notes:
    - `abstract_train_state` should be a tree of ShapeDtypeStruct matching TrainState.
    - `data_state` and `meta` are JSON dicts.

    :param ocp.CheckpointManager manager: Orbax checkpoint manager.
    :param Any abstract_train_state: ShapeDtypeStruct tree for restoration target.
    :raises FileNotFoundError: If no checkpoints exist.
    :return tuple: (step, train_state, data_state, meta).
    """

    import orbax.checkpoint as ocp

    latest = manager.latest_step()
    if latest is None:
        raise FileNotFoundError(f"No checkpoints found in {manager.directory}")

    restored = manager.restore(
        latest,
        args=ocp.args.Composite(
            train_state=ocp.args.StandardRestore(abstract_train_state),
            data_state=ocp.args.JsonRestore(),
            meta=ocp.args.JsonRestore(),
        ),
    )

    # Orbax returns a dict-like mapping for Composite.
    train_state = restored["train_state"]
    data_state = restored.get("data_state")
    meta = restored.get("meta")

    return int(latest), train_state, data_state, meta


def restore_at_step(
    manager: ocp.CheckpointManager,
    *,
    step: int,
    abstract_train_state: Any,
) -> tuple[int, Any, dict[str, Any] | None, dict[str, Any] | None]:
    """Restore checkpoint at a specific step.

    :param ocp.CheckpointManager manager: Orbax checkpoint manager.
    :param int step: Step number to restore.
    :param Any abstract_train_state: ShapeDtypeStruct tree for restoration target.
    :return tuple: (step, train_state, data_state, meta).
    """
    import orbax.checkpoint as ocp

    step = int(step)
    restored = manager.restore(
        step,
        args=ocp.args.Composite(
            train_state=ocp.args.StandardRestore(abstract_train_state),
            data_state=ocp.args.JsonRestore(),
            meta=ocp.args.JsonRestore(),
        ),
    )
    train_state = restored["train_state"]
    data_state = restored.get("data_state")
    meta = restored.get("meta")
    return step, train_state, data_state, meta


def check_resume_compat(cfg: Config, meta: dict[str, Any] | None) -> None:
    """Validate checkpoint metadata against current config.

    :param Config cfg: Current training configuration.
    :param meta: Checkpoint metadata dict (or None if missing).
    :raises RuntimeError: If meta is missing or config mismatches are found.
    """

    if meta is None:
        raise RuntimeError("Checkpoint meta is missing; cannot verify resume compatibility.")

    meta_cfg = meta.get("config")
    meta_fp = meta.get("data_fingerprint")
    if not isinstance(meta_cfg, dict) or not isinstance(meta_fp, dict):
        raise RuntimeError(
            "Checkpoint meta is missing config/data_fingerprint; cannot verify resume compatibility."
        )

    errors: list[str] = []
    warnings: list[str] = []

    def _cmp(path: str, cur: Any, prev: Any, *, severity: str) -> None:
        """Compare current and previous values, appending to errors or warnings.

        :param str path: Config path being compared.
        :param Any cur: Current config value.
        :param Any prev: Previous (checkpoint) config value.
        :param str severity: Either "error" or "warning".
        """
        if cur != prev:
            msg = f"{path} mismatch (checkpoint={prev!r}, current={cur!r})"
            if severity == "error":
                errors.append(msg)
            else:
                warnings.append(msg)

    cur_fp = data_fingerprint(cfg)

    # Data source comparisons.
    src_prev = meta_fp.get("source") or {}
    src_cur = cur_fp.get("source") or {}
    _cmp("data.source.backend", src_cur.get("backend"), src_prev.get("backend"), severity="error")

    if src_cur.get("backend") == "hf":
        _cmp("data.hf_dataset", src_cur.get("dataset"), src_prev.get("dataset"), severity="error")
        _cmp("data.hf_name", src_cur.get("name"), src_prev.get("name"), severity="error")
        _cmp("data.hf_split", src_cur.get("split"), src_prev.get("split"), severity="error")
        _cmp("data.text_key", src_cur.get("text_key"), src_prev.get("text_key"), severity="error")
        _cmp("data.shuffle", src_cur.get("shuffle"), src_prev.get("shuffle"), severity="error")
        _cmp("data.seed", src_cur.get("seed"), src_prev.get("seed"), severity="error")
        _cmp(
            "data.shuffle_buffer_size",
            src_cur.get("shuffle_buffer_size"),
            src_prev.get("shuffle_buffer_size"),
            severity="warning",
        )
    elif src_cur.get("backend") == "local_text":
        _cmp(
            "data.local_text_hash",
            src_cur.get("local_text_hash"),
            src_prev.get("local_text_hash"),
            severity="error",
        )
        _cmp("data.repeat", src_cur.get("repeat"), src_prev.get("repeat"), severity="error")

    # Tokenizer comparisons.
    tok_prev = meta_fp.get("tokenizer") or {}
    tok_cur = cur_fp.get("tokenizer") or {}
    _cmp("tokenizer.kind", tok_cur.get("kind"), tok_prev.get("kind"), severity="error")

    if tok_cur.get("kind") == "hf":
        _cmp(
            "tokenizer.hf_name_or_path",
            tok_cur.get("hf_name_or_path"),
            tok_prev.get("hf_name_or_path"),
            severity="error",
        )
        _cmp(
            "tokenizer.hf_use_fast",
            tok_cur.get("hf_use_fast"),
            tok_prev.get("hf_use_fast"),
            severity="error",
        )
        _cmp(
            "tokenizer.hf_trust_remote_code",
            tok_cur.get("hf_trust_remote_code"),
            tok_prev.get("hf_trust_remote_code"),
            severity="error",
        )
    elif tok_cur.get("kind") == "byte":
        _cmp(
            "tokenizer.byte_offset",
            tok_cur.get("byte_offset"),
            tok_prev.get("byte_offset"),
            severity="error",
        )

    _cmp("tokenizer.add_bos", tok_cur.get("add_bos"), tok_prev.get("add_bos"), severity="error")
    _cmp("tokenizer.add_eos", tok_cur.get("add_eos"), tok_prev.get("add_eos"), severity="error")
    _cmp(
        "tokenizer.max_doc_tokens",
        tok_cur.get("max_doc_tokens"),
        tok_prev.get("max_doc_tokens"),
        severity="error",
    )
    _cmp(
        "tokenizer.vocab_size_multiple",
        tok_cur.get("vocab_size_multiple"),
        tok_prev.get("vocab_size_multiple"),
        severity="error",
    )
    _cmp(
        "tokenizer.auto_set_special_tokens",
        tok_cur.get("auto_set_special_tokens"),
        tok_prev.get("auto_set_special_tokens"),
        severity="error",
    )

    # Packing/loss behavior comparisons.
    pack_prev = meta_fp.get("packing") or {}
    pack_cur = cur_fp.get("packing") or {}
    _cmp(
        "data.mask_boundary_loss",
        pack_cur.get("mask_boundary_loss"),
        pack_prev.get("mask_boundary_loss"),
        severity="error",
    )
    _cmp(
        "data.train_on_eos",
        pack_cur.get("train_on_eos"),
        pack_prev.get("train_on_eos"),
        severity="error",
    )

    # Batch shape invariants.
    _cmp("train.seq_len", cur_fp.get("seq_len"), meta_fp.get("seq_len"), severity="error")
    _cmp(
        "train.batch_size",
        cur_fp.get("batch_size"),
        meta_fp.get("batch_size"),
        severity="error",
    )
    _cmp(
        "train.grad_accum",
        cur_fp.get("grad_accum"),
        meta_fp.get("grad_accum"),
        severity="error",
    )

    # Model/optimizer comparisons.
    cur_cfg = cfg.to_dict()
    model_prev = meta_cfg.get("model") or {}
    model_cur = cur_cfg.get("model") or {}
    for key in sorted(set(model_prev) | set(model_cur)):
        _cmp(f"model.{key}", model_cur.get(key), model_prev.get(key), severity="error")

    optim_prev = meta_cfg.get("optim") or {}
    optim_cur = cur_cfg.get("optim") or {}
    for key in sorted(set(optim_prev) | set(optim_cur)):
        _cmp(f"optim.{key}", optim_cur.get(key), optim_prev.get(key), severity="error")

    if warnings:
        logger.warning(
            "Resume config warnings:\n%s",
            "\n".join(f"- {msg}" for msg in warnings),
        )

    if errors:
        detail = "\n".join(f"- {msg}" for msg in errors)
        raise RuntimeError(f"Resume config mismatch:\n{detail}")
