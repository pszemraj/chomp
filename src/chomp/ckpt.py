# SPDX-License-Identifier: Apache-2.0

"""Checkpointing (Orbax) for chomp.

Senior-engineer stance:
- "Resume" is not a nice-to-have. It's a contract.
- If you can't restore train_state (params+opt_state+rng+step) you don't have
  a training system; you have a demo.

This module is intentionally small and opinionated. It wraps Orbax in a way
that keeps the rest of the codebase boring.

We save three logical things:
- train_state: arrays-only pytree (TrainState)
- data_state: iterator state via Grain checkpoint handler
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

from chomp import __version__ as _chomp_version
from chomp.config import Config, decay_horizon_from_values
from chomp.data import data_fingerprint

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CheckpointMeta:
    """Metadata stored alongside checkpoints.

    Keep this JSON-serializable.
    """

    step: int
    timestamp: str
    tokens_seen: int | None

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
    *,
    step: int,
    config: dict[str, Any],
    data_fingerprint: dict[str, Any],
    tokens_seen: int | None = None,
) -> CheckpointMeta:
    """Build checkpoint metadata with version info and config snapshot.

    :param int step: Current training step.
    :param dict[str, Any] config: Full config dict for reproducibility.
    :param dict[str, Any] data_fingerprint: Data pipeline fingerprint.
    :param int | None tokens_seen: Optional cumulative token count for resume accounting.
    :return CheckpointMeta: Populated metadata object.
    """
    import platform

    return CheckpointMeta(
        step=int(step),
        timestamp=datetime.now().isoformat(timespec="seconds"),
        python=platform.python_version(),
        jax=_safe_version("jax"),
        orbax=_safe_version("orbax-checkpoint"),
        chomp=_chomp_version,
        megalodon_jax=_safe_version("megalodon-jax"),
        tokens_seen=int(tokens_seen) if tokens_seen is not None else None,
        config=config,
        data_fingerprint=data_fingerprint,
    )


def default_ckpt_dir(run_dir: Path) -> Path:
    """Return the default checkpoint directory for a run.

    :param Path run_dir: Run directory path.
    :return Path: Path to checkpoints subdirectory.
    """
    return run_dir / "checkpoints"


def resolve_ckpt_root(cfg: Config, run_dir: Path) -> Path:
    """Resolve the checkpoint root directory for a run.

    checkpoint.root_dir wins when set (relative paths resolve against
    run_dir); otherwise the default `<run_dir>/checkpoints`.

    :param Config cfg: Training configuration.
    :param Path run_dir: Run directory path.
    :return Path: Checkpoint root directory.
    """
    if cfg.checkpoint.root_dir:
        root = Path(cfg.checkpoint.root_dir)
        return root if root.is_absolute() else run_dir / root
    return default_ckpt_dir(run_dir)


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


def _checkpoint_target(data_iter: Any) -> Any:
    """Return the iterator object to pass to Grain's checkpoint handler.

    :param Any data_iter: Training data iterator.
    :return Any: Iterator object compatible with Grain checkpointing.
    """
    if hasattr(data_iter, "checkpoint_target"):
        return data_iter.checkpoint_target()
    return data_iter


def save(
    manager: ocp.CheckpointManager,
    *,
    step: int,
    train_state: Any,
    data_iter: Any,
    meta: CheckpointMeta,
    force: bool = False,
) -> None:
    """Save a checkpoint.

    - `train_state` is saved via StandardSave (PyTree)
    - `data_state` via Grain's checkpoint handler
    - `meta` via JsonSave

    Data-iterator state is serialized synchronously inside `manager.save()`:
    grain's CheckpointHandler is not an Orbax AsyncCheckpointHandler, so the
    composite handler runs it in the blocking phase. Async checkpointing
    therefore cannot race the training loop advancing the iterator (pinned
    by test_grain_data_state_capture_is_synchronous).

    :param ocp.CheckpointManager manager: Orbax checkpoint manager.
    :param int step: Training step number.
    :param Any train_state: TrainState pytree (arrays only).
    :param Any data_iter: Data iterator to checkpoint via Grain's handler.
    :param CheckpointMeta meta: Checkpoint metadata.
    :param bool force: If True, force a save even if the step is off-interval.
    """

    import grain.checkpoint as gcp
    import orbax.checkpoint as ocp

    manager.save(
        int(step),
        args=ocp.args.Composite(
            train_state=ocp.args.StandardSave(train_state),
            data_state=gcp.CheckpointSave(_checkpoint_target(data_iter)),
            meta=ocp.args.JsonSave(meta.to_dict()),
        ),
        force=force,
    )


def restore_latest(
    manager: ocp.CheckpointManager,
    *,
    abstract_train_state: Any,
    data_iter: Any,
) -> tuple[int, Any, dict[str, Any] | None]:
    """Restore latest checkpoint.

    Notes:
    - `abstract_train_state` should be a tree of ShapeDtypeStruct matching TrainState.
    - `data_state` is restored via Grain's checkpoint handler.

    :param ocp.CheckpointManager manager: Orbax checkpoint manager.
    :param Any abstract_train_state: ShapeDtypeStruct tree for restoration target.
    :param Any data_iter: Data iterator to restore via Grain's handler.
    :raises FileNotFoundError: If no checkpoints exist.
    :return tuple: (step, train_state, meta).
    """

    latest = manager.latest_step()
    if latest is None:
        raise FileNotFoundError(f"No checkpoints found in {manager.directory}")

    return _restore_step(
        manager,
        step=int(latest),
        abstract_train_state=abstract_train_state,
        data_iter=data_iter,
    )


def restore_at_step(
    manager: ocp.CheckpointManager,
    *,
    step: int,
    abstract_train_state: Any,
    data_iter: Any,
) -> tuple[int, Any, dict[str, Any] | None]:
    """Restore checkpoint at a specific step.

    :param ocp.CheckpointManager manager: Orbax checkpoint manager.
    :param int step: Step number to restore.
    :param Any abstract_train_state: ShapeDtypeStruct tree for restoration target.
    :param Any data_iter: Data iterator to restore via Grain's handler.
    :return tuple: (step, train_state, meta).
    """

    return _restore_step(
        manager,
        step=int(step),
        abstract_train_state=abstract_train_state,
        data_iter=data_iter,
    )


def _restore_step(
    manager: ocp.CheckpointManager,
    *,
    step: int,
    abstract_train_state: Any,
    data_iter: Any,
) -> tuple[int, Any, dict[str, Any] | None]:
    """Restore checkpoint at the specified step.

    :param ocp.CheckpointManager manager: Orbax checkpoint manager.
    :param int step: Step number to restore.
    :param Any abstract_train_state: ShapeDtypeStruct tree for restoration target.
    :param Any data_iter: Data iterator to restore via Grain's handler.
    :return tuple: (step, train_state, meta).
    """
    import grain.checkpoint as gcp
    import orbax.checkpoint as ocp

    restored = manager.restore(
        step,
        args=ocp.args.Composite(
            train_state=ocp.args.StandardRestore(abstract_train_state),
            data_state=gcp.CheckpointRestore(_checkpoint_target(data_iter)),
            meta=ocp.args.JsonRestore(),
        ),
    )

    # Orbax returns a dict-like mapping for Composite.
    train_state = restored["train_state"]
    meta = restored.get("meta")
    return step, train_state, meta


def restore_params_only(step_dir: Path, abstract_params: Any) -> Any:
    """Restore only the params subtree from a checkpoint step directory.

    Inference-side helper: loads train_state/params without opt_state/rng/step,
    directly from a step dir (no CheckpointManager needed).

    :param Path step_dir: Checkpoint step directory (contains train_state/).
    :param Any abstract_params: ShapeDtypeStruct tree matching params.
    :raises FileNotFoundError: If the step dir has no train_state.
    :return Any: Restored params pytree.
    """
    import orbax.checkpoint as ocp

    train_state_dir = Path(step_dir) / "train_state"
    if not train_state_dir.exists():
        raise FileNotFoundError(
            f"train_state directory not found in {step_dir}. Is this a valid chomp checkpoint?"
        )

    # transforms={} drops saved subtrees absent from item (opt_state/rng/step)
    # without needing their structure; partial_restore=True cannot do that on
    # this orbax version (0.11.31 chokes on the pruned tree metadata).
    abstract_train_state = {"params": abstract_params}
    with ocp.PyTreeCheckpointer() as ckptr:
        restored = ckptr.restore(
            train_state_dir,
            args=ocp.args.PyTreeRestore(
                item=abstract_train_state,
                transforms={},
                restore_args=ocp.checkpoint_utils.construct_restore_args(abstract_train_state),
            ),
        )
    return restored["params"]


def check_resume_compat(
    cfg: Config, meta: dict[str, Any] | None, *, tokenizer_snapshot_hash: str | None = None
) -> None:
    """Validate checkpoint metadata against current config.

    :param Config cfg: Current training configuration.
    :param meta: Checkpoint metadata dict (or None if missing).
    :param str | None tokenizer_snapshot_hash: Optional tokenizer snapshot hash for strict checks.
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

    cur_fp = data_fingerprint(cfg, tokenizer_snapshot_hash=tokenizer_snapshot_hash)

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
    snap_prev = tok_prev.get("snapshot_sha256")
    snap_cur = tok_cur.get("snapshot_sha256")
    if snap_prev is not None or snap_cur is not None:
        _cmp(
            "tokenizer.snapshot_sha256",
            snap_cur,
            snap_prev,
            severity="error",
        )

    # Packing/loss behavior comparisons.
    pack_prev = meta_fp.get("packing") or {}
    pack_cur = cur_fp.get("packing") or {}
    _cmp(
        "data.packing_mode",
        pack_cur.get("mode"),
        pack_prev.get("mode"),
        severity="error",
    )
    _cmp(
        "data.packing_buffer_docs",
        pack_cur.get("buffer_docs"),
        pack_prev.get("buffer_docs"),
        severity="error",
    )
    _cmp(
        "data.packing_max_docs_per_bin",
        pack_cur.get("max_docs_per_bin"),
        pack_prev.get("max_docs_per_bin"),
        severity="error",
    )
    # Changing group_docs alters which documents each multipack cycle packs,
    # so a resumed run would diverge from the continuous one.
    _cmp(
        "data.packing_group_docs",
        pack_cur.get("group_docs"),
        pack_prev.get("group_docs"),
        severity="error",
    )
    # Changing strict_attention silently changes the training objective
    # (segment isolation + backend boundary masking) mid-run.
    _cmp(
        "data.packing_strict_attention",
        pack_cur.get("strict_attention"),
        pack_prev.get("strict_attention"),
        severity="error",
    )
    _cmp(
        "data.grain_prefetch",
        pack_cur.get("grain_prefetch"),
        pack_prev.get("grain_prefetch"),
        severity="warning",
    )
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
    # Changing this alters the iterator-state shape (window-shuffle layer is
    # config-gated), so a mismatched resume would KeyError or silently skip
    # the data-state restore.
    _cmp(
        "data.window_shuffle_windows",
        pack_cur.get("window_shuffle_windows"),
        pack_prev.get("window_shuffle_windows"),
        severity="error",
    )

    eval_prev = meta_fp.get("eval") or {}
    eval_cur = cur_fp.get("eval") or {}
    _cmp(
        "data.max_eval_samples",
        eval_cur.get("max_eval_samples"),
        eval_prev.get("max_eval_samples"),
        severity="error",
    )
    _cmp(
        "data.hf_eval_split",
        eval_cur.get("hf_eval_split"),
        eval_prev.get("hf_eval_split"),
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
    train_prev = meta_cfg.get("train") or {}
    train_cur = cur_cfg.get("train") or {}
    model_prev = meta_cfg.get("model") or {}
    model_cur = cur_cfg.get("model") or {}
    for key in sorted(set(model_prev) | set(model_cur)):
        _cmp(f"model.{key}", model_cur.get(key), model_prev.get(key), severity="error")

    optim_prev = meta_cfg.get("optim") or {}
    optim_cur = cur_cfg.get("optim") or {}
    optim_name_prev = optim_prev.get("name")
    optim_name_cur = optim_cur.get("name")
    for key in sorted(set(optim_prev) | set(optim_cur)):
        if key == "decay_steps":
            continue
        if key == "muon" and optim_name_prev != "muon" and optim_name_cur != "muon":
            continue
        _cmp(f"optim.{key}", optim_cur.get(key), optim_prev.get(key), severity="error")

    decay_prev = decay_horizon_from_values(
        steps=train_prev.get("steps"),
        warmup_steps=optim_prev.get("warmup_steps"),
        decay_steps=optim_prev.get("decay_steps"),
    )
    decay_cur = decay_horizon_from_values(
        steps=train_cur.get("steps"),
        warmup_steps=optim_cur.get("warmup_steps"),
        decay_steps=optim_cur.get("decay_steps"),
    )
    _cmp("optim.decay_steps_effective", decay_cur, decay_prev, severity="error")
    if optim_prev.get("decay_steps") != optim_cur.get("decay_steps") and decay_cur == decay_prev:
        warnings.append(
            "optim.decay_steps changed but effective schedule horizon is unchanged "
            f"(prev={optim_prev.get('decay_steps')!r}, cur={optim_cur.get('decay_steps')!r})"
        )

    if train_cur.get("steps") != train_prev.get("steps"):
        warnings.append(
            "train.steps mismatch (checkpoint="
            f"{train_prev.get('steps')!r}, current={train_cur.get('steps')!r})"
        )

    if warnings:
        logger.warning(
            "Resume config warnings:\n%s",
            "\n".join(f"- {msg}" for msg in warnings),
        )

    if errors:
        detail = "\n".join(f"- {msg}" for msg in errors)
        raise RuntimeError(f"Resume config mismatch:\n{detail}")
