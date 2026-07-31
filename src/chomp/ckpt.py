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

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import orbax.checkpoint as ocp

from chomp.config import (
    Config,
    decay_horizon_from_values,
    derived_deterministic,
    strict_packed_segments,
)

logger = logging.getLogger(__name__)

_MISSING = object()
CHECKPOINT_META_SCHEMA_VERSION = 1


def megalodon_jax_identity() -> dict[str, Any]:
    """Return the installed Megalodon-JAX distribution identity.

    :return dict[str, Any]: Distribution version and optional PEP 610 source identity.
    """
    dist = metadata.distribution("megalodon-jax")
    identity: dict[str, Any] = {
        "distribution": "megalodon-jax",
        "version": dist.version,
    }
    direct_url_text = dist.read_text("direct_url.json")
    if direct_url_text is not None:
        direct_url = json.loads(direct_url_text)
        vcs_info = direct_url.get("vcs_info") or {}
        identity["direct_url"] = {
            "url": direct_url.get("url"),
            "vcs": vcs_info.get("vcs"),
            "requested_revision": vcs_info.get("requested_revision"),
            "commit_id": vcs_info.get("commit_id"),
        }
    return identity


def _effective_deterministic_from_mapping(config: dict[str, Any]) -> bool | None:
    """Resolve stored dropout determinism when enough config fields exist.

    :param dict[str, Any] config: Configuration mapping from checkpoint metadata.
    :return bool | None: Effective determinism, or None for incomplete legacy metadata.
    """
    train = config.get("train")
    model = config.get("model")
    if not isinstance(train, dict) or not isinstance(model, dict):
        return None
    if "deterministic" not in train:
        return None
    configured = train["deterministic"]
    if configured is not None:
        return configured if isinstance(configured, bool) else None

    dropout_fields = (
        ("dropout",)
        if model.get("backend") == "dummy"
        else ("dropout", "attention_dropout", "hidden_dropout")
    )
    if any(field not in model for field in dropout_fields):
        return None
    return all(model[field] == 0.0 for field in dropout_fields)


@dataclass(frozen=True)
class CheckpointMeta:
    """Metadata stored alongside checkpoints.

    Keep this JSON-serializable.
    """

    # This is a semantic resume record, not a general environment manifest.
    # Source trees, other packages, devices, and XLA flags remain external.

    schema_version: int
    step: int
    timestamp: str
    tokens_seen: int
    megalodon_jax: dict[str, Any]

    # Config snapshot used for semantic resume checks.
    config: dict[str, Any]

    # Minimal data fingerprint
    data_fingerprint: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to a JSON-serializable dictionary.

        :return dict[str, Any]: All fields as a nested dict.
        """
        return asdict(self)


def build_meta(
    *,
    step: int,
    config: dict[str, Any],
    data_fingerprint: dict[str, Any],
    tokens_seen: int,
) -> CheckpointMeta:
    """Build checkpoint metadata with version info and config snapshot.

    :param int step: Current training step.
    :param dict[str, Any] config: Full config dict for reproducibility.
    :param dict[str, Any] data_fingerprint: Data pipeline fingerprint.
    :param int tokens_seen: Cumulative loss-token count for exact resume accounting.
    :return CheckpointMeta: Populated metadata object.
    """
    return CheckpointMeta(
        schema_version=CHECKPOINT_META_SCHEMA_VERSION,
        step=int(step),
        timestamp=datetime.now().isoformat(timespec="seconds"),
        tokens_seen=int(tokens_seen),
        megalodon_jax=megalodon_jax_identity(),
        config=config,
        data_fingerprint=data_fingerprint,
    )


def default_ckpt_dir(run_dir: Path) -> Path:
    """Return the default checkpoint directory for a run.

    :param Path run_dir: Run directory path.
    :return Path: Path to checkpoints subdirectory.
    """
    return run_dir / "checkpoints"


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


def _train_state_step(train_state: Any) -> int:
    """Read an integer step from a restored or live train state.

    :param Any train_state: Object or mapping containing ``step``.
    :raises RuntimeError: If the step is missing or not scalar-convertible.
    :return int: Scalar optimizer step.
    """
    try:
        value = train_state["step"] if isinstance(train_state, dict) else train_state.step
        import jax

        return int(jax.device_get(value))
    except Exception as exc:
        raise RuntimeError("Checkpoint train_state has no valid scalar step") from exc


def validate_checkpoint_steps(
    *, directory_step: int, meta: dict[str, Any] | CheckpointMeta | None, train_state: Any | None
) -> None:
    """Require directory, metadata, and train-state steps to agree.

    :param int directory_step: CheckpointManager step/directory selector.
    :param meta: Restored metadata mapping or live CheckpointMeta.
    :param train_state: Restored/live state, or None for metadata-only preflight.
    :raises RuntimeError: If any required step is invalid or mismatched.
    """
    if meta is None:
        raise RuntimeError("Checkpoint metadata is missing; cannot validate step consistency")
    meta_step_raw = meta.step if isinstance(meta, CheckpointMeta) else meta.get("step")
    if isinstance(meta_step_raw, bool) or not isinstance(meta_step_raw, int):
        raise RuntimeError(f"Checkpoint metadata step is invalid: {meta_step_raw!r}")
    state_step = None if train_state is None else _train_state_step(train_state)
    mismatches = []
    if int(meta_step_raw) != int(directory_step):
        mismatches.append(f"metadata={meta_step_raw}")
    if state_step is not None and state_step != int(directory_step):
        mismatches.append(f"train_state={state_step}")
    if mismatches:
        raise RuntimeError(
            f"Checkpoint step mismatch: directory={int(directory_step)}, " + ", ".join(mismatches)
        )


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
    Grain 0.2.15's CheckpointHandler is synchronous, so the composite handler
    runs it in the blocking phase. Async checkpointing therefore cannot race
    the training loop advancing the iterator.

    :param ocp.CheckpointManager manager: Orbax checkpoint manager.
    :param int step: Training step number.
    :param Any train_state: TrainState pytree (arrays only).
    :param Any data_iter: Data iterator to checkpoint via Grain's handler.
    :param CheckpointMeta meta: Checkpoint metadata.
    :param bool force: If True, force a save even if the step is off-interval.
    """

    import grain.checkpoint as gcp
    import orbax.checkpoint as ocp

    validate_checkpoint_steps(directory_step=int(step), meta=meta, train_state=train_state)
    accepted = manager.save(
        int(step),
        args=ocp.args.Composite(
            train_state=ocp.args.StandardSave(train_state),
            data_state=gcp.CheckpointSave(data_iter.checkpoint_target()),
            meta=ocp.args.JsonSave(meta.to_dict()),
        ),
        force=force,
    )
    if accepted is not True:
        raise RuntimeError(
            f"CheckpointManager rejected save for step {int(step)} (returned {accepted!r})"
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

    import grain.checkpoint as gcp
    import orbax.checkpoint as ocp

    step = int(step)
    restored = manager.restore(
        step,
        args=ocp.args.Composite(
            train_state=ocp.args.StandardRestore(abstract_train_state),
            data_state=gcp.CheckpointRestore(data_iter.checkpoint_target()),
            meta=ocp.args.JsonRestore(),
        ),
    )

    train_state = restored["train_state"]
    meta = restored.get("meta")
    validate_checkpoint_steps(directory_step=step, meta=meta, train_state=train_state)
    return step, train_state, meta


def restore_train_state_at_step(
    manager: ocp.CheckpointManager,
    *,
    step: int,
    abstract_train_state: Any,
) -> tuple[Any, dict[str, Any] | None]:
    """Restore train state and metadata without touching the data iterator.

    :param ocp.CheckpointManager manager: Checkpoint manager.
    :param int step: Step number to restore.
    :param Any abstract_train_state: Shape/dtype tree for the train state.
    :return tuple[Any, dict[str, Any] | None]: Restored train state and metadata.
    """
    import orbax.checkpoint as ocp

    step = int(step)
    restored = manager.restore(
        step,
        args=ocp.args.Composite(
            train_state=ocp.args.StandardRestore(abstract_train_state),
            meta=ocp.args.JsonRestore(),
        ),
    )
    train_state = restored["train_state"]
    meta = restored.get("meta")
    validate_checkpoint_steps(directory_step=step, meta=meta, train_state=train_state)
    return train_state, meta


def restore_data_state_at_step(
    manager: ocp.CheckpointManager,
    *,
    step: int,
    data_iter: Any,
) -> None:
    """Restore only the checkpointed data-iterator state.

    :param ocp.CheckpointManager manager: Checkpoint manager.
    :param int step: Step number to restore.
    :param Any data_iter: Data iterator restored through Grain.
    """
    import grain.checkpoint as gcp
    import orbax.checkpoint as ocp

    manager.restore(
        int(step),
        args=ocp.args.Composite(
            data_state=gcp.CheckpointRestore(data_iter.checkpoint_target()),
        ),
    )


def restore_meta_at_step(manager: ocp.CheckpointManager, *, step: int) -> dict[str, Any] | None:
    """Restore checkpoint metadata without loading model or data state.

    :param ocp.CheckpointManager manager: Orbax checkpoint manager.
    :param int step: Checkpoint step number.
    :return dict[str, Any] | None: JSON checkpoint metadata.
    """
    import orbax.checkpoint as ocp

    restored = manager.restore(
        int(step),
        args=ocp.args.Composite(meta=ocp.args.JsonRestore()),
    )
    meta = restored.get("meta")
    validate_checkpoint_steps(directory_step=int(step), meta=meta, train_state=None)
    return meta


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

    # The caller's abstract params are the deliberate inference contract:
    # training resumes use check_resume_compat, while generate may supply an
    # explicit config override and incompatible parameter trees fail restore.
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
    cfg: Config,
    meta: dict[str, Any] | None,
) -> None:
    """Validate checkpoint metadata against current config.

    :param Config cfg: Current training configuration.
    :param meta: Checkpoint metadata dict (or None if missing).
    :raises RuntimeError: If meta is missing or config mismatches are found.
    """

    from chomp.data import data_fingerprint

    if meta is None:
        raise RuntimeError("Checkpoint meta is missing; cannot verify resume compatibility.")

    meta_cfg = meta.get("config")
    meta_fp = meta.get("data_fingerprint")
    if not isinstance(meta_cfg, dict) or not isinstance(meta_fp, dict):
        raise RuntimeError(
            "Checkpoint meta is missing config/data_fingerprint; cannot verify resume compatibility."
        )
    tokens_seen = meta.get("tokens_seen")
    if isinstance(tokens_seen, bool) or not isinstance(tokens_seen, int) or tokens_seen < 0:
        raise RuntimeError(
            "Checkpoint meta has missing or invalid tokens_seen; cannot resume exact accounting."
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
            prev_text = "<missing>" if prev is _MISSING else repr(prev)
            cur_text = "<missing>" if cur is _MISSING else repr(cur)
            msg = f"{path} mismatch (checkpoint={prev_text}, current={cur_text})"
            if severity == "error":
                errors.append(msg)
            else:
                warnings.append(msg)

    def _cmp_mapping(
        prefix: str,
        cur: dict[str, Any],
        prev: dict[str, Any],
        *,
        keys: set[str] | None = None,
        severity: str = "error",
        labels: dict[str, str] | None = None,
    ) -> None:
        """Compare selected mapping keys with consistent missing-key handling.

        :param str prefix: Default dotted path prefix.
        :param dict[str, Any] cur: Current values.
        :param dict[str, Any] prev: Checkpoint values.
        :param set[str] | None keys: Active keys, or all current keys when omitted.
        :param str severity: ``error`` or ``warning``.
        :param dict[str, str] | None labels: Optional full path overrides by key.
        """
        selected = set(cur) if keys is None else keys
        for key in sorted(selected):
            path = (labels or {}).get(key, f"{prefix}.{key}" if prefix else key)
            _cmp(
                path,
                cur.get(key, _MISSING),
                prev.get(key, _MISSING),
                severity=severity,
            )

    cur_fp = data_fingerprint(cfg)
    semantic_severity = "error" if cfg.checkpoint.resume_compat == "strict" else "warning"

    if cfg.model.backend == "megalodon":
        _cmp(
            "megalodon_jax",
            megalodon_jax_identity(),
            meta.get("megalodon_jax", _MISSING),
            severity=semantic_severity,
        )

    _cmp_mapping(
        "",
        cur_fp,
        meta_fp,
        keys={"seq_len", "batch_size", "grad_accum"},
        labels={
            "seq_len": "train.seq_len",
            "batch_size": "train.batch_size",
            "grad_accum": "train.grad_accum",
        },
        severity=semantic_severity,
    )

    # Fingerprint sections contain only active backend/mode keys. Every current
    # key must be present because missing metadata cannot establish equality.
    src_prev = meta_fp.get("source") or {}
    src_cur = cur_fp.get("source") or {}
    _cmp_mapping(
        "data",
        src_cur,
        src_prev,
        labels={
            "backend": "data.source.backend",
            "dataset": "data.hf_dataset",
            "name": "data.hf_name",
            "split": "data.hf_split",
            "revision": "data.hf_revision",
            "eval_holdout_fraction": "data.hf_eval_holdout_fraction",
        },
        severity=semantic_severity,
    )

    tok_prev = meta_fp.get("tokenizer") or {}
    tok_cur = cur_fp.get("tokenizer") or {}
    tokenizer_inert = {
        "hf_name_or_path",
        "hf_use_fast",
        "hf_trust_remote_code",
        "vocab_size_multiple",
        "auto_set_special_tokens",
    }
    _cmp_mapping(
        "tokenizer",
        tok_cur,
        tok_prev,
        keys=set(tok_cur) - tokenizer_inert,
        severity=semantic_severity,
    )

    # Fingerprinted packing knobs change data order or the training objective.
    pack_prev = meta_fp.get("packing") or {}
    pack_cur = cur_fp.get("packing") or {}
    # Knobs whose DataConfig field carries the packing_ prefix; the rest are
    # top-level data.* fields recorded in the fingerprint's packing section.
    packing_prefixed = {"mode", "buffer_docs", "max_docs_per_bin", "group_docs", "strict_segments"}
    for key in sorted(pack_cur):
        label = f"data.packing_{key}" if key in packing_prefixed else f"data.{key}"
        _cmp(
            label,
            pack_cur.get(key, _MISSING),
            pack_prev.get(key, _MISSING),
            severity=semantic_severity,
        )

    # Eval tokens are rebuilt from the live stream on every start; selection
    # drift silently changes what eval_loss measures across the resume.
    eval_prev = meta_fp.get("eval") or {}
    eval_cur = cur_fp.get("eval") or {}
    _cmp_mapping(
        "data.eval",
        eval_cur,
        eval_prev,
        labels={"max_eval_samples": "data.max_eval_samples"},
        severity=semantic_severity,
    )

    # Model/optimizer comparisons.
    cur_cfg = cfg.to_dict()
    train_prev = meta_cfg.get("train") or {}
    train_cur = cur_cfg.get("train") or {}
    # Compare the executed dropout behavior: null and true are equivalent
    # when all active dropout rates are zero.
    previous_deterministic = _effective_deterministic_from_mapping(meta_cfg)
    _cmp(
        "train.deterministic_effective",
        derived_deterministic(cfg),
        _MISSING if previous_deterministic is None else previous_deterministic,
        severity=semantic_severity,
    )
    model_prev = meta_cfg.get("model") or {}
    model_cur = cur_cfg.get("model") or {}
    model_active = set(model_cur)
    if model_cur.get("backend") == "dummy":
        model_active = {
            "backend",
            "vocab_size",
            "d_model",
            "dropout",
            "pad_token_id",
            "bos_token_id",
            "eos_token_id",
        }
        model_structural = {"backend", "vocab_size", "d_model"}
    else:
        # Dummy-only width and fresh initialization/checkpoint implementation
        # are inert after parameter restore.
        model_active -= {
            "d_model",
            "init_mode",
            "use_checkpoint",
            "loss_chunk_size",
        }
        if not strict_packed_segments(cfg):
            model_active.discard("use_associative_segment_scan")
        model_structural = {
            "backend",
            "vocab_size",
            "model_dim",
            "num_layers",
            "z_dim",
            "value_dim",
            "ffn_hidden_dim",
            "cema_ndim",
            "swiglu",
            "rescale_nffn",
            "share_emb",
            "norm_affine",
            "output_size",
            "param_dtype",
        }
    model_structural &= model_active
    _cmp_mapping("model", model_cur, model_prev, keys=model_structural, severity="error")
    _cmp_mapping(
        "model",
        model_cur,
        model_prev,
        keys=model_active - model_structural,
        severity=semantic_severity,
    )

    optim_prev = meta_cfg.get("optim") or {}
    optim_cur = cur_cfg.get("optim") or {}
    optim_name_cur = optim_cur.get("name")
    _cmp_mapping("optim", optim_cur, optim_prev, keys={"name"}, severity="error")

    optim_value_keys = set(optim_cur) - {
        "name",
        "decay_steps",
        "adam",
        "muon",
    }
    _cmp_mapping(
        "optim",
        optim_cur,
        optim_prev,
        keys=optim_value_keys,
        severity=semantic_severity,
    )

    adam_prev = optim_prev.get("adam") or {}
    adam_cur = optim_cur.get("adam") or {}
    _cmp_mapping("optim.adam", adam_cur, adam_prev, severity=semantic_severity)

    if optim_name_cur == "muon":
        muon_prev = optim_prev.get("muon") or {}
        muon_cur = optim_cur.get("muon") or {}
        muon_structural = {"allow_all_2d", "allow_tied_embed"}
        _cmp_mapping(
            "optim.muon",
            muon_cur,
            muon_prev,
            keys=muon_structural,
            severity="error",
        )
        consistent_rms_prev = muon_prev.get("consistent_rms", _MISSING)
        consistent_rms_cur = muon_cur.get("consistent_rms", _MISSING)
        if consistent_rms_prev is _MISSING or consistent_rms_cur is _MISSING:
            _cmp(
                "optim.muon.consistent_rms",
                consistent_rms_cur,
                consistent_rms_prev,
                severity=semantic_severity,
            )
        else:
            if (muon_prev["consistent_rms"] is None) != (muon_cur["consistent_rms"] is None):
                _cmp(
                    "optim.muon.consistent_rms",
                    muon_cur["consistent_rms"],
                    muon_prev["consistent_rms"],
                    severity="error",
                )
            else:
                _cmp(
                    "optim.muon.consistent_rms",
                    muon_cur["consistent_rms"],
                    muon_prev["consistent_rms"],
                    severity=semantic_severity,
                )
        _cmp_mapping(
            "optim.muon",
            muon_cur,
            muon_prev,
            keys=set(muon_cur) - muon_structural - {"consistent_rms"},
            severity=semantic_severity,
        )

    schedule_fields = (
        ("train.steps", train_cur, train_prev, "steps"),
        ("optim.warmup_steps", optim_cur, optim_prev, "warmup_steps"),
        ("optim.decay_steps", optim_cur, optim_prev, "decay_steps"),
    )
    decay_keys_present = all(key in cur and key in prev for _, cur, prev, key in schedule_fields)
    for path, cur, prev, key in (schedule_fields[0], schedule_fields[2]):
        if key not in cur or key not in prev:
            _cmp(
                path,
                cur.get(key, _MISSING),
                prev.get(key, _MISSING),
                severity=semantic_severity,
            )
    if decay_keys_present:
        decay_prev = decay_horizon_from_values(
            steps=train_prev["steps"],
            warmup_steps=optim_prev["warmup_steps"],
            decay_steps=optim_prev["decay_steps"],
        )
        decay_cur = decay_horizon_from_values(
            steps=train_cur["steps"],
            warmup_steps=optim_cur["warmup_steps"],
            decay_steps=optim_cur["decay_steps"],
        )
        _cmp(
            "optim.decay_steps_effective",
            decay_cur,
            decay_prev,
            severity=semantic_severity,
        )
        if optim_prev["decay_steps"] != optim_cur["decay_steps"] and decay_cur == decay_prev:
            warnings.append(
                "optim.decay_steps changed but effective schedule horizon is unchanged "
                f"(prev={optim_prev['decay_steps']!r}, cur={optim_cur['decay_steps']!r})"
            )

    if "steps" in train_cur and "steps" in train_prev and train_cur["steps"] != train_prev["steps"]:
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
