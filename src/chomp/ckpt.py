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

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


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
        return asdict(self)


def _safe_version(pkg: str) -> str | None:
    try:
        import importlib.metadata as im

        return im.version(pkg)
    except Exception:
        return None


def build_meta(
    *, step: int, config: dict[str, Any], data_fingerprint: dict[str, Any]
) -> CheckpointMeta:
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
    return run_dir / "checkpoints"


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except FileNotFoundError:
            # async / concurrent cleanup
            pass
    return total


def make_manager(ckpt_dir: Path, *, max_to_keep: int, save_every: int, async_save: bool):
    """Create an Orbax CheckpointManager.

    We keep this wrapper here so Orbax API drift is contained.
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
    manager,
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
    manager,
    *,
    abstract_train_state: Any,
) -> tuple[int, Any, dict[str, Any] | None, dict[str, Any] | None]:
    """Restore latest checkpoint.

    Returns: (step, train_state, data_state, meta)

    Notes:
    - `abstract_train_state` should be a tree of ShapeDtypeStruct matching TrainState.
    - `data_state` and `meta` are JSON dicts.
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
    manager,
    *,
    step: int,
    abstract_train_state: Any,
) -> tuple[int, Any, dict[str, Any] | None, dict[str, Any] | None]:
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
