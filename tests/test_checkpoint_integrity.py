"""Checkpoint integrity tests (async saves + corrupt checkpoints)."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from chomp.ckpt import (
    build_meta,
    default_ckpt_dir,
    make_manager,
    restore_at_step,
    restore_latest,
    save,
)
from chomp.config import (
    CheckpointConfig,
    Config,
    DataConfig,
    LoggingConfig,
    ModelConfig,
    OptimConfig,
    TokenizerConfig,
    TrainConfig,
)
from chomp.data import build_train_iterator, data_fingerprint
from chomp.types import TrainState
from chomp.utils.tree import abstractify_tree, tree_allclose


def _base_cfg(run_dir: Path) -> Config:
    """Create a base config for checkpoint tests.

    :param Path run_dir: Run directory path.
    :return Config: Config configured for checkpoint tests.
    """
    return Config(
        model=ModelConfig(backend="dummy", vocab_size=256, d_model=16, dropout=0.0),
        data=DataConfig(
            backend="local_text",
            repeat=True,
            local_text="checkpoint integrity text\n",
            tokenizer=TokenizerConfig(kind="byte", byte_offset=0, add_bos=False, add_eos=False),
        ),
        train=TrainConfig(
            steps=1,
            batch_size=1,
            seq_len=8,
            grad_accum=1,
            jit=False,
            deterministic=True,
            allow_cpu=True,
        ),
        optim=OptimConfig(warmup_steps=0),
        checkpoint=CheckpointConfig(enabled=True, save_every=1, max_to_keep=2, async_save=False),
        logging=LoggingConfig(project="chomp", run_dir=str(run_dir), metrics_file="metrics.jsonl"),
    )


def _make_state() -> TrainState:
    """Create a minimal TrainState for testing.

    :return TrainState: Minimal training state.
    """
    return TrainState(
        step=jnp.array(1, dtype=jnp.int32),
        params={"w": jnp.array([1.0, 2.0], dtype=jnp.float32)},
        opt_state={"m": jnp.array([0.5], dtype=jnp.float32)},
        rng=jax.random.PRNGKey(0),
    )


def test_async_checkpoint_roundtrip(tmp_path: Path) -> None:
    """Async checkpoint save should roundtrip state correctly."""
    run_dir = tmp_path / "run_async"
    cfg = _base_cfg(run_dir)
    state = _make_state()
    data_it = build_train_iterator(cfg)
    ckpt_dir = default_ckpt_dir(run_dir)
    mgr = make_manager(ckpt_dir, max_to_keep=2, save_every=1, async_save=True)

    meta = build_meta(step=1, config=cfg.to_dict(), data_fingerprint=data_fingerprint(cfg))
    save(mgr, step=1, train_state=state, data_iter=data_it, meta=meta)
    mgr.wait_until_finished()

    abstract_state = abstractify_tree(state)
    data_it_restore = build_train_iterator(cfg)
    step, restored, _meta = restore_latest(
        mgr, abstract_train_state=abstract_state, data_iter=data_it_restore
    )
    assert step == 1
    assert tree_allclose(restored.params, state.params, rtol=0.0, atol=0.0)
    assert tree_allclose(restored.opt_state, state.opt_state, rtol=0.0, atol=0.0)


def test_latest_step_ignores_incomplete(tmp_path: Path) -> None:
    """Checkpoint manager should ignore incomplete checkpoint directories."""
    run_dir = tmp_path / "run_latest"
    cfg = _base_cfg(run_dir)
    state = _make_state()
    data_it = build_train_iterator(cfg)
    ckpt_dir = default_ckpt_dir(run_dir)
    mgr = make_manager(ckpt_dir, max_to_keep=2, save_every=1, async_save=False)

    meta = build_meta(step=1, config=cfg.to_dict(), data_fingerprint=data_fingerprint(cfg))
    save(mgr, step=1, train_state=state, data_iter=data_it, meta=meta)
    mgr.wait_until_finished()

    (ckpt_dir / "2").mkdir()
    assert mgr.latest_step() == 1


def test_corrupt_checkpoint_fails_restore(tmp_path: Path) -> None:
    """Corrupted checkpoint metadata should raise an error on restore."""
    run_dir = tmp_path / "run_corrupt"
    cfg = _base_cfg(run_dir)
    state = _make_state()
    data_it = build_train_iterator(cfg)
    ckpt_dir = default_ckpt_dir(run_dir)
    mgr = make_manager(ckpt_dir, max_to_keep=2, save_every=1, async_save=False)

    meta = build_meta(step=1, config=cfg.to_dict(), data_fingerprint=data_fingerprint(cfg))
    save(mgr, step=1, train_state=state, data_iter=data_it, meta=meta)
    mgr.wait_until_finished()

    corrupt_target = None
    for path in (ckpt_dir / "1").rglob("*"):
        if path.is_file() and path.name == "metadata":
            corrupt_target = path
            break
    assert corrupt_target is not None
    corrupt_target.write_text("{not: valid json", encoding="utf-8")

    abstract_state = abstractify_tree(state)
    with pytest.raises((ValueError, RuntimeError, KeyError, json.JSONDecodeError)):
        data_it_restore = build_train_iterator(cfg)
        restore_at_step(mgr, step=1, abstract_train_state=abstract_state, data_iter=data_it_restore)


def test_max_to_keep_prunes_checkpoints(tmp_path: Path) -> None:
    """Checkpoint manager should prune old checkpoints per max_to_keep."""
    run_dir = tmp_path / "run_prune"
    cfg = _base_cfg(run_dir)
    state = _make_state()
    data_it = build_train_iterator(cfg)
    ckpt_dir = default_ckpt_dir(run_dir)
    mgr = make_manager(ckpt_dir, max_to_keep=2, save_every=1, async_save=False)

    for step in (1, 2, 3):
        meta = build_meta(step=step, config=cfg.to_dict(), data_fingerprint=data_fingerprint(cfg))
        save(
            mgr,
            step=step,
            train_state=state,
            data_iter=data_it,
            meta=meta,
        )
        mgr.wait_until_finished()

    meta = build_meta(step=4, config=cfg.to_dict(), data_fingerprint=data_fingerprint(cfg))
    save(
        mgr,
        step=4,
        train_state=state,
        data_iter=data_it,
        meta=meta,
    )
    mgr.wait_until_finished()

    steps = sorted(int(p.name) for p in ckpt_dir.iterdir() if p.is_dir() and p.name.isdigit())
    assert steps == [3, 4]
