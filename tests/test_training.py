"""Training and checkpointing tests consolidated by module."""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import pytest
from _pytest.logging import LogCaptureFixture

from chomp.ckpt import (
    build_meta,
    check_resume_compat,
    default_ckpt_dir,
    make_manager,
    restore_at_step,
    restore_latest,
    restore_params_only,
    save,
)
from chomp.config import (
    CheckpointConfig,
    Config,
    DataConfig,
    DebugConfig,
    LoggingConfig,
    ModelConfig,
    OptimConfig,
    TokenizerConfig,
    TrainConfig,
    WandbConfig,
    strict_packed_segments,
)
from chomp.data import build_train_iterator, data_fingerprint, prepare_tokenizer_and_config
from chomp.model import build_model, supports_packed_attention, training_loss
from chomp.train import _build_checkpoint_manager, build_optimizer, init_train_state, run
from chomp.types import Batch, TrainState
from chomp.utils.tree import abstractify_tree, tree_allclose
from tests.helpers.config_factories import make_small_run_cfg


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


def _saved_step1_checkpoint(
    run_dir: Path, *, async_save: bool = False
) -> tuple[Config, TrainState, Any, Path]:
    """Build the standard save harness and write one step-1 checkpoint.

    :param Path run_dir: Run directory for the checkpoint.
    :param bool async_save: Whether the manager saves asynchronously.
    :return tuple: (cfg, saved_state, manager, ckpt_dir).
    """
    cfg = _base_cfg(run_dir)
    state = _make_state()
    data_it = build_train_iterator(cfg)
    ckpt_dir = default_ckpt_dir(run_dir)
    mgr = make_manager(ckpt_dir, max_to_keep=2, save_every=1, async_save=async_save)

    meta = build_meta(step=1, config=cfg.to_dict(), data_fingerprint=data_fingerprint(cfg))
    save(mgr, step=1, train_state=state, data_iter=data_it, meta=meta)
    mgr.wait_until_finished()
    return cfg, state, mgr, ckpt_dir


def test_async_checkpoint_roundtrip(tmp_path: Path) -> None:
    """Async checkpoint save should roundtrip state correctly."""
    cfg, state, mgr, _ckpt_dir = _saved_step1_checkpoint(tmp_path / "run_async", async_save=True)

    abstract_state = abstractify_tree(state)
    data_it_restore = build_train_iterator(cfg)
    step, restored, _meta = restore_latest(
        mgr, abstract_train_state=abstract_state, data_iter=data_it_restore
    )
    assert step == 1
    assert tree_allclose(restored.params, state.params, rtol=0.0, atol=0.0)
    assert tree_allclose(restored.opt_state, state.opt_state, rtol=0.0, atol=0.0)


def test_restore_params_only(tmp_path: Path) -> None:
    """Params-only restore (generate CLI path) matches the saved params exactly."""
    _cfg, state, _mgr, ckpt_dir = _saved_step1_checkpoint(tmp_path / "run_params_only")

    params = restore_params_only(ckpt_dir / "1", abstractify_tree(state.params))
    assert tree_allclose(params, state.params, rtol=0.0, atol=0.0)

    with pytest.raises(FileNotFoundError, match="train_state"):
        restore_params_only(ckpt_dir / "999", abstractify_tree(state.params))


def test_checkpoint_data_state_roundtrip(tmp_path: Path) -> None:
    """Checkpoint restore should resume the data iterator position."""
    run_dir = tmp_path / "run_data_state"
    cfg = _base_cfg(run_dir)
    cfg = replace(
        cfg,
        train=replace(
            cfg.train,
            steps=2,
            batch_size=1,
            seq_len=8,
            grad_accum=1,
            jit=False,
            deterministic=True,
            allow_cpu=True,
        ),
        data=replace(
            cfg.data,
            packing_mode="sequential",
            packing_buffer_docs=4,
            grain_prefetch=0,
        ),
    )
    cfg, tokenizer = prepare_tokenizer_and_config(cfg)

    data_it = build_train_iterator(cfg, tokenizer=tokenizer)
    next(data_it)
    next(data_it)

    ckpt_dir = default_ckpt_dir(run_dir)
    mgr = make_manager(
        ckpt_dir,
        max_to_keep=cfg.checkpoint.max_to_keep,
        save_every=cfg.checkpoint.save_every,
        async_save=cfg.checkpoint.async_save,
    )

    state = TrainState(
        step=jnp.array(2, dtype=jnp.int32),
        params={"w": jnp.array([1.0], dtype=jnp.float32)},
        opt_state={"m": jnp.array([0.5], dtype=jnp.float32)},
        rng=jax.random.PRNGKey(0),
    )
    meta = build_meta(step=2, config=cfg.to_dict(), data_fingerprint=data_fingerprint(cfg))
    save(mgr, step=2, train_state=state, data_iter=data_it, meta=meta)
    mgr.wait_until_finished()

    expected = next(data_it)
    data_it_restore = build_train_iterator(cfg, tokenizer=tokenizer)
    abstract_state = abstractify_tree(state)
    step, _restored, _meta = restore_latest(
        mgr, abstract_train_state=abstract_state, data_iter=data_it_restore
    )
    assert step == 2
    restored_batch = next(data_it_restore)
    assert tree_allclose(expected, restored_batch, rtol=0.0, atol=0.0)


def test_latest_step_ignores_incomplete(tmp_path: Path) -> None:
    """Checkpoint manager should ignore incomplete checkpoint directories."""
    _cfg, _state, mgr, ckpt_dir = _saved_step1_checkpoint(tmp_path / "run_latest")

    (ckpt_dir / "2").mkdir()
    assert mgr.latest_step() == 1


def test_corrupt_checkpoint_fails_restore(tmp_path: Path) -> None:
    """Corrupted checkpoint metadata should raise an error on restore."""
    cfg, state, mgr, ckpt_dir = _saved_step1_checkpoint(tmp_path / "run_corrupt")

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


def test_checkpoint_root_dir_resolves_relative_to_run_dir(tmp_path: Path) -> None:
    """Relative checkpoint.root_dir should resolve against run_dir."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    cfg = Config()
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, root_dir="ckpts"))

    manager = _build_checkpoint_manager(cfg, run_dir)

    assert manager is not None
    assert Path(manager.directory) == (run_dir / "ckpts").resolve()


def test_checkpoint_resume_advances_step(tmp_path: Path) -> None:
    """A saved checkpoint can be resumed and training continues."""
    cfg, config_src = make_small_run_cfg(tmp_path, decay_steps=2)
    run_dir = run(cfg, config_path=str(config_src), resume="none", dry_run=False)

    ckpt_dir = default_ckpt_dir(run_dir)
    assert (ckpt_dir / "2").exists(), "expected checkpoint at step 2"

    cfg_resume = replace(cfg, train=replace(cfg.train, steps=3))
    run_dir2 = run(cfg_resume, config_path=str(config_src), resume="latest", dry_run=False)
    assert run_dir2 == run_dir
    assert (ckpt_dir / "3").exists(), "expected checkpoint at step 3 after resume"


def test_checkpoint_restore_allows_forward(tmp_path: Path) -> None:
    """Restored params can run a forward/loss computation."""
    cfg, config_src = make_small_run_cfg(tmp_path, decay_steps=2)
    run_dir = run(cfg, config_path=str(config_src), resume="none", dry_run=False)

    cfg, tokenizer = prepare_tokenizer_and_config(cfg)
    params, static = build_model(cfg, key=jax.random.PRNGKey(0))
    tx, _ = build_optimizer(cfg, params)
    state0 = init_train_state(cfg, params=params, tx=tx, key=jax.random.PRNGKey(1))
    abstract_state = abstractify_tree(state0)

    data_it = build_train_iterator(cfg, tokenizer=tokenizer)
    ckpt_dir = default_ckpt_dir(run_dir)
    manager = make_manager(
        ckpt_dir,
        max_to_keep=cfg.checkpoint.max_to_keep,
        save_every=cfg.checkpoint.save_every,
        async_save=cfg.checkpoint.async_save,
    )
    step, state, _meta = restore_latest(
        manager, abstract_train_state=abstract_state, data_iter=data_it
    )
    assert step >= 1

    bsz = int(cfg.train.batch_size)
    seq_len = int(cfg.train.seq_len)
    input_ids = jnp.zeros((bsz, seq_len), dtype=jnp.int32)
    labels = input_ids.copy()
    attn = jnp.ones((bsz, seq_len), dtype=bool)
    segs = jnp.ones((bsz, seq_len), dtype=jnp.int32)
    pos = jnp.zeros((bsz, seq_len), dtype=jnp.int32)
    batch = Batch(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attn,
        segment_ids=segs,
        position_ids=pos,
    )

    loss = training_loss(state.params, static, batch=batch, deterministic=True, key=None)
    loss_val = float(jax.device_get(loss))
    assert math.isfinite(loss_val)


def test_checkpoint_saves_final_step(tmp_path: Path) -> None:
    """Final step should be checkpointed even if save_every does not divide steps."""
    cfg, config_src = make_small_run_cfg(tmp_path, decay_steps=2)
    cfg = replace(cfg, train=replace(cfg.train, steps=3))
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, save_every=2))

    run_dir = run(cfg, config_path=str(config_src), resume="none", dry_run=False)
    ckpt_dir = default_ckpt_dir(run_dir)

    assert (ckpt_dir / "2").exists(), "expected checkpoint at save interval"
    assert (ckpt_dir / "3").exists(), "expected final checkpoint at step 3"


def test_crash_between_fetch_and_step_skips_final_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash after batch fetch but before its step completes must not
    write a final checkpoint: the data iterator is one batch ahead of the
    train state there, and saving would silently skip that batch on resume.

    Interrupted at the worst moment + resumed must match continuous exactly.
    """
    from chomp.utils.devices import assert_batch_on_device as real_assert

    def _finish_cfg(cfg: Config) -> Config:
        cfg = replace(cfg, train=replace(cfg.train, steps=5))
        return replace(cfg, checkpoint=replace(cfg.checkpoint, save_every=2))

    # Continuous reference: 5 steps, periodic saves at 2 and 4, final at 5.
    cfg_cont, config_src = make_small_run_cfg(tmp_path, run_subdir="run_cont", decay_steps=5)
    cfg_cont = _finish_cfg(cfg_cont)
    run_dir_cont = run(cfg_cont, config_path=str(config_src), resume="none", dry_run=False)

    # Crashing run: identical data/seed; the placement check runs once per
    # loop iteration (after fetch, before train_step) — blow up on exactly
    # the 4th call, i.e. mid-step 4 with state at step 3 and the last
    # periodic save at step 2.
    cfg_crash, _ = make_small_run_cfg(tmp_path, run_subdir="run_crash", decay_steps=5)
    cfg_crash = _finish_cfg(cfg_crash)
    cfg_crash = replace(cfg_crash, debug=replace(cfg_crash.debug, check_device_every=1))

    calls = {"n": 0}

    def _exploding_assert(batch: Batch, *, allow_cpu: bool) -> None:
        calls["n"] += 1
        if calls["n"] == 4:
            raise RuntimeError("injected crash between batch fetch and train step")
        real_assert(batch, allow_cpu=allow_cpu)

    monkeypatch.setattr("chomp.train.assert_batch_on_device", _exploding_assert)
    with pytest.raises(RuntimeError, match="injected crash"):
        run(cfg_crash, config_path=str(config_src), resume="none", dry_run=False)

    ckpt_dir_crash = default_ckpt_dir(Path(cfg_crash.logging.run_dir))
    steps_on_disk = {
        int(p.name) for p in ckpt_dir_crash.iterdir() if p.is_dir() and p.name.isdigit()
    }
    assert steps_on_disk == {2}, (
        f"final checkpoint must be skipped in the misaligned window, found {steps_on_disk}"
    )

    # Resume from the periodic checkpoint and finish; batches 3-5 replay.
    run(cfg_crash, config_path=str(config_src), resume="latest", dry_run=False)

    # Bit-exact resume contract: both step-5 train states identical.
    cfg_ref, tokenizer = prepare_tokenizer_and_config(cfg_cont)
    params, static = build_model(cfg_ref, key=jax.random.PRNGKey(0))
    tx, _ = build_optimizer(cfg_ref, params)
    abstract_state = abstractify_tree(
        init_train_state(cfg_ref, params=params, tx=tx, key=jax.random.PRNGKey(1))
    )

    states = []
    for run_dir in (run_dir_cont, Path(cfg_crash.logging.run_dir)):
        mgr = make_manager(default_ckpt_dir(run_dir), max_to_keep=2, save_every=2, async_save=False)
        _, state, _ = restore_at_step(
            mgr,
            step=5,
            abstract_train_state=abstract_state,
            data_iter=build_train_iterator(cfg_ref, tokenizer=tokenizer),
        )
        states.append(state)

    assert int(jax.device_get(states[0].step)) == 5
    assert tree_allclose(states[0].params, states[1].params, rtol=0.0, atol=0.0)
    assert tree_allclose(states[0].opt_state, states[1].opt_state, rtol=0.0, atol=0.0)


def test_grain_data_state_capture_is_synchronous() -> None:
    """ckpt.save() relies on grain serializing iterator state in the blocking
    phase of manager.save(); if grain's handler ever grows an async_save,
    the data stream could advance before capture and this contract breaks.
    """
    import grain.checkpoint as gcp
    from orbax.checkpoint import AsyncCheckpointHandler

    assert not issubclass(gcp.CheckpointHandler, AsyncCheckpointHandler)


def test_final_checkpoint_failure_fails_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    """A failed final save must not let training exit successfully.

    And when training itself crashed, the checkpoint failure is logged as
    secondary without masking the original exception.
    """
    cfg, config_src = make_small_run_cfg(tmp_path, run_subdir="run_ckpt_fail", decay_steps=3)
    cfg = replace(cfg, train=replace(cfg.train, steps=3))
    # save_every > steps: no periodic save, only the finally-block final save.
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, save_every=5))

    def _failing_save(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("disk full (injected)")

    monkeypatch.setattr("chomp.train.save", _failing_save)
    with pytest.raises(RuntimeError, match="checkpoint finalization failed"):
        run(cfg, config_path=str(config_src), resume="none", dry_run=False)

    # Crash path: the original exception wins; the save failure is logged.
    cfg2, _ = make_small_run_cfg(tmp_path, run_subdir="run_ckpt_fail2", decay_steps=3)
    cfg2 = replace(cfg2, train=replace(cfg2.train, steps=3))
    cfg2 = replace(cfg2, checkpoint=replace(cfg2.checkpoint, save_every=5))

    calls = {"n": 0}

    def _nan_boom(metrics: dict[str, Any], *, step: int) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("injected nan at step 2")

    monkeypatch.setattr("chomp.train._check_finite_metrics", _nan_boom)
    with (
        caplog.at_level(logging.ERROR, logger="chomp.train"),
        pytest.raises(RuntimeError, match="injected nan"),
    ):
        run(cfg2, config_path=str(config_src), resume="none", dry_run=False)
    assert any("Final checkpoint save failed" in rec.message for rec in caplog.records)


def test_resume_rejects_seq_len_mismatch(tmp_path: Path) -> None:
    """Resuming with different seq_len should raise RuntimeError."""
    base = Config(
        model=ModelConfig(backend="dummy", vocab_size=256, d_model=32, dropout=0.0),
        data=DataConfig(
            backend="local_text",
            repeat=True,
            local_text="Deterministic local text for resume mismatch test.\n",
            tokenizer=TokenizerConfig(kind="byte", byte_offset=0, add_bos=False, add_eos=False),
        ),
        train=TrainConfig(
            seed=0,
            steps=0,
            batch_size=2,
            seq_len=16,
            grad_accum=2,
            jit=False,
            deterministic=True,
            allow_cpu=True,
            log_every=1000,
        ),
        optim=OptimConfig(lr=1e-3, weight_decay=0.0, grad_clip_norm=0.0, warmup_steps=0),
        checkpoint=CheckpointConfig(enabled=True, save_every=1, max_to_keep=2, async_save=False),
        debug=DebugConfig(nan_check=True, check_device_every=0),
        logging=LoggingConfig(
            project="chomp", run_dir=None, metrics_file="metrics.jsonl", level="INFO"
        ),
    )

    run_dir = tmp_path / "run"
    cfg_a = replace(
        base,
        logging=replace(base.logging, run_dir=str(run_dir)),
        train=replace(base.train, steps=2),
    )
    run(cfg_a, config_path=None, resume="none")

    cfg_b = replace(
        base,
        logging=replace(base.logging, run_dir=str(run_dir)),
        train=replace(base.train, steps=3, seq_len=32),
    )
    with pytest.raises(RuntimeError, match="Resume config mismatch"):
        run(cfg_b, config_path=None, resume="latest")


def test_dry_run_compiles_single_step(tmp_path: Path) -> None:
    """Dry run should compile one step, write config, but not metrics."""
    run_dir = tmp_path / "dry_run"
    cfg = Config(
        model=ModelConfig(backend="dummy", vocab_size=128, d_model=32, dropout=0.0),
        data=DataConfig(
            backend="local_text",
            repeat=True,
            local_text="dry run text\n" * 8,
            max_eval_samples=4,
            tokenizer=TokenizerConfig(kind="byte", byte_offset=0, add_bos=False, add_eos=False),
        ),
        train=TrainConfig(
            steps=5,
            batch_size=1,
            seq_len=8,
            grad_accum=1,
            jit=True,
            deterministic=True,
            allow_cpu=True,
            log_every=1000,
            eval_every=0,
        ),
        optim=OptimConfig(warmup_steps=0),
        checkpoint=CheckpointConfig(enabled=False),
        logging=LoggingConfig(
            project="chomp",
            run_dir=str(run_dir),
            metrics_file="metrics.jsonl",
            wandb=WandbConfig(enabled=True),
        ),
    )

    run(cfg, config_path=None, resume="none", dry_run=True)

    assert (run_dir / "config_resolved.json").exists()
    assert not (run_dir / cfg.logging.metrics_file).exists()

    data = json.loads((run_dir / "config_resolved.json").read_text())
    assert data["derived"]["optim"]["decay_steps_effective"] == cfg.train.steps


def test_deterministic_checkpointing_warns(tmp_path: Path, caplog: LogCaptureFixture) -> None:
    """Deterministic mode should warn when use_checkpoint is enabled.

    :param Path tmp_path: Temporary directory for the run artifacts.
    :param LogCaptureFixture caplog: Log capture fixture.
    """
    run_dir = tmp_path / "dry_run_warn"
    cfg = Config(
        model=ModelConfig(
            backend="dummy",
            vocab_size=128,
            d_model=32,
            dropout=0.0,
            use_checkpoint=True,
        ),
        data=DataConfig(
            backend="local_text",
            repeat=True,
            local_text="dry run text\n" * 8,
            max_eval_samples=4,
            tokenizer=TokenizerConfig(kind="byte", byte_offset=0, add_bos=False, add_eos=False),
        ),
        train=TrainConfig(
            steps=5,
            batch_size=1,
            seq_len=8,
            grad_accum=1,
            jit=True,
            deterministic=True,
            allow_cpu=True,
            log_every=1000,
            eval_every=0,
        ),
        optim=OptimConfig(warmup_steps=0),
        checkpoint=CheckpointConfig(enabled=False),
        logging=LoggingConfig(
            project="chomp",
            run_dir=str(run_dir),
            metrics_file="metrics.jsonl",
            wandb=WandbConfig(enabled=False),
        ),
    )

    caplog.set_level(logging.WARNING)
    run(cfg, config_path=None, resume="none", dry_run=True)
    assert any(
        "checkpointing" in rec.message.lower() and "deterministic" in rec.message.lower()
        for rec in caplog.records
    )


class DummyWandbRun:
    """Minimal W&B stub to capture finish calls and logs."""

    def __init__(self) -> None:
        """Initialize captured logs, finish calls, and summary state."""
        self.finish_calls: list[int] = []
        self.logged: list[tuple[int | None, dict[str, Any]]] = []
        self.summary: dict[str, Any] = {}

    def log(self, row: dict[str, Any], *, step: int | None = None) -> None:
        """Record a metrics row and its optional step."""
        self.logged.append((step, row))

    def finish(self, *, exit_code: int = 0) -> None:
        """Record the finish exit code."""
        self.finish_calls.append(exit_code)


class DummyIter:
    """Single-batch iterator for crash tests."""

    def __init__(self) -> None:
        """Initialize the iterator in a not-yet-consumed state."""
        self._done = False

    def __iter__(self) -> DummyIter:
        return self

    def __next__(self) -> Batch:
        if self._done:
            raise StopIteration
        self._done = True
        zeros = jnp.zeros((1, 1, 8), dtype=jnp.int32)
        attn = jnp.ones((1, 1, 8), dtype=bool)
        return Batch(
            input_ids=zeros,
            labels=zeros,
            attention_mask=attn,
            segment_ids=zeros,
            position_ids=zeros,
        )

    def get_stats(self) -> dict[str, Any]:
        """Return empty iterator stats for crash tests."""
        return {}

    def get_state(self) -> dict[str, Any]:
        """Return empty iterator state for crash tests."""
        return {}

    def set_state(self, state: dict[str, Any]) -> None:
        """Accept state restores without changing iterator behavior."""
        _ = state


def test_training_crash_marks_wandb_failed_and_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Crashes should write a metrics row and finish W&B with exit_code=1.

    :param Path tmp_path: Temporary directory for run output.
    :param pytest.MonkeyPatch monkeypatch: Pytest monkeypatch fixture.
    """
    run_dir = tmp_path / "run"
    dummy_wandb = DummyWandbRun()

    def boom_make_train_step(*args: Any, **kwargs: Any) -> Any:
        """Return a train step that always raises a crash error."""

        def boom(state: Any, batch: Any) -> Any:
            """Raise a deterministic crash to exercise failure handling."""
            raise RuntimeError("kaboom")

        return boom

    monkeypatch.setattr("chomp.train.make_train_step", boom_make_train_step)
    monkeypatch.setattr("chomp.train.build_train_iterator", lambda *args, **kwargs: DummyIter())
    monkeypatch.setattr("chomp.train._maybe_init_wandb", lambda *args, **kwargs: dummy_wandb)

    cfg = Config(
        model=ModelConfig(backend="dummy", vocab_size=32, d_model=8, dropout=0.0),
        data=DataConfig(
            backend="local_text",
            local_text="boom",
            repeat=True,
            max_eval_samples=0,
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
            log_every=1,
            eval_every=0,
            generate_every=0,
        ),
        optim=OptimConfig(warmup_steps=0),
        checkpoint=CheckpointConfig(enabled=False),
        logging=LoggingConfig(
            run_dir=str(run_dir),
            wandb=replace(WandbConfig(), enabled=True),
        ),
        debug=DebugConfig(nan_check=False, check_device_every=0),
    )

    with pytest.raises(RuntimeError, match="kaboom"):
        run(cfg, config_path=None, resume="none", dry_run=False, max_steps=1)

    assert dummy_wandb.finish_calls == [1]

    metrics_path = run_dir / cfg.logging.metrics_file
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    assert any(row.get("crash") for row in rows)

    log_text = (run_dir / cfg.logging.log_file).read_text()
    assert "Training crashed" in log_text


def test_crash_does_not_save_future_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Crashes should not write a checkpoint for the next step."""
    run_dir = tmp_path / "run"

    def boom_make_train_step(*args: Any, **kwargs: Any) -> Any:
        """Return a train step that always raises a crash error."""

        def boom(state: Any, batch: Any) -> Any:
            """Raise a deterministic crash to exercise failure handling."""
            raise RuntimeError("kaboom")

        return boom

    monkeypatch.setattr("chomp.train.make_train_step", boom_make_train_step)
    monkeypatch.setattr("chomp.train.build_train_iterator", lambda *args, **kwargs: DummyIter())

    cfg = Config(
        model=ModelConfig(backend="dummy", vocab_size=32, d_model=8, dropout=0.0),
        data=DataConfig(
            backend="local_text",
            local_text="boom",
            repeat=True,
            max_eval_samples=0,
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
            log_every=1,
            eval_every=0,
            generate_every=0,
        ),
        optim=OptimConfig(warmup_steps=0),
        checkpoint=CheckpointConfig(enabled=True, save_every=1, max_to_keep=2, async_save=False),
        logging=LoggingConfig(run_dir=str(run_dir)),
        debug=DebugConfig(nan_check=False, check_device_every=0),
    )

    with pytest.raises(RuntimeError, match="kaboom"):
        run(cfg, config_path=None, resume="none", dry_run=False, max_steps=1)

    ckpt_dir = default_ckpt_dir(run_dir)
    assert ckpt_dir.exists()
    assert not (ckpt_dir / "1").exists()


def test_train_repeat_false_exits_cleanly(tmp_path: Path) -> None:
    """Training should exit cleanly and log data_exhausted when data ends."""
    cfg = Config(
        model=ModelConfig(backend="dummy", vocab_size=256, d_model=32, dropout=0.0),
        data=DataConfig(
            backend="local_text",
            repeat=False,
            local_text="short local text to exhaust\n",
            max_eval_samples=0,
            tokenizer=TokenizerConfig(kind="byte", byte_offset=0, add_bos=False, add_eos=False),
        ),
        train=TrainConfig(
            seed=0,
            steps=5,
            batch_size=1,
            seq_len=8,
            grad_accum=1,
            jit=False,
            deterministic=True,
            allow_cpu=True,
            log_every=1,
            eval_every=0,
        ),
        optim=OptimConfig(lr=1e-3, weight_decay=0.0, grad_clip_norm=0.0, warmup_steps=0),
        checkpoint=CheckpointConfig(enabled=False),
        debug=DebugConfig(nan_check=True, check_device_every=0),
        logging=LoggingConfig(project="chomp", run_dir=str(tmp_path / "run")),
    )

    run(cfg, config_path=None, resume="none")

    metrics_path = Path(cfg.logging.run_dir) / cfg.logging.metrics_file
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert any(row.get("data_exhausted") for row in rows)


def test_tokens_seen_matches_exact_loss_tokens(tmp_path: Path) -> None:
    """tokens_seen should equal cumulative exact loss_tokens from compiled metrics."""
    cfg = Config(
        model=ModelConfig(backend="dummy", vocab_size=512, d_model=32, dropout=0.0),
        data=DataConfig(
            backend="local_text",
            repeat=True,
            local_text="token accounting check\n",
            mask_boundary_loss=True,
            train_on_eos=True,
            tokenizer=TokenizerConfig(kind="byte", byte_offset=4, add_bos=True, add_eos=True),
        ),
        train=TrainConfig(
            seed=0,
            steps=4,
            batch_size=1,
            seq_len=8,
            grad_accum=2,
            jit=False,
            deterministic=True,
            allow_cpu=True,
            log_every=1,
            eval_every=0,
        ),
        optim=OptimConfig(lr=1e-3, weight_decay=0.0, grad_clip_norm=0.0, warmup_steps=0),
        checkpoint=CheckpointConfig(enabled=False),
        debug=DebugConfig(nan_check=True, check_device_every=0),
        logging=LoggingConfig(project="chomp", run_dir=str(tmp_path / "run")),
    )

    run(cfg, config_path=None, resume="none")

    metrics_path = Path(cfg.logging.run_dir) / cfg.logging.metrics_file
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    train_rows = [row for row in rows if "loss_tokens" in row]
    assert len(train_rows) == cfg.train.steps

    cumulative = 0
    for row in train_rows:
        loss_tokens = int(row["loss_tokens"])
        assert loss_tokens > 0
        cumulative += loss_tokens
        assert int(row["tokens_seen"]) == cumulative


@pytest.mark.parametrize("mode", ["bin", "multipack"])
def test_strict_packed_guard_raises_when_backend_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Strict packed modes (bin and multipack) fail fast without backend support."""
    cfg = Config(
        model=ModelConfig(backend="dummy", vocab_size=256, d_model=32, dropout=0.0),
        data=DataConfig(
            backend="local_text",
            repeat=True,
            local_text="strict packed guard\n",
            packing_mode=mode,
            packing_group_docs=2,
            packing_buffer_docs=4,
            packing_strict_segments=True,
            max_eval_samples=0,
            tokenizer=TokenizerConfig(kind="byte", byte_offset=0, add_bos=False, add_eos=False),
        ),
        train=TrainConfig(
            seed=0,
            steps=1,
            batch_size=1,
            seq_len=8,
            grad_accum=1,
            jit=False,
            deterministic=True,
            allow_cpu=True,
            log_every=1,
            eval_every=0,
        ),
        optim=OptimConfig(lr=1e-3, weight_decay=0.0, grad_clip_norm=0.0, warmup_steps=0),
        checkpoint=CheckpointConfig(enabled=False),
        debug=DebugConfig(nan_check=True, check_device_every=0),
        logging=LoggingConfig(project="chomp", run_dir=str(tmp_path / "run_guard")),
    )

    monkeypatch.setattr("chomp.train.supports_packed_attention", lambda params, static: False)
    with pytest.raises(RuntimeError, match="Strict segment isolation"):
        run(cfg, config_path=None, resume="none")


def test_resume_compat_rejects_multipack_knob_changes(tmp_path: Path) -> None:
    """Resume must reject changed packing_group_docs / packing_strict_segments.

    group_docs changes which documents each multipack cycle packs (data-order
    divergence); strict_segments silently changes the training objective.
    """
    cfg = _base_cfg(tmp_path / "run_compat")
    cfg = replace(
        cfg,
        data=replace(
            cfg.data,
            packing_mode="multipack",
            packing_group_docs=8,
            packing_strict_segments=True,
            max_eval_samples=0,
        ),
    )
    meta = {"config": cfg.to_dict(), "data_fingerprint": data_fingerprint(cfg)}

    check_resume_compat(cfg, meta)  # identical config resumes cleanly

    changed_group = replace(cfg, data=replace(cfg.data, packing_group_docs=16))
    with pytest.raises(RuntimeError, match="packing_group_docs"):
        check_resume_compat(changed_group, meta)

    changed_strict = replace(cfg, data=replace(cfg.data, packing_strict_segments=False))
    with pytest.raises(RuntimeError, match="packing_strict_segments"):
        check_resume_compat(changed_strict, meta)

    binc = replace(cfg, data=replace(cfg.data, packing_mode="bin", packing_buffer_docs=8))
    bin_meta = {"config": binc.to_dict(), "data_fingerprint": data_fingerprint(binc)}
    bin_changed = replace(binc, data=replace(binc.data, packing_strict_segments=False))
    with pytest.raises(RuntimeError, match="packing_strict_segments"):
        check_resume_compat(bin_changed, bin_meta)


def test_resume_compat_ignores_inert_packing_knobs(tmp_path: Path) -> None:
    """Editing a packing knob the active mode never consumes must not block resume.

    The fingerprint records mode-specific knobs only for the active
    packing_mode, so e.g. group_docs drift under 'sequential' (or
    buffer_docs drift under 'multipack') is invisible to compat checks.
    """
    cfg = _base_cfg(tmp_path / "run_inert")
    assert cfg.data.packing_mode == "sequential"
    meta = {"config": cfg.to_dict(), "data_fingerprint": data_fingerprint(cfg)}

    drifted = replace(
        cfg,
        data=replace(
            cfg.data,
            packing_buffer_docs=cfg.data.packing_buffer_docs + 1,
            packing_group_docs=cfg.data.packing_group_docs + 1,
            packing_strict_segments=not cfg.data.packing_strict_segments,
            packing_max_docs_per_bin=7,
        ),
    )
    check_resume_compat(drifted, meta)  # must not raise

    mp = replace(cfg, data=replace(cfg.data, packing_mode="multipack", packing_group_docs=8))
    mp_meta = {"config": mp.to_dict(), "data_fingerprint": data_fingerprint(mp)}
    mp_drifted = replace(
        mp, data=replace(mp.data, packing_buffer_docs=mp.data.packing_buffer_docs + 1)
    )
    check_resume_compat(mp_drifted, mp_meta)  # bin-only knob is inert here


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda c: replace(c, data=replace(c.data, grain_prefetch=c.data.grain_prefetch + 1)),
            "grain_prefetch",
        ),
        (
            lambda c: replace(c, data=replace(c.data, window_shuffle_windows=64)),
            "window_shuffle_windows",
        ),
        (
            lambda c: replace(
                c, data=replace(c.data, mask_boundary_loss=not c.data.mask_boundary_loss)
            ),
            "mask_boundary_loss",
        ),
        (
            lambda c: replace(c, data=replace(c.data, train_on_eos=not c.data.train_on_eos)),
            "train_on_eos",
        ),
        (
            lambda c: replace(c, train=replace(c.train, deterministic=not c.train.deterministic)),
            "deterministic",
        ),
    ],
    ids=["grain_prefetch", "window_shuffle", "mask_boundary", "train_on_eos", "deterministic"],
)
def test_resume_compat_rejects_stream_and_objective_drift(
    tmp_path: Path, mutate: Any, match: str
) -> None:
    """Every knob that changes data order, iterator-state shape, or the
    objective must hard-error on resume, not warn."""
    cfg = _base_cfg(tmp_path / "run_drift")
    meta = {"config": cfg.to_dict(), "data_fingerprint": data_fingerprint(cfg)}
    with pytest.raises(RuntimeError, match=match):
        check_resume_compat(mutate(cfg), meta)


def test_resume_compat_rejects_hf_shuffle_buffer_drift(tmp_path: Path) -> None:
    """shuffle_buffer_size drives HF shuffled document order — hard error."""
    cfg = _base_cfg(tmp_path / "run_sbuf")
    cfg = replace(
        cfg,
        data=replace(
            cfg.data,
            backend="hf",
            hf_dataset="dummy",
            hf_name="dummy",
            hf_split="train",
            shuffle=True,
            shuffle_buffer_size=10_000,
        ),
    )
    meta = {"config": cfg.to_dict(), "data_fingerprint": data_fingerprint(cfg)}
    drifted = replace(cfg, data=replace(cfg.data, shuffle_buffer_size=200_000))
    with pytest.raises(RuntimeError, match="shuffle_buffer_size"):
        check_resume_compat(drifted, meta)


def test_supports_packed_attention_requires_capability_flag() -> None:
    """Capability check keys on supports_segment_reset, not compute_loss signature.

    A backend that accepts segment_ids/position_ids but does not advertise the
    flag (megalodon-jax < 0.1.2: attention-only isolation, CEMA/TimestepNorm
    state leaking across packed boundaries) must be rejected.
    """
    import equinox as eqx

    cfg = Config(model=ModelConfig(backend="dummy", vocab_size=64, d_model=16, dropout=0.0))
    params, static = build_model(cfg, key=jax.random.PRNGKey(0))
    assert supports_packed_attention(params, static)

    class _LegacyLM(eqx.Module):
        """Pre-0.1.2 shape: packed kwargs in the signature, no capability flag."""

        w: jax.Array

        def compute_loss(
            self,
            input_ids: jax.Array,
            labels: jax.Array,
            attention_mask: jax.Array | None = None,
            segment_ids: jax.Array | None = None,
            position_ids: jax.Array | None = None,
        ) -> jax.Array:
            _ = (input_ids, labels, attention_mask, segment_ids, position_ids)
            return jnp.zeros(())

    legacy_params, legacy_static = eqx.partition(_LegacyLM(w=jnp.zeros(1)), eqx.is_array)
    assert not supports_packed_attention(legacy_params, legacy_static)


def test_strict_packed_segments_covers_multi_document_modes(tmp_path: Path) -> None:
    """Every mode that packs unrelated documents together requires isolation.

    bin previously trained non-strict by silent omission: it FFD-packs
    multiple documents per sequence exactly like multipack, so leaving it out
    of the strict predicate meant cross-document CEMA/TimestepNorm bleed on
    the default path.
    """
    cfg = _base_cfg(tmp_path / "run_pred")

    def _mode(mode: str, strict: bool = True) -> Config:
        return replace(
            cfg, data=replace(cfg.data, packing_mode=mode, packing_strict_segments=strict)
        )

    assert strict_packed_segments(_mode("bin"))
    assert strict_packed_segments(_mode("multipack"))
    assert not strict_packed_segments(_mode("sequential"))
    assert not strict_packed_segments(_mode("bin", strict=False))
    assert not strict_packed_segments(_mode("multipack", strict=False))


def test_training_loss_passes_segments_iff_packed() -> None:
    """segment_ids/position_ids reach the backend exactly when packed attention is on."""
    import equinox as eqx

    calls: dict[str, Any] = {}

    class _SpyLM(eqx.Module):
        """Backend spy recording which packed kwargs arrive."""

        w: jax.Array
        supports_segment_reset: ClassVar[bool] = True

        def compute_loss(
            self,
            input_ids: jax.Array,
            labels: jax.Array,
            attention_mask: jax.Array | None = None,
            deterministic: bool = True,
            key: jax.Array | None = None,
            segment_ids: jax.Array | None = None,
            position_ids: jax.Array | None = None,
        ) -> jax.Array:
            _ = (input_ids, labels, attention_mask, deterministic, key)
            calls["segment_ids"] = segment_ids
            calls["position_ids"] = position_ids
            return jnp.zeros(())

    params, static = eqx.partition(_SpyLM(w=jnp.zeros(1)), eqx.is_array)
    micro = Batch(
        input_ids=jnp.zeros((1, 8), dtype=jnp.int32),
        labels=jnp.zeros((1, 8), dtype=jnp.int32),
        attention_mask=jnp.ones((1, 8), dtype=bool),
        segment_ids=jnp.ones((1, 8), dtype=jnp.int32),
        position_ids=jnp.zeros((1, 8), dtype=jnp.int32),
    )

    training_loss(
        params, static, batch=micro, deterministic=True, key=None, use_packed_attention=True
    )
    assert calls["segment_ids"] is not None
    assert calls["position_ids"] is not None

    training_loss(
        params, static, batch=micro, deterministic=True, key=None, use_packed_attention=False
    )
    assert calls["segment_ids"] is None
    assert calls["position_ids"] is None


def test_megalodon_version_floor_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any megalodon model build must reject megalodon-jax < 0.1.2 outright.

    The pyproject git URL cannot carry a version specifier, so a stale
    environment is only caught here — for every mode, not just strict packing.
    """
    import importlib.metadata as real_metadata

    pytest.importorskip("megalodon_jax")

    cfg = Config(
        model=ModelConfig(
            backend="megalodon",
            vocab_size=64,
            model_dim=32,
            num_layers=1,
            num_heads=1,
            z_dim=16,
            value_dim=32,
            ffn_hidden_dim=64,
            cema_ndim=4,
            chunk_size=8,
            norm_num_groups=4,
        )
    )

    real_version = real_metadata.version

    def _stale_version(name: str) -> str:
        if name == "megalodon-jax":
            return "0.1.1"
        return real_version(name)

    monkeypatch.setattr("importlib.metadata.version", _stale_version)
    with pytest.raises(RuntimeError, match="requires megalodon-jax >= 0.1.2"):
        build_model(cfg, key=jax.random.PRNGKey(0))

    def _missing_version(name: str) -> str:
        if name == "megalodon-jax":
            raise real_metadata.PackageNotFoundError(name)
        return real_version(name)

    monkeypatch.setattr("importlib.metadata.version", _missing_version)
    with pytest.raises(RuntimeError, match="Cannot verify"):
        build_model(cfg, key=jax.random.PRNGKey(0))


def test_megalodon_backend_advertises_segment_reset() -> None:
    """The installed megalodon-jax must expose the full-isolation capability flag."""
    pytest.importorskip("megalodon_jax")
    cfg = Config(
        model=ModelConfig(
            backend="megalodon",
            vocab_size=64,
            model_dim=32,
            num_layers=1,
            num_heads=1,
            z_dim=16,
            value_dim=32,
            ffn_hidden_dim=64,
            cema_ndim=4,
            chunk_size=8,
            norm_num_groups=4,
        )
    )
    params, static = build_model(cfg, key=jax.random.PRNGKey(0))
    assert supports_packed_attention(params, static)
