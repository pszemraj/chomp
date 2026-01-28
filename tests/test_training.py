"""Training and checkpointing tests consolidated by module."""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import pytest
from _pytest.logging import LogCaptureFixture

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
    DebugConfig,
    LoggingConfig,
    ModelConfig,
    OptimConfig,
    TokenizerConfig,
    TrainConfig,
    WandbConfig,
    load_config,
)
from chomp.data import build_train_iterator, data_fingerprint, prepare_tokenizer_and_config
from chomp.model import build_model, training_loss
from chomp.train import _build_checkpoint_manager, build_optimizer, init_train_state, run
from chomp.types import Batch, TrainState
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


def _test_async_checkpoint_roundtrip(tmp_path: Path) -> None:
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


def _test_checkpoint_data_state_roundtrip(tmp_path: Path) -> None:
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


def _test_latest_step_ignores_incomplete(tmp_path: Path) -> None:
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


def _test_corrupt_checkpoint_fails_restore(tmp_path: Path) -> None:
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


def _test_max_to_keep_prunes_checkpoints(tmp_path: Path) -> None:
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


def _test_checkpoint_root_dir_resolves_relative_to_run_dir(tmp_path: Path) -> None:
    """Relative checkpoint.root_dir should resolve against run_dir."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    cfg = Config()
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, root_dir="ckpts"))

    manager = _build_checkpoint_manager(cfg, run_dir)

    assert manager is not None
    assert Path(manager.directory) == (run_dir / "ckpts").resolve()


def _small_cfg(tmp_path: Path) -> tuple[Config, Path]:
    """Return a tiny local_text config for fast checkpoint tests.

    :param Path tmp_path: Temporary directory provided by pytest.
    :return tuple[Config, Path]: (config, config_path) for the smoke run.
    """
    config_src = Path(__file__).resolve().parents[1] / "configs" / "debug_smoke.yaml"
    cfg = load_config(str(config_src))

    cfg = replace(
        cfg,
        train=replace(
            cfg.train,
            steps=2,
            batch_size=1,
            seq_len=16,
            grad_accum=1,
            jit=False,
            deterministic=True,
            allow_cpu=True,
            log_every=1,
            eval_every=0,
            generate_every=0,
        ),
        data=replace(
            cfg.data,
            backend="local_text",
            repeat=True,
            packing_mode="sequential",
            packing_buffer_docs=4,
            grain_prefetch=0,
            local_text="hello from chomp",
        ),
        checkpoint=replace(
            cfg.checkpoint,
            enabled=True,
            save_every=1,
            max_to_keep=2,
            async_save=False,
        ),
        optim=replace(
            cfg.optim,
            warmup_steps=0,
            decay_steps=2,
        ),
        logging=replace(
            cfg.logging,
            run_dir=str(tmp_path / "run"),
            console_use_rich=False,
        ),
        debug=replace(
            cfg.debug,
            nan_check=True,
            check_device_every=0,
        ),
    )
    return cfg, config_src


def _test_checkpoint_resume_advances_step(tmp_path: Path) -> None:
    """A saved checkpoint can be resumed and training continues."""
    cfg, config_src = _small_cfg(tmp_path)
    run_dir = run(cfg, config_path=str(config_src), resume="none", dry_run=False)

    ckpt_dir = default_ckpt_dir(run_dir)
    assert (ckpt_dir / "2").exists(), "expected checkpoint at step 2"

    cfg_resume = replace(cfg, train=replace(cfg.train, steps=3))
    run_dir2 = run(cfg_resume, config_path=str(config_src), resume="latest", dry_run=False)
    assert run_dir2 == run_dir
    assert (ckpt_dir / "3").exists(), "expected checkpoint at step 3 after resume"


def _test_checkpoint_restore_allows_forward(tmp_path: Path) -> None:
    """Restored params can run a forward/loss computation."""
    cfg, config_src = _small_cfg(tmp_path)
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
    batch = Batch(input_ids=input_ids, labels=labels, attention_mask=attn, segment_ids=segs)

    loss = training_loss(state.params, static, batch=batch, deterministic=True, key=None)
    loss_val = float(jax.device_get(loss))
    assert math.isfinite(loss_val)


def _test_checkpoint_saves_final_step(tmp_path: Path) -> None:
    """Final step should be checkpointed even if save_every does not divide steps."""
    cfg, config_src = _small_cfg(tmp_path)
    cfg = replace(cfg, train=replace(cfg.train, steps=3))
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, save_every=2))

    run_dir = run(cfg, config_path=str(config_src), resume="none", dry_run=False)
    ckpt_dir = default_ckpt_dir(run_dir)

    assert (ckpt_dir / "2").exists(), "expected checkpoint at save interval"
    assert (ckpt_dir / "3").exists(), "expected final checkpoint at step 3"


def _test_resume_rejects_seq_len_mismatch(tmp_path: Path) -> None:
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


def _test_dry_run_compiles_single_step(tmp_path: Path) -> None:
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


def _test_deterministic_checkpointing_warns(tmp_path: Path, caplog: LogCaptureFixture) -> None:
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
        return Batch(input_ids=zeros, labels=zeros, attention_mask=attn, segment_ids=zeros)

    def get_stats(self) -> dict[str, Any]:
        """Return empty iterator stats for crash tests."""
        return {}

    def get_state(self) -> dict[str, Any]:
        """Return empty iterator state for crash tests."""
        return {}

    def set_state(self, state: dict[str, Any]) -> None:
        """Accept state restores without changing iterator behavior."""
        _ = state


def _test_training_crash_marks_wandb_failed_and_logs(
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


def _test_crash_does_not_save_future_checkpoint(
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


def _test_train_repeat_false_exits_cleanly(tmp_path: Path) -> None:
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


class TestCheckpointing:
    """Checkpoint manager behavior and invariants."""

    def test_async_roundtrip(self, tmp_path: Path) -> None:
        """Async checkpoint saves should roundtrip correctly."""
        _test_async_checkpoint_roundtrip(tmp_path)

    def test_data_state_roundtrip(self, tmp_path: Path) -> None:
        """Checkpoint restore should resume data iterator position."""
        _test_checkpoint_data_state_roundtrip(tmp_path)

    def test_latest_step_ignores_incomplete(self, tmp_path: Path) -> None:
        """Incomplete checkpoint directories should be ignored."""
        _test_latest_step_ignores_incomplete(tmp_path)

    def test_corrupt_fails_restore(self, tmp_path: Path) -> None:
        """Corrupt checkpoint metadata should fail restore."""
        _test_corrupt_checkpoint_fails_restore(tmp_path)

    def test_max_to_keep_prunes(self, tmp_path: Path) -> None:
        """Checkpoint pruning should respect max_to_keep."""
        _test_max_to_keep_prunes_checkpoints(tmp_path)

    def test_root_dir_relative_to_run_dir(self, tmp_path: Path) -> None:
        """Relative checkpoint roots should resolve under the run directory."""
        _test_checkpoint_root_dir_resolves_relative_to_run_dir(tmp_path)

    def test_saves_final_step(self, tmp_path: Path) -> None:
        """Final steps should be checkpointed even off interval."""
        _test_checkpoint_saves_final_step(tmp_path)


class TestResume:
    """Resume semantics and safety checks."""

    def test_advances_step(self, tmp_path: Path) -> None:
        """Resuming from latest should advance the checkpoint step."""
        _test_checkpoint_resume_advances_step(tmp_path)

    def test_restore_allows_forward(self, tmp_path: Path) -> None:
        """Restored checkpoints should support a forward/loss pass."""
        _test_checkpoint_restore_allows_forward(tmp_path)

    def test_rejects_seq_len_mismatch(self, tmp_path: Path) -> None:
        """Resume should reject changed shape-critical config like seq_len."""
        _test_resume_rejects_seq_len_mismatch(tmp_path)


class TestTrainingLoop:
    """Training loop smoke tests and failure handling."""

    def test_dry_run_compiles_single_step(self, tmp_path: Path) -> None:
        """Dry runs should compile a single step and exit cleanly."""
        _test_dry_run_compiles_single_step(tmp_path)

    def test_deterministic_checkpointing_warns(
        self, tmp_path: Path, caplog: LogCaptureFixture
    ) -> None:
        """Deterministic runs should warn when checkpointing is enabled."""
        _test_deterministic_checkpointing_warns(tmp_path, caplog)

    def test_crash_marks_wandb_failed_and_logs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Crashes should mark W&B failed and emit a metrics row."""
        _test_training_crash_marks_wandb_failed_and_logs(tmp_path, monkeypatch)

    def test_crash_does_not_save_future_checkpoint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Crashes should not emit a future-step checkpoint."""
        _test_crash_does_not_save_future_checkpoint(tmp_path, monkeypatch)

    def test_repeat_false_exits_cleanly(self, tmp_path: Path) -> None:
        """Non-repeating data should exit cleanly when exhausted."""
        _test_train_repeat_false_exits_cleanly(tmp_path)
