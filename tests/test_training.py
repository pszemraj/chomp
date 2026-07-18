"""Training and checkpointing tests consolidated by module."""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import threading
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from _pytest.logging import LogCaptureFixture

from chomp.ckpt import (
    CheckpointMeta,
    build_meta,
    default_ckpt_dir,
    make_manager,
    restore_at_step,
    restore_params_only,
    save,
    validate_checkpoint_steps,
)
from chomp.ckpt import (
    check_resume_compat as _check_resume_compat,
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
    resolve_window_shuffle_rows,
    strict_packed_segments,
)
from chomp.data import (
    ZeroLossTokensError,
    build_train_iterator,
    data_fingerprint,
    prepare_tokenizer_and_config,
)
from chomp.model import build_model, supports_packed_segments, training_loss
from chomp.train import (
    _METRICS_FILE_DROP,
    _WANDB_DROP,
    TrainingPreempted,
    _build_checkpoint_manager,
    _project_metrics,
    _StopSignalState,
    build_optimizer,
    init_train_state,
    run,
)
from chomp.types import Batch, TrainState
from chomp.utils.io import RunDirectoryLock
from chomp.utils.tree import abstractify_tree
from tests.helpers.config_factories import make_small_run_cfg
from tests.helpers.io import read_jsonl


class _FakeStopSignal:
    """Mutable signal state for deterministic preemption tests."""

    def __init__(self, reason: str = "SIGTERM", exit_code: int = 143) -> None:
        self.requested = False
        self.reason = reason
        self.exit_code = exit_code

    def __enter__(self) -> _FakeStopSignal:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


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


def check_resume_compat(
    cfg: Config,
    meta: dict[str, Any] | None,
    *,
    tokenizer_snapshot_hash: str | None = None,
    parameter_manifest_hash: str = "test-parameter-manifest",
) -> None:
    """Call resume validation with the test model-manifest identity.

    :param Config cfg: Current configuration.
    :param meta: Checkpoint metadata.
    :param str | None tokenizer_snapshot_hash: Current tokenizer snapshot hash.
    :param str parameter_manifest_hash: Current parameter-manifest hash.
    """
    _check_resume_compat(
        cfg,
        meta,
        tokenizer_snapshot_hash=tokenizer_snapshot_hash,
        parameter_manifest_hash=parameter_manifest_hash,
    )


def _checkpoint_record(cfg: Config, *, step: int = 0, tokens_seen: int = 0) -> CheckpointMeta:
    """Build production-format metadata for resume compatibility tests.

    :param Config cfg: Configuration captured by the checkpoint.
    :param int step: Completed training step.
    :param int tokens_seen: Cumulative loss-token count.
    :return CheckpointMeta: Production checkpoint metadata record.
    """
    return build_meta(
        step=step,
        config=cfg.to_dict(),
        data_fingerprint=data_fingerprint(cfg),
        parameter_manifest_hash="test-parameter-manifest",
        tokens_seen=tokens_seen,
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


@pytest.fixture
def track_checkpoint_manager() -> Iterator[Callable[[Any], Any]]:
    """Track direct-test checkpoint managers and close them after each test."""
    managers: list[Any] = []

    def _track(manager: Any) -> Any:
        """Register and return one checkpoint manager."""
        managers.append(manager)
        return manager

    yield _track
    for manager in reversed(managers):
        manager.close()


def _saved_step1_checkpoint(
    run_dir: Path,
    track_checkpoint_manager: Callable[[Any], Any],
    *,
    async_save: bool = False,
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
    mgr = track_checkpoint_manager(
        make_manager(ckpt_dir, max_to_keep=2, save_every=1, async_save=async_save)
    )

    meta = _checkpoint_record(cfg, step=1)
    save(mgr, step=1, train_state=state, data_iter=data_it, meta=meta)
    mgr.wait_until_finished()
    return cfg, state, mgr, ckpt_dir


def test_async_checkpoint_roundtrip(
    tmp_path: Path, track_checkpoint_manager: Callable[[Any], Any]
) -> None:
    """Async checkpoint save should roundtrip state correctly."""
    cfg, state, mgr, _ckpt_dir = _saved_step1_checkpoint(
        tmp_path / "run_async", track_checkpoint_manager, async_save=True
    )

    abstract_state = abstractify_tree(state)
    data_it_restore = build_train_iterator(cfg)
    step, restored, _meta = restore_at_step(
        mgr, step=1, abstract_train_state=abstract_state, data_iter=data_it_restore
    )
    assert step == 1
    assert eqx.tree_equal(restored.params, state.params)
    assert eqx.tree_equal(restored.opt_state, state.opt_state)


def test_checkpoint_step_consistency_rejects_all_mismatches(tmp_path: Path) -> None:
    """Directory, metadata, and train-state steps form one indivisible identity."""
    cfg = _base_cfg(tmp_path / "run_step_consistency")
    state = _make_state()
    meta = _checkpoint_record(cfg, step=1)
    validate_checkpoint_steps(directory_step=1, meta=meta, train_state=state)

    with pytest.raises(RuntimeError, match="metadata=1"):
        validate_checkpoint_steps(directory_step=2, meta=meta, train_state=state)
    wrong_state = TrainState(
        step=jnp.array(2), params=state.params, opt_state=state.opt_state, rng=state.rng
    )
    with pytest.raises(RuntimeError, match="train_state=2"):
        validate_checkpoint_steps(directory_step=1, meta=meta, train_state=wrong_state)


def test_checkpoint_manager_false_save_result_is_failure(tmp_path: Path) -> None:
    """A manager that declines a save must not be reported as checkpointed."""

    class _RejectingManager:
        """CheckpointManager stub that declines the request without raising."""

        def save(self, *args: Any, **kwargs: Any) -> bool:
            """Return the Orbax not-saved signal."""
            return False

    cfg = _base_cfg(tmp_path / "run_rejected_save")
    with pytest.raises(RuntimeError, match="rejected save"):
        save(
            _RejectingManager(),  # type: ignore[arg-type]
            step=1,
            train_state=_make_state(),
            data_iter=build_train_iterator(cfg),
            meta=_checkpoint_record(cfg, step=1),
        )


def test_restore_params_only(
    tmp_path: Path, track_checkpoint_manager: Callable[[Any], Any]
) -> None:
    """Params-only restore (generate CLI path) matches the saved params exactly."""
    _cfg, state, _mgr, ckpt_dir = _saved_step1_checkpoint(
        tmp_path / "run_params_only", track_checkpoint_manager
    )

    params = restore_params_only(ckpt_dir / "1", abstractify_tree(state.params))
    assert eqx.tree_equal(params, state.params)

    with pytest.raises(FileNotFoundError, match="train_state"):
        restore_params_only(ckpt_dir / "999", abstractify_tree(state.params))


def test_checkpoint_data_state_roundtrip(
    tmp_path: Path, track_checkpoint_manager: Callable[[Any], Any]
) -> None:
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
    mgr = track_checkpoint_manager(
        make_manager(
            ckpt_dir,
            max_to_keep=cfg.checkpoint.max_to_keep,
            save_every=cfg.checkpoint.save_every,
            async_save=cfg.checkpoint.async_save,
        )
    )

    state = TrainState(
        step=jnp.array(2, dtype=jnp.int32),
        params={"w": jnp.array([1.0], dtype=jnp.float32)},
        opt_state={"m": jnp.array([0.5], dtype=jnp.float32)},
        rng=jax.random.PRNGKey(0),
    )
    meta = _checkpoint_record(cfg, step=2)
    save(mgr, step=2, train_state=state, data_iter=data_it, meta=meta)
    mgr.wait_until_finished()

    expected = next(data_it)
    data_it_restore = build_train_iterator(cfg, tokenizer=tokenizer)
    abstract_state = abstractify_tree(state)
    step, _restored, _meta = restore_at_step(
        mgr, step=2, abstract_train_state=abstract_state, data_iter=data_it_restore
    )
    assert step == 2
    restored_batch = next(data_it_restore)
    assert eqx.tree_equal(expected, restored_batch)


def test_corrupt_checkpoint_fails_restore(
    tmp_path: Path, track_checkpoint_manager: Callable[[Any], Any]
) -> None:
    """Corrupted checkpoint metadata should raise an error on restore."""
    cfg, state, mgr, ckpt_dir = _saved_step1_checkpoint(
        tmp_path / "run_corrupt", track_checkpoint_manager
    )

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


def test_max_to_keep_prunes_checkpoints(
    tmp_path: Path, track_checkpoint_manager: Callable[[Any], Any]
) -> None:
    """Checkpoint manager should prune old checkpoints per max_to_keep."""
    run_dir = tmp_path / "run_prune"
    cfg = _base_cfg(run_dir)
    state = _make_state()
    data_it = build_train_iterator(cfg)
    ckpt_dir = default_ckpt_dir(run_dir)
    mgr = track_checkpoint_manager(
        make_manager(ckpt_dir, max_to_keep=2, save_every=1, async_save=False)
    )

    for step in (1, 2, 3):
        state = TrainState(
            step=jnp.array(step),
            params=state.params,
            opt_state=state.opt_state,
            rng=state.rng,
        )
        meta = _checkpoint_record(cfg, step=step)
        save(
            mgr,
            step=step,
            train_state=state,
            data_iter=data_it,
            meta=meta,
        )
        mgr.wait_until_finished()

    state = TrainState(
        step=jnp.array(4), params=state.params, opt_state=state.opt_state, rng=state.rng
    )
    meta = _checkpoint_record(cfg, step=4)
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


def test_checkpoint_root_dir_resolves_relative_to_run_dir(
    tmp_path: Path, track_checkpoint_manager: Callable[[Any], Any]
) -> None:
    """Relative checkpoint.root_dir should resolve against run_dir."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    cfg = Config()
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, root_dir="ckpts"))

    manager = track_checkpoint_manager(_build_checkpoint_manager(cfg, run_dir))

    assert manager is not None
    assert Path(manager.directory) == (run_dir / "ckpts").resolve()


def test_external_checkpoint_root_is_locked_owned_and_resumable(
    tmp_path: Path,
) -> None:
    """One external Orbax tree must belong to exactly one inactive or active run."""
    checkpoint_root = tmp_path / "external-checkpoints"
    cfg, config_src = make_small_run_cfg(tmp_path, run_subdir="owner-run", decay_steps=1)
    cfg = replace(
        cfg,
        checkpoint=replace(cfg.checkpoint, root_dir=str(checkpoint_root)),
    )
    run_dir = Path(cfg.logging.run_dir or "")

    with (
        RunDirectoryLock(checkpoint_root, resource_name="Checkpoint root"),
        pytest.raises(RuntimeError, match="Checkpoint root is already active"),
    ):
        run(cfg, config_path=str(config_src), resume="none", dry_run=False)
    assert not run_dir.exists()
    assert not checkpoint_root.exists()

    assert run(cfg, config_path=str(config_src), resume="none", dry_run=False) == run_dir
    marker = json.loads((checkpoint_root / ".chomp-owner.json").read_text())
    assert marker == {"schema_version": 1, "run_dir": str(run_dir.resolve())}
    assert run(cfg, config_path=str(config_src), resume="latest", dry_run=False) == run_dir

    other_cfg, _ = make_small_run_cfg(tmp_path, run_subdir="other-run", decay_steps=1)
    other_cfg = replace(
        other_cfg,
        checkpoint=replace(other_cfg.checkpoint, root_dir=str(checkpoint_root)),
    )
    with pytest.raises(RuntimeError, match="belongs to run"):
        run(other_cfg, config_path=str(config_src), resume="none", dry_run=False)
    assert not Path(other_cfg.logging.run_dir or "").exists()


def test_run_closes_manager_and_preflights_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run must close its manager and reject drift before restoring iterator state."""
    import orbax.checkpoint as ocp

    close_calls: list[int] = []
    real_close = ocp.CheckpointManager.close

    def _tracked_close(manager: Any) -> None:
        """Record close calls while preserving Orbax cleanup behavior."""
        close_calls.append(id(manager))
        real_close(manager)

    monkeypatch.setattr(ocp.CheckpointManager, "close", _tracked_close)
    cfg, config_src = make_small_run_cfg(tmp_path, decay_steps=1)
    run(cfg, config_path=str(config_src), resume="none", dry_run=False)
    assert len(close_calls) == 1

    restore_calls = 0

    def _unexpected_full_restore(*args: Any, **kwargs: Any) -> Any:
        """Fail if incompatible metadata reaches model/data restoration."""
        nonlocal restore_calls
        restore_calls += 1
        raise AssertionError("full restore ran before compatibility validation")

    monkeypatch.setattr("chomp.train.restore_at_step", _unexpected_full_restore)
    incompatible = replace(cfg, data=replace(cfg.data, local_text="different corpus"))
    close_calls.clear()

    with pytest.raises(RuntimeError, match="local_text_hash"):
        run(incompatible, config_path=str(config_src), resume="latest", dry_run=False)

    assert restore_calls == 0
    assert len(close_calls) == 1


def test_resume_requires_existing_tokenizer_snapshot(tmp_path: Path) -> None:
    """A missing tokenizer snapshot must fail before mutating the run directory."""
    cfg, config_src = make_small_run_cfg(tmp_path, decay_steps=1)
    run_dir = run(cfg, config_path=str(config_src), resume="none", dry_run=False)
    tokenizer_dir = run_dir / "tokenizer"
    shutil.rmtree(tokenizer_dir)
    resume_record = run_dir / "config_resume.json"
    assert not resume_record.exists()

    with pytest.raises(FileNotFoundError, match="tokenizer snapshot is missing"):
        run(cfg, config_path=str(config_src), resume="latest", dry_run=False)

    assert not tokenizer_dir.exists()
    assert not resume_record.exists()


def test_checkpoint_saves_final_step(tmp_path: Path) -> None:
    """Final step should be checkpointed even if save_every does not divide steps."""
    cfg, config_src = make_small_run_cfg(tmp_path, decay_steps=2)
    cfg = replace(cfg, train=replace(cfg.train, steps=3))
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, save_every=2))

    run_dir = run(cfg, config_path=str(config_src), resume="none", dry_run=False)
    ckpt_dir = default_ckpt_dir(run_dir)

    assert (ckpt_dir / "2").exists(), "expected checkpoint at save interval"
    assert (ckpt_dir / "3").exists(), "expected final checkpoint at step 3"


def test_explicit_resume_rejects_older_retained_step(tmp_path: Path) -> None:
    """In-place rollback must not collide with newer finalized checkpoints."""
    cfg, config_src = make_small_run_cfg(tmp_path, decay_steps=4)
    cfg = replace(cfg, train=replace(cfg.train, steps=4))
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, max_to_keep=4))
    run_dir = run(cfg, config_path=str(config_src), resume="none", dry_run=False)

    with pytest.raises(RuntimeError, match="newer step 4 already exists"):
        run(cfg, config_path=str(config_src), resume=3, dry_run=False)

    ckpt_dir = default_ckpt_dir(run_dir)
    assert {int(path.name) for path in ckpt_dir.iterdir() if path.name.isdigit()} == {1, 2, 3, 4}


def test_stop_signal_state_records_reason_and_restores_handlers() -> None:
    """The signal bridge should defer work and restore the process handler."""
    previous = signal.getsignal(signal.SIGTERM)
    state = _StopSignalState()

    with state:
        state._handle(signal.SIGTERM, None)
        assert state.requested
        assert state.reason == "SIGTERM"

    assert signal.getsignal(signal.SIGTERM) is previous


def test_stop_signal_state_warns_when_off_main_thread(caplog: LogCaptureFixture) -> None:
    """Skipping handler install off the main thread must not be silent.

    Preemption stops triggering a final checkpoint in that case; without a
    log line the feature becomes a silent no-op.
    """
    previous = signal.getsignal(signal.SIGTERM)
    state = _StopSignalState()

    def _enter_and_exit() -> None:
        with state:
            pass

    with caplog.at_level(logging.WARNING, logger="chomp.train"):
        thread = threading.Thread(target=_enter_and_exit)
        thread.start()
        thread.join()

    assert signal.getsignal(signal.SIGTERM) is previous
    assert any("handlers were not" in record.message for record in caplog.records)


def test_preemption_finishes_one_step_and_writes_aligned_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cooperative stop after train_step should save that exact completed step."""
    import chomp.train as train_mod

    cfg, config_src = make_small_run_cfg(tmp_path, decay_steps=5)
    cfg = replace(cfg, train=replace(cfg.train, steps=5))
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, save_every=100))

    stop = _FakeStopSignal()
    monkeypatch.setattr(train_mod, "_StopSignalState", lambda: stop)
    real_make_train_step = train_mod.make_train_step

    def _make_stopping_step(*args: Any, **kwargs: Any) -> Any:
        step = real_make_train_step(*args, **kwargs)

        def _step(state: TrainState, batch: Batch) -> Any:
            result = step(state, batch)
            stop.requested = True
            return result

        return _step

    monkeypatch.setattr(train_mod, "make_train_step", _make_stopping_step)

    with pytest.raises(TrainingPreempted) as exc_info:
        run(cfg, config_path=str(config_src), resume="none", dry_run=False)
    assert exc_info.value.signal_name == "SIGTERM"
    assert exc_info.value.exit_code == 143
    run_dir = exc_info.value.run_dir

    rows = read_jsonl(run_dir / cfg.logging.metrics_file)
    assert any(
        row.get("preemption_requested")
        and row.get("preemption_signal") == "SIGTERM"
        and row.get("step") == 1
        for row in rows
    )
    ckpt_dir = default_ckpt_dir(run_dir)
    steps = {int(path.name) for path in ckpt_dir.iterdir() if path.name.isdigit()}
    assert steps == {1}


def test_preemption_during_final_logging_tail_is_not_lost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A signal after the last post-update poll must still report preemption."""
    import chomp.train as train_mod

    cfg, config_src = make_small_run_cfg(tmp_path, decay_steps=3)
    cfg = replace(cfg, train=replace(cfg.train, steps=3, log_every=1))
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, save_every=100))

    stop = _FakeStopSignal()
    monkeypatch.setattr(train_mod, "_StopSignalState", lambda: stop)
    real_write = train_mod.MetricsWriter.write

    def _write_and_signal(writer: Any, row: dict[str, Any]) -> None:
        real_write(writer, row)
        if row.get("step") == 3 and "loss" in row:
            stop.requested = True

    monkeypatch.setattr(train_mod.MetricsWriter, "write", _write_and_signal)

    with pytest.raises(TrainingPreempted) as exc_info:
        run(cfg, config_path=str(config_src), resume="none", dry_run=False)

    assert exc_info.value.signal_name == "SIGTERM"
    rows = read_jsonl(exc_info.value.run_dir / cfg.logging.metrics_file)
    assert any(row.get("preemption_requested") and row.get("step") == 3 for row in rows)
    assert (default_ckpt_dir(exc_info.value.run_dir) / "3").exists()


def test_run_enforces_device_before_artifact_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public Python API must enforce the GPU policy before writing a run."""
    cfg, config_src = make_small_run_cfg(tmp_path)
    cfg = replace(cfg, train=replace(cfg.train, allow_cpu=False))
    run_dir = Path(cfg.logging.run_dir or "")
    calls: list[bool] = []

    def _reject_device(*, allow_cpu: bool) -> None:
        calls.append(allow_cpu)
        raise RuntimeError("injected non-CUDA backend")

    monkeypatch.setattr("chomp.train.validate_default_device", _reject_device)

    with pytest.raises(RuntimeError, match="injected non-CUDA backend"):
        run(cfg, config_path=str(config_src), resume="none", dry_run=False)

    assert calls == [False]
    assert not run_dir.exists()


def test_run_lock_fails_before_fresh_artifact_setup(tmp_path: Path) -> None:
    """A competing owner must prevent even fresh config/tokenizer artifact writes."""
    cfg, config_src = make_small_run_cfg(tmp_path)
    run_dir = Path(cfg.logging.run_dir or "")

    with RunDirectoryLock(run_dir), pytest.raises(RuntimeError, match="already active"):
        run(cfg, config_path=str(config_src), resume="none", dry_run=False)

    assert not run_dir.exists()


@pytest.mark.parametrize("grain_prefetch", [0, 1])
def test_exact_eof_after_batch_boundary_saves_final_checkpoint(
    tmp_path: Path, grain_prefetch: int
) -> None:
    """Exact EOF after a completed batch should still save the final checkpoint."""
    cfg, config_src = make_small_run_cfg(tmp_path, local_text="x" * 48, decay_steps=10)
    cfg = replace(cfg, train=replace(cfg.train, steps=10, grad_accum=1))
    cfg = replace(
        cfg,
        data=replace(
            cfg.data,
            repeat=False,
            window_shuffle_tokens=0,
            grain_prefetch=grain_prefetch,
        ),
    )
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, save_every=2))

    run_dir = run(cfg, config_path=str(config_src), resume="none", dry_run=False)

    metrics_path = run_dir / cfg.logging.metrics_file
    rows = read_jsonl(metrics_path)
    assert any(row.get("data_exhausted") and row.get("step") == 3 for row in rows)

    ckpt_dir = default_ckpt_dir(run_dir)
    steps_on_disk = {int(p.name) for p in ckpt_dir.iterdir() if p.is_dir() and p.name.isdigit()}
    assert steps_on_disk == {2, 3}, (
        f"exact EOF must save the aligned final checkpoint, found {steps_on_disk}"
    )


def test_crash_between_fetch_and_step_skips_final_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    track_checkpoint_manager: Callable[[Any], Any],
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
        init_train_state(params=params, tx=tx, key=jax.random.PRNGKey(1))
    )

    states = []
    for run_dir in (run_dir_cont, Path(cfg_crash.logging.run_dir)):
        mgr = track_checkpoint_manager(
            make_manager(default_ckpt_dir(run_dir), max_to_keep=2, save_every=2, async_save=False)
        )
        _, state, _ = restore_at_step(
            mgr,
            step=5,
            abstract_train_state=abstract_state,
            data_iter=build_train_iterator(cfg_ref, tokenizer=tokenizer),
        )
        states.append(state)

    assert int(jax.device_get(states[0].step)) == 5
    assert eqx.tree_equal(states[0].params, states[1].params)
    assert eqx.tree_equal(states[0].opt_state, states[1].opt_state)


def test_finite_partial_batch_trains_and_saves_aligned_checkpoint(tmp_path: Path) -> None:
    """A usable finite tail must train before an aligned final checkpoint."""
    # One 116-char doc -> 116 byte tokens (offset 0, no BOS/EOS; varied bytes
    # so windows differ): seven full seq_len=16 rows plus one padded four-token
    # row. grad_accum=2 therefore produces four optimizer batches.
    text = "".join(chr(97 + (i * 7) % 26) for i in range(116))
    cfg, config_src = make_small_run_cfg(tmp_path, local_text=text, decay_steps=10)
    cfg = replace(cfg, train=replace(cfg.train, steps=10, grad_accum=2))
    cfg = replace(
        cfg,
        data=replace(
            cfg.data,
            repeat=False,
        ),
    )
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, save_every=2))

    run_dir = run(cfg, config_path=str(config_src), resume="none", dry_run=False)

    metrics_path = run_dir / cfg.logging.metrics_file
    rows = read_jsonl(metrics_path)
    assert any(row.get("data_exhausted") for row in rows)
    assert len([row for row in rows if row.get("step") == 4 and "loss" in row]) == 1

    ckpt_dir = default_ckpt_dir(run_dir)
    steps_on_disk = {int(p.name) for p in ckpt_dir.iterdir() if p.is_dir() and p.name.isdigit()}
    assert steps_on_disk == {2, 4}

    # Resume sees exact EOF at the saved aligned iterator state; it performs no
    # additional optimizer step and retains the final checkpoint.
    run(cfg, config_path=str(config_src), resume="latest", dry_run=False)
    rows = read_jsonl(metrics_path)
    assert len([row for row in rows if row.get("step") == 4 and "loss" in row]) == 1
    steps_on_disk = {int(p.name) for p in ckpt_dir.iterdir() if p.is_dir() and p.name.isdigit()}
    assert steps_on_disk == {2, 4}


def test_zero_loss_batch_does_not_mutate_training_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    """A zero-objective batch must fail before train_step or final checkpointing."""
    import chomp.train as train_mod

    run_dir = tmp_path / "run_zero_loss"
    cfg = Config(
        model=ModelConfig(backend="dummy", vocab_size=256, d_model=16, dropout=0.0),
        data=DataConfig(
            backend="local_text",
            local_text="a",
            repeat=True,
            window_shuffle_tokens=0,
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
            eval_every=0,
        ),
        optim=OptimConfig(warmup_steps=0),
        checkpoint=CheckpointConfig(enabled=True, save_every=1, async_save=False),
        logging=LoggingConfig(run_dir=str(run_dir)),
        debug=DebugConfig(check_device_every=0),
    )

    captured: dict[str, Any] = {}
    real_build = train_mod._build_model_state

    def _capture_initial_state(config: Config) -> Any:
        """Snapshot the initial state that must remain untouched."""
        result = real_build(config)
        captured["state"] = result[4]
        captured["snapshot"] = jax.tree_util.tree_map(lambda value: jnp.array(value), result[4])
        return result

    train_step_calls = 0

    def _spy_make_train_step(*args: Any, **kwargs: Any) -> Any:
        """Return a step function that records any forbidden invocation."""

        def _step(state: TrainState, batch: Batch) -> Any:
            nonlocal train_step_calls
            train_step_calls += 1
            return state, {}

        return _step

    monkeypatch.setattr(train_mod, "_build_model_state", _capture_initial_state)
    monkeypatch.setattr(train_mod, "make_train_step", _spy_make_train_step)

    with (
        caplog.at_level(logging.WARNING, logger="chomp.train"),
        pytest.raises(ZeroLossTokensError, match="zero valid loss tokens"),
    ):
        run(cfg, config_path=None, resume="none")

    assert train_step_calls == 0
    assert eqx.tree_equal(captured["state"], captured["snapshot"])
    ckpt_dir = default_ckpt_dir(run_dir)
    saved_steps = (
        {path.name for path in ckpt_dir.iterdir() if path.is_dir() and path.name.isdigit()}
        if ckpt_dir.exists()
        else set()
    )
    assert saved_steps == set()
    assert any("Skipping final checkpoint" in record.getMessage() for record in caplog.records)


def test_resume_bit_exact_with_prefetch_and_window_shuffle(
    tmp_path: Path, track_checkpoint_manager: Callable[[Any], Any]
) -> None:
    """Interrupted + resumed must match continuous bit-exactly with the full
    iterator stack engaged: grain_prefetch > 0 and window shuffle enabled.

    This pins the sharpest resume question: when the prefetch thread has
    pulled batches ahead of the consumer, the serialized iterator state must
    represent the consumer-visible position, not the advanced parent — and
    the window shuffle must replay identically from a checkpoint taken
    mid-window. Every window is distinct (varied text, period coprime with
    seq_len), so a skipped or reordered batch cannot cancel out.
    """
    # 101 varied byte tokens; gcd(101, 16) = 1, so packed windows repeat only
    # every 101 windows — far beyond the 12 this test consumes.
    text = "".join(chr(97 + (i * 7) % 26) for i in range(101))

    def _make_cfg(subdir: str, steps: int) -> tuple[Config, Path]:
        cfg, config_src = make_small_run_cfg(tmp_path, run_subdir=subdir, decay_steps=6)
        cfg = replace(cfg, train=replace(cfg.train, steps=steps, grad_accum=2))
        cfg = replace(
            cfg,
            data=replace(
                cfg.data,
                local_text=text,
                grain_prefetch=2,
                window_shuffle_tokens=128,
            ),
        )
        # save_every > steps: the interrupted run's step-3 checkpoint is the
        # finally-block final save, taken while prefetch is ahead and 6
        # windows into shuffle block 0 (blocks span 8 windows; 2 per step).
        return replace(cfg, checkpoint=replace(cfg.checkpoint, save_every=4)), config_src

    cfg_cont, config_src = _make_cfg("run_pf_cont", steps=6)
    run_dir_cont = run(cfg_cont, config_path=str(config_src), resume="none", dry_run=False)

    cfg_int, _ = _make_cfg("run_pf_int", steps=3)
    run_dir_int = run(cfg_int, config_path=str(config_src), resume="none", dry_run=False)
    cfg_resume, _ = _make_cfg("run_pf_int", steps=6)
    resumed_run_dir = run(cfg_resume, config_path=str(config_src), resume="latest", dry_run=False)
    assert resumed_run_dir == run_dir_int

    # Per-step losses agree exactly across the resume boundary (steps 4-6 ran
    # from the restored mid-window prefetching iterator).
    def _losses(run_dir: Path) -> dict[int, float]:
        rows = read_jsonl(run_dir / "metrics.jsonl")
        return {int(r["step"]): r["loss"] for r in rows if "loss" in r and "step" in r}

    losses_cont = _losses(run_dir_cont)
    losses_int = _losses(run_dir_int)
    assert set(losses_cont) == set(losses_int) == {1, 2, 3, 4, 5, 6}
    # Teeth: distinct windows produce distinct step losses, so an off-by-one
    # in the replayed stream could not produce equal sequences by accident.
    assert len(set(losses_cont.values())) > 1
    assert losses_cont == losses_int

    # And the step-6 train states are bit-identical.
    cfg_ref, tokenizer = prepare_tokenizer_and_config(cfg_cont)
    params, _static = build_model(cfg_ref, key=jax.random.PRNGKey(0))
    tx, _ = build_optimizer(cfg_ref, params)
    abstract_state = abstractify_tree(
        init_train_state(params=params, tx=tx, key=jax.random.PRNGKey(1))
    )
    states = []
    for run_dir in (run_dir_cont, run_dir_int):
        mgr = track_checkpoint_manager(
            make_manager(default_ckpt_dir(run_dir), max_to_keep=2, save_every=4, async_save=False)
        )
        _, state, _ = restore_at_step(
            mgr,
            step=6,
            abstract_train_state=abstract_state,
            data_iter=build_train_iterator(cfg_ref, tokenizer=tokenizer),
        )
        states.append(state)
    assert eqx.tree_equal(states[0].params, states[1].params)
    assert eqx.tree_equal(states[0].opt_state, states[1].opt_state)


def test_resume_bit_exact_through_exhaustion_flush(
    tmp_path: Path, track_checkpoint_manager: Callable[[Any], Any]
) -> None:
    """Interrupted + resumed must match continuous bit-exactly when the run
    ends in an FFD end-of-stream flush, with window shuffle + prefetch engaged.

    This is the integration claim behind the flush feature: grain's window
    shuffle replays its parent from the block start on resume, so the resumed
    process re-drives the packer through StopIteration -> finish() -> flush.
    If the replayed flush produced different windows than the continuous run,
    step-3 losses or the final states would diverge.
    """
    # One 84-byte doc -> segments [16]*5 + [4] at seq_len=16. With
    # bins_per_pack = grad_accum*batch_size = 2 and buffer_docs=4, two pack
    # cycles emit 4 windows and leave [16, 4] pending below threshold — only
    # the exhaustion flush turns those into windows 5 and 6. Teeth: step 3
    # trains on flushed windows, so without the flush both runs would end at
    # step 2 and the step-set assertion below fails.
    text = "".join(chr(97 + (i * 7) % 26) for i in range(84))

    def _make_cfg(subdir: str, steps: int) -> tuple[Config, Path]:
        cfg, config_src = make_small_run_cfg(tmp_path, run_subdir=subdir, decay_steps=3)
        cfg = replace(cfg, train=replace(cfg.train, steps=steps, grad_accum=2))
        cfg = replace(
            cfg,
            data=replace(
                cfg.data,
                local_text=text,
                repeat=False,
                packing_mode="bin",
                grain_prefetch=2,
                window_shuffle_tokens=128,
            ),
        )
        # save_every > steps: only the finally-block final save runs, so the
        # interrupted run checkpoints exactly at step 2.
        return replace(cfg, checkpoint=replace(cfg.checkpoint, save_every=4)), config_src

    cfg_cont, config_src = _make_cfg("run_flush_cont", steps=3)
    run_dir_cont = run(cfg_cont, config_path=str(config_src), resume="none", dry_run=False)

    cfg_int, _ = _make_cfg("run_flush_int", steps=2)
    run_dir_int = run(cfg_int, config_path=str(config_src), resume="none", dry_run=False)
    cfg_resume, _ = _make_cfg("run_flush_int", steps=3)
    run(cfg_resume, config_path=str(config_src), resume="latest", dry_run=False)

    def _losses(run_dir: Path) -> dict[int, float]:
        rows = read_jsonl(run_dir / "metrics.jsonl")
        return {int(r["step"]): r["loss"] for r in rows if "loss" in r and "step" in r}

    losses_cont = _losses(run_dir_cont)
    losses_int = _losses(run_dir_int)
    # Step 3 exists only because the flush emitted windows 5 and 6.
    assert set(losses_cont) == set(losses_int) == {1, 2, 3}
    assert len(set(losses_cont.values())) > 1
    assert losses_cont == losses_int

    cfg_ref, tokenizer = prepare_tokenizer_and_config(cfg_cont)
    params, _static = build_model(cfg_ref, key=jax.random.PRNGKey(0))
    tx, _ = build_optimizer(cfg_ref, params)
    abstract_state = abstractify_tree(
        init_train_state(params=params, tx=tx, key=jax.random.PRNGKey(1))
    )
    states = []
    for run_dir in (run_dir_cont, run_dir_int):
        mgr = track_checkpoint_manager(
            make_manager(default_ckpt_dir(run_dir), max_to_keep=2, save_every=4, async_save=False)
        )
        _, state, _ = restore_at_step(
            mgr,
            step=3,
            abstract_train_state=abstract_state,
            data_iter=build_train_iterator(cfg_ref, tokenizer=tokenizer),
        )
        states.append(state)
    assert eqx.tree_equal(states[0].params, states[1].params)
    assert eqx.tree_equal(states[0].opt_state, states[1].opt_state)


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

    # W&B telemetry must agree with the raised failure: finish() is called
    # with a nonzero exit code, not 0-then-raise.
    finish_codes: list[int] = []

    class _FakeWandbRun:
        summary: dict[str, Any] = {}

        def log(self, *args: Any, **kwargs: Any) -> None:
            pass

        def finish(self, exit_code: int = 0) -> None:
            finish_codes.append(int(exit_code))

    monkeypatch.setattr("chomp.train.save", _failing_save)
    monkeypatch.setattr("chomp.train._maybe_init_wandb", lambda *a, **k: _FakeWandbRun())
    with pytest.raises(RuntimeError, match="run finalization failed"):
        run(cfg, config_path=str(config_src), resume="none", dry_run=False)
    assert finish_codes == [1], (
        f"W&B must record the checkpoint finalization failure, got {finish_codes}"
    )
    monkeypatch.setattr("chomp.train._maybe_init_wandb", lambda *a, **k: None)

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


def _poison_loss_at_step(monkeypatch: pytest.MonkeyPatch, poison_step: int) -> None:
    """Wrap make_train_step so metrics['loss'] is NaN at one chosen step.

    :param pytest.MonkeyPatch monkeypatch: Fixture used to patch chomp.train.
    :param int poison_step: 1-based step whose loss becomes NaN.
    """
    import chomp.train as train_mod

    real_make = train_mod.make_train_step

    def _poisoned_make(cfg: Config, **kwargs: Any) -> Any:
        step_fn = real_make(cfg, **kwargs)

        def wrapped(state: Any, batch: Batch) -> tuple[Any, dict[str, Any]]:
            new_state, metrics = step_fn(state, batch)
            metrics = dict(metrics)
            metrics["loss"] = jnp.where(new_state.step == poison_step, jnp.nan, metrics["loss"])
            return new_state, metrics

        return wrapped

    monkeypatch.setattr("chomp.train.make_train_step", _poisoned_make)


def _poison_state_at_step(monkeypatch: pytest.MonkeyPatch, *, poison_step: int, field: str) -> None:
    """Inject NaNs into post-update parameters or optimizer state.

    :param pytest.MonkeyPatch monkeypatch: Fixture used to patch chomp.train.
    :param int poison_step: One-based step whose state becomes non-finite.
    :param str field: TrainState field to poison: params or opt_state.
    """
    import chomp.train as train_mod

    real_make = train_mod.make_train_step

    def _poisoned_make(cfg: Config, **kwargs: Any) -> Any:
        step_fn = real_make(cfg, **kwargs)

        def wrapped(state: Any, batch: Batch) -> tuple[Any, dict[str, Any]]:
            new_state, metrics = step_fn(state, batch)
            target = getattr(new_state, field)
            poisoned = jax.tree_util.tree_map(
                lambda leaf: (
                    jnp.where(
                        new_state.step == poison_step,
                        jnp.full_like(leaf, jnp.nan),
                        leaf,
                    )
                    if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact)
                    else leaf
                ),
                target,
            )
            values = {
                "step": new_state.step,
                "params": new_state.params,
                "opt_state": new_state.opt_state,
                "rng": new_state.rng,
                field: poisoned,
            }
            return TrainState(**values), metrics

        return wrapped

    monkeypatch.setattr("chomp.train.make_train_step", _poisoned_make)


def test_periodic_save_step_forces_finite_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    """A save step that is not a logging step must still run the finite
    check before writing: without the forced sync, a NaN step landing on the
    save cadence would be persisted as a resume point.
    """
    cfg, config_src = make_small_run_cfg(tmp_path, run_subdir="run_nan_save", decay_steps=5)
    # log_every=1000: step 3 is a save step but not a logging step.
    cfg = replace(cfg, train=replace(cfg.train, steps=5, log_every=1000))
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, save_every=3))
    _poison_loss_at_step(monkeypatch, poison_step=3)

    with (
        caplog.at_level(logging.ERROR, logger="chomp.train"),
        pytest.raises(RuntimeError, match="Non-finite loss at step 3"),
    ):
        run(cfg, config_path=str(config_src), resume="none", dry_run=False)

    ckpt_dir = default_ckpt_dir(Path(cfg.logging.run_dir))
    steps_on_disk = (
        {int(p.name) for p in ckpt_dir.iterdir() if p.is_dir() and p.name.isdigit()}
        if ckpt_dir.exists()
        else set()
    )
    assert steps_on_disk == set(), f"NaN step must never reach disk, found {steps_on_disk}"
    # The finally-block validation also refused to write the poisoned state.
    assert any("Skipping final checkpoint at step" in rec.getMessage() for rec in caplog.records)


def test_final_checkpoint_refuses_nonfinite_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    """A NaN on a step that neither logs nor saves is only caught at exit:
    the final-save validation must skip the write, keep the last good
    periodic checkpoint as latest, and fail the run loudly.
    """
    cfg, config_src = make_small_run_cfg(tmp_path, run_subdir="run_nan_final", decay_steps=5)
    # Step 5 (the last step) neither logs nor saves, so the in-loop finite
    # check never sees it; only the finally-block validation can.
    cfg = replace(cfg, train=replace(cfg.train, steps=5, log_every=1000))
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, save_every=2))
    _poison_loss_at_step(monkeypatch, poison_step=5)

    with (
        caplog.at_level(logging.ERROR, logger="chomp.train"),
        pytest.raises(RuntimeError, match="run finalization failed: Non-finite loss at step 5"),
    ):
        run(cfg, config_path=str(config_src), resume="none", dry_run=False)

    ckpt_dir = default_ckpt_dir(Path(cfg.logging.run_dir))
    steps_on_disk = {int(p.name) for p in ckpt_dir.iterdir() if p.is_dir() and p.name.isdigit()}
    assert steps_on_disk == {2, 4}, f"latest must stay the last good save, found {steps_on_disk}"
    assert any("Skipping final checkpoint at step" in rec.getMessage() for rec in caplog.records)


@pytest.mark.parametrize(
    ("field", "poison_step", "save_every", "expected_steps", "match"),
    [
        ("params", 3, 3, set(), "Non-finite parameters at step 3"),
        ("opt_state", 5, 2, {2, 4}, "Non-finite optimizer state at step 5"),
    ],
)
def test_checkpoint_refuses_nonfinite_post_update_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    poison_step: int,
    save_every: int,
    expected_steps: set[int],
    match: str,
) -> None:
    """Finite pre-update metrics cannot authorize a poisoned TrainState save.

    :param Path tmp_path: Temporary test directory.
    :param pytest.MonkeyPatch monkeypatch: Fixture used to patch the train step.
    :param str field: TrainState partition poisoned after the optimizer update.
    :param int poison_step: Step whose state becomes non-finite.
    :param int save_every: Periodic checkpoint cadence for this case.
    :param set[int] expected_steps: Last known-good checkpoints expected on disk.
    :param str match: Expected validation failure text.
    """
    cfg, config_src = make_small_run_cfg(
        tmp_path,
        run_subdir=f"run_nonfinite_{field}",
        decay_steps=5,
    )
    cfg = replace(cfg, train=replace(cfg.train, steps=5, log_every=1000))
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, save_every=save_every))
    _poison_state_at_step(monkeypatch, poison_step=poison_step, field=field)

    with pytest.raises(RuntimeError, match=match):
        run(cfg, config_path=str(config_src), resume="none", dry_run=False)

    ckpt_dir = default_ckpt_dir(Path(cfg.logging.run_dir))
    steps_on_disk = (
        {int(path.name) for path in ckpt_dir.iterdir() if path.name.isdigit()}
        if ckpt_dir.exists()
        else set()
    )
    assert steps_on_disk == expected_steps


def test_checkpoint_disabled_run_rejects_nonfinite_final_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Final metric validity remains a run invariant without checkpointing."""
    cfg, config_src = make_small_run_cfg(
        tmp_path,
        run_subdir="run_nonfinite_no_checkpoint",
        decay_steps=2,
    )
    cfg = replace(cfg, train=replace(cfg.train, steps=2, log_every=1000))
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, enabled=False))
    _poison_loss_at_step(monkeypatch, poison_step=2)

    with pytest.raises(RuntimeError, match="Non-finite loss at step 2"):
        run(cfg, config_path=str(config_src), resume="none", dry_run=False)


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


def test_dry_run_compiles_single_step(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Dry run should compile one step, write config, but not metrics."""
    from chomp.data.grain import GrainTrainBatchIterator

    profile_events: list[str] = []
    close_calls: list[int] = []
    real_close = GrainTrainBatchIterator.close

    def _tracked_close(iterator: GrainTrainBatchIterator) -> None:
        """Record and preserve data-iterator cleanup."""
        close_calls.append(id(iterator))
        real_close(iterator)

    monkeypatch.setattr(GrainTrainBatchIterator, "close", _tracked_close)
    monkeypatch.setattr("chomp.train.start_trace", lambda _: profile_events.append("start"))
    monkeypatch.setattr("chomp.train.stop_trace", lambda: profile_events.append("stop"))
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
            profile=True,
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
    manifest = json.loads((run_dir / "parameter-manifest.json").read_text())
    assert manifest["group_counts"] == {"adam": 2}
    assert {entry["family"] for entry in manifest["arrays"]} == {"embedding", "projection"}
    assert not (run_dir / cfg.logging.metrics_file).exists()
    assert profile_events == ["start", "stop"]

    data = json.loads((run_dir / "config_resolved.json").read_text())
    assert data["derived"]["optim"]["decay_steps_effective"] == cfg.train.steps
    assert len(close_calls) == 1


def test_deterministic_checkpointing_warns(tmp_path: Path, caplog: LogCaptureFixture) -> None:
    """Deterministic mode should warn when use_checkpoint is enabled.

    :param Path tmp_path: Temporary directory for the run artifacts.
    :param LogCaptureFixture caplog: Log capture fixture.
    """
    run_dir = tmp_path / "dry_run_warn"
    cfg = Config(
        model=ModelConfig(
            backend="megalodon",
            vocab_size=128,
            model_dim=32,
            num_layers=1,
            num_heads=1,
            z_dim=16,
            value_dim=32,
            ffn_hidden_dim=64,
            cema_ndim=4,
            chunk_size=8,
            norm_num_groups=4,
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


def test_metrics_sinks_receive_distinct_projections() -> None:
    """Local and W&B telemetry should retain their sink-specific details."""
    row = {
        "step": 7,
        "loss": 1.5,
        "wall_time_s": 2.0,
        "packing_tokens": 11,
        "device_memory_gb": 3.0,
        "peak_memory_gb": 4.0,
    }

    local = _project_metrics(row, drop=_METRICS_FILE_DROP)
    wandb = _project_metrics(row, drop=_WANDB_DROP)

    assert local == {"step": 7, "loss": 1.5, "peak_memory_gb": 4.0}
    assert wandb == {
        "loss": 1.5,
        "wall_time_s": 2.0,
        "packing_tokens": 11,
        "device_memory_gb": 3.0,
    }


def test_training_crash_marks_wandb_failed_and_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Crashes should write a metrics row and finish W&B with exit_code=1.

    :param Path tmp_path: Temporary directory for run output.
    :param pytest.MonkeyPatch monkeypatch: Pytest monkeypatch fixture.
    """
    run_dir = tmp_path / "run"
    dummy_wandb = DummyWandbRun()
    from chomp.data.grain import GrainTrainBatchIterator

    close_calls: list[int] = []
    real_close = GrainTrainBatchIterator.close

    def _tracked_close(iterator: GrainTrainBatchIterator) -> None:
        """Record and preserve data-iterator cleanup on the crash path."""
        close_calls.append(id(iterator))
        real_close(iterator)

    monkeypatch.setattr(GrainTrainBatchIterator, "close", _tracked_close)

    def boom_make_train_step(*args: Any, **kwargs: Any) -> Any:
        """Return a train step that always raises a crash error."""

        def boom(state: Any, batch: Any) -> Any:
            """Raise a deterministic crash to exercise failure handling."""
            raise RuntimeError("kaboom")

        return boom

    monkeypatch.setattr("chomp.train.make_train_step", boom_make_train_step)
    monkeypatch.setattr("chomp.train._maybe_init_wandb", lambda *args, **kwargs: dummy_wandb)

    cfg = Config(
        model=ModelConfig(backend="dummy", vocab_size=256, d_model=8, dropout=0.0),
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
        run(cfg, config_path=None, resume="none", dry_run=False)

    assert dummy_wandb.finish_calls == [1]
    assert len(close_calls) == 1

    metrics_path = run_dir / cfg.logging.metrics_file
    rows = read_jsonl(metrics_path)
    assert any(row.get("crash") for row in rows)

    log_text = (run_dir / cfg.logging.log_file).read_text()
    assert "Training crashed" in log_text


def test_tokens_seen_matches_host_counts_between_sync_points(tmp_path: Path) -> None:
    """Host counts stay exact while intermediate optimizer steps remain asynchronous."""
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
            log_every=4,
            eval_every=0,
        ),
        optim=OptimConfig(lr=1e-3, weight_decay=0.0, grad_clip_norm=0.0, warmup_steps=0),
        checkpoint=CheckpointConfig(enabled=False),
        debug=DebugConfig(nan_check=True, check_device_every=0),
        logging=LoggingConfig(project="chomp", run_dir=str(tmp_path / "run")),
    )

    resolved, tokenizer = prepare_tokenizer_and_config(cfg)
    iterator = build_train_iterator(resolved, tokenizer=tokenizer)
    expected_counts = []
    for _ in range(cfg.train.steps):
        _ = next(iterator)
        expected_counts.append(iterator.get_loss_tokens())

    run(cfg, config_path=None, resume="none")

    metrics_path = Path(cfg.logging.run_dir) / cfg.logging.metrics_file
    rows = read_jsonl(metrics_path)
    train_rows = [row for row in rows if "loss_tokens" in row]
    assert len(train_rows) == 1
    assert train_rows[0]["packing_utilization"] > 0
    assert int(train_rows[0]["loss_tokens"]) == expected_counts[-1]
    assert int(train_rows[0]["tokens_seen"]) == sum(expected_counts)


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

    monkeypatch.setattr("chomp.train.supports_packed_segments", lambda params, static: False)
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
    meta = _checkpoint_record(cfg).to_dict()

    check_resume_compat(cfg, meta)  # identical config resumes cleanly

    changed_group = replace(cfg, data=replace(cfg.data, packing_group_docs=16))
    with pytest.raises(RuntimeError, match="packing_group_docs"):
        check_resume_compat(changed_group, meta)

    changed_strict = replace(cfg, data=replace(cfg.data, packing_strict_segments=False))
    with pytest.raises(RuntimeError, match="packing_strict_segments"):
        check_resume_compat(changed_strict, meta)

    binc = replace(cfg, data=replace(cfg.data, packing_mode="bin", packing_buffer_docs=8))
    bin_meta = _checkpoint_record(binc).to_dict()
    bin_changed = replace(binc, data=replace(binc.data, packing_strict_segments=False))
    with pytest.raises(RuntimeError, match="packing_strict_segments"):
        check_resume_compat(bin_changed, bin_meta)


def test_resume_compat_hard_gates_parameter_manifest(tmp_path: Path) -> None:
    """Resume must reject any change to trainable, optimizer, or decay assignments."""
    cfg = _base_cfg(tmp_path / "run_manifest_compat")
    meta = _checkpoint_record(cfg).to_dict()

    check_resume_compat(cfg, meta, parameter_manifest_hash="test-parameter-manifest")
    with pytest.raises(RuntimeError, match="parameter_manifest_hash"):
        check_resume_compat(cfg, meta, parameter_manifest_hash="changed-parameter-manifest")

    del meta["parameter_manifest_hash"]
    with pytest.raises(RuntimeError, match="parameter_manifest_hash"):
        check_resume_compat(cfg, meta, parameter_manifest_hash="test-parameter-manifest")


def _git(repo: Path, *args: str) -> None:
    """Run a git command in a test repository with a fixed identity."""
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=chomp-test",
            "-c",
            "user.email=chomp-test@example.com",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def test_source_revision_flags_untracked_and_tracked_changes(tmp_path: Path) -> None:
    """Source identity must flag untracked, unstaged, and staged src/ changes.

    `git diff` alone misses a brand-new uncommitted module, letting the strict
    runtime-identity resume gate pass on exactly the source drift it exists to
    catch.
    """
    from chomp.ckpt import _source_revision_for

    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    tracked = repo / "src" / "module.py"
    tracked.write_text("x = 1\n")
    _git(repo, "init", "--quiet")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "init")

    clean = _source_revision_for(repo)
    assert not clean.startswith("package:")
    assert "+dirty." not in clean

    untracked = repo / "src" / "new_module.py"
    untracked.write_text("y = 2\n")
    untracked_identity = _source_revision_for(repo)
    assert untracked_identity.startswith(f"{clean}+dirty.")
    untracked.write_text("y = 3\n")
    assert _source_revision_for(repo) != untracked_identity
    untracked.unlink()
    assert _source_revision_for(repo) == clean

    tracked.write_text("x = 2\n")
    unstaged_identity = _source_revision_for(repo)
    assert unstaged_identity.startswith(f"{clean}+dirty.")
    _git(repo, "add", "-A")
    assert _source_revision_for(repo) == unstaged_identity

    outside_scope = repo / "notes.txt"
    outside_scope.write_text("not source\n")
    _git(repo, "reset", "--quiet")
    _git(repo, "checkout", "--quiet", "--", "src")
    assert _source_revision_for(repo) == clean


def test_resume_compat_hard_gates_runtime_identity(tmp_path: Path) -> None:
    """Dependency, source, platform, and accelerator drift must reject resume."""
    cfg = _base_cfg(tmp_path / "run_runtime_compat")
    meta = _checkpoint_record(cfg).to_dict()
    meta["runtime"]["packages"]["jaxlib"] = "incompatible"

    with pytest.raises(RuntimeError, match="runtime.packages"):
        check_resume_compat(cfg, meta)

    del meta["runtime"]
    with pytest.raises(RuntimeError, match="runtime identity"):
        check_resume_compat(cfg, meta)


def test_resume_compat_ignores_inert_packing_knobs(tmp_path: Path) -> None:
    """Editing a packing knob the active mode never consumes must not block resume.

    The fingerprint records mode-specific knobs only for the active
    packing_mode, so e.g. group_docs drift under 'sequential' (or
    buffer_docs drift under 'multipack') is invisible to compat checks.
    """
    cfg = _base_cfg(tmp_path / "run_inert")
    assert cfg.data.packing_mode == "sequential"
    meta = _checkpoint_record(cfg).to_dict()

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
    mp_meta = _checkpoint_record(mp).to_dict()
    mp_drifted = replace(
        mp, data=replace(mp.data, packing_buffer_docs=mp.data.packing_buffer_docs + 1)
    )
    check_resume_compat(mp_drifted, mp_meta)  # bin-only knob is inert here


@pytest.mark.parametrize("section", ["source", "tokenizer", "packing", "eval"])
def test_resume_compat_checks_unknown_fingerprint_keys(tmp_path: Path, section: str) -> None:
    """A fingerprint key recorded on only one side must error, never be skipped.

    Sections are compared over the union of recorded keys, so a field added to
    ``data_fingerprint`` cannot bypass compatibility checking.
    """
    cfg = _base_cfg(tmp_path / "run_unknown_key")
    meta = _checkpoint_record(cfg).to_dict()
    meta["data_fingerprint"][section]["future_knob"] = 3

    with pytest.raises(RuntimeError, match="future_knob"):
        check_resume_compat(cfg, meta)


@pytest.mark.parametrize("tokens_seen", [None, -1, True, 1.5])
def test_resume_compat_requires_valid_token_count(tmp_path: Path, tokens_seen: Any) -> None:
    """Exact resume must reject absent, negative, boolean, or non-integer counts."""
    cfg = _base_cfg(tmp_path / "run_invalid_tokens")
    meta = _checkpoint_record(cfg).to_dict()
    meta["tokens_seen"] = tokens_seen

    with pytest.raises(RuntimeError, match="invalid tokens_seen"):
        check_resume_compat(cfg, meta)


def test_resume_compat_warns_on_gpu_determinism_drift(
    tmp_path: Path, caplog: LogCaptureFixture
) -> None:
    """Kernel-determinism drift across a resume boundary warns, not blocks.

    Deterministic kernels are opt-in (they cost throughput); drift changes
    low-order step numerics only, never the data or the objective. But it is
    the one resume-relevant setting living in XLA_FLAGS instead of config,
    so the fingerprint comparison is the only place a user learns about it.
    """
    cfg = _base_cfg(tmp_path / "run_det_ops")
    meta = _checkpoint_record(cfg).to_dict()
    # Whatever this process's effective setting is (True on GPU hosts via
    # conftest, None on CPU-only), record something else in the checkpoint.
    meta["data_fingerprint"]["xla_gpu_deterministic_ops"] = not bool(
        meta["data_fingerprint"]["xla_gpu_deterministic_ops"]
    )

    with caplog.at_level(logging.WARNING, logger="chomp.ckpt"):
        check_resume_compat(cfg, meta)  # must not raise
    assert any("xla_gpu_deterministic_ops" in rec.message for rec in caplog.records)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda c: replace(c, data=replace(c.data, grain_prefetch=c.data.grain_prefetch + 1)),
            "grain_prefetch",
        ),
        (
            lambda c: replace(c, data=replace(c.data, window_shuffle_tokens=64)),
            "window_shuffle_rows",
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
    meta = _checkpoint_record(cfg).to_dict()
    with pytest.raises(RuntimeError, match=match):
        check_resume_compat(mutate(cfg), meta)


def test_resume_compat_device_put_drift(tmp_path: Path, caplog: LogCaptureFixture) -> None:
    """device_put does not change sample order, so plain drift only warns —
    but with prefetch active it moves device transfers into the prefetch
    thread whose serialized state a restore must line up against, so the
    mismatch hardens to an error."""
    cfg = _base_cfg(tmp_path / "run_dput")
    meta = _checkpoint_record(cfg).to_dict()
    drifted = replace(cfg, data=replace(cfg.data, device_put=not cfg.data.device_put))
    with caplog.at_level(logging.WARNING, logger="chomp.ckpt"):
        check_resume_compat(drifted, meta)  # must not raise
    assert any("device_put" in rec.message for rec in caplog.records)

    pf = replace(cfg, data=replace(cfg.data, grain_prefetch=2))
    pf_meta = _checkpoint_record(pf).to_dict()
    pf_drifted = replace(pf, data=replace(pf.data, device_put=not pf.data.device_put))
    with pytest.raises(RuntimeError, match="device_put"):
        check_resume_compat(pf_drifted, pf_meta)


@pytest.mark.parametrize(
    ("field", "initial", "changed", "match"),
    [
        ("shuffle_buffer_size", 10_000, 200_000, "shuffle_buffer_size"),
        ("shuffle_buffer_bytes", 1024, 2048, "shuffle_buffer_bytes"),
        ("hf_revision", "abc123", "def456", "hf_revision"),
        ("repeat", True, False, "data.repeat"),
    ],
)
def test_resume_compat_rejects_hf_source_drift(
    tmp_path: Path, field: str, initial: Any, changed: Any, match: str
) -> None:
    """Every HF field that changes source order or identity is a hard error."""
    cfg = _base_cfg(tmp_path / f"run_{field}")
    data = replace(
        cfg.data,
        backend="hf",
        hf_dataset="dummy",
        hf_name="dummy",
        hf_split="train",
        shuffle=True,
        **{field: initial},
    )
    cfg = replace(cfg, data=data)
    meta = _checkpoint_record(cfg).to_dict()
    drifted = replace(cfg, data=replace(cfg.data, **{field: changed}))

    with pytest.raises(RuntimeError, match=match):
        check_resume_compat(drifted, meta)


def test_resume_compat_ignores_inert_shuffle_values(tmp_path: Path) -> None:
    """Only effective shuffle behavior belongs in the resume identity."""
    cfg = _base_cfg(tmp_path / "run_inert_shuffle")
    raw_drift = replace(
        cfg,
        data=replace(cfg.data, window_shuffle_tokens=cfg.data.window_shuffle_tokens + 1),
    )
    assert resolve_window_shuffle_rows(raw_drift) == resolve_window_shuffle_rows(cfg)
    check_resume_compat(raw_drift, _checkpoint_record(cfg).to_dict())

    hf = replace(
        cfg,
        data=replace(
            cfg.data,
            backend="hf",
            hf_dataset="dummy",
            hf_name="dummy",
            hf_split="train",
            shuffle=False,
            window_shuffle_tokens=0,
        ),
    )
    inert_hf_drift = replace(
        hf,
        data=replace(
            hf.data,
            shuffle_buffer_size=hf.data.shuffle_buffer_size + 1,
            shuffle_buffer_bytes=hf.data.shuffle_buffer_bytes + 1,
            seed=hf.data.seed + 1,
        ),
    )
    check_resume_compat(inert_hf_drift, _checkpoint_record(hf).to_dict())


def test_resume_compat_rejects_pipeline_schema_drift(tmp_path: Path) -> None:
    """Implementation-version drift must fail even when config fields match."""
    cfg = _base_cfg(tmp_path / "run_schema")
    meta = _checkpoint_record(cfg).to_dict()
    del meta["data_fingerprint"]["data_pipeline_schema_version"]

    with pytest.raises(RuntimeError, match="data_pipeline_schema_version"):
        check_resume_compat(cfg, meta)


def test_resume_compat_rejects_local_window_shuffle_seed_drift(tmp_path: Path) -> None:
    """Local window-shuffle replay must reject a changed data seed."""
    cfg = _base_cfg(tmp_path / "run_window_seed")
    assert cfg.data.backend == "local_text"
    assert cfg.data.window_shuffle_tokens > 0
    meta = _checkpoint_record(cfg).to_dict()

    drifted = replace(cfg, data=replace(cfg.data, seed=cfg.data.seed + 1))
    with pytest.raises(RuntimeError, match="window_shuffle_seed"):
        check_resume_compat(drifted, meta)

    disabled = replace(cfg, data=replace(cfg.data, window_shuffle_tokens=0))
    disabled_meta = _checkpoint_record(disabled).to_dict()
    disabled_drifted = replace(disabled, data=replace(disabled.data, seed=disabled.data.seed + 1))
    check_resume_compat(disabled_drifted, disabled_meta)


def test_supports_packed_segments_requires_capability_flag() -> None:
    """Capability check keys on supports_segment_reset, not compute_loss signature.

    A backend that accepts segment_ids/position_ids but does not advertise the
    flag (legacy megalodon-jax: attention-only isolation, CEMA/TimestepNorm
    state leaking across packed boundaries) must be rejected.
    """
    cfg = Config(model=ModelConfig(backend="dummy", vocab_size=64, d_model=16, dropout=0.0))
    params, static = build_model(cfg, key=jax.random.PRNGKey(0))
    assert supports_packed_segments(params, static)

    class _LegacyLM(eqx.Module):
        """Legacy shape: packed kwargs in the signature, no capability flag."""

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
    assert not supports_packed_segments(legacy_params, legacy_static)


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
    """Strict packing passes segments and lets the backend derive positions."""
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
            loss_chunk_size: int | None = None,
        ) -> jax.Array:
            _ = (input_ids, labels, attention_mask, deterministic, key)
            calls["segment_ids"] = segment_ids
            calls["position_ids"] = position_ids
            calls["loss_chunk_size"] = loss_chunk_size
            return jnp.zeros(())

    params, static = eqx.partition(_SpyLM(w=jnp.zeros(1)), eqx.is_array)
    micro = Batch(
        input_ids=jnp.zeros((1, 8), dtype=jnp.int32),
        labels=jnp.zeros((1, 8), dtype=jnp.int32),
        segment_ids=jnp.ones((1, 8), dtype=jnp.int32),
    )

    training_loss(
        params,
        static,
        batch=micro,
        deterministic=True,
        key=None,
        use_packed_segments=True,
        loss_chunk_size=7,
    )
    assert calls["segment_ids"] is not None
    assert calls["position_ids"] is None
    assert calls["loss_chunk_size"] == 7

    training_loss(
        params, static, batch=micro, deterministic=True, key=None, use_packed_segments=False
    )
    assert calls["segment_ids"] is None
    assert calls["position_ids"] is None
    assert calls["loss_chunk_size"] is None


def test_megalodon_version_floor_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any Megalodon model build must reject releases older than 0.2.1."""
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
            return "0.2.0"
        return real_version(name)

    monkeypatch.setattr("importlib.metadata.version", _stale_version)
    with pytest.raises(RuntimeError, match="requires megalodon-jax >= 0.2.1"):
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
    assert supports_packed_segments(params, static)
