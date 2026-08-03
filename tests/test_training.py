"""Training and checkpointing tests consolidated by module."""

from __future__ import annotations

import json
import logging
import os
import signal
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
    CHECKPOINT_META_SCHEMA_VERSION,
    CheckpointMeta,
    build_meta,
    default_ckpt_dir,
    make_manager,
    megalodon_jax_identity,
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
    build_config,
    resolve_window_shuffle_rows,
    strict_packed_segments,
)
from chomp.data import (
    ZeroLossTokensError,
    build_train_iterator,
    data_fingerprint,
    prepare_tokenizer_and_config,
)
from chomp.model import build_model, loss_sum_and_count, supports_packed_segments
from chomp.train import (
    _METRICS_FILE_DROP,
    _WANDB_DROP,
    TrainingPreempted,
    _project_metrics,
    _restore_eval_status,
    _StopSignalState,
    _sync_metrics_and_validate_loss_tokens,
    build_optimizer,
    init_train_state,
    make_train_step,
    run,
)
from chomp.types import Batch, TrainState
from chomp.utils.tree import abstractify_tree
from tests.helpers.config_factories import (
    make_pipeline_cfg,
    make_small_run_cfg,
    make_tiny_megalodon_model,
)
from tests.helpers.io import read_jsonl

_TEST_TOKENIZER_IDENTITY = {
    "manifest_version": 1,
    "sha256": "test-tokenizer-identity",
}
_TEST_EVAL_STATUS = {
    "eval_disabled": False,
    "eval_failure_count": 0,
    "eval_last_failure_step": None,
    "eval_last_failure_type": None,
    "eval_last_success_step": None,
}


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
    cfg = make_pipeline_cfg(
        seq_len=8,
        vocab_size=256,
        local_text="checkpoint integrity text\n",
        tokenizer=TokenizerConfig(kind="byte", byte_offset=0, add_bos=False, add_eos=False),
    )
    return replace(
        cfg,
        model=replace(cfg.model, d_model=16),
        checkpoint=CheckpointConfig(enabled=True, save_every=1, max_to_keep=2, async_save=False),
        logging=LoggingConfig(project="chomp", run_dir=str(run_dir), metrics_file="metrics.jsonl"),
    )


def check_resume_compat(
    cfg: Config,
    meta: dict[str, Any] | None,
    *,
    tokenizer_identity: dict[str, Any] = _TEST_TOKENIZER_IDENTITY,
) -> None:
    """Call resume validation for a test checkpoint.

    :param Config cfg: Current configuration.
    :param meta: Checkpoint metadata.
    :param dict[str, Any] tokenizer_identity: Effective tokenizer identity.
    """
    _check_resume_compat(cfg, meta, tokenizer_identity=tokenizer_identity)


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
        tokens_seen=tokens_seen,
        eval_status=_TEST_EVAL_STATUS,
        tokenizer_identity=_TEST_TOKENIZER_IDENTITY,
    )


@pytest.mark.parametrize(
    ("direct_url", "expected_direct_url"),
    [
        (None, None),
        (
            {
                "url": "https://github.com/example/megalodon-jax.git",
                "vcs_info": {
                    "vcs": "git",
                    "requested_revision": "correct-loss-sums",
                    "commit_id": "a" * 40,
                },
            },
            {
                "url": "https://github.com/example/megalodon-jax.git",
                "vcs": "git",
                "requested_revision": "correct-loss-sums",
                "commit_id": "a" * 40,
            },
        ),
    ],
    ids=["pypi", "vcs"],
)
def test_megalodon_jax_identity_reads_distribution_and_pep610(
    monkeypatch: pytest.MonkeyPatch,
    direct_url: dict[str, Any] | None,
    expected_direct_url: dict[str, Any] | None,
) -> None:
    """Distribution identity includes exact version and available PEP 610 VCS fields.

    :param pytest.MonkeyPatch monkeypatch: Importlib metadata patch fixture.
    :param dict[str, Any] | None direct_url: Simulated PEP 610 payload.
    :param dict[str, Any] | None expected_direct_url: Expected normalized source fields.
    """

    class _Distribution:
        """Minimal installed-distribution metadata stub."""

        version = "0.2.2"

        def read_text(self, filename: str) -> str | None:
            """Return the simulated direct URL metadata.

            :param str filename: Requested distribution metadata filename.
            :return str | None: Serialized PEP 610 payload when configured.
            """
            assert filename == "direct_url.json"
            return None if direct_url is None else json.dumps(direct_url)

    def _distribution(name: str) -> _Distribution:
        """Return the simulated Megalodon-JAX distribution.

        :param str name: Requested distribution name.
        :return _Distribution: Metadata stub.
        """
        assert name == "megalodon-jax"
        return _Distribution()

    monkeypatch.setattr("chomp.ckpt.metadata.distribution", _distribution)
    identity = megalodon_jax_identity()

    assert identity["distribution"] == "megalodon-jax"
    assert identity["version"] == "0.2.2"
    if expected_direct_url is None:
        assert "direct_url" not in identity
    else:
        assert identity["direct_url"] == expected_direct_url


def test_checkpoint_meta_records_schema_and_backend_identity(tmp_path: Path) -> None:
    """Every new checkpoint record carries its schema and installed backend identity."""
    meta = _checkpoint_record(_base_cfg(tmp_path / "run_identity")).to_dict()

    assert meta["schema_version"] == CHECKPOINT_META_SCHEMA_VERSION
    assert meta["eval_status"] == _TEST_EVAL_STATUS
    assert meta["megalodon_jax"] == megalodon_jax_identity()
    assert meta["tokenizer_identity"] == _TEST_TOKENIZER_IDENTITY


@pytest.mark.parametrize("prior_schema", ["missing", CHECKPOINT_META_SCHEMA_VERSION + 1])
def test_resume_requires_supported_checkpoint_meta_schema(
    tmp_path: Path,
    caplog: LogCaptureFixture,
    prior_schema: str | int,
) -> None:
    """A missing or unknown metadata schema cannot establish strict compatibility.

    :param Path tmp_path: Temporary run-directory root.
    :param LogCaptureFixture caplog: Captured warn-mode compatibility log.
    :param str | int prior_schema: Missing marker or unsupported schema version.
    """
    strict_cfg = replace(
        _base_cfg(tmp_path / f"run_schema_{prior_schema}"),
        checkpoint=replace(CheckpointConfig(), resume_compat="strict"),
    )
    meta = _checkpoint_record(strict_cfg).to_dict()
    if prior_schema == "missing":
        meta.pop("schema_version")
    else:
        meta["schema_version"] = prior_schema

    with pytest.raises(RuntimeError, match=r"checkpoint_meta\.schema_version"):
        check_resume_compat(strict_cfg, meta)

    warn_cfg = replace(
        strict_cfg,
        checkpoint=replace(strict_cfg.checkpoint, resume_compat="warn"),
    )
    with caplog.at_level(logging.WARNING, logger="chomp.ckpt"):
        check_resume_compat(warn_cfg, meta)
    assert "checkpoint_meta.schema_version mismatch" in caplog.text


def test_strict_resume_requires_persisted_eval_status(
    caplog: LogCaptureFixture,
) -> None:
    """Missing evaluation policy state is unproven in strict and reset in warn mode.

    :param LogCaptureFixture caplog: Captured warn-mode compatibility log.
    """
    with pytest.raises(RuntimeError, match="eval_status"):
        _restore_eval_status({"step": 4}, resume_compat="strict")

    with caplog.at_level(logging.WARNING, logger="chomp.train"):
        status = _restore_eval_status({"step": 4}, resume_compat="warn")
    assert status == _TEST_EVAL_STATUS
    assert "eval_status is missing or invalid" in caplog.text


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


def _checkpoint_steps(run_dir: Path) -> set[int]:
    """Return numeric checkpoint steps present in a run directory.

    :param Path run_dir: Training run directory.
    :return set[int]: Completed numeric checkpoint directories, or an empty set.
    """
    ckpt_dir = default_ckpt_dir(run_dir)
    if not ckpt_dir.exists():
        return set()
    return {int(path.name) for path in ckpt_dir.iterdir() if path.is_dir() and path.name.isdigit()}


def _losses_by_step(run_dir: Path) -> dict[int, float]:
    """Read logged training losses keyed by optimizer step.

    :param Path run_dir: Training run directory.
    :return dict[int, float]: Loss values from the run's metrics file.
    """
    rows = read_jsonl(run_dir / "metrics.jsonl")
    return {int(row["step"]): row["loss"] for row in rows if "loss" in row and "step" in row}


def _restore_run_states(
    cfg: Config,
    run_dirs: tuple[Path, Path],
    *,
    step: int,
    save_every: int,
    track_checkpoint_manager: Callable[[Any], Any],
) -> list[TrainState]:
    """Restore matching train states from two completed runs.

    :param Config cfg: Shared model and data configuration.
    :param tuple[Path, Path] run_dirs: Run directories to compare.
    :param int step: Checkpoint step to restore.
    :param int save_every: Manager cadence used by the runs.
    :param track_checkpoint_manager: Fixture callback that closes managers after the test.
    :return list[TrainState]: Restored states in run-directory order.
    """
    resolved, tokenizer = prepare_tokenizer_and_config(cfg)
    params, _static = build_model(resolved, key=jax.random.PRNGKey(0))
    tx, _ = build_optimizer(resolved, params)
    abstract_state = abstractify_tree(
        init_train_state(params=params, tx=tx, key=jax.random.PRNGKey(1))
    )
    states: list[TrainState] = []
    for run_dir in run_dirs:
        manager = track_checkpoint_manager(
            make_manager(
                default_ckpt_dir(run_dir),
                max_to_keep=2,
                save_every=save_every,
                async_save=False,
            )
        )
        _, state, _ = restore_at_step(
            manager,
            step=step,
            abstract_train_state=abstract_state,
            data_iter=build_train_iterator(resolved, tokenizer=tokenizer),
        )
        states.append(state)
    return states


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

    for step in (1, 2, 3, 4):
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

    assert _checkpoint_steps(run_dir) == {3, 4}


def test_run_closes_manager_and_preflights_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run must reject strict drift before constructing data or a restore manager."""
    import orbax.checkpoint as ocp

    close_calls: list[int] = []
    real_close = ocp.CheckpointManager.close

    def _tracked_close(manager: Any) -> None:
        """Record close calls while preserving Orbax cleanup behavior."""
        close_calls.append(id(manager))
        real_close(manager)

    monkeypatch.setattr(ocp.CheckpointManager, "close", _tracked_close)
    cfg = make_small_run_cfg(tmp_path, decay_steps=1)
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, resume_compat="strict"))
    run(cfg, config_path=None, resume="none", dry_run=False)
    assert len(close_calls) == 1

    restore_calls = 0
    data_calls = 0

    def _unexpected_full_restore(*args: Any, **kwargs: Any) -> Any:
        """Fail if incompatible metadata reaches model/data restoration."""
        nonlocal restore_calls
        restore_calls += 1
        raise AssertionError("full restore ran before compatibility validation")

    monkeypatch.setattr("chomp.train.restore_train_state_at_step", _unexpected_full_restore)

    def _unexpected_data_construction(*args: Any, **kwargs: Any) -> Any:
        """Fail if strict incompatibility reaches eval or training data setup."""
        nonlocal data_calls
        data_calls += 1
        raise AssertionError("data construction ran before compatibility validation")

    monkeypatch.setattr("chomp.train.load_or_create_eval_tokens", _unexpected_data_construction)
    monkeypatch.setattr("chomp.train.build_train_iterator", _unexpected_data_construction)
    incompatible = replace(cfg, data=replace(cfg.data, local_text="different corpus"))
    close_calls.clear()

    with pytest.raises(RuntimeError, match="local_text"):
        run(incompatible, config_path=None, resume="latest", dry_run=False)

    assert restore_calls == 0
    assert data_calls == 0
    assert close_calls == []


@pytest.mark.parametrize("resume_compat", ["warn", "strict"])
def test_resume_rejects_unrestorable_data_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resume_compat: str
) -> None:
    """A Grain restore failure must never restart behind restored train state."""
    cfg = make_small_run_cfg(tmp_path, decay_steps=2)
    cfg = replace(
        cfg,
        train=replace(cfg.train, steps=1),
        checkpoint=replace(cfg.checkpoint, resume_compat=resume_compat),
    )
    run_dir = run(cfg, config_path=None, resume="none", dry_run=False)

    resumed = replace(cfg, train=replace(cfg.train, steps=2))

    def _incompatible_data_state(*args: Any, **kwargs: Any) -> None:
        """Inject an unclassified iterator-state restore failure."""
        raise ValueError("incompatible Grain state")

    monkeypatch.setattr("chomp.train.restore_data_state_at_step", _incompatible_data_state)
    with pytest.raises(ValueError, match="incompatible Grain state"):
        run(resumed, config_path=None, resume="latest", dry_run=False)

    assert _checkpoint_steps(run_dir) == {1}


def test_checkpoint_saves_final_step(tmp_path: Path) -> None:
    """Final step should be checkpointed even if save_every does not divide steps."""
    cfg = make_small_run_cfg(tmp_path, decay_steps=2)
    cfg = replace(cfg, train=replace(cfg.train, steps=3))
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, save_every=2))

    run_dir = run(cfg, config_path=None, resume="none", dry_run=False)
    ckpt_dir = default_ckpt_dir(run_dir)

    assert (ckpt_dir / "2").exists(), "expected checkpoint at save interval"
    assert (ckpt_dir / "3").exists(), "expected final checkpoint at step 3"


def test_explicit_resume_rejects_older_retained_step(tmp_path: Path) -> None:
    """In-place rollback must not collide with newer finalized checkpoints."""
    cfg = make_small_run_cfg(tmp_path, decay_steps=4)
    cfg = replace(cfg, train=replace(cfg.train, steps=4))
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, max_to_keep=4))
    run_dir = run(cfg, config_path=None, resume="none", dry_run=False)

    with pytest.raises(RuntimeError, match="newer step 4 already exists"):
        run(cfg, config_path=None, resume=3, dry_run=False)

    assert _checkpoint_steps(run_dir) == {1, 2, 3, 4}


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

    cfg = make_small_run_cfg(tmp_path, decay_steps=5)
    cfg = replace(cfg, train=replace(cfg.train, steps=5))
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, save_every=100))

    stop = _FakeStopSignal()
    dummy_wandb = DummyWandbRun()
    monkeypatch.setattr(train_mod, "_StopSignalState", lambda: stop)
    monkeypatch.setattr(train_mod, "_maybe_init_wandb", lambda *args, **kwargs: dummy_wandb)
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
        run(cfg, config_path=None, resume="none", dry_run=False)
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
    assert _checkpoint_steps(run_dir) == {1}
    assert dummy_wandb.finish_calls == [143]


def test_preemption_before_first_batch_writes_resumable_step_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh pre-first-batch stop must leave a resumable checkpoint."""
    import chomp.train as train_mod

    cfg = make_small_run_cfg(tmp_path)
    stop = _FakeStopSignal()
    stop.requested = True
    monkeypatch.setattr(train_mod, "_StopSignalState", lambda: stop)

    with pytest.raises(TrainingPreempted) as exc_info:
        run(cfg, config_path=None, resume="none", dry_run=False)

    run_dir = exc_info.value.run_dir
    assert _checkpoint_steps(run_dir) == {0}

    stop.requested = False
    assert run(cfg, config_path=None, resume="latest", dry_run=False) == run_dir
    assert _checkpoint_steps(run_dir) == {1, 2}


def test_preemption_during_final_logging_tail_is_not_lost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A signal after the last post-update poll must still report preemption."""
    import chomp.train as train_mod

    cfg = make_small_run_cfg(tmp_path, decay_steps=3)
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
        run(cfg, config_path=None, resume="none", dry_run=False)

    assert exc_info.value.signal_name == "SIGTERM"
    rows = read_jsonl(exc_info.value.run_dir / cfg.logging.metrics_file)
    assert any(row.get("preemption_requested") and row.get("step") == 3 for row in rows)
    assert (default_ckpt_dir(exc_info.value.run_dir) / "3").exists()


def test_preemption_during_finalization_is_not_lost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A signal during teardown must replace the pending success return.

    :param Path tmp_path: Temporary directory for run artifacts.
    :param pytest.MonkeyPatch monkeypatch: Fixture used to inject the stop request.
    """
    import chomp.train as train_mod

    cfg = make_small_run_cfg(tmp_path, decay_steps=1)
    cfg = replace(cfg, train=replace(cfg.train, steps=1))
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, save_every=100))
    stop = _FakeStopSignal()
    monkeypatch.setattr(train_mod, "_StopSignalState", lambda: stop)
    real_finish = train_mod._finish_run_telemetry

    def _finish_and_signal(*args: Any, **kwargs: Any) -> None:
        """Request preemption after telemetry teardown completes.

        :param Any args: Positional telemetry finalization arguments.
        :param Any kwargs: Keyword telemetry finalization arguments.
        """
        real_finish(*args, **kwargs)
        stop.requested = True

    monkeypatch.setattr(train_mod, "_finish_run_telemetry", _finish_and_signal)

    with pytest.raises(TrainingPreempted) as exc_info:
        run(cfg, config_path=None, resume="none", dry_run=False)

    assert exc_info.value.signal_name == "SIGTERM"
    assert exc_info.value.exit_code == 143
    assert (default_ckpt_dir(exc_info.value.run_dir) / "1").exists()


def test_run_enforces_device_before_artifact_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public Python API must enforce the GPU policy before writing a run."""
    cfg = make_small_run_cfg(tmp_path)
    cfg = replace(cfg, train=replace(cfg.train, allow_cpu=False))
    run_dir = Path(cfg.logging.run_dir or "")
    calls: list[bool] = []

    def _reject_device(*, allow_cpu: bool) -> None:
        calls.append(allow_cpu)
        raise RuntimeError("injected non-CUDA backend")

    monkeypatch.setattr("chomp.train.validate_default_device", _reject_device)

    with pytest.raises(RuntimeError, match="injected non-CUDA backend"):
        run(cfg, config_path=None, resume="none", dry_run=False)

    assert calls == [False]
    assert not run_dir.exists()


def test_training_rejects_dense_attention_window_without_blocking_model_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sliding-window config remains usable for inference but not training.

    :param Path tmp_path: Temporary directory for potential run artifacts.
    :param pytest.MonkeyPatch monkeypatch: Device-policy call guard.
    """
    cfg = make_small_run_cfg(tmp_path, run_subdir="dense_window")
    cfg = replace(
        cfg,
        model=make_tiny_megalodon_model(
            vocab_size=256,
            chunk_size=8,
            attention_window=8,
        ),
        train=replace(cfg.train, allow_cpu=False),
    )
    params, _ = build_model(cfg, key=jax.random.PRNGKey(0))
    assert params is not None

    def _unexpected_device_check(*, allow_cpu: bool) -> None:
        """Fail if the training-only semantic guard runs too late."""
        raise AssertionError(f"unexpected device validation: {allow_cpu}")

    monkeypatch.setattr("chomp.train.validate_default_device", _unexpected_device_check)
    run_dir = Path(cfg.logging.run_dir or "")
    with pytest.raises(ValueError) as exc_info:
        run(cfg, config_path=None, resume="none", dry_run=False)

    message = str(exc_info.value)
    assert "dense O(L²)" in message
    assert "model.chunk_size" in message
    assert "future upstream work" in message
    assert not run_dir.exists()


@pytest.mark.parametrize("grain_prefetch", [0, 1])
def test_exact_eof_after_batch_boundary_saves_final_checkpoint(
    tmp_path: Path, grain_prefetch: int
) -> None:
    """Exact EOF after a completed batch should still save the final checkpoint."""
    cfg = make_small_run_cfg(tmp_path, local_text="x" * 48, decay_steps=10)
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

    run_dir = run(cfg, config_path=None, resume="none", dry_run=False)

    metrics_path = run_dir / cfg.logging.metrics_file
    rows = read_jsonl(metrics_path)
    assert any(row.get("data_exhausted") and row.get("step") == 3 for row in rows)

    steps_on_disk = _checkpoint_steps(run_dir)
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
    from chomp.train import make_train_step as real_make_train_step

    def _finish_cfg(cfg: Config) -> Config:
        cfg = replace(cfg, train=replace(cfg.train, steps=5))
        return replace(cfg, checkpoint=replace(cfg.checkpoint, save_every=2))

    # Continuous reference: 5 steps, periodic saves at 2 and 4, final at 5.
    cfg_cont = make_small_run_cfg(tmp_path, run_subdir="run_cont", decay_steps=5)
    cfg_cont = _finish_cfg(cfg_cont)
    run_dir_cont = run(cfg_cont, config_path=None, resume="none", dry_run=False)

    # Crashing run: identical data/seed; blow up on the 4th step call, after
    # its batch has been fetched but before the optimizer update, with state
    # at step 3 and the last periodic save at step 2.
    cfg_crash = make_small_run_cfg(tmp_path, run_subdir="run_crash", decay_steps=5)
    cfg_crash = _finish_cfg(cfg_crash)

    calls = {"n": 0}

    def _make_exploding_train_step(*args: Any, **kwargs: Any) -> Any:
        step = real_make_train_step(*args, **kwargs)

        def _step(state: TrainState, batch: Batch) -> Any:
            calls["n"] += 1
            if calls["n"] == 4:
                raise RuntimeError("injected crash between batch fetch and train step")
            return step(state, batch)

        return _step

    monkeypatch.setattr("chomp.train.make_train_step", _make_exploding_train_step)
    with pytest.raises(RuntimeError, match="injected crash"):
        run(cfg_crash, config_path=None, resume="none", dry_run=False)

    steps_on_disk = _checkpoint_steps(Path(cfg_crash.logging.run_dir))
    assert steps_on_disk == {2}, (
        f"final checkpoint must be skipped in the misaligned window, found {steps_on_disk}"
    )

    # Resume from the periodic checkpoint and finish; batches 3-5 replay.
    run(cfg_crash, config_path=None, resume="latest", dry_run=False)

    # Bit-exact resume contract: both step-5 train states identical.
    states = _restore_run_states(
        cfg_cont,
        (run_dir_cont, Path(cfg_crash.logging.run_dir)),
        step=5,
        save_every=2,
        track_checkpoint_manager=track_checkpoint_manager,
    )

    assert int(jax.device_get(states[0].step)) == 5
    assert eqx.tree_equal(states[0].params, states[1].params)
    assert eqx.tree_equal(states[0].opt_state, states[1].opt_state)


def test_finite_partial_batch_trains_and_saves_aligned_checkpoint(tmp_path: Path) -> None:
    """A usable finite tail must train before an aligned final checkpoint."""
    # One 116-char doc -> 116 byte tokens (offset 0, no BOS/EOS; varied bytes
    # so windows differ): seven full seq_len=16 rows plus one padded four-token
    # row. grad_accum=2 therefore produces four optimizer batches.
    text = "".join(chr(97 + (i * 7) % 26) for i in range(116))
    cfg = make_small_run_cfg(tmp_path, local_text=text, decay_steps=10)
    cfg = replace(cfg, train=replace(cfg.train, steps=10, grad_accum=2))
    cfg = replace(
        cfg,
        data=replace(
            cfg.data,
            repeat=False,
        ),
    )
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, save_every=2))

    run_dir = run(cfg, config_path=None, resume="none", dry_run=False)

    metrics_path = run_dir / cfg.logging.metrics_file
    rows = read_jsonl(metrics_path)
    assert any(row.get("data_exhausted") for row in rows)
    assert len([row for row in rows if row.get("step") == 4 and "loss" in row]) == 1

    assert _checkpoint_steps(run_dir) == {2, 4}

    # Resume sees exact EOF at the saved aligned iterator state; it performs no
    # additional optimizer step and retains the final checkpoint.
    run(cfg, config_path=None, resume="latest", dry_run=False)
    rows = read_jsonl(metrics_path)
    assert len([row for row in rows if row.get("step") == 4 and "loss" in row]) == 1
    assert _checkpoint_steps(run_dir) == {2, 4}


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
        debug=DebugConfig(),
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
    assert _checkpoint_steps(run_dir) == set()
    assert any("Skipping final checkpoint" in record.getMessage() for record in caplog.records)


def test_resume_bit_exact_with_prefetch_and_window_shuffle(
    tmp_path: Path, track_checkpoint_manager: Callable[[Any], Any]
) -> None:
    """Disabling prefetch on resume must preserve bit-exact window replay.

    This pins the sharpest resume question: when the prefetch thread has
    pulled batches ahead of the consumer, the serialized iterator state must
    represent the consumer-visible position, not the advanced parent. Restoring
    that checkpoint without the prefetch wrapper must replay the same shuffled
    windows. Every window is distinct (varied text, period coprime with seq_len),
    so a skipped or reordered batch cannot cancel out.

    :param Path tmp_path: Temporary directory for continuous and resumed runs.
    :param track_checkpoint_manager: Fixture callback that closes restore managers.
    """
    # 101 varied byte tokens; gcd(101, 16) = 1, so packed windows repeat only
    # every 101 windows — far beyond the 12 this test consumes.
    text = "".join(chr(97 + (i * 7) % 26) for i in range(101))

    def _make_cfg(subdir: str, steps: int) -> Config:
        cfg = make_small_run_cfg(tmp_path, run_subdir=subdir, decay_steps=6)
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
        return replace(cfg, checkpoint=replace(cfg.checkpoint, save_every=4))

    cfg_cont = _make_cfg("run_pf_cont", steps=6)
    run_dir_cont = run(cfg_cont, config_path=None, resume="none", dry_run=False)

    cfg_int = _make_cfg("run_pf_int", steps=3)
    run_dir_int = run(cfg_int, config_path=None, resume="none", dry_run=False)
    cfg_resume = _make_cfg("run_pf_int", steps=6)
    cfg_resume = replace(
        cfg_resume,
        data=replace(cfg_resume.data, grain_prefetch=0),
    )
    resumed_run_dir = run(cfg_resume, config_path=None, resume="latest", dry_run=False)
    assert resumed_run_dir == run_dir_int

    # Per-step losses agree exactly across the resume boundary (steps 4-6 ran
    # from the restored mid-window iterator after prefetch was disabled).
    losses_cont = _losses_by_step(run_dir_cont)
    losses_int = _losses_by_step(run_dir_int)
    assert set(losses_cont) == set(losses_int) == {1, 2, 3, 4, 5, 6}
    # Teeth: distinct windows produce distinct step losses, so an off-by-one
    # in the replayed stream could not produce equal sequences by accident.
    assert len(set(losses_cont.values())) > 1
    assert losses_cont == losses_int

    # And the step-6 train states are bit-identical.
    states = _restore_run_states(
        cfg_cont,
        (run_dir_cont, run_dir_int),
        step=6,
        save_every=4,
        track_checkpoint_manager=track_checkpoint_manager,
    )
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

    def _make_cfg(subdir: str, steps: int) -> Config:
        cfg = make_small_run_cfg(tmp_path, run_subdir=subdir, decay_steps=3)
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
        return replace(cfg, checkpoint=replace(cfg.checkpoint, save_every=4))

    cfg_cont = _make_cfg("run_flush_cont", steps=3)
    run_dir_cont = run(cfg_cont, config_path=None, resume="none", dry_run=False)

    cfg_int = _make_cfg("run_flush_int", steps=2)
    run_dir_int = run(cfg_int, config_path=None, resume="none", dry_run=False)
    cfg_resume = _make_cfg("run_flush_int", steps=3)
    run(cfg_resume, config_path=None, resume="latest", dry_run=False)

    losses_cont = _losses_by_step(run_dir_cont)
    losses_int = _losses_by_step(run_dir_int)
    # Step 3 exists only because the flush emitted windows 5 and 6.
    assert set(losses_cont) == set(losses_int) == {1, 2, 3}
    assert len(set(losses_cont.values())) > 1
    assert losses_cont == losses_int

    states = _restore_run_states(
        cfg_cont,
        (run_dir_cont, run_dir_int),
        step=3,
        save_every=4,
        track_checkpoint_manager=track_checkpoint_manager,
    )
    assert eqx.tree_equal(states[0].params, states[1].params)
    assert eqx.tree_equal(states[0].opt_state, states[1].opt_state)


def test_final_checkpoint_failure_fails_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    """A failed final save must not let training exit successfully.

    And when training itself crashed, the checkpoint failure is logged as
    secondary without masking the original exception.
    """
    cfg = make_small_run_cfg(tmp_path, run_subdir="run_ckpt_fail", decay_steps=3)
    cfg = replace(cfg, train=replace(cfg.train, steps=3))
    # save_every > steps: no periodic save, only the finally-block final save.
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, save_every=5))

    def _failing_save(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("disk full (injected)")

    # W&B telemetry must agree with the raised failure: finish() is called
    # with a nonzero exit code, not 0-then-raise.
    dummy_wandb = DummyWandbRun()
    monkeypatch.setattr("chomp.train.save", _failing_save)
    monkeypatch.setattr("chomp.train._maybe_init_wandb", lambda *a, **k: dummy_wandb)
    with pytest.raises(RuntimeError, match="run finalization failed"):
        run(cfg, config_path=None, resume="none", dry_run=False)
    assert dummy_wandb.finish_calls == [1], (
        f"W&B must record the checkpoint finalization failure, got {dummy_wandb.finish_calls}"
    )
    monkeypatch.setattr("chomp.train._maybe_init_wandb", lambda *a, **k: None)

    # Crash path: the original exception wins; the save failure is logged.
    cfg2 = make_small_run_cfg(tmp_path, run_subdir="run_ckpt_fail2", decay_steps=3)
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
        run(cfg2, config_path=None, resume="none", dry_run=False)
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


def _shift_reported_loss_tokens(
    monkeypatch: pytest.MonkeyPatch,
    shifts: dict[int, int],
) -> None:
    """Shift selected device token-count metrics without changing the update.

    :param pytest.MonkeyPatch monkeypatch: Fixture used to patch chomp.train.
    :param dict[int, int] shifts: Per-step integer changes to reported device counts.
    """
    import chomp.train as train_mod

    real_make = train_mod.make_train_step

    def _shifted_make(cfg: Config, **kwargs: Any) -> Any:
        step_fn = real_make(cfg, **kwargs)

        def wrapped(state: TrainState, batch: Batch) -> tuple[TrainState, dict[str, Any]]:
            new_state, metrics = step_fn(state, batch)
            shift = jnp.zeros_like(metrics["token_sum"])
            for step, amount in shifts.items():
                shift = jnp.where(new_state.step == step, amount, shift)
            metrics = dict(metrics)
            metrics["token_sum"] = metrics["token_sum"] + shift
            return new_state, metrics

        return wrapped

    monkeypatch.setattr(train_mod, "make_train_step", _shifted_make)


def test_loss_token_check_covers_each_step_between_syncs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opposite intermediate count errors cannot cancel inside a sync interval.

    :param Path tmp_path: Temporary directory for run artifacts.
    :param pytest.MonkeyPatch monkeypatch: Device-count injection fixture.
    """
    cfg = make_small_run_cfg(tmp_path, run_subdir="run_token_check_interval", decay_steps=4)
    cfg = replace(
        cfg,
        train=replace(cfg.train, steps=4, log_every=4, eval_every=0),
        checkpoint=replace(cfg.checkpoint, enabled=False),
    )
    _shift_reported_loss_tokens(monkeypatch, {2: 1, 3: -1})

    with pytest.raises(RuntimeError, match="loss-token count mismatch at step 2"):
        run(cfg, config_path=None, resume="none", dry_run=False)


def test_loss_token_check_reports_missing_device_metric() -> None:
    """A missing compiled count should not masquerade as a numeric mismatch."""
    with pytest.raises(
        RuntimeError, match="Missing required training metric 'token_sum' at step 2"
    ):
        _sync_metrics_and_validate_loss_tokens({}, [(2, 10, None)])


def test_loss_token_check_drains_partial_interval_before_final_save(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finalization rejects an unchecked intermediate count and its checkpoint.

    :param Path tmp_path: Temporary directory for run artifacts.
    :param pytest.MonkeyPatch monkeypatch: Device-count injection fixture.
    """
    cfg = make_small_run_cfg(tmp_path, run_subdir="run_token_check_final", decay_steps=3)
    cfg = replace(
        cfg,
        train=replace(cfg.train, steps=3, log_every=1000, eval_every=0),
        checkpoint=replace(cfg.checkpoint, save_every=100),
    )
    _shift_reported_loss_tokens(monkeypatch, {2: 1})

    with pytest.raises(RuntimeError, match="loss-token count mismatch at step 2"):
        run(cfg, config_path=None, resume="none", dry_run=False)

    assert _checkpoint_steps(Path(cfg.logging.run_dir or "")) == set()


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
    cfg = make_small_run_cfg(tmp_path, run_subdir="run_nan_save", decay_steps=5)
    # log_every=1000: step 3 is a save step but not a logging step.
    cfg = replace(cfg, train=replace(cfg.train, steps=5, log_every=1000))
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, save_every=3))
    _poison_loss_at_step(monkeypatch, poison_step=3)

    with (
        caplog.at_level(logging.ERROR, logger="chomp.train"),
        pytest.raises(RuntimeError, match="Non-finite loss at step 3"),
    ):
        run(cfg, config_path=None, resume="none", dry_run=False)

    steps_on_disk = _checkpoint_steps(Path(cfg.logging.run_dir))
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
    cfg = make_small_run_cfg(tmp_path, run_subdir="run_nan_final", decay_steps=5)
    # Step 5 (the last step) neither logs nor saves, so the in-loop finite
    # check never sees it; only the finally-block validation can.
    cfg = replace(cfg, train=replace(cfg.train, steps=5, log_every=1000))
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, save_every=2))
    _poison_loss_at_step(monkeypatch, poison_step=5)

    with (
        caplog.at_level(logging.ERROR, logger="chomp.train"),
        pytest.raises(RuntimeError, match="run finalization failed: Non-finite loss at step 5"),
    ):
        run(cfg, config_path=None, resume="none", dry_run=False)

    steps_on_disk = _checkpoint_steps(Path(cfg.logging.run_dir))
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
    cfg = make_small_run_cfg(
        tmp_path,
        run_subdir=f"run_nonfinite_{field}",
        decay_steps=5,
    )
    cfg = replace(cfg, train=replace(cfg.train, steps=5, log_every=1000))
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, save_every=save_every))
    _poison_state_at_step(monkeypatch, poison_step=poison_step, field=field)

    with pytest.raises(RuntimeError, match=match):
        run(cfg, config_path=None, resume="none", dry_run=False)

    steps_on_disk = _checkpoint_steps(Path(cfg.logging.run_dir))
    assert steps_on_disk == expected_steps


def test_checkpoint_disabled_run_rejects_nonfinite_final_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Final metric validity remains a run invariant without checkpointing."""
    cfg = make_small_run_cfg(
        tmp_path,
        run_subdir="run_nonfinite_no_checkpoint",
        decay_steps=2,
    )
    cfg = replace(cfg, train=replace(cfg.train, steps=2, log_every=1000))
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, enabled=False))
    _poison_loss_at_step(monkeypatch, poison_step=2)

    with pytest.raises(RuntimeError, match="Non-finite loss at step 2"):
        run(cfg, config_path=None, resume="none", dry_run=False)


def test_resume_warns_for_seq_len_mismatch(tmp_path: Path, caplog: LogCaptureFixture) -> None:
    """A changed batch shape warns but remains resumable when state shapes fit."""
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
        debug=DebugConfig(nan_check=True),
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
    with caplog.at_level(logging.WARNING, logger="chomp.ckpt"):
        assert run(cfg_b, config_path=None, resume="latest") == run_dir
    assert "train.seq_len" in caplog.text


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
    assert not (run_dir / cfg.logging.metrics_file).exists()
    assert profile_events == ["start", "stop"]

    data = json.loads((run_dir / "config_resolved.json").read_text())
    assert data["derived"]["optim"]["decay_steps_effective"] == cfg.train.steps
    assert data["derived"]["megalodon_jax"] == megalodon_jax_identity()
    assert len(close_calls) == 1


def test_eval_collection_failure_disables_eval_without_stopping_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    """Unavailable diagnostic data must not prevent optimizer steps."""
    cfg = make_small_run_cfg(tmp_path, run_subdir="run_eval_setup_failure", decay_steps=2)
    cfg = replace(
        cfg,
        data=replace(cfg.data, max_eval_samples=2),
        train=replace(
            cfg.train,
            steps=2,
            eval_every=1,
            eval_failure_policy="disable",
        ),
    )

    def _fail_eval_setup(*args: Any, **kwargs: Any) -> list[list[int]]:
        """Inject an unavailable or malformed eval source."""
        raise RuntimeError("broken validation split")

    monkeypatch.setattr("chomp.train.load_or_create_eval_tokens", _fail_eval_setup)
    with caplog.at_level(logging.WARNING, logger="chomp.train"):
        run_dir = run(cfg, config_path=None, resume="none", dry_run=False)

    assert "disabling evaluation" in caplog.text
    assert _checkpoint_steps(run_dir) == {1, 2}
    rows = read_jsonl(run_dir / cfg.logging.metrics_file)
    assert all(row["eval_disabled"] is True for row in rows)
    assert all(row["eval_failure_count"] == 1 for row in rows)
    assert all(row["eval_last_failure_step"] == 0 for row in rows)
    assert all(row["eval_last_failure_type"] == "RuntimeError" for row in rows)
    assert all(row["eval_last_success_step"] is None for row in rows)


def test_resumed_eval_setup_failure_records_and_persists_checkpoint_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setup disable state must use the resume step and prevent later retries.

    :param Path tmp_path: Temporary directory for run artifacts.
    :param pytest.MonkeyPatch monkeypatch: Eval-setup failure injection.
    """
    cfg = make_small_run_cfg(tmp_path, run_subdir="run_eval_setup_resume", decay_steps=3)
    cfg = replace(
        cfg,
        data=replace(cfg.data, max_eval_samples=2),
        train=replace(
            cfg.train,
            steps=1,
            eval_every=2,
            eval_failure_policy="disable",
        ),
    )
    run_dir = run(cfg, config_path=None, resume="none", dry_run=False)

    setup_calls = 0

    def _fail_eval_setup(*args: Any, **kwargs: Any) -> list[list[int]]:
        """Count and fail evaluation initialization."""
        nonlocal setup_calls
        setup_calls += 1
        raise RuntimeError("broken validation split after resume")

    monkeypatch.setattr("chomp.train.load_or_create_eval_tokens", _fail_eval_setup)
    run(cfg, config_path=None, resume="latest", dry_run=False)
    run(cfg, config_path=None, resume="latest", dry_run=False)
    assert setup_calls == 0

    resumed = replace(cfg, train=replace(cfg.train, steps=2))
    run(resumed, config_path=None, resume="latest", dry_run=False)
    continued = replace(cfg, train=replace(cfg.train, steps=3))
    run(continued, config_path=None, resume="latest", dry_run=False)

    assert setup_calls == 1
    rows = read_jsonl(run_dir / cfg.logging.metrics_file)
    resumed_rows = [row for row in rows if row["step"] >= 2]
    assert resumed_rows
    assert all(row["eval_disabled"] is True for row in resumed_rows)
    assert all(row["eval_failure_count"] == 1 for row in resumed_rows)
    assert all(row["eval_last_failure_step"] == 1 for row in resumed_rows)
    assert all(row["eval_last_failure_type"] == "RuntimeError" for row in resumed_rows)
    assert all(row["eval_last_success_step"] is None for row in resumed_rows)


def test_eval_batch_failure_disables_future_evals_without_stopping_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    """A degenerate eval tail batch must not crash an otherwise valid run."""
    import chomp.train as train_mod

    cfg = make_small_run_cfg(tmp_path, run_subdir="run_eval_batch_failure", decay_steps=2)
    cfg = replace(
        cfg,
        data=replace(cfg.data, max_eval_samples=2),
        train=replace(
            cfg.train,
            steps=2,
            eval_every=1,
            eval_failure_policy="disable",
        ),
        logging=replace(
            cfg.logging,
            wandb=replace(cfg.logging.wandb, enabled=True),
        ),
    )
    dummy_wandb = DummyWandbRun()
    monkeypatch.setattr("chomp.train._maybe_init_wandb", lambda *args, **kwargs: dummy_wandb)
    real_build_eval_iterator = train_mod.build_eval_iterator
    build_calls = 0

    def _build_broken_eval_iterator(config: Config, *, tokens: list[list[int]]) -> Iterator[Batch]:
        """Yield one usable batch, then reproduce a zero-token tail failure."""
        nonlocal build_calls
        build_calls += 1
        iterator = iter(real_build_eval_iterator(config, tokens=tokens))
        yield next(iterator)
        raise ZeroLossTokensError("tail batch contains zero valid loss tokens")

    monkeypatch.setattr(train_mod, "build_eval_iterator", _build_broken_eval_iterator)
    with caplog.at_level(logging.WARNING, logger="chomp.train"):
        run_dir = run(cfg, config_path=None, resume="none", dry_run=False)

    assert build_calls == 1
    assert "Evaluation failed at step 1; disabling evaluation" in caplog.text
    assert _checkpoint_steps(run_dir) == {1, 2}
    rows = read_jsonl(run_dir / cfg.logging.metrics_file)
    assert all(row["eval_disabled"] is True for row in rows)
    assert all(row["eval_failure_count"] == 1 for row in rows)
    assert all(row["eval_last_failure_step"] == 1 for row in rows)
    assert all(row["eval_last_failure_type"] == "ZeroLossTokensError" for row in rows)
    assert all(row["eval_last_success_step"] is None for row in rows)
    telemetry_rows = [row for _, row in dummy_wandb.logged if "eval_disabled" in row]
    assert telemetry_rows
    assert all(row["eval_disabled"] is True for row in telemetry_rows)
    assert all(row["eval_failure_count"] == 1 for row in telemetry_rows)

    resumed = replace(cfg, train=replace(cfg.train, steps=3))
    run(resumed, config_path=None, resume="latest", dry_run=False)
    assert build_calls == 1
    resumed_rows = read_jsonl(run_dir / cfg.logging.metrics_file)
    assert resumed_rows[-1]["eval_disabled"] is True
    assert resumed_rows[-1]["eval_failure_count"] == 1
    assert resumed_rows[-1]["eval_last_failure_step"] == 1


def test_eval_collection_failure_is_fatal_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default policy must fail before training when eval collection fails.

    :param Path tmp_path: Temporary directory for run artifacts.
    :param pytest.MonkeyPatch monkeypatch: Eval-setup failure injection.
    """
    cfg = make_small_run_cfg(tmp_path, run_subdir="run_eval_setup_fatal")
    cfg = replace(
        cfg,
        data=replace(cfg.data, max_eval_samples=2),
        train=replace(cfg.train, eval_every=1),
    )

    def _fail_eval_setup(*args: Any, **kwargs: Any) -> list[list[int]]:
        """Raise a deterministic evaluation initialization failure."""
        raise RuntimeError("broken validation split")

    monkeypatch.setattr("chomp.train.load_or_create_eval_tokens", _fail_eval_setup)

    with pytest.raises(RuntimeError, match="broken validation split"):
        run(cfg, config_path=None, resume="none", dry_run=False)


def test_eval_runtime_failure_is_fatal_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default policy must propagate scheduled evaluation failures.

    :param Path tmp_path: Temporary directory for run artifacts.
    :param pytest.MonkeyPatch monkeypatch: Eval-runtime failure injection.
    """
    cfg = make_small_run_cfg(tmp_path, run_subdir="run_eval_runtime_fatal")
    cfg = replace(
        cfg,
        data=replace(cfg.data, max_eval_samples=2),
        train=replace(cfg.train, steps=2, eval_every=1),
    )

    def _fail_eval_iterator(config: Config, *, tokens: list[list[int]]) -> Iterator[Batch]:
        """Raise while assembling the scheduled evaluation pass."""
        _ = (config, tokens)
        raise RuntimeError("broken eval batch")

    monkeypatch.setattr("chomp.train.build_eval_iterator", _fail_eval_iterator)

    run_dir = Path(cfg.logging.run_dir or "")
    with pytest.raises(RuntimeError, match="broken eval batch"):
        run(cfg, config_path=None, resume="none", dry_run=False)

    rows = read_jsonl(run_dir / cfg.logging.metrics_file)
    crash = rows[-1]
    assert crash["crash"] is True
    assert crash["eval_disabled"] is False
    assert crash["eval_failure_count"] == 1
    assert crash["eval_last_failure_step"] == 1
    assert crash["eval_last_failure_type"] == "RuntimeError"
    assert crash["eval_last_success_step"] is None


def test_fatal_eval_failure_must_succeed_before_resume_can_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A checkpointed fatal eval failure remains owed on same-target resume.

    :param Path tmp_path: Temporary directory for run artifacts.
    :param pytest.MonkeyPatch monkeypatch: Eval failure injection fixture.
    """
    import chomp.train as train_mod

    cfg = make_small_run_cfg(tmp_path, run_subdir="run_eval_fatal_resume")
    cfg = replace(
        cfg,
        data=replace(cfg.data, max_eval_samples=2),
        train=replace(cfg.train, steps=1, eval_every=1),
    )
    real_build_eval_iterator = train_mod.build_eval_iterator
    real_make_train_step = train_mod.make_train_step
    eval_attempts = 0
    events: list[str] = []

    def _fail_first_eval(config: Config, *, tokens: list[list[int]]) -> Iterator[Batch]:
        """Fail the original attempt and serve the identical resumed evaluation."""
        nonlocal eval_attempts
        eval_attempts += 1
        events.append("eval")
        if eval_attempts == 1:
            raise RuntimeError("injected fatal eval failure")
        yield from real_build_eval_iterator(config, tokens=tokens)

    def _tracking_make_train_step(cfg: Config, **kwargs: Any) -> Any:
        """Record optimizer calls so resume ordering is observable."""
        step_fn = real_make_train_step(cfg, **kwargs)

        def wrapped(state: TrainState, batch: Batch) -> tuple[TrainState, dict[str, Any]]:
            events.append("train")
            return step_fn(state, batch)

        return wrapped

    monkeypatch.setattr(train_mod, "build_eval_iterator", _fail_first_eval)
    monkeypatch.setattr(train_mod, "make_train_step", _tracking_make_train_step)

    with pytest.raises(RuntimeError, match="injected fatal eval failure"):
        run(cfg, config_path=None, resume="none", dry_run=False)

    run_dir = Path(cfg.logging.run_dir or "")
    assert _checkpoint_steps(run_dir) == {1}

    # The selected checkpoint is already at the target. Returning without
    # retrying here would silently turn the failed logical run into success.
    events.clear()
    run(cfg, config_path=None, resume="latest", dry_run=False)

    assert eval_attempts == 2
    assert events == ["eval"]
    recovered = read_jsonl(run_dir / cfg.logging.metrics_file)[-1]
    assert recovered["step"] == 1
    assert recovered["eval_loss"] > 0.0
    assert recovered["eval_failure_count"] == 1
    assert recovered["eval_last_failure_step"] == 1
    assert recovered["eval_last_success_step"] == 1

    # The checkpoint still conservatively records the owed attempt, so an
    # extended resume repeats eval before consuming its next training batch.
    events.clear()
    extended = replace(cfg, train=replace(cfg.train, steps=2))
    run(extended, config_path=None, resume="latest", dry_run=False)
    assert eval_attempts == 3
    assert events == ["eval", "train"]


@pytest.mark.parametrize(
    ("field", "match"),
    [
        ("params", "Non-finite parameters at step 1"),
        ("opt_state", "Non-finite optimizer state at step 1"),
    ],
)
def test_dry_run_rejects_nonfinite_post_update_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    match: str,
) -> None:
    """Dry-run success requires finite parameters and optimizer state.

    :param Path tmp_path: Temporary directory for run artifacts.
    :param pytest.MonkeyPatch monkeypatch: Fixture used to poison the train step.
    :param str field: TrainState field to poison after the update.
    :param str match: Expected validation failure text.
    """
    cfg = make_small_run_cfg(tmp_path, run_subdir=f"dry_run_nonfinite_{field}")
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, enabled=False))
    _poison_state_at_step(monkeypatch, poison_step=1, field=field)

    with pytest.raises(RuntimeError, match=match):
        run(cfg, config_path=None, resume="none", dry_run=True)


def test_deterministic_checkpointing_warns(tmp_path: Path, caplog: LogCaptureFixture) -> None:
    """Deterministic mode should warn when use_checkpoint is enabled.

    :param Path tmp_path: Temporary directory for the run artifacts.
    :param LogCaptureFixture caplog: Log capture fixture.
    """
    run_dir = tmp_path / "dry_run_warn"
    cfg = Config(
        model=make_tiny_megalodon_model(use_checkpoint=True),
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
        debug=DebugConfig(nan_check=False),
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


def test_tokens_seen_matches_host_counts_between_sync_points(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Host counts stay exact while intermediate optimizer steps remain asynchronous.

    :param Path tmp_path: Temporary directory for run artifacts.
    :param pytest.MonkeyPatch monkeypatch: Sync-boundary recorder fixture.
    """
    import chomp.train as train_mod

    real_sync = train_mod._sync_metrics_and_validate_loss_tokens
    queued_step_groups: list[list[int]] = []

    def _record_sync(metrics: dict[str, Any], checks: list[tuple[int, int, Any]]) -> dict[str, Any]:
        """Record queued step groups before delegating to the real synchronization."""
        if checks:
            queued_step_groups.append([step for step, _, _ in checks])
        return real_sync(metrics, checks)

    monkeypatch.setattr(train_mod, "_sync_metrics_and_validate_loss_tokens", _record_sync)
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
        debug=DebugConfig(nan_check=True),
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
    assert [int(row["step"]) for row in train_rows] == [1, 4]
    assert float(train_rows[0]["first_step_compile_time_s"]) >= 0.0
    assert "first_step_compile_time_s" not in train_rows[1]
    assert all(row["packing_utilization"] > 0 for row in train_rows)
    assert [int(row["loss_tokens"]) for row in train_rows] == [
        expected_counts[0],
        expected_counts[-1],
    ]
    assert [int(row["tokens_seen"]) for row in train_rows] == [
        expected_counts[0],
        sum(expected_counts),
    ]
    assert queued_step_groups == [[1], [2, 3, 4]]


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
        debug=DebugConfig(nan_check=True),
        logging=LoggingConfig(project="chomp", run_dir=str(tmp_path / "run_guard")),
    )

    monkeypatch.setattr("chomp.train.supports_packed_segments", lambda params, static: False)
    with pytest.raises(RuntimeError, match="Strict segment isolation"):
        run(cfg, config_path=None, resume="none")


def _megalodon_resume_cfg(
    run_dir: Path,
    *,
    packing_mode: str = "sequential",
    packing_strict_segments: bool = True,
    resume_compat: str = "strict",
) -> Config:
    """Build a tiny Megalodon config for resume-identity tests.

    :param Path run_dir: Run directory recorded in the config.
    :param str packing_mode: Active data packing mode.
    :param bool packing_strict_segments: Whether packed segment resets execute.
    :param str resume_compat: Resume comparison policy.
    :return Config: Tiny Megalodon resume configuration.
    """
    cfg = _base_cfg(run_dir)
    return replace(
        cfg,
        model=make_tiny_megalodon_model(vocab_size=256, chunk_size=8),
        data=replace(
            cfg.data,
            packing_mode=packing_mode,
            packing_strict_segments=packing_strict_segments,
        ),
        checkpoint=replace(cfg.checkpoint, resume_compat=resume_compat),
    )


@pytest.mark.parametrize("prior_identity", ["missing", "flat", "changed"])
def test_megalodon_strict_resume_requires_structured_backend_identity(
    tmp_path: Path,
    prior_identity: str,
    caplog: LogCaptureFixture,
) -> None:
    """Missing, legacy-flat, or changed backend identity is never proven equal.

    :param Path tmp_path: Temporary run-directory root.
    :param str prior_identity: Checkpoint identity shape to test.
    :param LogCaptureFixture caplog: Captured warn-mode compatibility log.
    """
    strict_cfg = _megalodon_resume_cfg(tmp_path / f"run_backend_{prior_identity}")
    meta = _checkpoint_record(strict_cfg).to_dict()
    if prior_identity == "missing":
        meta.pop("megalodon_jax")
    elif prior_identity == "flat":
        meta["megalodon_jax"] = "0.2.2"
    else:
        meta["megalodon_jax"] = {
            **meta["megalodon_jax"],
            "version": "0.2.1",
        }

    with pytest.raises(RuntimeError, match="megalodon_jax"):
        check_resume_compat(strict_cfg, meta)

    warn_cfg = replace(
        strict_cfg,
        checkpoint=replace(strict_cfg.checkpoint, resume_compat="warn"),
    )
    with caplog.at_level(logging.WARNING, logger="chomp.ckpt"):
        check_resume_compat(warn_cfg, meta)
    assert "megalodon_jax mismatch" in caplog.text


def test_dummy_resume_does_not_require_megalodon_identity(tmp_path: Path) -> None:
    """Backend identity is enforced only when Megalodon will execute."""
    cfg = _base_cfg(tmp_path / "run_dummy_backend_identity")
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, resume_compat="strict"))
    meta = _checkpoint_record(cfg).to_dict()
    meta.pop("megalodon_jax")

    check_resume_compat(cfg, meta)


@pytest.mark.parametrize("prior_identity", ["missing", "changed"])
def test_resume_requires_checkpoint_bound_tokenizer_identity(
    tmp_path: Path,
    prior_identity: str,
    caplog: LogCaptureFixture,
) -> None:
    """Missing or changed tokenizer identity is not proven equal.

    :param Path tmp_path: Temporary run-directory root.
    :param str prior_identity: Checkpoint identity shape to test.
    :param LogCaptureFixture caplog: Captured warn-mode compatibility log.
    """
    strict_cfg = replace(
        _base_cfg(tmp_path / f"run_tokenizer_{prior_identity}"),
        checkpoint=replace(CheckpointConfig(), resume_compat="strict"),
    )
    meta = _checkpoint_record(strict_cfg).to_dict()
    if prior_identity == "missing":
        meta.pop("tokenizer_identity")
    else:
        meta["tokenizer_identity"] = {
            **meta["tokenizer_identity"],
            "sha256": "different",
        }

    with pytest.raises(RuntimeError, match="tokenizer_identity"):
        check_resume_compat(strict_cfg, meta)

    warn_cfg = replace(
        strict_cfg,
        checkpoint=replace(strict_cfg.checkpoint, resume_compat="warn"),
    )
    with caplog.at_level(logging.WARNING, logger="chomp.ckpt"):
        check_resume_compat(warn_cfg, meta)
    assert "tokenizer_identity mismatch" in caplog.text


def test_segment_scan_resume_semantics_are_contextual(
    tmp_path: Path,
    caplog: LogCaptureFixture,
) -> None:
    """Segmented CEMA implementation matters exactly when reset execution is active.

    :param Path tmp_path: Temporary run-directory root.
    :param LogCaptureFixture caplog: Captured warn-mode compatibility log.
    """
    for mode in ("bin", "multipack"):
        strict_cfg = _megalodon_resume_cfg(
            tmp_path / f"run_scan_{mode}",
            packing_mode=mode,
        )
        meta = _checkpoint_record(strict_cfg).to_dict()
        drifted = replace(
            strict_cfg,
            model=replace(
                strict_cfg.model,
                use_associative_segment_scan=not strict_cfg.model.use_associative_segment_scan,
            ),
        )
        with pytest.raises(RuntimeError, match="use_associative_segment_scan"):
            check_resume_compat(drifted, meta)

        missing = _checkpoint_record(strict_cfg).to_dict()
        del missing["config"]["model"]["use_associative_segment_scan"]
        with pytest.raises(RuntimeError, match="use_associative_segment_scan"):
            check_resume_compat(strict_cfg, missing)

        warn_cfg = replace(
            drifted,
            checkpoint=replace(drifted.checkpoint, resume_compat="warn"),
        )
        with caplog.at_level(logging.WARNING, logger="chomp.ckpt"):
            check_resume_compat(warn_cfg, meta)
        assert "use_associative_segment_scan" in caplog.text
        caplog.clear()

    inert_configs = (
        _megalodon_resume_cfg(tmp_path / "run_scan_sequential"),
        _megalodon_resume_cfg(
            tmp_path / "run_scan_nonstrict",
            packing_mode="bin",
            packing_strict_segments=False,
        ),
    )
    for cfg in inert_configs:
        meta = _checkpoint_record(cfg).to_dict()
        drifted = replace(
            cfg,
            model=replace(
                cfg.model,
                use_associative_segment_scan=not cfg.model.use_associative_segment_scan,
            ),
        )
        check_resume_compat(drifted, meta)


@pytest.mark.parametrize(
    ("field", "changed_value", "deterministic"),
    [("loss_chunk_size", 4, True), ("use_checkpoint", True, False)],
)
def test_megalodon_runtime_resume_semantics_are_active(
    tmp_path: Path,
    caplog: LogCaptureFixture,
    field: str,
    changed_value: Any,
    deterministic: bool,
) -> None:
    """Active Megalodon runtime choices follow strict/warn resume policy.

    :param Path tmp_path: Temporary run-directory root.
    :param LogCaptureFixture caplog: Captured warn-mode compatibility log.
    :param str field: Megalodon runtime field to change.
    :param Any changed_value: Value that differs from the checkpoint config.
    :param bool deterministic: Effective deterministic execution setting.
    """
    strict_cfg = _megalodon_resume_cfg(tmp_path / f"run_runtime_{field}")
    strict_cfg = replace(
        strict_cfg,
        train=replace(strict_cfg.train, deterministic=deterministic),
    )
    meta = _checkpoint_record(strict_cfg).to_dict()
    drifted = replace(
        strict_cfg,
        model=replace(strict_cfg.model, **{field: changed_value}),
    )

    with pytest.raises(RuntimeError, match=field):
        check_resume_compat(drifted, meta)

    missing = _checkpoint_record(strict_cfg).to_dict()
    del missing["config"]["model"][field]
    with pytest.raises(RuntimeError, match=field):
        check_resume_compat(strict_cfg, missing)

    warn_cfg = replace(
        drifted,
        checkpoint=replace(drifted.checkpoint, resume_compat="warn"),
    )
    with caplog.at_level(logging.WARNING, logger="chomp.ckpt"):
        check_resume_compat(warn_cfg, meta)
    assert field in caplog.text


def test_use_checkpoint_resume_semantics_are_inert_when_disabled(
    tmp_path: Path,
) -> None:
    """Rematerialization choice is inert when deterministic execution disables it.

    :param Path tmp_path: Temporary run-directory root.
    """
    cfg = _megalodon_resume_cfg(tmp_path / "run_checkpoint_inert")
    cfg = replace(cfg, train=replace(cfg.train, deterministic=True))
    meta = _checkpoint_record(cfg).to_dict()
    drifted = replace(
        cfg,
        model=replace(cfg.model, use_checkpoint=not cfg.model.use_checkpoint),
    )
    missing = _checkpoint_record(cfg).to_dict()
    del missing["config"]["model"]["use_checkpoint"]

    check_resume_compat(drifted, meta)
    check_resume_compat(cfg, missing)


def test_eval_failure_policy_follows_resume_compatibility(
    tmp_path: Path,
    caplog: LogCaptureFixture,
) -> None:
    """Evaluation failure policy changes reject in strict and warn otherwise.

    :param Path tmp_path: Temporary run-directory root.
    :param LogCaptureFixture caplog: Captured warn-mode compatibility log.
    """
    strict_cfg = replace(
        _base_cfg(tmp_path / "run_eval_failure_policy"),
        checkpoint=replace(CheckpointConfig(), resume_compat="strict"),
    )
    meta = _checkpoint_record(strict_cfg).to_dict()
    drifted = replace(
        strict_cfg,
        train=replace(strict_cfg.train, eval_failure_policy="disable"),
    )

    with pytest.raises(RuntimeError, match="train.eval_failure_policy"):
        check_resume_compat(drifted, meta)

    warn_cfg = replace(
        drifted,
        checkpoint=replace(drifted.checkpoint, resume_compat="warn"),
    )
    with caplog.at_level(logging.WARNING, logger="chomp.ckpt"):
        check_resume_compat(warn_cfg, meta)
    assert "train.eval_failure_policy mismatch" in caplog.text


def test_resume_compat_warns_for_multipack_knob_changes(
    tmp_path: Path, caplog: LogCaptureFixture
) -> None:
    """Default resume warns for changed packing order and objective knobs.

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
    with caplog.at_level(logging.WARNING, logger="chomp.ckpt"):
        check_resume_compat(changed_group, meta)
    assert "packing_group_docs" in caplog.text

    caplog.clear()
    changed_strict = replace(cfg, data=replace(cfg.data, packing_strict_segments=False))
    with caplog.at_level(logging.WARNING, logger="chomp.ckpt"):
        check_resume_compat(changed_strict, meta)
    assert "packing_strict_segments" in caplog.text

    binc = replace(cfg, data=replace(cfg.data, packing_mode="bin", packing_buffer_docs=8))
    bin_meta = _checkpoint_record(binc).to_dict()
    bin_changed = replace(binc, data=replace(binc.data, packing_strict_segments=False))
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="chomp.ckpt"):
        check_resume_compat(bin_changed, bin_meta)
    assert "packing_strict_segments" in caplog.text


@pytest.mark.parametrize(
    ("optimizer_name", "section", "active"),
    [
        ("adamw", "adam", True),
        ("adamw", "muon", False),
        ("muon", "adam", True),
        ("muon", "muon", True),
    ],
)
def test_resume_compat_tracks_consumed_optimizer_config(
    tmp_path: Path,
    optimizer_name: str,
    section: str,
    active: bool,
    caplog: LogCaptureFixture,
) -> None:
    """Optimizer value drift warns only when the active transform consumes it.

    Muon remains a hybrid: non-Muon leaves use AdamW, so ``optim.adam`` is
    active in both modes. Only ``optim.muon`` under plain AdamW is inert.

    :param Path tmp_path: Temporary run directory root.
    :param str optimizer_name: Active optimizer mode.
    :param str section: Nested optimizer section to change.
    :param bool active: Whether the changed section is consumed in this mode.
    :param LogCaptureFixture caplog: Captured resume warnings.
    """
    cfg = _base_cfg(tmp_path / f"run_{optimizer_name}_{section}")
    cfg = replace(cfg, optim=replace(cfg.optim, name=optimizer_name))
    meta = _checkpoint_record(cfg).to_dict()

    if section == "adam":
        drifted = replace(
            cfg,
            optim=replace(cfg.optim, adam=replace(cfg.optim.adam, b1=0.8)),
        )
    else:
        drifted = replace(
            cfg,
            optim=replace(cfg.optim, muon=replace(cfg.optim.muon, momentum=0.9)),
        )

    with caplog.at_level(logging.WARNING, logger="chomp.ckpt"):
        check_resume_compat(drifted, meta)
    assert (f"optim.{section}" in caplog.text) is active


def test_resume_compat_ignores_inert_dummy_model_config(tmp_path: Path) -> None:
    """Megalodon-only model fields must not block a DummyLM smoke resume.

    :param Path tmp_path: Temporary run directory root.
    """
    cfg = _base_cfg(tmp_path / "run_inert_dummy_model")
    meta = _checkpoint_record(cfg).to_dict()

    inert_drift = replace(cfg, model=replace(cfg.model, model_dim=cfg.model.model_dim + 1))
    check_resume_compat(inert_drift, meta)

    active_drift = replace(cfg, model=replace(cfg.model, d_model=cfg.model.d_model + 1))
    with pytest.raises(RuntimeError, match="model.d_model"):
        check_resume_compat(active_drift, meta)


def test_resume_compat_rejects_active_dummy_muon_routing_change(tmp_path: Path) -> None:
    """Dummy embedding routing changes must fail before optimizer-state restore."""
    cfg = _base_cfg(tmp_path / "run_dummy_muon_routing")
    cfg = replace(
        cfg,
        model=replace(cfg.model, share_emb=False),
        optim=replace(
            cfg.optim,
            name="muon",
            muon=replace(
                cfg.optim.muon,
                allow_all_2d=False,
                allow_tied_embed=True,
            ),
        ),
    )
    meta = _checkpoint_record(cfg).to_dict()
    drifted = replace(cfg, model=replace(cfg.model, share_emb=True))

    with pytest.raises(RuntimeError, match="model.share_emb"):
        check_resume_compat(drifted, meta)


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


@pytest.mark.parametrize("tokens_seen", [None, -1, True, 1.5])
def test_resume_compat_requires_valid_token_count(tmp_path: Path, tokens_seen: Any) -> None:
    """Exact resume must reject absent, negative, boolean, or non-integer counts."""
    cfg = _base_cfg(tmp_path / "run_invalid_tokens")
    meta = _checkpoint_record(cfg).to_dict()
    meta["tokens_seen"] = tokens_seen

    with pytest.raises(RuntimeError, match="invalid tokens_seen"):
        check_resume_compat(cfg, meta)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
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
    ids=["window_shuffle", "mask_boundary", "train_on_eos", "deterministic"],
)
def test_resume_compat_warns_for_stream_and_objective_drift(
    tmp_path: Path, mutate: Any, match: str, caplog: LogCaptureFixture
) -> None:
    """Data-order and objective changes warn without blocking default resume."""
    cfg = _base_cfg(tmp_path / "run_drift")
    meta = _checkpoint_record(cfg).to_dict()
    with caplog.at_level(logging.WARNING, logger="chomp.ckpt"):
        check_resume_compat(mutate(cfg), meta)
    assert match in caplog.text


def test_resume_compat_strict_rejects_semantic_drift(tmp_path: Path) -> None:
    """Strict mode remains available for exact data/objective continuation."""
    cfg = _base_cfg(tmp_path / "run_strict_drift")
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, resume_compat="strict"))
    meta = _checkpoint_record(cfg).to_dict()
    drifted = replace(cfg, data=replace(cfg.data, train_on_eos=not cfg.data.train_on_eos))

    with pytest.raises(RuntimeError, match="train_on_eos"):
        check_resume_compat(drifted, meta)


def test_resume_compat_compares_effective_determinism(tmp_path: Path) -> None:
    """Equivalent null/true dropout behavior must not block strict resume."""
    cfg = _base_cfg(tmp_path / "run_effective_determinism")
    cfg = replace(
        cfg,
        train=replace(cfg.train, deterministic=None),
        checkpoint=replace(cfg.checkpoint, resume_compat="strict"),
    )
    meta = _checkpoint_record(cfg).to_dict()

    explicit_true = replace(cfg, train=replace(cfg.train, deterministic=True))
    check_resume_compat(explicit_true, meta)

    effective_change = replace(cfg, train=replace(cfg.train, deterministic=False))
    with pytest.raises(RuntimeError, match="train.deterministic_effective"):
        check_resume_compat(effective_change, meta)


def test_resume_compat_allows_schedule_and_optimizer_value_changes(
    tmp_path: Path, caplog: LogCaptureFixture
) -> None:
    """Extending a run and lowering its LR must warn rather than refuse resume."""
    cfg = _base_cfg(tmp_path / "run_extend")
    cfg = replace(
        cfg, train=replace(cfg.train, steps=10), optim=replace(cfg.optim, decay_steps=None)
    )
    meta = _checkpoint_record(cfg).to_dict()
    drifted = replace(
        cfg,
        train=replace(cfg.train, steps=20),
        optim=replace(
            cfg.optim,
            lr=cfg.optim.lr / 2,
            weight_decay=cfg.optim.weight_decay / 2,
            grad_clip_norm=cfg.optim.grad_clip_norm / 2,
            warmup_steps=cfg.optim.warmup_steps + 1,
            min_lr_ratio=0.1,
            adam=replace(cfg.optim.adam, b1=0.8),
        ),
    )

    with caplog.at_level(logging.WARNING, logger="chomp.ckpt"):
        check_resume_compat(drifted, meta)

    for field in (
        "train.steps",
        "optim.decay_steps_effective",
        "optim.lr",
        "optim.weight_decay",
        "optim.grad_clip_norm",
        "optim.warmup_steps",
        "optim.min_lr_ratio",
        "optim.adam.b1",
    ):
        assert field in caplog.text


def test_resume_compat_rejects_optimizer_structure_change(tmp_path: Path) -> None:
    """Changing optimizer families still fails before incompatible state restore."""
    cfg = _base_cfg(tmp_path / "run_optim_structure")
    meta = _checkpoint_record(cfg).to_dict()
    drifted = replace(cfg, optim=replace(cfg.optim, name="muon"))

    with pytest.raises(RuntimeError, match="optim.name"):
        check_resume_compat(drifted, meta)


def test_resume_compat_ignores_inert_fields(tmp_path: Path, caplog: LogCaptureFixture) -> None:
    """Restore-inert knobs must not produce compatibility warnings."""
    cfg = _base_cfg(tmp_path / "run_old_meta")
    meta = _checkpoint_record(cfg).to_dict()
    drifted = replace(
        cfg,
        model=replace(cfg.model, init_mode="xavier", use_checkpoint=True),
        data=replace(
            cfg.data,
            tokenizer=replace(
                cfg.data.tokenizer,
                hf_use_fast=not cfg.data.tokenizer.hf_use_fast,
                hf_trust_remote_code=not cfg.data.tokenizer.hf_trust_remote_code,
                vocab_size_multiple=cfg.data.tokenizer.vocab_size_multiple * 2,
            ),
        ),
    )

    with caplog.at_level(logging.WARNING, logger="chomp.ckpt"):
        check_resume_compat(drifted, meta)
    assert "Resume config warnings" not in caplog.text


@pytest.mark.parametrize(
    ("path", "match"),
    [
        (("data_fingerprint", "source", "local_text"), "data.local_text"),
        (("data_fingerprint", "tokenizer", "add_eos"), "tokenizer.add_eos"),
        (
            ("data_fingerprint", "packing", "window_shuffle_rows"),
            "data.window_shuffle_rows",
        ),
        (("data_fingerprint", "eval", "max_eval_samples"), "data.max_eval_samples"),
        (("config", "model", "dropout"), "model.dropout"),
        (("config", "optim", "lr"), "optim.lr"),
        (("config", "optim", "adam", "b1"), "optim.adam.b1"),
        (("config", "train", "deterministic"), "train.deterministic_effective"),
        (("config", "train", "eval_failure_policy"), "train.eval_failure_policy"),
        (("config", "train", "steps"), "train.steps"),
        (("config", "optim", "decay_steps"), "optim.decay_steps"),
    ],
    ids=[
        "source",
        "tokenizer",
        "packing",
        "eval",
        "model",
        "optimizer",
        "adam",
        "determinism",
        "eval_failure_policy",
        "steps",
        "schedule",
    ],
)
def test_strict_resume_rejects_missing_active_fields(
    tmp_path: Path, path: tuple[str, ...], match: str
) -> None:
    """Strict compatibility must treat a missing active value as unknown.

    :param Path tmp_path: Temporary run directory root.
    :param tuple[str, ...] path: Metadata path to delete.
    :param str match: Expected compatibility error path.
    """
    cfg = _base_cfg(tmp_path / "run_missing_active")
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, resume_compat="strict"))
    meta = _checkpoint_record(cfg).to_dict()
    parent = meta
    for part in path[:-1]:
        parent = parent[part]
    del parent[path[-1]]

    with pytest.raises(RuntimeError, match=match) as exc_info:
        check_resume_compat(cfg, meta)
    assert "checkpoint=<missing>" in str(exc_info.value)


def test_resume_compat_warns_for_eval_selection_drift(
    tmp_path: Path, caplog: LogCaptureFixture
) -> None:
    """Eval selection drift warns without blocking training resume."""
    cfg = _base_cfg(tmp_path / "run_eval_drift")
    cfg = replace(cfg, data=replace(cfg.data, max_eval_samples=4))
    meta = _checkpoint_record(cfg).to_dict()

    drifted = replace(cfg, data=replace(cfg.data, max_eval_samples=8))
    with caplog.at_level(logging.WARNING, logger="chomp.ckpt"):
        check_resume_compat(drifted, meta)
    assert "max_eval_samples" in caplog.text

    hf = replace(
        cfg,
        data=replace(
            cfg.data,
            backend="hf",
            hf_dataset="dummy",
            hf_name="dummy",
            hf_split="train",
            hf_eval_split=None,
            shuffle=False,
        ),
    )
    hf_meta = _checkpoint_record(hf).to_dict()
    split_drift = replace(hf, data=replace(hf.data, hf_eval_split="validation"))
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="chomp.ckpt"):
        check_resume_compat(split_drift, hf_meta)
    assert "data.eval" in caplog.text


def test_resume_compat_ignores_eval_selection_when_disabled(tmp_path: Path) -> None:
    """Eval selection knobs are inert while max_eval_samples=0."""
    cfg = _base_cfg(tmp_path / "run_eval_inert")
    cfg = replace(cfg, data=replace(cfg.data, max_eval_samples=0))
    meta = _checkpoint_record(cfg).to_dict()

    drifted = replace(cfg, data=replace(cfg.data, hf_eval_split="validation"))
    check_resume_compat(drifted, meta)


@pytest.mark.parametrize(
    ("mode", "lookahead_name"),
    [
        ("bin", "packing_buffer_docs"),
        ("multipack", "packing_group_docs"),
    ],
)
def test_strict_resume_detects_eval_effective_lookahead_change(
    tmp_path: Path,
    mode: str,
    lookahead_name: str,
) -> None:
    """Strict resume must reject lookahead drift that changes only eval packing."""
    cfg = _base_cfg(tmp_path / f"run_eval_lookahead_{mode}")
    cfg = replace(
        cfg,
        train=replace(cfg.train, batch_size=2, grad_accum=4),
        data=replace(
            cfg.data,
            packing_mode=mode,
            max_eval_samples=4,
            **{lookahead_name: 2},
        ),
        checkpoint=replace(cfg.checkpoint, resume_compat="strict"),
    )
    meta = _checkpoint_record(cfg).to_dict()
    drifted = replace(
        cfg,
        data=replace(cfg.data, **{lookahead_name: 4}),
    )

    with pytest.raises(RuntimeError, match="data.eval.packing_lookahead_docs"):
        check_resume_compat(drifted, meta)


@pytest.mark.parametrize(
    ("mode", "lookahead_name"),
    [
        ("bin", "packing_buffer_docs"),
        ("multipack", "packing_group_docs"),
    ],
)
def test_lookahead_change_is_inert_when_eval_disabled(
    tmp_path: Path,
    mode: str,
    lookahead_name: str,
) -> None:
    """Lookahead drift below the train clamp is inert without eval data."""
    cfg = _base_cfg(tmp_path / f"run_no_eval_lookahead_{mode}")
    cfg = replace(
        cfg,
        train=replace(cfg.train, batch_size=2, grad_accum=4),
        data=replace(
            cfg.data,
            packing_mode=mode,
            max_eval_samples=0,
            **{lookahead_name: 2},
        ),
        checkpoint=replace(cfg.checkpoint, resume_compat="strict"),
    )
    meta = _checkpoint_record(cfg).to_dict()
    drifted = replace(
        cfg,
        data=replace(cfg.data, **{lookahead_name: 4}),
    )

    assert data_fingerprint(drifted) == data_fingerprint(cfg)
    check_resume_compat(drifted, meta)


@pytest.mark.parametrize(
    ("field", "initial", "changed", "match"),
    [
        ("shuffle_buffer_size", 10_000, 200_000, "shuffle_buffer_size"),
        ("shuffle_buffer_bytes", 1024, 2048, "shuffle_buffer_bytes"),
        ("hf_revision", "abc123", "def456", "hf_revision"),
        ("repeat", True, False, "data.repeat"),
    ],
)
def test_resume_compat_warns_for_hf_source_drift(
    tmp_path: Path,
    field: str,
    initial: Any,
    changed: Any,
    match: str,
    caplog: LogCaptureFixture,
) -> None:
    """HF source-order and identity changes warn in the default mode."""
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

    with caplog.at_level(logging.WARNING, logger="chomp.ckpt"):
        check_resume_compat(drifted, meta)
    assert match in caplog.text


def test_resume_compat_ignores_inert_shuffle_values(tmp_path: Path) -> None:
    """Only effective shuffle behavior belongs in the resume identity."""
    cfg = _base_cfg(tmp_path / "run_inert_shuffle")
    cfg = replace(cfg, data=replace(cfg.data, window_shuffle_tokens=64))
    raw_drift = replace(
        cfg,
        data=replace(
            cfg.data,
            window_shuffle_tokens=cfg.data.window_shuffle_tokens + 1,
            window_shuffle_max_rows=cfg.data.window_shuffle_max_rows + 1,
        ),
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


def test_resume_compat_warns_for_local_window_shuffle_seed_drift(
    tmp_path: Path, caplog: LogCaptureFixture
) -> None:
    """Local window-shuffle seed drift warns in the default mode."""
    cfg = _base_cfg(tmp_path / "run_window_seed")
    cfg = replace(cfg, data=replace(cfg.data, window_shuffle_tokens=64))
    assert cfg.data.backend == "local_text"
    assert cfg.data.window_shuffle_tokens > 0
    meta = _checkpoint_record(cfg).to_dict()

    drifted = replace(cfg, data=replace(cfg.data, seed=cfg.data.seed + 1))
    with caplog.at_level(logging.WARNING, logger="chomp.ckpt"):
        check_resume_compat(drifted, meta)
    assert "window_shuffle_seed" in caplog.text

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


def test_loss_sum_adapter_passes_segments_iff_packed() -> None:
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
            reduction: str = "mean",
            return_valid_count: bool = False,
            loss_chunk_size: int | None = None,
        ) -> jax.Array | tuple[jax.Array, jax.Array]:
            _ = (input_ids, labels, attention_mask, deterministic, key)
            calls["segment_ids"] = segment_ids
            calls["position_ids"] = position_ids
            calls["reduction"] = reduction
            calls["return_valid_count"] = return_valid_count
            calls["loss_chunk_size"] = loss_chunk_size
            loss = jnp.zeros((), dtype=jnp.float32)
            count = jnp.array(7, dtype=jnp.int32)
            return (loss, count) if return_valid_count else loss

    params, static = eqx.partition(_SpyLM(w=jnp.zeros(1)), eqx.is_array)
    micro = Batch(
        input_ids=jnp.zeros((1, 8), dtype=jnp.int32),
        labels=jnp.zeros((1, 8), dtype=jnp.int32),
        segment_ids=jnp.ones((1, 8), dtype=jnp.int32),
    )

    loss_sum_and_count(
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
    assert calls["reduction"] == "sum"
    assert calls["return_valid_count"] is True
    assert calls["loss_chunk_size"] == 7

    loss_sum_and_count(
        params, static, batch=micro, deterministic=True, key=None, use_packed_segments=False
    )
    assert calls["segment_ids"] is None
    assert calls["position_ids"] is None
    assert calls["loss_chunk_size"] is None

    with pytest.raises(TypeError, match="unexpected keyword argument 'cache'"):
        loss_sum_and_count(  # type: ignore[call-arg]
            params,
            static,
            batch=micro,
            deterministic=True,
            key=None,
            cache=None,
        )


def test_megalodon_backend_advertises_segment_reset() -> None:
    """The installed megalodon-jax must expose the full-isolation capability flag."""
    pytest.importorskip("megalodon_jax")
    cfg = Config(model=make_tiny_megalodon_model(vocab_size=64))
    params, static = build_model(cfg, key=jax.random.PRNGKey(0))
    assert supports_packed_segments(params, static)


def test_run_pins_resolved_dataset_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    patch_hf_load_dataset: Callable[..., dict[str, int]],
) -> None:
    """A ref revision is resolved to a commit before artifacts and fingerprints.

    Users write refs like "main"; checkpointed runs must record the resolved
    commit so resume compares content identity instead of a mutable name.
    """
    patch_hf_load_dataset({"train": [{"text": "abcdefgh"} for _ in range(64)]})
    monkeypatch.setattr("chomp.train.resolve_dataset_revision", lambda dataset, revision: "a" * 40)
    cfg = make_small_run_cfg(tmp_path, run_subdir="run_resolve")
    cfg = replace(
        cfg,
        data=replace(
            cfg.data,
            backend="hf",
            hf_dataset="dummy",
            hf_name="dummy",
            hf_split="train",
            hf_revision="main",
            shuffle=False,
            repeat=True,
            max_eval_samples=0,
        ),
    )

    run_dir = run(cfg, config_path=None, resume="none", dry_run=False)

    resolved = json.loads((run_dir / "config_resolved.json").read_text())
    assert resolved["data"]["hf_revision"] == "a" * 40
    assert resolved["data"]["hf_requested_revision"] == "main"
    # The same resolved cfg object feeds checkpoint meta and data_fingerprint,
    # so the recorded source identity is the commit, not the ref.
    assert data_fingerprint(build_config(resolved))["source"]["revision"] == "a" * 40


def test_resume_reuses_pinned_dataset_revision_without_hub_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    patch_hf_load_dataset: Callable[..., dict[str, int]],
) -> None:
    """Resume must reuse the run's commit even when the YAML still says main."""
    patch_hf_load_dataset({"train": [{"text": "abcdefgh"} for _ in range(64)]})
    resolve_calls = 0

    def _resolve_once(dataset: str, revision: str | None) -> str:
        """Resolve the fresh run and fail if resume calls the Hub resolver."""
        nonlocal resolve_calls
        resolve_calls += 1
        if resolve_calls > 1:
            raise AssertionError("resume unexpectedly resolved the live Hub ref")
        assert (dataset, revision) == ("dummy", "main")
        return "b" * 40

    monkeypatch.setattr("chomp.train.resolve_dataset_revision", _resolve_once)
    cfg = make_small_run_cfg(tmp_path, run_subdir="run_pinned_resume", decay_steps=2)
    cfg = replace(
        cfg,
        data=replace(
            cfg.data,
            backend="hf",
            hf_dataset="dummy",
            hf_name="dummy",
            hf_split="train",
            hf_revision="main",
            shuffle=False,
            max_eval_samples=0,
        ),
        train=replace(cfg.train, steps=1),
    )
    run_dir = run(cfg, config_path=None, resume="none", dry_run=False)
    (run_dir / "config_resolved.json").unlink()

    resumed = replace(cfg, train=replace(cfg.train, steps=2))
    assert run(resumed, config_path=None, resume="latest", dry_run=False) == run_dir
    assert resolve_calls == 1


def test_resume_uses_tokenizer_snapshot_after_source_disappears(
    tmp_path: Path,
    local_bert_tokenizer: Path,
) -> None:
    """Resume must execute only the run-local tokenizer snapshot.

    :param Path tmp_path: Temporary run-directory root.
    :param Path local_bert_tokenizer: Deterministic local tokenizer source.
    """
    cfg = make_small_run_cfg(tmp_path, run_subdir="run_local_tokenizer", decay_steps=2)
    tokenizer = TokenizerConfig(
        kind="hf",
        hf_name_or_path=str(local_bert_tokenizer),
        hf_use_fast=True,
        hf_trust_remote_code=False,
        vocab_size_multiple=1,
        auto_set_special_tokens=True,
        add_bos=False,
        add_eos=False,
    )
    cfg = replace(
        cfg,
        data=replace(cfg.data, tokenizer=tokenizer),
        train=replace(cfg.train, steps=1),
    )
    run_dir = run(cfg, config_path=None, resume="none", dry_run=False)
    local_bert_tokenizer.rename(tmp_path / "source-unavailable")

    resumed = replace(cfg, train=replace(cfg.train, steps=2))
    assert run(resumed, config_path=None, resume="latest", dry_run=False) == run_dir


def test_warn_resume_honors_new_hf_ref_and_reuses_selected_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    patch_hf_load_dataset: Callable[..., dict[str, int]],
    caplog: LogCaptureFixture,
) -> None:
    """A deliberate ref change should run once, then become the resume identity."""
    patch_hf_load_dataset({"train": [{"text": "abcdefgh"} for _ in range(64)]})
    revisions = {"main": "a" * 40, "branch-b": "b" * 40}
    resolve_calls: list[str | None] = []

    def _resolve(dataset: str, revision: str | None) -> str:
        """Record each mutable ref resolved through the Hub boundary."""
        assert dataset == "dummy"
        assert revision is not None
        resolve_calls.append(revision)
        return revisions[revision]

    monkeypatch.setattr("chomp.train.resolve_dataset_revision", _resolve)
    cfg = make_small_run_cfg(tmp_path, run_subdir="run_changed_ref", decay_steps=3)
    cfg = replace(
        cfg,
        data=replace(
            cfg.data,
            backend="hf",
            hf_dataset="dummy",
            hf_name="dummy",
            hf_revision="main",
            shuffle=False,
            max_eval_samples=0,
        ),
        train=replace(cfg.train, steps=1),
    )
    run_dir = run(cfg, config_path=None, resume="none", dry_run=False)

    changed = replace(
        cfg,
        data=replace(cfg.data, hf_revision="branch-b"),
        train=replace(cfg.train, steps=2),
    )
    with caplog.at_level(logging.WARNING, logger="chomp.ckpt"):
        run(changed, config_path=None, resume="latest", dry_run=False)
    assert "data.hf_revision" in caplog.text

    continued = replace(changed, train=replace(changed.train, steps=3))
    run(continued, config_path=None, resume="latest", dry_run=False)

    assert resolve_calls == ["main", "branch-b"]
    meta = json.loads((run_dir / "checkpoints" / "3" / "meta" / "metadata").read_text())
    assert meta["config"]["data"]["hf_revision"] == "b" * 40
    assert meta["config"]["data"]["hf_requested_revision"] == "branch-b"


def test_strict_resume_rejects_new_hf_commit_in_same_repository(
    tmp_path: Path,
    patch_hf_load_dataset: Callable[..., dict[str, int]],
) -> None:
    """Strict compatibility should reject a deliberate same-repo commit change."""
    patch_hf_load_dataset({"train": [{"text": "abcdefgh"} for _ in range(64)]})
    cfg = make_small_run_cfg(tmp_path, run_subdir="run_strict_revision", decay_steps=2)
    cfg = replace(
        cfg,
        data=replace(
            cfg.data,
            backend="hf",
            hf_dataset="dummy",
            hf_name="dummy",
            hf_revision="1" * 40,
            shuffle=False,
            max_eval_samples=0,
        ),
        train=replace(cfg.train, steps=1),
        checkpoint=replace(cfg.checkpoint, resume_compat="strict"),
    )
    run(cfg, config_path=None, resume="none", dry_run=False)

    changed = replace(
        cfg,
        data=replace(cfg.data, hf_revision="2" * 40),
        train=replace(cfg.train, steps=2),
    )
    with pytest.raises(RuntimeError, match="data.hf_revision"):
        run(changed, config_path=None, resume="latest", dry_run=False)


def test_dry_run_skips_dataset_revision_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dry-run setup must not add an eager Hub revision request."""
    cfg = make_small_run_cfg(tmp_path, run_subdir="run_dry_revision")
    cfg = replace(
        cfg,
        data=replace(cfg.data, backend="hf", hf_dataset="dummy", hf_revision="main"),
    )

    def _unexpected_resolve(dataset: str, revision: str | None) -> str:
        raise AssertionError(f"unexpected Hub resolution for {dataset}@{revision}")

    monkeypatch.setattr("chomp.train.resolve_dataset_revision", _unexpected_resolve)
    monkeypatch.setattr(
        "chomp.train._run_impl", lambda config, **kwargs: Path(config.logging.run_dir)
    )

    assert run(cfg, config_path=None, resume="none", dry_run=True) == Path(
        cfg.logging.run_dir or ""
    )


def test_train_step_does_not_recompile_across_steps(caplog: LogCaptureFixture) -> None:
    """Fixed [A, B, T] shapes must compile train_step exactly once.

    The first call must produce compile logs (proving detection works); a
    second call with same-shape, different-content batches must produce none.
    """
    cfg = Config(
        model=ModelConfig(backend="dummy", vocab_size=256, d_model=32, dropout=0.0),
        data=DataConfig(
            backend="local_text",
            repeat=True,
            local_text="The quick brown fox jumps over the lazy dog.\n",
            tokenizer=TokenizerConfig(kind="byte", byte_offset=0, add_bos=False, add_eos=False),
        ),
        train=TrainConfig(
            seed=0,
            steps=2,
            batch_size=1,
            seq_len=16,
            grad_accum=2,
            jit=True,
            allow_cpu=True,
            deterministic=True,
        ),
        optim=OptimConfig(lr=1e-3, weight_decay=0.0, grad_clip_norm=1.0, warmup_steps=0),
    )
    key = jax.random.PRNGKey(cfg.train.seed)
    key, k_model = jax.random.split(key)
    params, static = build_model(cfg, key=k_model)
    tx, sched = build_optimizer(cfg, params)
    state = init_train_state(params=params, tx=tx, key=key)
    step = make_train_step(cfg, static=static, tx=tx, lr_schedule=sched)

    it = build_train_iterator(cfg)
    first = jax.device_put(next(it))
    second = jax.device_put(next(it))

    def compile_messages() -> list[str]:
        return [
            r.getMessage()
            for r in caplog.records
            if "Compiling" in r.getMessage() or "Finished tracing" in r.getMessage()
        ]

    with caplog.at_level(logging.DEBUG, logger="jax"), jax.log_compiles(True):
        state, _ = step(state, first)
        jax.block_until_ready(state.params)
        assert compile_messages(), "no compile logs on first call; detection is broken"
        caplog.clear()
        state, _ = step(state, second)
        jax.block_until_ready(state.params)
        assert not compile_messages(), compile_messages()
