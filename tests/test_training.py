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
from chomp.model import build_model, supports_packed_segments, training_loss
from chomp.train import (
    _METRICS_FILE_DROP,
    _WANDB_DROP,
    _build_checkpoint_manager,
    _project_metrics,
    build_optimizer,
    init_train_state,
    run,
)
from chomp.types import Batch, TrainState
from chomp.utils.tree import abstractify_tree
from tests.helpers.assertions import tree_allclose
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
    state0 = init_train_state(params=params, tx=tx, key=jax.random.PRNGKey(1))
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
            window_shuffle_windows=0,
            grain_prefetch=grain_prefetch,
        ),
    )
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, save_every=2))

    run_dir = run(cfg, config_path=str(config_src), resume="none", dry_run=False)

    metrics_path = run_dir / cfg.logging.metrics_file
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert any(row.get("data_exhausted") and row.get("step") == 3 for row in rows)

    ckpt_dir = default_ckpt_dir(run_dir)
    steps_on_disk = {int(p.name) for p in ckpt_dir.iterdir() if p.is_dir() and p.name.isdigit()}
    assert steps_on_disk == {2, 3}, (
        f"exact EOF must save the aligned final checkpoint, found {steps_on_disk}"
    )


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
        init_train_state(params=params, tx=tx, key=jax.random.PRNGKey(1))
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


def test_exhaustion_mid_assembly_skips_final_checkpoint(tmp_path: Path) -> None:
    """StopIteration during batch fetch must not write a final checkpoint.

    Batch assembly pops A*B windows one at a time, so exhaustion can strike
    after part of the stream/packer state was consumed (and a popped window
    discarded). "Nothing was consumed" does not hold; a final checkpoint
    there would pair the old train state with a partially-advanced iterator.
    """
    # One 116-char doc -> 116 byte tokens (offset 0, no BOS/EOS; varied bytes
    # so windows differ and the loss-replay check below has teeth): 7 poppable
    # seq_len=16 windows. grad_accum=2 -> batches 1-3 eat 6 windows; batch 4
    # pops window 7, then dies on window 8 with 4 leftover tokens.
    # max_doc_tokens must be raised: null resolves to 4*seq_len=64 and would
    # truncate the doc to 4 windows.
    text = "".join(chr(97 + (i * 7) % 26) for i in range(116))
    cfg, config_src = make_small_run_cfg(tmp_path, local_text=text, decay_steps=10)
    cfg = replace(cfg, train=replace(cfg.train, steps=10, grad_accum=2))
    cfg = replace(
        cfg,
        data=replace(
            cfg.data,
            repeat=False,
            tokenizer=replace(cfg.data.tokenizer, max_doc_tokens=128),
        ),
    )
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, save_every=2))

    run_dir = run(cfg, config_path=str(config_src), resume="none", dry_run=False)

    metrics_path = run_dir / cfg.logging.metrics_file
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert any(row.get("data_exhausted") for row in rows)
    loss_step3 = [row["loss"] for row in rows if row.get("step") == 3 and "loss" in row]
    assert len(loss_step3) == 1

    ckpt_dir = default_ckpt_dir(run_dir)
    steps_on_disk = {int(p.name) for p in ckpt_dir.iterdir() if p.is_dir() and p.name.isdigit()}
    assert steps_on_disk == {2}, (
        f"exhaustion mid-assembly must skip the final checkpoint, found {steps_on_disk}"
    )

    # Resume from the aligned periodic checkpoint: batch 3 replays bit-exactly
    # (same loss), then the stream exhausts again without new checkpoints.
    run(cfg, config_path=str(config_src), resume="latest", dry_run=False)
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    loss_step3_replayed = [row["loss"] for row in rows if row.get("step") == 3 and "loss" in row]
    assert len(loss_step3_replayed) == 2
    assert loss_step3_replayed[0] == loss_step3_replayed[1]
    steps_on_disk = {int(p.name) for p in ckpt_dir.iterdir() if p.is_dir() and p.name.isdigit()}
    assert steps_on_disk == {2}


def test_resume_bit_exact_with_prefetch_and_window_shuffle(tmp_path: Path) -> None:
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
    # max_doc_tokens must be raised: null resolves to 4*seq_len=64 and would
    # shorten the period.
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
                window_shuffle_windows=8,
                tokenizer=replace(cfg.data.tokenizer, max_doc_tokens=256),
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
    run(cfg_resume, config_path=str(config_src), resume="latest", dry_run=False)

    # Per-step losses agree exactly across the resume boundary (steps 4-6 ran
    # from the restored mid-window prefetching iterator).
    def _losses(run_dir: Path) -> dict[int, float]:
        rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
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
        mgr = make_manager(default_ckpt_dir(run_dir), max_to_keep=2, save_every=4, async_save=False)
        _, state, _ = restore_at_step(
            mgr,
            step=6,
            abstract_train_state=abstract_state,
            data_iter=build_train_iterator(cfg_ref, tokenizer=tokenizer),
        )
        states.append(state)
    assert tree_allclose(states[0].params, states[1].params, rtol=0.0, atol=0.0)
    assert tree_allclose(states[0].opt_state, states[1].opt_state, rtol=0.0, atol=0.0)


def test_resume_bit_exact_through_exhaustion_flush(tmp_path: Path) -> None:
    """Interrupted + resumed must match continuous bit-exactly when the run
    ends in an FFD end-of-stream flush, with window shuffle + prefetch engaged.

    This is the integration claim behind the flush feature: grain's window
    shuffle replays its parent from the block start on resume, so the resumed
    process re-drives the packer through StopIteration -> finish() -> flush.
    If the replayed flush produced different windows than the continuous run,
    step-3 losses or the final states would diverge.
    """
    # One 84-byte doc -> segments [16]*5 + [4] at seq_len=16 (max_doc_tokens
    # raised past the 4*seq_len=64 default so nothing truncates). With
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
                window_shuffle_windows=8,
                tokenizer=replace(cfg.data.tokenizer, max_doc_tokens=256),
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
        rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
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
        mgr = make_manager(default_ckpt_dir(run_dir), max_to_keep=2, save_every=4, async_save=False)
        _, state, _ = restore_at_step(
            mgr,
            step=3,
            abstract_train_state=abstract_state,
            data_iter=build_train_iterator(cfg_ref, tokenizer=tokenizer),
        )
        states.append(state)
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
    with pytest.raises(RuntimeError, match="checkpoint finalization failed"):
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
        pytest.raises(
            RuntimeError, match="checkpoint finalization failed: Non-finite loss at step 5"
        ),
    ):
        run(cfg, config_path=str(config_src), resume="none", dry_run=False)

    ckpt_dir = default_ckpt_dir(Path(cfg.logging.run_dir))
    steps_on_disk = {int(p.name) for p in ckpt_dir.iterdir() if p.is_dir() and p.name.isdigit()}
    assert steps_on_disk == {2, 4}, f"latest must stay the last good save, found {steps_on_disk}"
    assert any("Skipping final checkpoint at step" in rec.getMessage() for rec in caplog.records)


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

    metrics_path = run_dir / cfg.logging.metrics_file
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    assert any(row.get("crash") for row in rows)

    log_text = (run_dir / cfg.logging.log_file).read_text()
    assert "Training crashed" in log_text


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


def test_resume_compat_checks_unknown_packing_fingerprint_keys(tmp_path: Path) -> None:
    """A packing key recorded on only one side must error, never be skipped.

    The packing section is compared over the union of recorded keys, so a
    knob added to data_fingerprint (or one written by a different chomp
    version) can never bypass compat checking — the failure mode of a
    hand-enumerated comparison list.
    """
    cfg = _base_cfg(tmp_path / "run_unknown_key")
    meta = {"config": cfg.to_dict(), "data_fingerprint": data_fingerprint(cfg)}
    meta["data_fingerprint"]["packing"]["future_knob"] = 3

    with pytest.raises(RuntimeError, match="future_knob"):
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
    meta = {"config": cfg.to_dict(), "data_fingerprint": data_fingerprint(cfg)}
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


def test_resume_compat_device_put_drift(tmp_path: Path, caplog: LogCaptureFixture) -> None:
    """device_put does not change sample order, so plain drift only warns —
    but with prefetch active it moves device transfers into the prefetch
    thread whose serialized state a restore must line up against, so the
    mismatch hardens to an error."""
    cfg = _base_cfg(tmp_path / "run_dput")
    meta = {"config": cfg.to_dict(), "data_fingerprint": data_fingerprint(cfg)}
    drifted = replace(cfg, data=replace(cfg.data, device_put=not cfg.data.device_put))
    with caplog.at_level(logging.WARNING, logger="chomp.ckpt"):
        check_resume_compat(drifted, meta)  # must not raise
    assert any("device_put" in rec.message for rec in caplog.records)

    pf = replace(cfg, data=replace(cfg.data, grain_prefetch=2))
    pf_meta = {"config": pf.to_dict(), "data_fingerprint": data_fingerprint(pf)}
    pf_drifted = replace(pf, data=replace(pf.data, device_put=not pf.data.device_put))
    with pytest.raises(RuntimeError, match="device_put"):
        check_resume_compat(pf_drifted, pf_meta)


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


def test_resume_compat_rejects_local_window_shuffle_seed_drift(tmp_path: Path) -> None:
    """Local window-shuffle replay must reject a changed data seed."""
    cfg = _base_cfg(tmp_path / "run_window_seed")
    assert cfg.data.backend == "local_text"
    assert cfg.data.window_shuffle_windows > 0
    meta = {"config": cfg.to_dict(), "data_fingerprint": data_fingerprint(cfg)}

    drifted = replace(cfg, data=replace(cfg.data, seed=cfg.data.seed + 1))
    with pytest.raises(RuntimeError, match="window_shuffle_seed"):
        check_resume_compat(drifted, meta)

    disabled = replace(cfg, data=replace(cfg.data, window_shuffle_windows=0))
    disabled_meta = {
        "config": disabled.to_dict(),
        "data_fingerprint": data_fingerprint(disabled),
    }
    disabled_drifted = replace(disabled, data=replace(disabled.data, seed=disabled.data.seed + 1))
    check_resume_compat(disabled_drifted, disabled_meta)


def test_resume_compat_rejects_hf_repeat_drift(tmp_path: Path) -> None:
    """repeat decides epoch rollover vs stream termination — hard error for
    HF streams, not just local_text."""
    cfg = _base_cfg(tmp_path / "run_repeat")
    cfg = replace(
        cfg,
        data=replace(
            cfg.data,
            backend="hf",
            hf_dataset="dummy",
            hf_name="dummy",
            hf_split="train",
            repeat=True,
        ),
    )
    meta = {"config": cfg.to_dict(), "data_fingerprint": data_fingerprint(cfg)}
    drifted = replace(cfg, data=replace(cfg.data, repeat=False))
    with pytest.raises(RuntimeError, match="data.repeat"):
        check_resume_compat(drifted, meta)


def test_supports_packed_segments_requires_capability_flag() -> None:
    """Capability check keys on supports_segment_reset, not compute_loss signature.

    A backend that accepts segment_ids/position_ids but does not advertise the
    flag (megalodon-jax < 0.1.2: attention-only isolation, CEMA/TimestepNorm
    state leaking across packed boundaries) must be rejected.
    """
    import equinox as eqx

    cfg = Config(model=ModelConfig(backend="dummy", vocab_size=64, d_model=16, dropout=0.0))
    params, static = build_model(cfg, key=jax.random.PRNGKey(0))
    assert supports_packed_segments(params, static)

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
    """segment_ids/position_ids reach the backend exactly when strict packed segments are on."""
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
        params, static, batch=micro, deterministic=True, key=None, use_packed_segments=True
    )
    assert calls["segment_ids"] is not None
    assert calls["position_ids"] is not None

    training_loss(
        params, static, batch=micro, deterministic=True, key=None, use_packed_segments=False
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
    assert supports_packed_segments(params, static)
