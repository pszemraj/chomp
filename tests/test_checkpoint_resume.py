"""Checkpoint resume + forward smoke tests."""

from __future__ import annotations

import math
import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

from chomp.ckpt import default_ckpt_dir, make_manager, restore_latest
from chomp.config import Config, load_config
from chomp.data import build_train_iterator, prepare_tokenizer_and_config
from chomp.model import build_model, training_loss
from chomp.train import build_optimizer, init_train_state, run
from chomp.types import Batch
from chomp.utils.tree import abstractify_tree


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


def test_checkpoint_resume_advances_step(tmp_path: Path) -> None:
    """A saved checkpoint can be resumed and training continues."""
    cfg, config_src = _small_cfg(tmp_path)
    run_dir = run(cfg, config_path=str(config_src), resume="none", dry_run=False)

    ckpt_dir = default_ckpt_dir(run_dir)
    assert (ckpt_dir / "2").exists(), "expected checkpoint at step 2"

    cfg_resume = replace(cfg, train=replace(cfg.train, steps=3))
    run_dir2 = run(cfg_resume, config_path=str(config_src), resume="latest", dry_run=False)
    assert run_dir2 == run_dir
    assert (ckpt_dir / "3").exists(), "expected checkpoint at step 3 after resume"


def test_checkpoint_restore_allows_forward(tmp_path: Path) -> None:
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


def test_checkpoint_saves_final_step(tmp_path: Path) -> None:
    """Final step should be checkpointed even if save_every does not divide steps."""
    cfg, config_src = _small_cfg(tmp_path)
    cfg = replace(cfg, train=replace(cfg.train, steps=3))
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, save_every=2))

    run_dir = run(cfg, config_path=str(config_src), resume="none", dry_run=False)
    ckpt_dir = default_ckpt_dir(run_dir)

    assert (ckpt_dir / "2").exists(), "expected checkpoint at save interval"
    assert (ckpt_dir / "3").exists(), "expected final checkpoint at step 3"
