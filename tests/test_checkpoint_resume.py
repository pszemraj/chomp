"""Checkpoint + resume contract test.

We want to catch the class of bugs where:
- you *think* you're resuming
- but some part of train_state (or data iterator) silently resets

Test strategy:
- Run K steps, saving every step.
- Resume from latest and run to N steps.
- Separately run a continuous N-step run.
- Restore final checkpoints and compare params + optimizer state.

We generate data via local_text through the real tokenize+pack pipeline.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax

from chomp.ckpt import default_ckpt_dir, make_manager, restore_at_step
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
)
from chomp.data.pipeline import build_train_iterator
from chomp.model import build_model
from chomp.train import build_optimizer, init_train_state, run
from chomp.utils.tree import tree_allclose


def _abstractify(tree):
    def to_struct(x):
        return jax.ShapeDtypeStruct(x.shape, x.dtype)

    return jax.tree_util.tree_map(to_struct, tree)


def _restore_state(run_dir: Path, cfg: Config, step: int):
    # Build skeleton state
    key = jax.random.PRNGKey(cfg.train.seed)
    key, k_model = jax.random.split(key)
    params, _static = build_model(cfg, key=k_model)
    tx, _sched = build_optimizer(cfg, params)
    state0 = init_train_state(cfg, params=params, tx=tx, key=key)
    abstract_state = _abstractify(state0)

    ckpt_dir = default_ckpt_dir(run_dir)
    mgr = make_manager(ckpt_dir, max_to_keep=5, save_every=1, async_save=False)
    data_it = build_train_iterator(cfg)
    _, state, _meta = restore_at_step(
        mgr, step=step, abstract_train_state=abstract_state, data_iter=data_it
    )
    return state


def test_resume_matches_continuous(tmp_path: Path):
    K = 4
    N = 6

    base = Config(
        model=ModelConfig(backend="dummy", vocab_size=256, d_model=32, dropout=0.0),
        data=DataConfig(
            backend="local_text",
            repeat=True,
            local_text="Deterministic local text for checkpoint test.\n",
            tokenizer=TokenizerConfig(kind="byte", byte_offset=0, add_bos=False, add_eos=False),
        ),
        train=TrainConfig(
            seed=0,
            steps=N,
            batch_size=2,
            seq_len=16,
            grad_accum=2,
            jit=False,
            deterministic=True,
            allow_cpu=True,
            log_every=1000,
        ),
        optim=OptimConfig(lr=1e-3, weight_decay=0.0, grad_clip_norm=0.0, warmup_steps=0),
        checkpoint=CheckpointConfig(enabled=True, save_every=1, max_to_keep=5, async_save=False),
        debug=DebugConfig(nan_check=True, check_device_every=0),
        logging=LoggingConfig(
            project="chomp", run_dir=None, metrics_file="metrics.jsonl", level="INFO"
        ),
    )

    # --- Interrupted + resume run ---
    run_a = tmp_path / "run_a"
    cfg_a = replace(base, logging=replace(base.logging, run_dir=str(run_a)))
    run(cfg_a, config_path=None, resume="none", max_steps=K)
    run(cfg_a, config_path=None, resume="latest")

    state_a = _restore_state(run_a, cfg_a, step=N)

    # --- Continuous run ---
    run_b = tmp_path / "run_b"
    cfg_b = replace(base, logging=replace(base.logging, run_dir=str(run_b)))
    run(cfg_b, config_path=None, resume="none")

    state_b = _restore_state(run_b, cfg_b, step=N)

    plat = jax.devices()[0].platform
    if plat == "cpu":
        assert tree_allclose(state_a.params, state_b.params, rtol=0.0, atol=0.0)
        assert tree_allclose(state_a.opt_state, state_b.opt_state, rtol=0.0, atol=0.0)
    else:
        # GPUs can be non-bit-exact depending on kernels.
        assert tree_allclose(state_a.params, state_b.params, rtol=1e-5, atol=1e-5)
        assert tree_allclose(state_a.opt_state, state_b.opt_state, rtol=1e-5, atol=1e-5)
