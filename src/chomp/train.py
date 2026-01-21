"""Training loop + compiled train step.

This module is the heart of chomp.

Design rules (hard-earned):
1) **Compile once**. If shapes change, you lose.
2) **TrainState is arrays-only**. If you stash python objects in state, you lose.
3) **Grad accumulation happens inside the compiled step** via `lax.scan`.
   This avoids Python overhead and keeps optimizer updates correct.
4) **Real data**. Synthetic batches are a bootstrap tool, not a training system.

Phases 0–2:
- dummy or Megalodon model backend
- Optax AdamW with warmup+cosine schedule
- scan-based grad accumulation

Phase 3:
- Orbax checkpointing + resume contract

Initial Phase 4 chunk:
- Minimal HF streaming + tokenize + pack iterator (no Grain yet)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from chomp.ckpt import build_meta, default_ckpt_dir, make_manager, restore_at_step, restore_latest, save
from chomp.config import Config, derived_deterministic
from chomp.data import build_train_iterator, data_fingerprint
from chomp.model import build_model, training_loss
from chomp.types import Batch, TrainState
from chomp.utils.devices import assert_batch_on_device
from chomp.utils.io import MetricsWriter, create_run_dir
from chomp.utils.profiling import start_trace, step_annotation, stop_trace
from chomp.utils.tree import param_count


def _weight_decay_mask(params: Any) -> Any:
    """Heuristic: apply weight decay to matrices (ndim >= 2), not to biases/scales."""

    def mask_one(x):
        if not hasattr(x, "ndim"):
            return False
        return x.ndim >= 2

    return jax.tree_util.tree_map(mask_one, params)


def build_optimizer(cfg: Config, params: Any) -> tuple[optax.GradientTransformation, Callable[[jax.Array], jax.Array]]:
    """Create Optax optimizer + schedule function (for logging)."""

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.optim.lr,
        warmup_steps=cfg.optim.warmup_steps,
        decay_steps=cfg.optim.total_steps,
        end_value=0.0,
    )

    transforms = []
    if cfg.optim.grad_clip_norm and cfg.optim.grad_clip_norm > 0:
        transforms.append(optax.clip_by_global_norm(cfg.optim.grad_clip_norm))

    transforms.append(
        optax.adamw(
            learning_rate=schedule,
            weight_decay=cfg.optim.weight_decay,
            mask=_weight_decay_mask(params),
        )
    )

    tx = optax.chain(*transforms)
    return tx, schedule


def init_train_state(cfg: Config, *, params: Any, tx: optax.GradientTransformation, key: jax.Array) -> TrainState:
    opt_state = tx.init(params)
    return TrainState(step=jnp.array(0, dtype=jnp.int32), params=params, opt_state=opt_state, rng=key)


def _abstractify_tree(tree: Any) -> Any:
    """Convert a pytree of arrays to ShapeDtypeStruct for Orbax restore."""

    def to_struct(x):
        return jax.ShapeDtypeStruct(x.shape, x.dtype)

    return jax.tree_util.tree_map(to_struct, tree)


def make_train_step(
    cfg: Config,
    *,
    static: Any,
    tx: optax.GradientTransformation,
    lr_schedule: Callable[[jax.Array], jax.Array],
) -> Callable[[TrainState, Batch], tuple[TrainState, dict[str, jax.Array]]]:
    """Build the compiled train_step.

    The resulting function:
      - consumes TrainState and Batch (fixed shape)
      - performs grad accumulation via `lax.scan`
      - applies exactly one optimizer update

    NOTE: We close over `static`, `tx`, and small config constants. This is fine.
    Do not close over dynamic shapes or python objects.
    """

    deterministic = derived_deterministic(cfg)
    grad_accum = int(cfg.train.grad_accum)

    def micro_loss(params: Any, input_ids: jax.Array, labels: jax.Array, attn: jax.Array, key: jax.Array | None):
        micro = Batch(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attn.astype(bool),
        )
        return training_loss(params, static, batch=micro, deterministic=deterministic, key=key)

    loss_and_grad = eqx.filter_value_and_grad(micro_loss)

    def train_step(state: TrainState, batch: Batch) -> tuple[TrainState, dict[str, jax.Array]]:
        # Split RNG: one for next state, one to generate per-micro dropout keys
        rng, step_key = jax.random.split(state.rng)
        micro_keys = jax.random.split(step_key, grad_accum)

        # Init accumulators
        loss0 = jnp.zeros((), dtype=jnp.float32)
        grad0 = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), state.params)

        def body(carry, inputs):
            loss_sum, grad_sum = carry
            in_ids, labs, attn, k = inputs
            loss, grads = loss_and_grad(state.params, in_ids, labs, attn, k)
            loss_sum = loss_sum + loss.astype(jnp.float32)
            grad_sum = jax.tree_util.tree_map(lambda a, b: a + b, grad_sum, grads)
            return (loss_sum, grad_sum), None

        (loss_sum, grad_sum), _ = jax.lax.scan(
            body,
            (loss0, grad0),
            (batch.input_ids, batch.labels, batch.attention_mask, micro_keys),
        )

        loss = loss_sum / grad_accum
        grads = jax.tree_util.tree_map(lambda g: g / grad_accum, grad_sum)

        grad_norm = optax.global_norm(grads)
        updates, new_opt_state = tx.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)

        new_state = TrainState(step=state.step + 1, params=new_params, opt_state=new_opt_state, rng=rng)

        lr = lr_schedule(state.step)
        metrics = {
            "loss": loss,
            "grad_norm": grad_norm.astype(jnp.float32),
            "lr": lr.astype(jnp.float32),
        }
        return new_state, metrics

    if cfg.train.jit:
        train_step = eqx.filter_jit(train_step)
    return train_step


def run(
    cfg: Config,
    *,
    config_path: str | None = None,
    resume: Literal["none", "latest"] | int = "none",
) -> Path:
    """Run a training job and return the run directory.

    Resume contract:
    - resume requires logging.run_dir to be set (existing run directory)
    - we restore both train_state and data iterator state when present

    `resume`:
      - "none": start fresh
      - "latest": restore latest checkpoint
      - int: restore specific checkpoint step
    """

    allow_existing = resume != "none"

    # Prepare run dir
    run_dir = create_run_dir(cfg, config_path=config_path, allow_existing=allow_existing)
    metrics_path = run_dir / cfg.logging.metrics_file

    # Optional profiling
    if cfg.train.profile:
        trace_dir = cfg.train.profile_dir or str(run_dir / "trace")
        Path(trace_dir).mkdir(parents=True, exist_ok=True)
        start_trace(trace_dir)

    # Build model
    key = jax.random.PRNGKey(cfg.train.seed)
    key, k_model = jax.random.split(key)
    params, static = build_model(cfg, key=k_model)

    # Log param count once
    n_params = param_count(params)
    print(f"[chomp] params: {n_params:,}")

    # Optim + state skeleton
    tx, schedule = build_optimizer(cfg, params)
    state0 = init_train_state(cfg, params=params, tx=tx, key=key)
    abstract_state = _abstractify_tree(state0)

    # Data iterator (host-side)
    data_it = build_train_iterator(cfg)

    # Checkpoint manager
    manager = None
    if cfg.checkpoint.enabled:
        ckpt_dir = Path(cfg.checkpoint.root_dir) if cfg.checkpoint.root_dir else default_ckpt_dir(run_dir)
        manager = make_manager(
            ckpt_dir,
            max_to_keep=cfg.checkpoint.max_to_keep,
            save_every=cfg.checkpoint.save_every,
            async_save=cfg.checkpoint.async_save,
        )

    # Restore if requested
    state = state0
    if resume != "none":
        if manager is None:
            raise RuntimeError("resume requested but checkpointing is disabled")
        if resume == "latest":
            step_r, state, data_state, _meta = restore_latest(manager, abstract_train_state=abstract_state)
        else:
            step_r, state, data_state, _meta = restore_at_step(manager, step=int(resume), abstract_train_state=abstract_state)

        print(f"[chomp] resumed from checkpoint step {step_r}")
        if data_state is not None:
            data_it.set_state(data_state)

    train_step = make_train_step(cfg, static=static, tx=tx, lr_schedule=schedule)

    # Training loop
    t_compile = None
    t0 = time.perf_counter()

    # Determine starting step from TrainState
    start_step = int(jax.device_get(state.step))
    if start_step >= cfg.train.steps:
        print(f"[chomp] start_step ({start_step}) >= train.steps ({cfg.train.steps}); nothing to do")
        return run_dir

    tokens_per_step = int(cfg.train.grad_accum) * int(cfg.train.batch_size) * int(cfg.train.seq_len)

    with MetricsWriter(metrics_path) as mw:
        for _ in tqdm(range(start_step, cfg.train.steps), desc="train", dynamic_ncols=True):
            # Fetch batch (host) and (optionally) device_put
            batch = next(data_it)
            if not cfg.data.device_put:
                batch = jax.device_put(batch)

            # Batch placement validation (real check)
            if cfg.debug.check_device_every > 0 and (int(jax.device_get(state.step)) % cfg.debug.check_device_every == 0):
                assert_batch_on_device(batch, allow_cpu=cfg.train.allow_cpu)

            # Step (compile happens on first call)
            with step_annotation("train_step"):
                t1 = time.perf_counter()
                state, metrics = train_step(state, batch)
                # Synchronize for accurate timing
                metrics_host = jax.device_get(metrics)
                t2 = time.perf_counter()

            if t_compile is None:
                t_compile = t2 - t1

            step_i = int(jax.device_get(state.step))

            # Debug NaN check
            if cfg.debug.nan_check:
                loss_f = float(metrics_host["loss"])
                if not (loss_f == loss_f) or loss_f in (float("inf"), float("-inf")):
                    raise RuntimeError(f"Non-finite loss at step {step_i}: {loss_f}")

            # Checkpoint save (after state updated)
            if manager is not None and (step_i % int(cfg.checkpoint.save_every) == 0):
                meta = build_meta(step=step_i, config=cfg.to_dict(), data_fingerprint=data_fingerprint(cfg))
                save(
                    manager,
                    step=step_i,
                    train_state=state,
                    data_state=data_it.get_state(),
                    meta=meta,
                )

            # Log
            if (step_i % cfg.train.log_every) == 0:
                row = {
                    "step": int(step_i),
                    "loss": float(metrics_host["loss"]),
                    "grad_norm": float(metrics_host["grad_norm"]),
                    "lr": float(metrics_host["lr"]),
                    "tokens_seen": int(step_i) * tokens_per_step,
                    "wall_time_s": time.perf_counter() - t0,
                }
                if step_i == (start_step + 1) and t_compile is not None:
                    row["first_step_compile_time_s"] = float(t_compile)
                    # Best-effort memory stats
                    try:
                        ms = jax.local_devices()[0].memory_stats()
                        if "peak_bytes_in_use" in ms:
                            row["peak_memory_gb"] = float(ms["peak_bytes_in_use"]) / 1e9
                    except Exception:
                        pass

                mw.write(row)

    if manager is not None:
        manager.wait_until_finished()

    if cfg.train.profile:
        stop_trace()

    return run_dir
