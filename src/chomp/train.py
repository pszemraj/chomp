# SPDX-License-Identifier: Apache-2.0

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
- Optax AdamW or Muon with warmup+cosine schedule
- scan-based grad accumulation

Phase 3:
- Orbax checkpointing + resume contract

Phases 4–5:
- HF streaming + tokenize + pack iterator wrapped in Grain
"""

from __future__ import annotations

import contextlib
import logging
import math
import random
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from chomp.ckpt import (
    build_meta,
    check_resume_compat,
    default_ckpt_dir,
    make_manager,
    restore_at_step,
    restore_latest,
    save,
)
from chomp.config import Config, derived_deterministic, resolve_decay_horizon
from chomp.data import (
    build_eval_iterator,
    build_generation_text_stream,
    build_train_iterator,
    data_fingerprint,
    load_or_create_eval_texts,
    load_tokenizer_snapshot,
    prepare_tokenizer_and_config,
    save_tokenizer_snapshot,
    tokenizer_snapshot_hash,
)
from chomp.model import build_model, training_loss
from chomp.types import IGNORE_INDEX, Batch, TrainState
from chomp.utils.devices import assert_batch_on_device
from chomp.utils.io import MetricsWriter, add_file_logging, create_run_dir
from chomp.utils.profiling import start_trace, step_annotation, stop_trace
from chomp.utils.tree import abstractify_tree, param_count

logger = logging.getLogger(__name__)


def _count_tokens(labels: jax.Array, attention_mask: jax.Array | None) -> jax.Array:
    """Count valid tokens after the causal shift (for correct GA normalization).

    :param jax.Array labels: Label tensor of shape [B, T].
    :param attention_mask: Optional mask tensor of shape [B, T].
    :return jax.Array: Scalar count of valid (non-ignored, non-masked) tokens.
    """

    shift_labels = labels[:, 1:]
    valid = shift_labels != IGNORE_INDEX
    if attention_mask is not None:
        valid = valid & attention_mask[:, 1:].astype(bool)
    return jnp.sum(valid, dtype=jnp.int32)


def _check_finite_metrics(metrics: dict[str, Any], *, step: int) -> None:
    """Fail fast if loss/grad_norm are non-finite.

    :param dict[str, Any] metrics: Dictionary containing 'loss' and 'grad_norm' values.
    :param int step: Current training step (for error messages).
    :raises RuntimeError: If loss or grad_norm is NaN or Inf.
    """

    for name in ("loss", "grad_norm"):
        value = float(metrics[name])
        if not math.isfinite(value):
            raise RuntimeError(f"Non-finite {name} at step {step}: {value}")


def _device_memory_stats_gb() -> dict[str, float]:
    """Best-effort device memory stats in GB (if available).

    :return dict[str, float]: Keys include device_memory_gb and peak_memory_gb when present.
    """

    try:
        ms = jax.local_devices()[0].memory_stats()
    except Exception:
        return {}
    if not ms:
        return {}

    stats: dict[str, float] = {}
    if "bytes_in_use" in ms:
        stats["device_memory_gb"] = float(ms["bytes_in_use"]) / 1e9
    if "peak_bytes_in_use" in ms:
        stats["peak_memory_gb"] = float(ms["peak_bytes_in_use"]) / 1e9
    return stats


@dataclass(frozen=True)
class GenerationSettings:
    """Resolved generation settings for periodic sampling."""

    every: int
    input_len: int
    max_new_tokens: int
    temperature: float | None
    top_k: int | None
    top_p: float | None


def _resolve_generation_settings(cfg: Config) -> GenerationSettings | None:
    """Resolve generation defaults from config and model settings.

    :param Config cfg: Training configuration.
    :return GenerationSettings | None: Resolved settings or None if disabled.
    """
    every = int(cfg.train.generate_every)
    if every <= 0:
        return None
    input_len = cfg.train.generate_input_len
    if input_len is None:
        input_len = max(1, int(cfg.train.seq_len) // 2)
    max_new = cfg.train.generate_max_tokens
    if max_new is None:
        max_new = int(cfg.model.chunk_size) + 16
    return GenerationSettings(
        every=every,
        input_len=int(input_len),
        max_new_tokens=int(max_new),
        temperature=cfg.train.generate_temperature,
        top_k=cfg.train.generate_top_k,
        top_p=cfg.train.generate_top_p,
    )


def _setup_run_dir_and_tokenizer(
    cfg: Config,
    *,
    config_path: str | None,
    allow_existing: bool,
    dry_run: bool,
) -> tuple[
    Config,
    Any,
    Path,
    Path,
    list[list[int]],
    GenerationSettings | None,
    Any | None,
    jax.Array | None,
    random.Random | None,
]:
    """Prepare run directory, tokenizer snapshot, eval tokens, and generation stream.

    :param Config cfg: Training configuration.
    :param str | None config_path: Optional config path for run_dir bookkeeping.
    :param bool allow_existing: Whether to reuse an existing run directory.
    :param bool dry_run: If True, skip heavy data/generation setup.
    :return tuple[Config, Any, Path, Path, list[list[int]], GenerationSettings | None, Any | None, jax.Array | None, random.Random | None]:
        Updated config, tokenizer, run/metrics paths, eval tokens, generation settings, stream, key, and RNG.
    """
    tokenizer = None
    if allow_existing and cfg.logging.run_dir is not None:
        run_dir_hint = Path(cfg.logging.run_dir)
        tok_dir = run_dir_hint / "tokenizer"
        if tok_dir.exists():
            tokenizer = load_tokenizer_snapshot(run_dir_hint, cfg)
        else:
            logger.warning(
                "Resume requested but tokenizer snapshot is missing at %s; rebuilding from config.",
                tok_dir,
            )

    cfg, tokenizer = prepare_tokenizer_and_config(cfg, tokenizer=tokenizer)

    run_dir = create_run_dir(cfg, config_path=config_path, allow_existing=allow_existing)
    if cfg.logging.log_file is not None:
        add_file_logging(run_dir / cfg.logging.log_file, level=cfg.logging.level)
    metrics_path = run_dir / cfg.logging.metrics_file
    save_tokenizer_snapshot(run_dir, cfg, tokenizer, allow_existing=allow_existing)

    eval_tokens = [] if dry_run else load_or_create_eval_texts(cfg, tokenizer=tokenizer)

    gen_settings: GenerationSettings | None = None
    gen_stream = None
    gen_key = None
    gen_rng = None
    if not dry_run:
        gen_settings = _resolve_generation_settings(cfg)
        if gen_settings is not None and cfg.model.backend != "megalodon":
            logger.debug(
                "Generation is enabled but model.backend=%s; skipping generation.",
                cfg.model.backend,
            )
            gen_settings = None
        if gen_settings is not None:
            try:
                gen_stream = build_generation_text_stream(cfg, seed_offset=1)
                gen_key = jax.random.PRNGKey(cfg.train.seed + 1234)
                gen_rng = random.Random(cfg.train.seed + 5678)
            except Exception as exc:
                logger.warning("Failed to initialize generation stream: %s", exc)
                gen_settings = None

    return (
        cfg,
        tokenizer,
        run_dir,
        metrics_path,
        eval_tokens,
        gen_settings,
        gen_stream,
        gen_key,
        gen_rng,
    )


def _maybe_init_wandb(cfg: Config, *, run_dir: Path, dry_run: bool) -> Any | None:
    """Initialize W&B if enabled, otherwise return None.

    :param Config cfg: Training configuration.
    :param Path run_dir: Run directory path.
    :param bool dry_run: Whether this is a dry run.
    :return Any | None: W&B run object or None if disabled.
    """
    wandb_cfg = cfg.logging.wandb
    if wandb_cfg.enabled and wandb_cfg.mode != "disabled" and not dry_run:
        try:
            import wandb
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError(
                "wandb is enabled but not installed. Install with `pip install wandb`."
            ) from exc

        tags = list(wandb_cfg.tags) if wandb_cfg.tags else None
        wandb_run = wandb.init(
            project=wandb_cfg.project or cfg.logging.project,
            entity=wandb_cfg.entity,
            name=wandb_cfg.run_name or run_dir.name,
            mode=wandb_cfg.mode,
            config=cfg.to_dict(),
            tags=tags,
        )
        cfg_path = run_dir / "config_original.yaml"
        if cfg_path.exists():
            artifact = wandb.Artifact(f"{run_dir.name}-config", type="config")
            artifact.add_file(str(cfg_path), name="config_original.yaml")
            wandb_run.log_artifact(artifact)
        else:
            logging.getLogger(__name__).info(
                "config_original.yaml not found; skipping W&B artifact."
            )
        return wandb_run
    if dry_run and wandb_cfg.enabled:
        logging.getLogger(__name__).info("dry_run: skipping W&B initialization.")
    return None


def _maybe_start_profile(cfg: Config, *, run_dir: Path) -> bool:
    """Start profiling if enabled; returns True if started.

    :param Config cfg: Training configuration.
    :param Path run_dir: Run directory path.
    :return bool: True if profiling was started.
    """
    if cfg.train.profile:
        trace_dir = cfg.train.profile_dir or str(run_dir / "trace")
        Path(trace_dir).mkdir(parents=True, exist_ok=True)
        start_trace(trace_dir)
        return True
    return False


def _build_model_state(
    cfg: Config,
) -> tuple[
    Any, Any, optax.GradientTransformation, Callable[[jax.Array], jax.Array], TrainState, Any
]:
    """Build model, optimizer, and initial TrainState.

    :param Config cfg: Training configuration.
    :return tuple[Any, Any, optax.GradientTransformation, Callable[[jax.Array], jax.Array], TrainState, Any]:
        Params, static, optimizer, LR schedule, train state, and abstract state.
    """
    key = jax.random.PRNGKey(cfg.train.seed)
    key, k_model = jax.random.split(key)
    params, static = build_model(cfg, key=k_model)
    tx, schedule = build_optimizer(cfg, params)
    state0 = init_train_state(cfg, params=params, tx=tx, key=key)
    abstract_state = abstractify_tree(state0)
    return params, static, tx, schedule, state0, abstract_state


def _build_checkpoint_manager(cfg: Config, run_dir: Path) -> Any | None:
    """Create checkpoint manager when enabled.

    :param Config cfg: Training configuration.
    :param Path run_dir: Run directory path.
    :return Any | None: Checkpoint manager or None if disabled.
    """
    if not cfg.checkpoint.enabled:
        return None
    if cfg.checkpoint.root_dir:
        ckpt_dir = Path(cfg.checkpoint.root_dir)
        if not ckpt_dir.is_absolute():
            ckpt_dir = run_dir / ckpt_dir
    else:
        ckpt_dir = default_ckpt_dir(run_dir)
    return make_manager(
        ckpt_dir,
        max_to_keep=cfg.checkpoint.max_to_keep,
        save_every=cfg.checkpoint.save_every,
        async_save=cfg.checkpoint.async_save,
    )


def _maybe_restore_state(
    *,
    resume: Literal["none", "latest"] | int,
    manager: Any | None,
    state0: TrainState,
    abstract_state: Any,
    data_it: Any,
    cfg: Config,
    tokenizer_hash: str | None,
) -> tuple[TrainState, dict[str, Any] | None]:
    """Restore state if requested, otherwise return the initial state.

    :param Literal["none", "latest"] | int resume: Resume selector.
    :param Any | None manager: Checkpoint manager.
    :param TrainState state0: Initial state.
    :param Any abstract_state: Abstract train state for restore shape.
    :param Any data_it: Data iterator to restore.
    :param Config cfg: Training configuration.
    :param str | None tokenizer_hash: Optional tokenizer snapshot hash for resume checks.
    :return tuple: (TrainState, meta) where meta is checkpoint metadata if restored.
    """
    if resume == "none":
        return state0, None
    if manager is None:
        raise RuntimeError("resume requested but checkpointing is disabled")

    if resume == "latest":
        step_r, state, meta = restore_latest(
            manager, abstract_train_state=abstract_state, data_iter=data_it
        )
    else:
        step_r, state, meta = restore_at_step(
            manager, step=int(resume), abstract_train_state=abstract_state, data_iter=data_it
        )

    print(f"[chomp] resumed from checkpoint step {step_r}")
    check_resume_compat(cfg, meta, tokenizer_snapshot_hash=tokenizer_hash)
    return state, meta


def _trim_trailing_token(tokens: list[int], token_id: int | None) -> list[int]:
    """Trim trailing token_id values from a token list.

    :param list[int] tokens: Token list to trim.
    :param int | None token_id: Token id to remove from the tail.
    :return list[int]: Trimmed token list.
    """
    if token_id is None:
        return tokens
    end = len(tokens)
    while end > 0 and tokens[end - 1] == token_id:
        end -= 1
    return tokens[:end]


def _select_prompt_tokens(
    tokens: list[int],
    *,
    input_len: int,
    eos_token_id: int | None,
    rng: random.Random,
) -> list[int]:
    """Select a prompt slice from tokenized text.

    :param list[int] tokens: Tokenized text.
    :param int input_len: Target prompt length.
    :param int | None eos_token_id: EOS token id for trimming, if any.
    :param random.Random rng: RNG used to choose prefix/suffix.
    :return list[int]: Prompt token slice.
    """
    tokens = _trim_trailing_token(tokens, eos_token_id)
    if not tokens:
        return []
    if len(tokens) <= input_len:
        return tokens
    if rng.random() < 0.5:
        return tokens[:input_len]
    return tokens[-input_len:]


def _safe_decode(tokenizer: Any, tokens: list[int], *, label: str) -> str:
    """Decode tokens with best-effort logging.

    :param Any tokenizer: Tokenizer with a decode method.
    :param list[int] tokens: Tokens to decode.
    :param str label: Label for error logging context.
    :return str: Decoded text or placeholder on failure.
    """
    try:
        return tokenizer.decode(tokens, skip_special_tokens=True)
    except Exception as exc:
        logger.warning("Generation %s decode failed: %s", label, exc)
        return "<decode failed>"


def _emit_generation_output(
    *,
    step: int,
    prompt_text: str,
    generated_text: str,
    use_rich: bool,
) -> None:
    """Print a generation sample to the console.

    :param int step: Training step number.
    :param str prompt_text: Prompt text to display.
    :param str generated_text: Generated continuation text.
    :param bool use_rich: Whether to render Rich panels.
    """
    if use_rich:
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.rule import Rule
        except Exception:
            use_rich = False
        else:
            console = Console()
            with tqdm.external_write_mode(file=sys.stdout, nolock=False):
                console.print(Rule(f"Step {step} | Generation"))
                console.print(Panel(prompt_text, title="Prompt", style="cyan"))
                console.print(Panel(generated_text, title="Generated", style="magenta"))
            return

    bar = "=" * 50
    tqdm.write(f"{bar} Step {step} {bar}")
    tqdm.write(f"Prompt: {prompt_text}")
    tqdm.write(f"Generated: {generated_text}")


def _format_console_row(
    *,
    step: int,
    loss: float,
    grad_norm: float,
    lr: float,
    lr_muon: float | None,
    step_time_s: float,
    tokens_per_sec: float,
    eval_loss: float | None,
    packing_util: float | None,
    device_mem_gb: float | None,
    peak_mem_gb: float | None,
) -> str:
    """Format a concise console line with key training metrics.

    :param int step: Training step number.
    :param float loss: Training loss value.
    :param float grad_norm: Gradient norm value.
    :param float lr: Learning rate value.
    :param float | None lr_muon: Optional Muon learning rate.
    :param float step_time_s: Step wall time in seconds.
    :param float tokens_per_sec: Throughput in tokens per second.
    :param float | None eval_loss: Optional eval loss.
    :param float | None packing_util: Optional packing utilization.
    :param float | None device_mem_gb: Optional device memory usage.
    :param float | None peak_mem_gb: Optional peak memory usage.
    :return str: Formatted console line.
    """

    parts = [
        f"step {step}",
        f"loss {loss:.4f}",
        f"grad {grad_norm:.2e}",
        f"lr {lr:.2e}",
        f"time {step_time_s:.3f}s",
        f"tok/s {tokens_per_sec:.0f}",
    ]
    if lr_muon is not None:
        parts.append(f"muon_lr {lr_muon:.2e}")
    if eval_loss is not None:
        parts.append(f"eval {eval_loss:.4f}")
    if packing_util is not None:
        parts.append(f"pack {packing_util:.3f}")
    if device_mem_gb is not None:
        parts.append(f"mem {device_mem_gb:.1f}GB")
    if peak_mem_gb is not None:
        parts.append(f"peak {peak_mem_gb:.1f}GB")
    return " | ".join(parts)


def _weight_decay_mask(params: Any) -> Any:
    """Apply weight decay to matrix-like parameters (ndim >= 2).

    :param Any params: Parameter pytree.
    :return Any: Boolean mask pytree with True for parameters that should have weight decay.
    """

    def mask_one(x: Any) -> bool:
        """Return True if x is a tensor with ndim >= 2.

        :param Any x: Leaf value from parameter pytree.
        :return bool: True if weight decay should apply.
        """
        if not hasattr(x, "ndim"):
            return False
        return x.ndim >= 2

    return jax.tree_util.tree_map(mask_one, params)


_MUON_WEIGHT_WHITELIST = (
    ".attn.wz.weight",
    ".attn.wv.weight",
    ".attn.wr.weight",
    ".attn.wh1.weight",
    ".attn.wh2.weight",
    ".ffn.fc1.weight",
    ".ffn.fc2.weight",
    ".ffn.fc3.weight",
    ".lm_head.weight",
)


def _flush_loggers() -> None:
    """Flush all log handlers to ensure crash logs are written."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        with contextlib.suppress(Exception):
            handler.flush()


def _path_to_str(path: tuple[Any, ...]) -> str:
    """Convert a JAX tree path to a dotted string.

    :param tuple[Any, ...] path: Path elements from tree_flatten_with_path.
    :return str: Dotted path string (with list indices in brackets).
    """
    parts: list[str] = []
    for key in path:
        if hasattr(key, "name"):
            parts.append(str(key.name))
        elif hasattr(key, "key"):
            parts.append(str(key.key))
        elif hasattr(key, "idx"):
            parts.append(f"[{key.idx}]")
        else:
            parts.append(str(key))
    return ".".join(parts)


def _is_muon_weight_path(path_str: str) -> bool:
    """Return True if a path refers to a muon-eligible weight matrix.

    :param str path_str: Dotted parameter path.
    :return bool: True if the path should use Muon.
    """
    if not path_str.endswith(".weight"):
        return False
    return any(token in path_str for token in _MUON_WEIGHT_WHITELIST)


def _muon_param_stats(params: Any, *, allow_all_2d: bool) -> tuple[int, int, int, int, list[str]]:
    """Return Muon/Adam tensor counts and a sample of Muon paths.

    :param Any params: Parameter pytree.
    :param bool allow_all_2d: If True, apply Muon to all 2D tensors.
    :return tuple: (muon_tensors, adam_tensors, muon_2d, total_2d, muon_paths).
    """
    flat, _ = jax.tree_util.tree_flatten_with_path(params)
    total_tensors = 0
    total_2d = 0
    muon_tensors = 0
    muon_2d = 0
    muon_paths: list[str] = []
    for path, leaf in flat:
        if not hasattr(leaf, "ndim"):
            continue
        total_tensors += 1
        if leaf.ndim == 2:
            total_2d += 1
        path_str = _path_to_str(path)
        use_muon = leaf.ndim == 2 and (allow_all_2d or _is_muon_weight_path(path_str))
        if use_muon:
            muon_tensors += 1
            muon_2d += 1
            muon_paths.append(path_str)
    adam_tensors = total_tensors - muon_tensors
    return muon_tensors, adam_tensors, muon_2d, total_2d, muon_paths


def _muon_weight_dim_numbers(params: Any, *, allow_all_2d: bool) -> Any:
    """Return Muon dimension specs for eligible parameters.

    :param Any params: Parameter pytree.
    :param bool allow_all_2d: If True, apply Muon to all 2D tensors.
    :return Any: Pytree of MuonDimensionNumbers (muon) or None (adam).
    """
    flat, treedef = jax.tree_util.tree_flatten_with_path(params)
    dim_nums: list[Any] = []
    for path, leaf in flat:
        if not hasattr(leaf, "ndim") or leaf.ndim != 2:
            dim_nums.append(None)
            continue
        if allow_all_2d:
            dim_nums.append(optax.contrib.MuonDimensionNumbers())
            continue
        path_str = _path_to_str(path)
        dim_nums.append(
            optax.contrib.MuonDimensionNumbers() if _is_muon_weight_path(path_str) else None
        )
    return treedef.unflatten(dim_nums)


def _muon_lr_from_adam(lr_adam: jax.Array, cfg: Config) -> jax.Array:
    """Return the Muon learning rate derived from the AdamW schedule.

    :param jax.Array lr_adam: AdamW learning rate for the current step.
    :param Config cfg: Training configuration.
    :return jax.Array: Muon learning rate (scaled).
    """
    return lr_adam * cfg.optim.muon_lr_scale


def _scale_updates(mask_fn: Callable[[Any], Any], scale: float) -> optax.GradientTransformation:
    """Scale masked updates by a fixed factor.

    :param Callable mask_fn: Function returning a boolean mask pytree.
    :param float scale: Scaling factor to apply where mask is True.
    :return optax.GradientTransformation: Stateless scaling transform.
    """

    def init_fn(params: Any) -> optax.EmptyState:
        del params
        return optax.EmptyState()

    def update_fn(
        updates: Any, state: optax.EmptyState, params: Any | None = None
    ) -> tuple[Any, optax.EmptyState]:
        mask_source = params if params is not None else updates
        mask = mask_fn(mask_source)

        def _apply_scale(m: bool, g: Any) -> Any:
            if g is None:
                return None
            return g * scale if m else g

        updates = jax.tree.map(_apply_scale, mask, updates, is_leaf=lambda node: node is None)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def build_optimizer(
    cfg: Config, params: Any
) -> tuple[optax.GradientTransformation, Callable[[jax.Array], jax.Array]]:
    """Create Optax optimizer + schedule function (for logging).

    :param Config cfg: Training configuration.
    :param Any params: Model parameters (used to build weight decay mask).
    :return tuple: (optimizer, lr_schedule) where lr_schedule maps step to learning rate.
    """

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.optim.lr,
        warmup_steps=cfg.optim.warmup_steps,
        decay_steps=resolve_decay_horizon(cfg),
        end_value=cfg.optim.lr * cfg.optim.min_lr_ratio,
    )

    transforms = []
    if cfg.optim.grad_clip_norm and cfg.optim.grad_clip_norm > 0:
        transforms.append(optax.clip_by_global_norm(cfg.optim.grad_clip_norm))

    if cfg.optim.name == "muon":
        allow_all_2d = cfg.optim.muon_allow_all_2d
        if cfg.model.backend != "megalodon":
            allow_all_2d = True
        muon_tensors, adam_tensors, muon_2d, total_2d, muon_paths = _muon_param_stats(
            params, allow_all_2d=allow_all_2d
        )
        logger.info(
            "Muon param split: %s muon / %s adam tensors; 2D coverage %s/%s",
            muon_tensors,
            adam_tensors,
            muon_2d,
            total_2d,
        )
        if muon_paths:
            sample = ", ".join(muon_paths[:5])
            logger.info("Muon sample params: %s", sample)

        if muon_2d == 0:
            logger.warning(
                "optim.name=muon selected but no muon-eligible parameters were found; "
                "falling back to AdamW for all parameters."
            )

        def label_fn(tree: Any) -> Any:
            """Return optimizer labels for each parameter leaf."""
            flat, treedef = jax.tree_util.tree_flatten_with_path(tree)
            labels: list[str] = []
            for path, leaf in flat:
                if not hasattr(leaf, "ndim"):
                    labels.append("adam")
                    continue
                path_str = _path_to_str(path)
                use_muon = leaf.ndim == 2 and (allow_all_2d or _is_muon_weight_path(path_str))
                labels.append("muon" if use_muon else "adam")
            return treedef.unflatten(labels)

        def muon_schedule(step: jax.Array) -> jax.Array:
            return _muon_lr_from_adam(schedule(step), cfg)

        def muon_dim_fn(tree: Any) -> Any:
            """Return Muon dimension numbers for masked Muon parameters."""
            return _muon_weight_dim_numbers(tree, allow_all_2d=True)

        muon_tx = optax.chain(
            optax.contrib.scale_by_muon(
                beta=cfg.optim.muon_momentum,
                ns_steps=cfg.optim.muon_ns_steps,
                nesterov=cfg.optim.muon_nesterov,
                weight_dimension_numbers=muon_dim_fn,
            ),
            optax.add_decayed_weights(cfg.optim.weight_decay, mask=_weight_decay_mask),
            optax.scale_by_learning_rate(muon_schedule),
        )
        adam_tx = optax.adamw(
            learning_rate=schedule,
            weight_decay=cfg.optim.weight_decay,
            mask=_weight_decay_mask,
        )
        transforms.append(optax.multi_transform({"muon": muon_tx, "adam": adam_tx}, label_fn))
    else:
        transforms.append(
            optax.adamw(
                learning_rate=schedule,
                weight_decay=cfg.optim.weight_decay,
                mask=_weight_decay_mask,
            )
        )

    tx = optax.chain(*transforms)
    return tx, schedule


def init_train_state(
    cfg: Config, *, params: Any, tx: optax.GradientTransformation, key: jax.Array
) -> TrainState:
    """Initialize a fresh TrainState at step 0.

    :param Config cfg: Training configuration (unused but kept for API consistency).
    :param Any params: Model parameters.
    :param optax.GradientTransformation tx: Optimizer transform.
    :param jax.Array key: PRNG key for dropout.
    :return TrainState: Initialized training state.
    """
    opt_state = tx.init(params)
    return TrainState(
        step=jnp.array(0, dtype=jnp.int32), params=params, opt_state=opt_state, rng=key
    )


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

    :param Config cfg: Training configuration.
    :param Any static: Static (non-differentiable) model components from eqx.partition.
    :param optax.GradientTransformation tx: Optimizer transform.
    :param lr_schedule: Function mapping step number to learning rate.
    :return Callable: Compiled train_step(state, batch) -> (new_state, metrics).
    """

    deterministic = derived_deterministic(cfg)
    grad_accum = int(cfg.train.grad_accum)

    def micro_loss(
        params: Any,
        input_ids: jax.Array,
        labels: jax.Array,
        attn: jax.Array,
        segs: jax.Array,
        key: jax.Array | None,
        token_count: jax.Array,
    ) -> jax.Array:
        """Compute token-weighted loss for a single micro-batch.

        :param Any params: Model parameters.
        :param jax.Array input_ids: Input token IDs [B, T].
        :param jax.Array labels: Label token IDs [B, T].
        :param jax.Array attn: Attention mask [B, T].
        :param jax.Array segs: Segment IDs [B, T].
        :param key: PRNG key for dropout, or None if deterministic.
        :param jax.Array token_count: Number of valid tokens for weighting.
        :return jax.Array: Weighted loss scalar.
        """
        micro = Batch(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attn.astype(bool),
            segment_ids=segs.astype(jnp.int32),
        )
        loss = training_loss(
            params,
            static,
            batch=micro,
            deterministic=deterministic,
            key=key,
        )
        return loss * token_count

    loss_and_grad = eqx.filter_value_and_grad(micro_loss)

    def train_step(state: TrainState, batch: Batch) -> tuple[TrainState, dict[str, jax.Array]]:
        """Execute one training step with scan-based gradient accumulation.

        :param TrainState state: Current training state.
        :param Batch batch: Input batch of shape [A, B, T].
        :return tuple: (new_state, metrics_dict).
        """
        # Split RNG: one for next state, one to generate per-micro dropout keys
        rng, step_key = jax.random.split(state.rng)
        micro_keys = jax.random.split(step_key, grad_accum)

        # Init accumulators
        loss0 = jnp.zeros((), dtype=jnp.float32)
        grad0 = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), state.params)
        token0 = jnp.zeros((), dtype=jnp.float32)

        def body(
            carry: tuple[jax.Array, Any, jax.Array],
            inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
        ) -> tuple[tuple[jax.Array, Any, jax.Array], None]:
            """Scan body: accumulate loss and gradients for one micro-batch.

            :param tuple carry: (loss_sum, grad_sum, token_sum) accumulators.
            :param tuple inputs: (input_ids, labels, attn, segs, key) for one micro-batch.
            :return tuple: (updated_carry, None).
            """
            loss_sum, grad_sum, token_sum = carry
            in_ids, labs, attn, segs, k = inputs
            token_count = _count_tokens(labs, attn).astype(jnp.float32)
            loss, grads = loss_and_grad(state.params, in_ids, labs, attn, segs, k, token_count)
            loss_sum = loss_sum + loss.astype(jnp.float32)
            grad_sum = jax.tree_util.tree_map(lambda a, b: a + b, grad_sum, grads)
            token_sum = token_sum + token_count
            return (loss_sum, grad_sum, token_sum), None

        (loss_sum, grad_sum, token_sum), _ = jax.lax.scan(
            body,
            (loss0, grad0, token0),
            (
                batch.input_ids,
                batch.labels,
                batch.attention_mask,
                batch.segment_ids,
                micro_keys,
            ),
        )

        token_denom = jnp.maximum(token_sum, 1.0)
        loss = loss_sum / token_denom
        grads = jax.tree_util.tree_map(lambda g: g / token_denom, grad_sum)

        grad_norm = optax.global_norm(grads)
        updates, new_opt_state = tx.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)

        new_state = TrainState(
            step=state.step + 1, params=new_params, opt_state=new_opt_state, rng=rng
        )

        lr = lr_schedule(state.step)
        metrics = {
            "loss": loss,
            "grad_norm": grad_norm.astype(jnp.float32),
            "lr": lr.astype(jnp.float32),
            "token_sum": token_sum.astype(jnp.float32),
        }
        return new_state, metrics

    if cfg.train.jit:
        train_step = eqx.filter_jit(train_step)
    return train_step


def make_eval_step(
    cfg: Config, *, static: Any
) -> Callable[[Any, Batch], tuple[jax.Array, jax.Array]]:
    """Build a compiled evaluation step.

    :param Config cfg: Training configuration.
    :param Any static: Static (non-differentiable) model components from eqx.partition.
    :return Callable: eval_step(params, batch) -> (loss_sum, token_sum).
    """

    def eval_step(params: Any, batch: Batch) -> tuple[jax.Array, jax.Array]:
        """Compute token-weighted loss sums for a batch.

        :param Any params: Model parameters.
        :param Batch batch: Input batch of shape [A, B, T].
        :return tuple: (loss_sum, token_sum) for the batch.
        """
        loss0 = jnp.zeros((), dtype=jnp.float32)
        token0 = jnp.zeros((), dtype=jnp.float32)

        def body(
            carry: tuple[jax.Array, jax.Array], xs: tuple[jax.Array, ...]
        ) -> tuple[tuple[jax.Array, jax.Array], None]:
            """Scan body that accumulates loss and token counts for eval.

            :param tuple carry: (loss_sum, token_sum) accumulators.
            :param tuple xs: (input_ids, labels, attn, segs) microbatch inputs.
            :return tuple: (updated_carry, None).
            """
            loss_sum, token_sum = carry
            input_ids, labels, attn, segs = xs
            micro = Batch(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attn.astype(bool),
                segment_ids=segs.astype(jnp.int32),
            )
            token_count = _count_tokens(labels, attn).astype(jnp.float32)
            loss = training_loss(
                params,
                static,
                batch=micro,
                deterministic=True,
                key=None,
            )
            return (loss_sum + loss * token_count, token_sum + token_count), None

        (loss_sum, token_sum), _ = jax.lax.scan(
            body,
            (loss0, token0),
            (
                batch.input_ids,
                batch.labels,
                batch.attention_mask,
                batch.segment_ids,
            ),
        )
        return loss_sum, token_sum

    if cfg.train.jit:
        eval_step = eqx.filter_jit(eval_step)
    return eval_step


def run(
    cfg: Config,
    *,
    config_path: str | None = None,
    resume: Literal["none", "latest"] | int = "none",
    dry_run: bool = False,
    max_steps: int | None = None,
) -> Path:
    """Run a training job and return the run directory.

    Resume contract:
    - resume requires logging.run_dir to be set (existing run directory)
    - we restore both train_state and data iterator state when present

    :param Config cfg: Fully validated training configuration.
    :param config_path: Optional path to the source YAML config file.
    :param resume: Resume mode - "none" (fresh), "latest", or specific step number.
    :param bool dry_run: If True, compile and run a single step, then exit early.
    :param max_steps: Optional cap on steps for this invocation (<= train.steps).
    :raises RuntimeError: If resume requested but checkpointing is disabled.
    :return Path: Path to the run directory.
    """

    if dry_run and resume != "none":
        raise RuntimeError("dry_run does not support resume; use a fresh run.")

    allow_existing = resume != "none"

    (
        cfg,
        tokenizer,
        run_dir,
        metrics_path,
        eval_tokens,
        gen_settings,
        gen_stream,
        gen_key,
        gen_rng,
    ) = _setup_run_dir_and_tokenizer(
        cfg,
        config_path=config_path,
        allow_existing=allow_existing,
        dry_run=dry_run,
    )
    if cfg.model.use_checkpoint and derived_deterministic(cfg):
        logger.warning(
            "train.deterministic=true disables activation checkpointing in megalodon-jax. "
            "Set train.deterministic=false (and keep dropout at 0.0 for deterministic math) "
            "to enable checkpointing."
        )
    tokenizer_hash = tokenizer_snapshot_hash(run_dir)

    wandb_run = _maybe_init_wandb(cfg, run_dir=run_dir, dry_run=dry_run)
    profile_enabled = _maybe_start_profile(cfg, run_dir=run_dir)

    params, static, tx, schedule, state0, abstract_state = _build_model_state(cfg)

    # Log param count once
    n_params = param_count(params)
    print(f"[chomp] params: {n_params:,}")

    # Data iterator (host-side)
    data_it = build_train_iterator(cfg, tokenizer=tokenizer)

    # Checkpoint manager
    manager = _build_checkpoint_manager(cfg, run_dir)

    # Restore if requested
    state, resume_meta = _maybe_restore_state(
        resume=resume,
        manager=manager,
        state0=state0,
        abstract_state=abstract_state,
        data_it=data_it,
        cfg=cfg,
        tokenizer_hash=tokenizer_hash,
    )

    train_step = make_train_step(cfg, static=static, tx=tx, lr_schedule=schedule)
    eval_every = int(cfg.train.eval_every)
    eval_step = None
    if eval_tokens and eval_every > 0:
        eval_step = make_eval_step(cfg, static=static)

    if dry_run:
        try:
            batch = next(data_it)
        except StopIteration as exc:
            raise RuntimeError("dry_run: data iterator exhausted before first batch") from exc
        data_stats = data_it.get_stats()
        if not cfg.data.device_put:
            batch = jax.device_put(batch)
        if cfg.debug.check_device_every > 0:
            assert_batch_on_device(batch, allow_cpu=cfg.train.allow_cpu)

        t1 = time.perf_counter()
        state, metrics = train_step(state, batch)
        metrics_host = jax.device_get(metrics)
        t2 = time.perf_counter()
        step_time_s = t2 - t1

        step_i = int(jax.device_get(state.step))
        if cfg.debug.nan_check:
            _check_finite_metrics(metrics_host, step=step_i)

        token_sum = float(metrics_host.get("token_sum", 0.0))
        tokens_per_sec = token_sum / step_time_s if step_time_s > 0 else 0.0
        lr_adam = float(metrics_host["lr"])
        lr_muon = _muon_lr_from_adam(lr_adam, cfg) if cfg.optim.name == "muon" else None
        mem_stats = _device_memory_stats_gb()
        packing_util = (
            float(data_stats["packing_utilization"])
            if data_stats and "packing_utilization" in data_stats
            else None
        )
        device_mem = float(mem_stats.get("device_memory_gb")) if mem_stats else None
        peak_mem = float(mem_stats.get("peak_memory_gb")) if mem_stats else None
        console_line = _format_console_row(
            step=step_i,
            loss=float(metrics_host["loss"]),
            grad_norm=float(metrics_host["grad_norm"]),
            lr=lr_adam,
            lr_muon=lr_muon,
            step_time_s=float(step_time_s),
            tokens_per_sec=float(tokens_per_sec),
            eval_loss=None,
            packing_util=packing_util,
            device_mem_gb=device_mem,
            peak_mem_gb=peak_mem,
        )
        print("[chomp] dry-run complete")
        print(console_line)

        if wandb_run is not None:
            wandb_run.finish()
        if manager is not None:
            manager.wait_until_finished()
        if profile_enabled:
            stop_trace()
        return run_dir

    # Training loop
    t_compile = None
    t0 = time.perf_counter()

    # Determine starting step from TrainState
    start_step = int(jax.device_get(state.step))
    target_steps = cfg.train.steps
    if max_steps is not None:
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")
        target_steps = min(target_steps, int(max_steps))

    if start_step >= target_steps:
        print(f"[chomp] start_step ({start_step}) >= target steps ({target_steps}); nothing to do")
        return run_dir

    console_every = int(cfg.train.log_every)
    host_step = int(start_step)
    tokens_seen_base = 0.0
    if resume_meta and resume_meta.get("tokens_seen") is not None:
        tokens_seen_base = float(resume_meta["tokens_seen"])
    tokens_seen_device = jnp.asarray(tokens_seen_base, dtype=jnp.float32)

    def _run_eval(params: Any) -> dict[str, Any]:
        """Run a full eval pass over the cached eval texts.

        :param Any params: Model parameters.
        :return dict[str, Any]: Eval metrics row with eval_loss and eval_tokens.
        """
        if eval_step is None or not eval_tokens:
            return {}
        total_loss = 0.0
        total_tokens = 0.0
        eval_it = build_eval_iterator(cfg, tokens=eval_tokens, tokenizer=tokenizer)
        while True:
            try:
                eval_batch = next(eval_it)
            except StopIteration:
                break
            if not cfg.data.device_put:
                eval_batch = jax.device_put(eval_batch)
            loss_sum, token_sum = eval_step(params, eval_batch)
            loss_sum_host, token_sum_host = jax.device_get((loss_sum, token_sum))
            total_loss += float(loss_sum_host)
            total_tokens += float(token_sum_host)

        if total_tokens <= 0:
            return {"eval_loss": None, "eval_tokens": 0}
        return {"eval_loss": total_loss / total_tokens, "eval_tokens": int(total_tokens)}

    def _run_generation_sample(step: int, params: Any) -> None:
        """Sample a prompt and run generation.

        :param int step: Current training step.
        :param Any params: Model parameters.
        """
        nonlocal gen_key, gen_rng, gen_settings, gen_stream
        if gen_settings is None or gen_stream is None or gen_rng is None:
            return
        try:
            item = next(gen_stream)
        except StopIteration:
            logger.warning("Generation stream exhausted; disabling generation.")
            gen_settings = None
            gen_stream = None
            return

        text = item if isinstance(item, str) else str(item)
        tokens = tokenizer.encode(text)
        prompt_tokens = _select_prompt_tokens(
            tokens,
            input_len=gen_settings.input_len,
            eos_token_id=int(cfg.model.eos_token_id),
            rng=gen_rng,
        )
        if not prompt_tokens:
            logger.debug("Generation prompt empty at step %d; skipping.", step)
            return

        try:
            from megalodon_jax import generate as mega_generate
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            logger.warning("Generation requested but megalodon_jax unavailable: %s", exc)
            gen_settings = None
            return

        model = eqx.combine(params, static)
        needs_key = gen_settings.temperature is None or gen_settings.temperature > 0
        key = gen_key if needs_key else None
        gen_kwargs: dict[str, Any] = {
            "bos_token_id": int(cfg.model.bos_token_id),
            "eos_token_id": int(cfg.model.eos_token_id),
        }
        if gen_settings.temperature is not None:
            gen_kwargs["temperature"] = float(gen_settings.temperature)
        if gen_settings.top_k is not None:
            gen_kwargs["top_k"] = int(gen_settings.top_k)
        if gen_settings.top_p is not None:
            gen_kwargs["top_p"] = float(gen_settings.top_p)

        prompt_ids = jnp.asarray(prompt_tokens, dtype=jnp.int32)[None, :]
        try:
            output_ids, _cache, next_key = mega_generate(
                model,
                prompt_ids,
                gen_settings.max_new_tokens,
                key=key,
                **gen_kwargs,
            )
        except Exception as exc:
            logger.warning("Generation failed at step %d: %s", step, exc)
            return

        if next_key is not None:
            gen_key = next_key

        output_host = jax.device_get(output_ids)
        output_tokens = [int(x) for x in output_host[0].tolist()]
        gen_tokens = output_tokens[len(prompt_tokens) :]

        prompt_text = _safe_decode(tokenizer, prompt_tokens, label="prompt")
        generated_text = _safe_decode(tokenizer, gen_tokens, label="generated")
        _emit_generation_output(
            step=step,
            prompt_text=prompt_text,
            generated_text=generated_text,
            use_rich=cfg.logging.console_use_rich,
        )

    exit_code = 0
    crash_reason = None
    crash_type = None
    crash_step = None

    try:
        with MetricsWriter(metrics_path) as mw:
            try:
                for _ in tqdm(range(start_step, target_steps), desc="train", dynamic_ncols=True):
                    # Fetch batch (host) and (optionally) device_put
                    try:
                        batch = next(data_it)
                    except StopIteration:
                        tokens_seen_host = int(jax.device_get(tokens_seen_device))
                        step_i = int(host_step)
                        row = {
                            "step": int(step_i),
                            "data_exhausted": True,
                            "tokens_seen": int(tokens_seen_host),
                            "wall_time_s": time.perf_counter() - t0,
                        }
                        mw.write(row)
                        if wandb_run is not None:
                            wandb_run.log(row, step=step_i)
                        print("[chomp] data exhausted; stopping early")
                        break
                    data_stats = data_it.get_stats()
                    if not cfg.data.device_put:
                        batch = jax.device_put(batch)

                    # Batch placement validation (real check)
                    if cfg.debug.check_device_every > 0 and (
                        host_step % cfg.debug.check_device_every == 0
                    ):
                        assert_batch_on_device(batch, allow_cpu=cfg.train.allow_cpu)

                    # Step (compile happens on first call)
                    with step_annotation("train_step"):
                        t1 = time.perf_counter()
                        state, metrics = train_step(state, batch)
                        tokens_seen_device = tokens_seen_device + metrics["token_sum"]

                    host_step += 1
                    step_i = int(host_step)

                    should_eval = (
                        eval_step is not None and eval_every > 0 and (step_i % eval_every) == 0
                    )
                    should_log = (step_i % cfg.train.log_every) == 0 or should_eval
                    should_console = (step_i % console_every) == 0 or should_eval
                    should_sync = should_log or should_console or (t_compile is None)

                    metrics_host = None
                    tokens_seen_host = None
                    step_time_s = None
                    tokens_per_sec = 0.0

                    if should_sync:
                        metrics_host, tokens_seen_host = jax.device_get(
                            (metrics, tokens_seen_device)
                        )
                        t2 = time.perf_counter()
                        step_time_s = t2 - t1
                        if t_compile is None:
                            t_compile = step_time_s
                        if cfg.debug.nan_check:
                            _check_finite_metrics(metrics_host, step=step_i)
                        token_sum = float(metrics_host.get("token_sum", 0.0))
                        tokens_per_sec = token_sum / step_time_s if step_time_s > 0 else 0.0

                    # Checkpoint save (after state updated)
                    if manager is not None and (step_i % int(cfg.checkpoint.save_every) == 0):
                        tokens_seen_ckpt = tokens_seen_host
                        if tokens_seen_ckpt is None:
                            tokens_seen_ckpt = int(jax.device_get(tokens_seen_device))
                        meta = build_meta(
                            step=step_i,
                            config=cfg.to_dict(),
                            data_fingerprint=data_fingerprint(
                                cfg, tokenizer_snapshot_hash=tokenizer_hash
                            ),
                            tokens_seen=tokens_seen_ckpt,
                        )
                        save(
                            manager,
                            step=step_i,
                            train_state=state,
                            data_iter=data_it,
                            meta=meta,
                        )

                    eval_row: dict[str, Any] = {}
                    if should_eval:
                        eval_row = _run_eval(state.params)

                    if (
                        gen_settings is not None
                        and gen_stream is not None
                        and (step_i % gen_settings.every) == 0
                    ):
                        _run_generation_sample(step_i, state.params)

                    # Log
                    mem_stats = _device_memory_stats_gb() if (should_log or should_console) else {}
                    if should_log:
                        if metrics_host is None:
                            metrics_host, tokens_seen_host = jax.device_get(
                                (metrics, tokens_seen_device)
                            )
                        if tokens_seen_host is None:
                            tokens_seen_host = int(jax.device_get(tokens_seen_device))
                        lr_adam = float(metrics_host["lr"])
                        lr_muon = (
                            _muon_lr_from_adam(lr_adam, cfg) if cfg.optim.name == "muon" else None
                        )
                        row = {
                            "step": int(step_i),
                            "loss": float(metrics_host["loss"]),
                            "grad_norm": float(metrics_host["grad_norm"]),
                            "lr": lr_adam,
                            "step_time_s": float(step_time_s),
                            "tokens_per_sec": float(tokens_per_sec),
                            "tokens_seen": int(tokens_seen_host),
                            "wall_time_s": time.perf_counter() - t0,
                        }
                        if lr_muon is not None:
                            row["lr_muon"] = float(lr_muon)
                        if data_stats:
                            row.update(data_stats)
                        if eval_row:
                            row.update(eval_row)
                        if mem_stats:
                            row.update(mem_stats)
                        if step_i == (start_step + 1) and t_compile is not None:
                            row["first_step_compile_time_s"] = float(t_compile)

                        local_drop = {
                            "wall_time_s",
                            "packing_tokens",
                            "packing_capacity",
                            "eval_tokens",
                            "device_memory_gb",
                        }
                        row_local = {k: v for k, v in row.items() if k not in local_drop}
                        mw.write(row_local)
                        if wandb_run is not None:
                            wandb_drop = {
                                "wall_time_s",
                                "step",
                                "peak_memory_gb",
                                "packing_tokens",
                                "packing_capacity",
                                "eval_tokens",
                                "device_memory_gb",
                            }
                            row_wandb = {k: v for k, v in row.items() if k not in wandb_drop}
                            wandb_run.log(row_wandb, step=step_i)
                    if should_console:
                        if metrics_host is None:
                            metrics_host = jax.device_get(metrics)
                        lr_adam = float(metrics_host["lr"])
                        lr_muon = (
                            _muon_lr_from_adam(lr_adam, cfg) if cfg.optim.name == "muon" else None
                        )
                        eval_loss = eval_row.get("eval_loss") if eval_row else None
                        packing_util = None
                        if data_stats and "packing_utilization" in data_stats:
                            packing_util = float(data_stats["packing_utilization"])
                        device_mem = None
                        peak_mem = None
                        if "device_memory_gb" in mem_stats:
                            device_mem = float(mem_stats["device_memory_gb"])
                        if "peak_memory_gb" in mem_stats:
                            peak_mem = float(mem_stats["peak_memory_gb"])
                        console_line = _format_console_row(
                            step=step_i,
                            loss=float(metrics_host["loss"]),
                            grad_norm=float(metrics_host["grad_norm"]),
                            lr=lr_adam,
                            lr_muon=lr_muon,
                            step_time_s=float(step_time_s),
                            tokens_per_sec=float(tokens_per_sec),
                            eval_loss=float(eval_loss) if eval_loss is not None else None,
                            packing_util=packing_util,
                            device_mem_gb=device_mem,
                            peak_mem_gb=peak_mem,
                        )
                        tqdm.write(console_line)
            except Exception as exc:
                exit_code = 1
                crash_type = type(exc).__name__
                crash_reason = str(exc)
                crash_step = int(host_step) + 1
                logger.exception("Training crashed at step %s", crash_step)
                row = {
                    "step": int(crash_step),
                    "crash": True,
                    "crash_type": crash_type,
                    "crash_reason": crash_reason,
                    "wall_time_s": time.perf_counter() - t0,
                }
                with contextlib.suppress(Exception):
                    tokens_seen_host = int(jax.device_get(tokens_seen_device))
                    row["tokens_seen"] = tokens_seen_host
                mw.write(row)
                if wandb_run is not None:
                    with contextlib.suppress(Exception):
                        wandb_run.summary["crashed"] = True
                        wandb_run.summary["crash_type"] = crash_type
                        wandb_run.summary["crash_reason"] = crash_reason
                        wandb_run.log(
                            {
                                "crash": True,
                                "crash_type": crash_type,
                                "crash_reason": crash_reason,
                            },
                            step=int(crash_step),
                        )
                _flush_loggers()
                raise
    finally:
        if wandb_run is not None:
            with contextlib.suppress(Exception):
                if crash_reason is not None:
                    wandb_run.summary["crashed"] = True
                    wandb_run.summary["crash_type"] = crash_type
                    wandb_run.summary["crash_reason"] = crash_reason
                wandb_run.finish(exit_code=exit_code)

        if manager is not None:
            with contextlib.suppress(Exception):
                manager.wait_until_finished()

        if profile_enabled:
            with contextlib.suppress(Exception):
                stop_trace()

        _flush_loggers()

    return run_dir
