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
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from optax.contrib import _muon as muon_contrib
from tqdm import tqdm

from chomp.ckpt import (
    build_meta,
    check_resume_compat,
    make_manager,
    resolve_ckpt_root,
    restore_at_step,
    restore_meta_at_step,
    save,
)
from chomp.config import (
    Config,
    derived_deterministic,
    resolve_decay_horizon,
    strict_packed_segments,
)
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
from chomp.model import (
    build_model,
    build_parameter_manifest,
    causal_loss_mask,
    generate_tokens,
    is_muon_eligible,
    parameter_decay_mask,
    supports_packed_segments,
    training_loss,
    write_parameter_manifest,
)
from chomp.types import IGNORE_INDEX, Batch, TrainState
from chomp.utils.devices import assert_batch_on_device
from chomp.utils.io import MetricsWriter, add_file_logging, create_run_dir
from chomp.utils.profiling import start_trace, step_annotation, stop_trace
from chomp.utils.tree import abstractify_tree, param_count, path_to_str

logger = logging.getLogger(__name__)


def _count_tokens(labels: jax.Array, attention_mask: jax.Array) -> jax.Array:
    """Count valid tokens after the causal shift (for correct GA normalization).

    :param jax.Array labels: Label tensor of shape [B, T].
    :param jax.Array attention_mask: Mask tensor of shape [B, T].
    :return jax.Array: Scalar count of valid (non-ignored, non-masked) tokens.
    """

    return jnp.sum(
        causal_loss_mask(labels, attention_mask, ignore_index=IGNORE_INDEX),
        dtype=jnp.int32,
    )


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
    list[list[int]] | None,
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
    :return tuple[Config, Any, Path, Path, list[list[int]] | None, GenerationSettings | None, Any | None, jax.Array | None, random.Random | None]:
        Updated config, tokenizer, run/metrics paths, optional deferred eval
        tokens, generation settings, stream, key, and RNG.
    """
    tokenizer = None
    if allow_existing and cfg.logging.run_dir is not None:
        run_dir_hint = Path(cfg.logging.run_dir)
        if not run_dir_hint.exists():
            raise RuntimeError(f"Resume requested but run directory does not exist: {run_dir_hint}")
        tok_dir = run_dir_hint / "tokenizer"
        if not tok_dir.is_dir():
            raise FileNotFoundError(
                f"Resume requested but tokenizer snapshot is missing at {tok_dir}. "
                "Refusing to rebuild it from a mutable tokenizer source or modify the run "
                "directory before checkpoint compatibility is known. Recover the original "
                "snapshot or start a new run."
            )
        tokenizer = load_tokenizer_snapshot(run_dir_hint, cfg)

    cfg, tokenizer = prepare_tokenizer_and_config(cfg, tokenizer=tokenizer)

    run_dir = create_run_dir(cfg, config_path=config_path, allow_existing=allow_existing)
    if cfg.logging.log_file is not None:
        add_file_logging(run_dir / cfg.logging.log_file, level=cfg.logging.level)
    metrics_path = run_dir / cfg.logging.metrics_file
    save_tokenizer_snapshot(run_dir, cfg, tokenizer, allow_existing=allow_existing)

    # run_dir pins the eval set: created once, persisted, and reloaded on
    # resume so evals stay comparable even if the upstream dataset drifts.
    # The recreation override can write a missing artifact, so defer that
    # entire operation until checkpoint compatibility accepts this resume.
    # None is the explicit "deferred" sentinel; [] remains "eval disabled."
    if dry_run:
        eval_tokens = []
    elif allow_existing and cfg.data.recreate_eval_cache:
        eval_tokens = None
    else:
        eval_tokens = load_or_create_eval_texts(
            cfg, tokenizer=tokenizer, run_dir=run_dir, resume=allow_existing
        )

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
    if wandb_cfg.enabled and not dry_run:
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


def _finish_run_telemetry(
    wandb_run: Any | None,
    *,
    profile_enabled: bool,
    exit_code: int = 0,
    crash_type: str | None = None,
    crash_reason: str | None = None,
) -> None:
    """Finish optional telemetry resources without masking run failures.

    :param Any | None wandb_run: Active W&B run, if any.
    :param bool profile_enabled: Whether a profiler trace was started.
    :param int exit_code: Process-style completion code reported to W&B.
    :param str | None crash_type: Exception type recorded for a failed run.
    :param str | None crash_reason: Exception message recorded for a failed run.
    """
    if wandb_run is not None:
        with contextlib.suppress(Exception):
            if crash_reason is not None:
                wandb_run.summary["crashed"] = True
                wandb_run.summary["crash_type"] = crash_type
                wandb_run.summary["crash_reason"] = crash_reason
            wandb_run.finish(exit_code=exit_code)
    if profile_enabled:
        with contextlib.suppress(Exception):
            stop_trace()


def _start_run_telemetry(cfg: Config, *, run_dir: Path, dry_run: bool) -> tuple[Any | None, bool]:
    """Start optional telemetry resources as one failure-safe lifecycle.

    :param Config cfg: Training configuration.
    :param Path run_dir: Run directory path.
    :param bool dry_run: Whether this is a dry run.
    :return tuple[Any | None, bool]: W&B run and profiler-started flag.
    """
    wandb_run = _maybe_init_wandb(cfg, run_dir=run_dir, dry_run=dry_run)
    try:
        return wandb_run, _maybe_start_profile(cfg, run_dir=run_dir)
    except Exception:
        _finish_run_telemetry(wandb_run, profile_enabled=False, exit_code=1)
        raise


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
    state0 = init_train_state(params=params, tx=tx, key=key)
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
    return make_manager(
        resolve_ckpt_root(cfg, run_dir),
        max_to_keep=cfg.checkpoint.max_to_keep,
        save_every=cfg.checkpoint.save_every,
        async_save=cfg.checkpoint.async_save,
    )


def _close_manager_after_failure(manager: Any | None, *, phase: str) -> None:
    """Close a checkpoint manager without masking an active primary failure.

    :param manager: Checkpoint manager, or None when checkpointing is disabled.
    :param str phase: Setup/finalization phase used in secondary-error logging.
    """
    if manager is None:
        return
    try:
        manager.close()
    except Exception:
        logger.exception("Closing the checkpoint manager after %s failure also failed", phase)


def _save_training_checkpoint(
    manager: Any,
    *,
    step: int,
    cfg: Config,
    tokenizer_hash: str | None,
    parameter_manifest_hash: str,
    tokens_seen: int,
    train_state: TrainState,
    data_iter: Any,
    force: bool = False,
) -> None:
    """Save train/data state with the standard chomp checkpoint metadata.

    :param manager: Checkpoint manager.
    :param int step: Completed training step to save.
    :param Config cfg: Training configuration.
    :param str | None tokenizer_hash: Hash of the run tokenizer snapshot.
    :param str parameter_manifest_hash: Hash of model/optimizer leaf assignments.
    :param int tokens_seen: Cumulative exact loss-token count.
    :param TrainState train_state: Train state to checkpoint.
    :param data_iter: Data iterator to checkpoint.
    :param bool force: Whether to force an off-cadence save.
    """
    meta = build_meta(
        step=step,
        config=cfg.to_dict(),
        data_fingerprint=data_fingerprint(cfg, tokenizer_snapshot_hash=tokenizer_hash),
        parameter_manifest_hash=parameter_manifest_hash,
        tokens_seen=int(tokens_seen),
    )
    save(
        manager,
        step=step,
        train_state=train_state,
        data_iter=data_iter,
        meta=meta,
        force=force,
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
    parameter_manifest_hash: str,
) -> tuple[TrainState, dict[str, Any] | None]:
    """Restore state if requested, otherwise return the initial state.

    :param Literal["none", "latest"] | int resume: Resume selector.
    :param Any | None manager: Checkpoint manager.
    :param TrainState state0: Initial state.
    :param Any abstract_state: Abstract train state for restore shape.
    :param Any data_it: Data iterator to restore.
    :param Config cfg: Training configuration.
    :param str | None tokenizer_hash: Optional tokenizer snapshot hash for resume checks.
    :param str parameter_manifest_hash: Current model/optimizer manifest hash.
    :return tuple: (TrainState, meta) where meta is checkpoint metadata if restored.
    """
    if resume == "none":
        return state0, None
    if manager is None:
        raise RuntimeError("resume requested but checkpointing is disabled")

    if resume == "latest":
        latest = manager.latest_step()
        if latest is None:
            raise FileNotFoundError(f"No checkpoints found in {manager.directory}")
        step_r = int(latest)
    else:
        step_r = int(resume)

    # Validate metadata before Grain restores/replays any iterator buffers.
    # A pipeline schema or source mismatch must fail without reading up to a
    # full document/packed shuffle window from an incompatible source.
    meta = restore_meta_at_step(manager, step=step_r)
    check_resume_compat(
        cfg,
        meta,
        tokenizer_snapshot_hash=tokenizer_hash,
        parameter_manifest_hash=parameter_manifest_hash,
    )
    _, state, restored_meta = restore_at_step(
        manager,
        step=step_r,
        abstract_train_state=abstract_state,
        data_iter=data_it,
    )
    print(f"[chomp] resumed from checkpoint step {step_r}")
    return state, restored_meta


def _trim_trailing_token(tokens: list[int], token_id: int | None) -> list[int]:
    """Trim trailing token_id values from a token list.

    :param list[int] tokens: Token list to trim.
    :param int | None token_id: Token id to remove from the tail.
    :return list[int]: Trimmed token list.
    """
    if token_id is None:
        return tokens
    out = list(tokens)
    while out and out[-1] == token_id:
        out.pop()
    return out


def _validate_packing_capabilities(cfg: Config, *, params: Any, static: Any) -> None:
    """Fail fast when strict packed semantics are requested but unsupported.

    :param Config cfg: Training configuration.
    :param Any params: Model parameters.
    :param Any static: Static model components.
    :raises RuntimeError: If strict segment isolation is requested but unavailable.
    """
    if not strict_packed_segments(cfg):
        return
    if supports_packed_segments(params, static):
        return
    raise RuntimeError(
        f"Strict segment isolation (packing_mode={cfg.data.packing_mode!r}) was "
        "requested but the model backend does not "
        "advertise full segment isolation (supports_segment_reset capability flag, "
        "megalodon-jax >= 0.1.2: attention + ComplexEMA + TimestepNorm reset at "
        "packed document boundaries). Set data.packing_strict_segments=false to "
        "run in non-strict mode or upgrade megalodon-jax."
    )


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


# Detailed pipeline and device metrics belong in W&B; the local file keeps the
# compact, resume-oriented training record. W&B supplies its own step axis and
# exposes current device memory instead of the process-lifetime peak.
_METRICS_FILE_DROP = frozenset(
    {"wall_time_s", "packing_tokens", "packing_capacity", "eval_tokens", "device_memory_gb"}
)
_WANDB_DROP = frozenset({"step", "peak_memory_gb"})


def _project_metrics(row: dict[str, Any], *, drop: frozenset[str]) -> dict[str, Any]:
    """Project a complete metrics row onto one telemetry sink.

    :param dict[str, Any] row: Complete metrics row.
    :param frozenset[str] drop: Keys omitted from the sink.
    :return dict[str, Any]: Metrics accepted by the sink.
    """
    return {key: value for key, value in row.items() if key not in drop}


def _console_row(
    cfg: Config,
    *,
    step: int,
    metrics_host: dict[str, Any],
    step_time_s: float,
    tokens_per_sec: float,
    eval_loss: float | None,
    data_stats: dict[str, Any] | None,
    mem_stats: dict[str, Any],
) -> str:
    """Format the console line from host metrics and iterator/memory stats.

    Shared by the dry-run path and the main loop so the two never drift.

    :param Config cfg: Training configuration.
    :param int step: Training step number.
    :param dict[str, Any] metrics_host: Host-side step metrics (loss/grad_norm/lr).
    :param float step_time_s: Step wall time in seconds.
    :param float tokens_per_sec: Throughput in tokens per second.
    :param float | None eval_loss: Optional eval loss.
    :param data_stats: Latest data iterator stats, if any.
    :param dict[str, Any] mem_stats: Device memory stats (possibly empty).
    :return str: Formatted console line.
    """
    lr_adam = float(metrics_host["lr"])
    parts = [
        f"step {step}",
        f"loss {float(metrics_host['loss']):.4f}",
        f"grad {float(metrics_host['grad_norm']):.2e}",
        f"lr {lr_adam:.2e}",
        f"time {float(step_time_s):.3f}s",
        f"tok/s {float(tokens_per_sec):.0f}",
    ]
    if cfg.optim.name == "muon":
        parts.append(f"muon_lr {float(_muon_lr_from_adam(lr_adam, cfg)):.2e}")
    if eval_loss is not None:
        parts.append(f"eval {float(eval_loss):.4f}")
    if data_stats and "packing_utilization" in data_stats:
        parts.append(f"pack {float(data_stats['packing_utilization']):.3f}")
    if "device_memory_gb" in mem_stats:
        parts.append(f"mem {float(mem_stats['device_memory_gb']):.1f}GB")
    if "peak_memory_gb" in mem_stats:
        parts.append(f"peak {float(mem_stats['peak_memory_gb']):.1f}GB")
    return " | ".join(parts)


def _flush_loggers() -> None:
    """Flush all log handlers to ensure crash logs are written."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        with contextlib.suppress(Exception):
            handler.flush()


def _muon_param_stats(
    params: Any, *, allow_all_2d: bool, allow_embed: bool
) -> tuple[int, int, int, list[str]]:
    """Return Muon/Adam tensor counts and a sample of Muon paths.

    :param Any params: Parameter pytree.
    :param bool allow_all_2d: If True, apply Muon to all 2D tensors.
    :param bool allow_embed: If True, allow Muon on tied embedding weights.
    :return tuple: (muon_tensors, adam_tensors, total_2d, muon_paths).
    """
    flat, _ = jax.tree_util.tree_flatten_with_path(params)
    total_tensors = 0
    total_2d = 0
    muon_tensors = 0
    muon_paths: list[str] = []
    for path, leaf in flat:
        if not hasattr(leaf, "ndim"):
            continue
        total_tensors += 1
        if leaf.ndim == 2:
            total_2d += 1
        path_str = path_to_str(path)
        if is_muon_eligible(leaf, path_str, allow_all_2d=allow_all_2d, allow_embed=allow_embed):
            muon_tensors += 1
            muon_paths.append(path_str)
    adam_tensors = total_tensors - muon_tensors
    return muon_tensors, adam_tensors, total_2d, muon_paths


def _muon_weight_dim_numbers(params: Any, *, allow_all_2d: bool, allow_embed: bool = False) -> Any:
    """Return Muon dimension specs for eligible parameters.

    :param Any params: Parameter pytree.
    :param bool allow_all_2d: If True, apply Muon to all 2D tensors.
    :param bool allow_embed: If True, allow Muon on tied embedding weights.
    :return Any: Pytree of MuonDimensionNumbers (muon) or None (adam).
    """
    muon_dims = optax.contrib.MuonDimensionNumbers(reduction_axis=(1,), output_axis=(0,))
    flat, treedef = jax.tree_util.tree_flatten_with_path(params)
    dim_nums = [
        muon_dims
        if is_muon_eligible(
            leaf, path_to_str(path), allow_all_2d=allow_all_2d, allow_embed=allow_embed
        )
        else None
        for path, leaf in flat
    ]
    return treedef.unflatten(dim_nums)


def _muon_lr_from_adam(lr_adam: jax.Array, cfg: Config) -> jax.Array:
    """Return the Muon learning rate derived from the AdamW schedule.

    :param jax.Array lr_adam: AdamW learning rate for the current step.
    :param Config cfg: Training configuration.
    :return jax.Array: Muon learning rate (scaled).
    """
    return lr_adam * cfg.optim.muon.lr_scale


def build_optimizer(
    cfg: Config, params: Any
) -> tuple[optax.GradientTransformation, Callable[[jax.Array], jax.Array]]:
    """Create Optax optimizer + schedule function (for logging).

    :param Config cfg: Training configuration.
    :param Any params: Model parameters (used to build weight decay mask).
    :return tuple: (optimizer, lr_schedule) where lr_schedule maps step to learning rate.
    """

    # Optax's warmup_cosine_decay_schedule expects decay_steps to be the total
    # schedule horizon INCLUDING warmup (cosine length is decay_steps - warmup_steps).
    # Our config treats optim.decay_steps as the post-warmup duration, so we
    # explicitly pass warmup + decay_duration here.
    schedule_horizon = resolve_decay_horizon(cfg)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.optim.lr,
        warmup_steps=cfg.optim.warmup_steps,
        decay_steps=schedule_horizon,
        end_value=cfg.optim.lr * cfg.optim.min_lr_ratio,
    )

    # NOTE: grad clipping is done manually in make_train_step to avoid
    # computing global_norm twice (once for clipping, once for logging).

    if cfg.optim.name == "muon":
        muon_cfg = cfg.optim.muon
        adam_cfg = cfg.optim.adam
        allow_all_2d = muon_cfg.allow_all_2d
        allow_embed = muon_cfg.allow_tied_embed
        if cfg.model.backend == "megalodon" and allow_all_2d:
            logger.warning(
                "optim.muon.allow_all_2d=true will apply Muon to all 2D tensors, including "
                "non-matmul parameters (e.g., embed.weight, attn.gamma/beta, cema.gamma_*)."
            )
        muon_tensors, adam_tensors, total_2d, muon_paths = _muon_param_stats(
            params, allow_all_2d=allow_all_2d, allow_embed=allow_embed
        )
        logger.info(
            "Muon param split: %s muon / %s adam tensors; 2D coverage %s/%s",
            muon_tensors,
            adam_tensors,
            muon_tensors,
            total_2d,
        )
        if muon_paths:
            sample = ", ".join(muon_paths[:5])
            logger.info("Muon sample params: %s", sample)

        if muon_tensors == 0:
            logger.warning(
                "optim.name=muon selected but no muon-eligible parameters were found; "
                "falling back to AdamW for all parameters."
            )

        def label_fn(tree: Any) -> Any:
            """Return optimizer labels for each parameter leaf.

            multi_transform requires a label for EVERY leaf, so non-array
            leaves label "adam" (is_muon_eligible is False for them).

            :param Any tree: Parameter pytree.
            :return Any: Pytree of labels ("muon" or "adam").
            """
            flat, treedef = jax.tree_util.tree_flatten_with_path(tree)
            labels = [
                "muon"
                if is_muon_eligible(
                    leaf, path_to_str(path), allow_all_2d=allow_all_2d, allow_embed=allow_embed
                )
                else "adam"
                for path, leaf in flat
            ]
            return treedef.unflatten(labels)

        def muon_schedule(step: jax.Array) -> jax.Array:
            """Return the Muon learning rate schedule.

            :param jax.Array step: Training step.
            :return jax.Array: Muon learning rate at step.
            """
            return _muon_lr_from_adam(schedule(step), cfg)

        muon_weight_decay = cfg.optim.weight_decay * muon_cfg.weight_decay_mult

        def muon_dim_fn(tree: Any) -> Any:
            """Return Muon dimension numbers for masked Muon parameters.

            :param Any tree: Parameter pytree.
            :return Any: Pytree of MuonDimensionNumbers for Muon params.
            """
            # The Muon transform only sees Muon-labeled leaves, so use all-2D mode.
            return _muon_weight_dim_numbers(tree, allow_all_2d=True)

        muon_transforms = [
            optax.contrib.scale_by_muon(
                ns_steps=muon_cfg.ns_steps,
                beta=muon_cfg.momentum,
                nesterov=muon_cfg.nesterov,
                weight_dimension_numbers=muon_dim_fn,
            )
        ]
        if muon_cfg.consistent_rms is not None:
            muon_transforms.append(
                muon_contrib.scale_by_shape(
                    weight_dimension_numbers=muon_dim_fn,
                    consistent_rms=muon_cfg.consistent_rms,
                )
            )
        muon_transforms.extend(
            [
                optax.add_decayed_weights(
                    muon_weight_decay, mask=lambda tree: parameter_decay_mask(cfg, tree)
                ),
                optax.scale_by_learning_rate(muon_schedule),
            ]
        )
        muon_tx = optax.chain(*muon_transforms)
        adam_tx = optax.adamw(
            learning_rate=schedule,
            b1=adam_cfg.b1,
            b2=adam_cfg.b2,
            eps=adam_cfg.eps,
            weight_decay=cfg.optim.weight_decay,
            mask=lambda tree: parameter_decay_mask(cfg, tree),
            nesterov=adam_cfg.nesterov,
        )
        tx = optax.multi_transform({"muon": muon_tx, "adam": adam_tx}, label_fn)
    else:
        adam_cfg = cfg.optim.adam
        tx = optax.adamw(
            learning_rate=schedule,
            b1=adam_cfg.b1,
            b2=adam_cfg.b2,
            eps=adam_cfg.eps,
            weight_decay=cfg.optim.weight_decay,
            mask=lambda tree: parameter_decay_mask(cfg, tree),
            nesterov=adam_cfg.nesterov,
        )

    return tx, schedule


def init_train_state(
    *, params: Any, tx: optax.GradientTransformation, key: jax.Array
) -> TrainState:
    """Initialize a fresh TrainState at step 0.

    :param Any params: Model parameters.
    :param optax.GradientTransformation tx: Optimizer transform.
    :param jax.Array key: PRNG key for dropout.
    :return TrainState: Initialized training state.
    """
    opt_state = tx.init(params)
    return TrainState(
        step=jnp.array(0, dtype=jnp.int32), params=params, opt_state=opt_state, rng=key
    )


def _micro_batch(
    input_ids: jax.Array,
    labels: jax.Array,
    attn: jax.Array,
    segs: jax.Array,
    pos_ids: jax.Array,
) -> Batch:
    """Assemble one [B, T] micro-batch inside a scan body (train and eval).

    :param jax.Array input_ids: Input token IDs [B, T].
    :param jax.Array labels: Label token IDs [B, T].
    :param jax.Array attn: Attention mask [B, T].
    :param jax.Array segs: Segment IDs [B, T].
    :param jax.Array pos_ids: Position IDs [B, T].
    :return Batch: Micro-batch with contract dtypes.
    """
    return Batch(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attn.astype(bool),
        segment_ids=segs.astype(jnp.int32),
        position_ids=pos_ids.astype(jnp.int32),
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
    clip_norm = float(cfg.optim.grad_clip_norm) if cfg.optim.grad_clip_norm else 0.0
    use_packed_segments = strict_packed_segments(cfg)
    # Harness-level optimizer math — micro-grad summation, token
    # normalization, and global-norm clipping — is always fp32. Deliberately
    # NOT cfg.model.accum_dtype: that knob feeds the model's *internal*
    # accumulation (attention/CEMA reductions), and pointing it at bf16 must
    # not silently degrade gradient accumulation across the scan.
    grad_accum_dtype = jnp.float32

    def micro_loss(
        params: Any,
        input_ids: jax.Array,
        labels: jax.Array,
        attn: jax.Array,
        segs: jax.Array,
        pos_ids: jax.Array,
        key: jax.Array | None,
        token_count: jax.Array,
    ) -> jax.Array:
        """Compute token-weighted loss for a single micro-batch.

        :param Any params: Model parameters.
        :param jax.Array input_ids: Input token IDs [B, T].
        :param jax.Array labels: Label token IDs [B, T].
        :param jax.Array attn: Attention mask [B, T].
        :param jax.Array segs: Segment IDs [B, T].
        :param jax.Array pos_ids: Position IDs [B, T].
        :param key: PRNG key for dropout, or None if deterministic.
        :param jax.Array token_count: Number of valid tokens for weighting.
        :return jax.Array: Weighted loss scalar.
        """
        micro = _micro_batch(input_ids, labels, attn, segs, pos_ids)
        loss = training_loss(
            params,
            static,
            batch=micro,
            deterministic=deterministic,
            key=key,
            use_packed_segments=use_packed_segments,
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

        # Init accumulators. Gradients accumulate in fp32, not param dtype:
        # with bf16 params, summing micro-grads in bf16 systematically drops
        # low-order bits across the scan.
        loss0 = jnp.zeros((), dtype=jnp.float32)
        grad0 = jax.tree_util.tree_map(
            lambda x: jnp.zeros(x.shape, dtype=grad_accum_dtype)
            if jnp.issubdtype(x.dtype, jnp.floating)
            else jnp.zeros_like(x),
            state.params,
        )
        token0 = jnp.zeros((), dtype=jnp.float32)

        def body(
            carry: tuple[jax.Array, Any, jax.Array],
            inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
        ) -> tuple[tuple[jax.Array, Any, jax.Array], None]:
            """Scan body: accumulate loss and gradients for one micro-batch.

            :param tuple carry: (loss_sum, grad_sum, token_sum) accumulators.
            :param tuple inputs: (input_ids, labels, attn, segs, pos_ids, key) for one micro-batch.
            :return tuple: (updated_carry, None).
            """
            loss_sum, grad_sum, token_sum = carry
            in_ids, labs, attn, segs, pos_ids, k = inputs
            token_count = _count_tokens(labs, attn).astype(jnp.float32)
            loss, grads = loss_and_grad(
                state.params, in_ids, labs, attn, segs, pos_ids, k, token_count
            )
            loss_sum = loss_sum + loss.astype(jnp.float32)
            grad_sum = jax.tree_util.tree_map(lambda a, b: a + b.astype(a.dtype), grad_sum, grads)
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
                batch.position_ids,
                micro_keys,
            ),
        )

        token_denom = jnp.maximum(token_sum, 1.0)
        loss = loss_sum / token_denom
        grads = jax.tree_util.tree_map(lambda g: g / token_denom, grad_sum)

        grad_norm = optax.tree.norm(grads)
        if clip_norm > 0:
            trigger = grad_norm > clip_norm
            grads = jax.tree_util.tree_map(
                lambda g: jnp.where(trigger, g * (clip_norm / jnp.maximum(grad_norm, 1e-6)), g),
                grads,
            )
        # Normalization + global-norm clipping ran in fp32 above; cast
        # back to param dtype only now so opt_state dtypes stay stable.
        grads = jax.tree_util.tree_map(
            lambda g, p: g.astype(p.dtype) if jnp.issubdtype(p.dtype, jnp.floating) else g,
            grads,
            state.params,
        )
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

    use_packed_segments = strict_packed_segments(cfg)

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
            :param tuple xs: (input_ids, labels, attn, segs, pos_ids) microbatch inputs.
            :return tuple: (updated_carry, None).
            """
            loss_sum, token_sum = carry
            input_ids, labels, attn, segs, pos_ids = xs
            micro = _micro_batch(input_ids, labels, attn, segs, pos_ids)
            token_count = _count_tokens(labels, attn).astype(jnp.float32)
            loss = training_loss(
                params,
                static,
                batch=micro,
                deterministic=True,
                key=None,
                use_packed_segments=use_packed_segments,
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
                batch.position_ids,
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
) -> Path:
    """Run a training job and return the run directory.

    Resume contract:
    - resume requires logging.run_dir to be set (existing run directory)
    - we restore both train_state and data iterator state when present

    :param Config cfg: Fully validated training configuration.
    :param config_path: Optional path to the source YAML config file.
    :param resume: Resume mode - "none" (fresh), "latest", or specific step number.
    :param bool dry_run: If True, compile and run a single step, then exit early.
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

    params, static, tx, schedule, state0, abstract_state = _build_model_state(cfg)
    _validate_packing_capabilities(cfg, params=params, static=static)
    parameter_manifest = build_parameter_manifest(cfg, params, static)
    parameter_manifest_hash = str(parameter_manifest["sha256"])
    logger.info(
        "Parameter manifest %s: %s",
        parameter_manifest_hash,
        parameter_manifest["group_counts"],
    )

    # Log param count once
    n_params = param_count(params)
    print(f"[chomp] params: {n_params:,}")

    # Data iterator (host-side)
    data_it = build_train_iterator(cfg, tokenizer=tokenizer)

    # Checkpoint manager
    manager = _build_checkpoint_manager(cfg, run_dir)

    try:
        # Restore if requested.
        state, resume_meta = _maybe_restore_state(
            resume=resume,
            manager=manager,
            state0=state0,
            abstract_state=abstract_state,
            data_it=data_it,
            cfg=cfg,
            tokenizer_hash=tokenizer_hash,
            parameter_manifest_hash=parameter_manifest_hash,
        )
        write_parameter_manifest(run_dir, parameter_manifest)

        if eval_tokens is None:
            # Reaching here means check_resume_compat accepted the restored
            # checkpoint. It is now safe for the explicit override to persist a
            # replacement cache without a rejected configuration poisoning the
            # run directory.
            eval_tokens = load_or_create_eval_texts(
                cfg, tokenizer=tokenizer, run_dir=run_dir, resume=True
            )

        train_step = make_train_step(cfg, static=static, tx=tx, lr_schedule=schedule)
        eval_every = int(cfg.train.eval_every)
        eval_step = None
        if eval_tokens and eval_every > 0:
            eval_step = make_eval_step(cfg, static=static)
    except BaseException:
        _close_manager_after_failure(manager, phase="training setup")
        raise

    if dry_run:
        try:
            wandb_run, profile_enabled = _start_run_telemetry(cfg, run_dir=run_dir, dry_run=True)
        except BaseException:
            _close_manager_after_failure(manager, phase="dry-run telemetry setup")
            raise
        try:
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
            console_line = _console_row(
                cfg,
                step=step_i,
                metrics_host=metrics_host,
                step_time_s=step_time_s,
                tokens_per_sec=tokens_per_sec,
                eval_loss=None,
                data_stats=data_stats,
                mem_stats=_device_memory_stats_gb(),
            )
            print("[chomp] dry-run complete")
            print(console_line)
        except BaseException:
            _close_manager_after_failure(manager, phase="dry run")
            _finish_run_telemetry(wandb_run, profile_enabled=profile_enabled, exit_code=1)
            raise
        else:
            try:
                if manager is not None:
                    manager.close()
            except BaseException:
                _finish_run_telemetry(wandb_run, profile_enabled=profile_enabled, exit_code=1)
                raise
            else:
                _finish_run_telemetry(wandb_run, profile_enabled=profile_enabled)
        return run_dir

    # Training loop
    t_compile = None
    t0 = time.perf_counter()

    # Determine starting step from TrainState
    start_step = int(jax.device_get(state.step))
    target_steps = cfg.train.steps

    if start_step >= target_steps:
        print(f"[chomp] start_step ({start_step}) >= target steps ({target_steps}); nothing to do")

    host_step = int(start_step)
    step_i = int(host_step)
    # Host-side Python int: avoids int32 overflow without jax_enable_x64.
    tokens_seen_count = 0 if resume_meta is None else int(resume_meta["tokens_seen"])
    last_saved_step: int = -1
    # True whenever `state` and `data_it` correspond to the same completed
    # step. False in the window between consuming a batch and finishing its
    # train step — a checkpoint written there would pair the old train state
    # with an advanced data stream and silently skip that batch on resume.
    data_state_aligned = True
    # Device metrics of the last completed train_step (always matching
    # `state`); the finally block re-validates them before the final save so
    # a non-finite step can never be persisted as "latest".
    metrics: dict[str, jax.Array] | None = None

    eval_batches_cache: list[Batch] = []

    def _run_eval(params: Any) -> dict[str, Any]:
        """Run a full eval pass over the cached eval texts.

        :param Any params: Model parameters.
        :return dict[str, Any]: Eval metrics row with eval_loss and eval_tokens.
        """
        if eval_step is None or not eval_tokens:
            return {}
        # Eval batches are deterministic (cached tokens, never shuffled), so
        # assemble them once and reuse across evals instead of re-running
        # tokenize/pack every eval_every steps. The iterator is built with
        # device_put disabled so the cache stays host-side regardless of
        # data.device_put; each eval transfers per batch, so device memory is
        # never held between evals.
        if not eval_batches_cache:
            host_cfg = dc_replace(cfg, data=dc_replace(cfg.data, device_put=False))
            eval_batches_cache.extend(build_eval_iterator(host_cfg, tokens=eval_tokens))
        total_loss = 0.0
        total_tokens = 0.0
        batch_count = 0
        for eval_batch in eval_batches_cache:
            batch_count += 1
            eval_batch = jax.device_put(eval_batch)
            loss_sum, token_sum = eval_step(params, eval_batch)
            loss_sum_host, token_sum_host = jax.device_get((loss_sum, token_sum))
            total_loss += float(loss_sum_host)
            total_tokens += float(token_sum_host)

        if batch_count == 0:
            raise RuntimeError(
                "Evaluation produced zero batches. "
                f"packing_mode={cfg.data.packing_mode!r}, "
                f"eval_rows_per_batch={int(cfg.train.batch_size)}, "
                f"eval_doc_count={len(eval_tokens)}. "
                "The eval set did not yield any usable packed window. Increase "
                "data.max_eval_samples or check tokenization and masking."
            )

        if total_tokens <= 0:
            # Batches exist but every label is masked out: broken boundary
            # masking, EOS suppression eating the whole set, or pathological
            # short docs. A null eval loss would hide it for the entire run.
            raise RuntimeError(
                f"Evaluation produced {batch_count} batch(es) but zero valid "
                "loss tokens: every label is masked. Check "
                "data.mask_boundary_loss / data.train_on_eos against the eval "
                "document lengths."
            )
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
            gen_tokens, next_key = generate_tokens(
                params,
                static,
                prompt_tokens=prompt_tokens,
                max_new_tokens=gen_settings.max_new_tokens,
                bos_token_id=cfg.model.bos_token_id,
                eos_token_id=cfg.model.eos_token_id,
                temperature=gen_settings.temperature,
                top_k=gen_settings.top_k,
                top_p=gen_settings.top_p,
                key=gen_key,
            )
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            logger.warning("Generation requested but megalodon_jax unavailable: %s", exc)
            gen_settings = None
            return
        except Exception as exc:
            logger.warning("Generation failed at step %d: %s", step, exc)
            return

        if next_key is not None:
            gen_key = next_key

        prompt_text = _safe_decode(tokenizer, prompt_tokens, label="prompt")
        generated_text = _safe_decode(tokenizer, gen_tokens, label="generated")
        _emit_generation_output(
            step=step,
            prompt_text=prompt_text,
            generated_text=generated_text,
            use_rich=cfg.logging.console_use_rich,
        )

    try:
        wandb_run, profile_enabled = _start_run_telemetry(cfg, run_dir=run_dir, dry_run=False)
    except BaseException:
        _close_manager_after_failure(manager, phase="telemetry setup")
        raise
    exit_code = 0
    crash_reason = None
    crash_type = None
    crash_step = None

    try:
        with MetricsWriter(metrics_path) as mw:
            try:
                for _ in tqdm(range(start_step, target_steps), desc="train", dynamic_ncols=True):
                    # Fetch batch (host) and (optionally) device_put
                    step_i = int(host_step) + 1
                    data_state_aligned = False
                    t_fetch = time.perf_counter()
                    try:
                        batch = next(data_it)
                    except StopIteration:
                        # Usable partial batches are padded and returned by
                        # assembly. A failed fetch therefore consumes zero
                        # windows and remains aligned with the last step.
                        data_state_aligned = True
                        tokens_seen_host = int(tokens_seen_count)
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
                    # Host-side input-pipeline time for this step: fetch (incl.
                    # tokenize/pack/shuffle backpressure) + stats + device_put.
                    # tokens_per_sec below is model-step throughput only; the
                    # e2e rate in the metrics row includes this wait.
                    data_wait_s = time.perf_counter() - t_fetch

                    # Batch placement validation (real check)
                    if cfg.debug.check_device_every > 0 and (
                        host_step % cfg.debug.check_device_every == 0
                    ):
                        assert_batch_on_device(batch, allow_cpu=cfg.train.allow_cpu)

                    # Step (compile happens on first call)
                    with step_annotation("train_step"):
                        t1 = time.perf_counter()
                        state, metrics = train_step(state, batch)
                        # Intentional per-step device sync: tokens_seen feeds
                        # checkpoint meta and must be exact for resume
                        # accounting. On a single device the host blocks at the
                        # next dispatch anyway, so estimating to stay async
                        # would trade exactness for nothing.
                        step_loss_tokens = int(jax.device_get(metrics["token_sum"]))
                        tokens_seen_count += step_loss_tokens

                    host_step = int(step_i)
                    data_state_aligned = True

                    should_eval = (
                        eval_step is not None and eval_every > 0 and (step_i % eval_every) == 0
                    )
                    should_log = (step_i % cfg.train.log_every) == 0 or should_eval
                    save_every = int(cfg.checkpoint.save_every)
                    save_interval = (
                        manager is not None and save_every > 0 and (step_i % save_every == 0)
                    )
                    # Save steps force a sync so the finite check below runs
                    # before the checkpoint write: a step with NaN/Inf metrics
                    # must never be persisted as a resume point, even when the
                    # save cadence does not land on a logging step.
                    should_sync = should_log or save_interval or (t_compile is None)

                    step_time_s = None
                    tokens_per_sec = 0.0

                    if should_sync:
                        metrics_host = jax.device_get(metrics)
                        t2 = time.perf_counter()
                        step_time_s = t2 - t1
                        if t_compile is None:
                            t_compile = step_time_s
                        if cfg.debug.nan_check:
                            _check_finite_metrics(metrics_host, step=step_i)
                        token_sum = float(metrics_host.get("token_sum", 0.0))
                        tokens_per_sec = token_sum / step_time_s if step_time_s > 0 else 0.0

                    # Checkpoint save (after state updated + finite-checked)
                    if save_interval:
                        _save_training_checkpoint(
                            manager,
                            step=step_i,
                            cfg=cfg,
                            tokenizer_hash=tokenizer_hash,
                            parameter_manifest_hash=parameter_manifest_hash,
                            tokens_seen=int(tokens_seen_count),
                            train_state=state,
                            data_iter=data_it,
                        )
                        last_saved_step = step_i

                    eval_row: dict[str, Any] = {}
                    if should_eval:
                        eval_row = _run_eval(state.params)

                    if (
                        gen_settings is not None
                        and gen_stream is not None
                        and (step_i % gen_settings.every) == 0
                    ):
                        _run_generation_sample(step_i, state.params)

                    # Log: metrics file, wandb, and console fire on the same
                    # steps by design (metrics_host is set — should_log
                    # implies should_sync).
                    if should_log:
                        mem_stats = _device_memory_stats_gb()
                        lr_adam = float(metrics_host["lr"])
                        row = {
                            "step": int(step_i),
                            "loss": float(metrics_host["loss"]),
                            "grad_norm": float(metrics_host["grad_norm"]),
                            "lr": lr_adam,
                            "loss_tokens": int(step_loss_tokens),
                            "step_time_s": float(step_time_s),
                            "data_wait_s": float(data_wait_s),
                            "tokens_per_sec": float(tokens_per_sec),
                            "tokens_per_sec_e2e": float(
                                token_sum / (step_time_s + data_wait_s)
                                if (step_time_s + data_wait_s) > 0
                                else 0.0
                            ),
                            "tokens_seen": int(tokens_seen_count),
                            "wall_time_s": time.perf_counter() - t0,
                        }
                        if cfg.optim.name == "muon":
                            row["lr_muon"] = float(_muon_lr_from_adam(lr_adam, cfg))
                        if data_stats:
                            row.update(data_stats)
                        if eval_row:
                            row.update(eval_row)
                        if mem_stats:
                            row.update(mem_stats)
                        if step_i == (start_step + 1) and t_compile is not None:
                            row["first_step_compile_time_s"] = float(t_compile)

                        mw.write(_project_metrics(row, drop=_METRICS_FILE_DROP))
                        if wandb_run is not None:
                            wandb_run.log(
                                _project_metrics(row, drop=_WANDB_DROP),
                                step=step_i,
                            )

                        eval_loss = eval_row.get("eval_loss") if eval_row else None
                        tqdm.write(
                            _console_row(
                                cfg,
                                step=step_i,
                                metrics_host=metrics_host,
                                step_time_s=step_time_s,
                                tokens_per_sec=tokens_per_sec,
                                eval_loss=eval_loss,
                                data_stats=data_stats,
                                mem_stats=mem_stats,
                            )
                        )
            except Exception as exc:
                exit_code = 1
                crash_type = type(exc).__name__
                crash_reason = str(exc)
                crash_step = int(step_i)
                logger.exception("Training crashed at step %s", crash_step)
                row = {
                    "step": int(crash_step),
                    "crash": True,
                    "crash_type": crash_type,
                    "crash_reason": crash_reason,
                    "wall_time_s": time.perf_counter() - t0,
                }
                row["tokens_seen"] = int(tokens_seen_count)
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
        # Final checkpoint: save if we did any training and the last step
        # wasn't already saved — but only when `state` and `data_it` are
        # aligned. In the fetched-but-unfinished window (Ctrl-C mid
        # train_step, device failure, placement check) the iterator is one
        # batch ahead of state.step; a checkpoint written there would
        # silently skip that batch on resume.
        # Checkpoint failures must never be silent: on a clean exit they are
        # re-raised (training must not exit successfully with an unwritten
        # checkpoint); on the crash path the original exception keeps
        # propagating and they are logged as secondary failures.
        exc_in_flight = sys.exc_info()[0] is not None
        ckpt_errors: list[Exception] = []

        final_step = None
        if manager is not None and not data_state_aligned:
            logger.warning(
                "Skipping final checkpoint: the data iterator is ahead of the "
                "train state (a fetched batch never completed its step, or the "
                "stream exhausted partway through batch assembly). Resume from "
                "the last periodic checkpoint%s.",
                f" (step {last_saved_step})" if last_saved_step >= 0 else "",
            )
        elif manager is not None:
            try:
                final_step = int(jax.device_get(state.step))
            except Exception as exc:
                logger.exception("Could not read state.step for the final checkpoint")
                ckpt_errors.append(exc)
        if (
            manager is not None
            and final_step is not None
            and final_step > start_step
            and final_step != last_saved_step
        ):
            final_state_valid = True
            if cfg.debug.nan_check and metrics is not None:
                # The in-loop finite check only runs on sync steps, and a
                # non-finite step may itself be why we are unwinding.
                # Re-validate the last completed step's metrics (they always
                # match `state`) so "latest" can never be a NaN tombstone.
                try:
                    _check_finite_metrics(jax.device_get(metrics), step=final_step)
                except Exception as exc:
                    final_state_valid = False
                    logger.error(
                        "Skipping final checkpoint at step %s (%s). Resume from "
                        "the last periodic checkpoint%s.",
                        final_step,
                        exc,
                        f" (step {last_saved_step})" if last_saved_step >= 0 else "",
                    )
                    ckpt_errors.append(exc)
            if final_state_valid:
                try:
                    _save_training_checkpoint(
                        manager,
                        step=final_step,
                        cfg=cfg,
                        tokenizer_hash=tokenizer_hash,
                        parameter_manifest_hash=parameter_manifest_hash,
                        tokens_seen=int(tokens_seen_count),
                        train_state=state,
                        data_iter=data_it,
                        force=True,
                    )
                except Exception as exc:
                    logger.exception("Final checkpoint save failed at step %s", final_step)
                    ckpt_errors.append(exc)

        if manager is not None:
            try:
                manager.close()
            except Exception as exc:
                logger.exception("Closing the checkpoint manager failed")
                ckpt_errors.append(exc)

        # Checkpoint finalization failures fail the run (raise below), so the
        # exit code W&B records must agree — finish() only after all
        # checkpoint work has had its chance to error.
        if ckpt_errors and not exc_in_flight and exit_code == 0:
            exit_code = 1
        _finish_run_telemetry(
            wandb_run,
            profile_enabled=profile_enabled,
            exit_code=exit_code,
            crash_type=crash_type,
            crash_reason=crash_reason,
        )

        _flush_loggers()

        if ckpt_errors and not exc_in_flight:
            raise RuntimeError(
                f"checkpoint finalization failed: {ckpt_errors[0]}"
            ) from ckpt_errors[0]

    return run_dir
