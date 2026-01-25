"""Generate subcommand for standalone text generation from checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click


def _find_checkpoint_dir(
    checkpoint_path: str, config_override: str | None = None
) -> tuple[Path, Path]:
    """Find checkpoint directory and config file.

    Supports:
    - Direct step directory: runs/my_run/checkpoints/500/
    - Run directory: runs/my_run/ (uses latest checkpoint)

    :param str checkpoint_path: Path to run_dir or step directory.
    :param config_override: Optional path to override config file.
    :raises click.ClickException: If no valid checkpoint found.
    :return tuple: (step_dir, run_dir)
    """
    path = Path(checkpoint_path).resolve()

    # Case 1: Direct step directory (has train_state subdirectory)
    if (path / "train_state").exists():
        # Walk up to find run_dir
        ckpt_parent = path.parent
        if ckpt_parent.name == "checkpoints":
            run_dir = ckpt_parent.parent
        else:
            run_dir = path.parent
        return path, run_dir

    # Case 2: Checkpoints directory
    if path.name == "checkpoints":
        steps = sorted([int(d.name) for d in path.iterdir() if d.is_dir() and d.name.isdigit()])
        if not steps:
            raise click.ClickException(f"No step directories found in {path}")
        latest_step = steps[-1]
        return path / str(latest_step), path.parent

    # Case 3: Run directory (look for checkpoints/latest or root_dir)
    ckpt_dir = path / "checkpoints"
    if config_override or (path / "config_resolved.json").exists():
        cfg = _load_config_from_checkpoint(path, config_override)
        if cfg.checkpoint.root_dir:
            ckpt_dir = Path(cfg.checkpoint.root_dir)
            if not ckpt_dir.is_absolute():
                # Always resolve relative paths against run_dir, not CWD
                ckpt_dir = (path / ckpt_dir).resolve()

    if ckpt_dir.exists():
        steps = sorted([int(d.name) for d in ckpt_dir.iterdir() if d.is_dir() and d.name.isdigit()])
        if not steps:
            raise click.ClickException(f"No step directories found in {ckpt_dir}")
        latest_step = steps[-1]
        return ckpt_dir / str(latest_step), path

    raise click.ClickException(
        f"Could not find checkpoint at {path}. "
        "Provide a run_dir, checkpoints dir, or step directory."
    )


def _load_config_from_checkpoint(run_dir: Path, config_override: str | None) -> Any:
    """Load config from checkpoint or override file.

    :param Path run_dir: Run directory containing config_resolved.json.
    :param config_override: Optional path to override config file.
    :raises click.ClickException: If config file not found.
    :return Config: Loaded configuration.
    """
    from chomp.config import _from_nested_dict, _resolve_variables, validate_config

    if config_override:
        config_path = Path(config_override)
        if not config_path.exists():
            raise click.ClickException(f"Config override not found: {config_path}")
        if config_path.suffix in {".yaml", ".yml"}:
            import yaml

            with config_path.open() as f:
                data = yaml.safe_load(f) or {}
        else:
            with config_path.open() as f:
                data = json.load(f) or {}
    else:
        config_path = run_dir / "config_resolved.json"
        if not config_path.exists():
            raise click.ClickException(
                f"config_resolved.json not found in {run_dir}. "
                "Use --config to provide a config file."
            )
        with config_path.open() as f:
            data = json.load(f) or {}

    data = _resolve_variables(data)
    cfg = _from_nested_dict(data)

    validate_config(cfg)
    return cfg


def _restore_params(step_dir: Path, abstract_params: Any) -> Any:
    """Restore just the params from a checkpoint.

    Uses PyTreeCheckpointer directly on the train_state directory to load only
    the params subtree, skipping opt_state/rng/step.

    :param Path step_dir: Checkpoint step directory.
    :param Any abstract_params: Abstract tree for params (ShapeDtypeStruct).
    :return Any: Restored params pytree.
    """
    import orbax.checkpoint as ocp

    # Build abstract structure with only params key
    abstract_train_state = {"params": abstract_params}

    # Use PyTreeCheckpointer directly on the train_state directory
    ckptr = ocp.PyTreeCheckpointer()
    train_state_dir = step_dir / "train_state"

    restored = ckptr.restore(
        train_state_dir,
        item=abstract_train_state,
        transforms={},
        restore_args=ocp.checkpoint_utils.construct_restore_args(abstract_train_state),
    )

    return restored["params"]


@click.command()
@click.argument("checkpoint", type=click.Path(exists=True))
@click.option(
    "--prompt",
    "-p",
    required=True,
    help="Text prompt for generation.",
)
@click.option(
    "--max-tokens",
    type=int,
    default=128,
    help="Maximum number of tokens to generate.",
)
@click.option(
    "--temperature",
    type=float,
    default=1.0,
    help="Sampling temperature. Use 0 for greedy decoding.",
)
@click.option(
    "--top-k",
    type=int,
    default=None,
    help="Top-k sampling (optional).",
)
@click.option(
    "--top-p",
    type=float,
    default=None,
    help="Nucleus sampling threshold (optional).",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Random seed for sampling.",
)
@click.option(
    "--config",
    "config_override",
    type=click.Path(exists=True),
    default=None,
    help="Override config file (defaults to checkpoint's config_resolved.json).",
)
def generate(
    checkpoint: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    seed: int,
    config_override: str | None,
) -> None:
    """Generate text from a trained checkpoint.

    :param str checkpoint: Path to run directory or checkpoint step directory.
    :param str prompt: Text prompt for generation.
    :param int max_tokens: Maximum number of tokens to generate.
    :param float temperature: Sampling temperature (0 for greedy).
    :param top_k: Top-k sampling cutoff (optional).
    :param top_p: Nucleus sampling threshold (optional).
    :param int seed: Random seed for sampling.
    :param config_override: Path to override config file (optional).
    """
    from chomp.utils.xla import configure_blackwell_xla_env

    # Configure XLA env quirks before JAX backend init.
    configure_blackwell_xla_env()

    # Deferred imports: must run after XLA env config
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from chomp.data.pipeline import build_tokenizer, resolve_tokenizer_config
    from chomp.model import build_model

    # Find checkpoint and load config
    step_dir, run_dir = _find_checkpoint_dir(checkpoint, config_override=config_override)
    click.echo(f"Loading checkpoint from: {step_dir}")

    cfg = _load_config_from_checkpoint(run_dir, config_override)

    if cfg.model.backend != "megalodon":
        raise click.ClickException(
            "generate only supports model.backend='megalodon'. "
            f"Found {cfg.model.backend!r} in the checkpoint config."
        )

    # Build tokenizer and resolve tokenizer-derived config fields
    # (vocab_size rounding, special token IDs) before model build
    tokenizer = build_tokenizer(cfg)
    cfg = resolve_tokenizer_config(cfg, tokenizer)

    # Build model skeleton for abstract shapes
    key = jax.random.key(seed)
    model_key, gen_key = jax.random.split(key)
    params, static = build_model(cfg, key=model_key)

    # Get abstract params for restoration
    abstract_params = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=getattr(x, "sharding", None)),
        params,
    )

    # Restore params from checkpoint
    click.echo("Restoring model parameters...")
    params = _restore_params(step_dir, abstract_params)

    # Import generation function
    try:
        from megalodon_jax import generate as mega_generate
    except ImportError as e:
        raise click.ClickException(
            "megalodon_jax is required for generation. Install with: pip install megalodon-jax"
        ) from e

    # Tokenize prompt
    prompt_tokens = tokenizer.encode(prompt)
    if not prompt_tokens:
        raise click.ClickException("Prompt tokenized to empty sequence")

    prompt_ids = jnp.asarray(prompt_tokens, dtype=jnp.int32)[None, :]

    # Build generation kwargs
    gen_kwargs: dict[str, Any] = {
        "bos_token_id": int(cfg.model.bos_token_id),
        "eos_token_id": int(cfg.model.eos_token_id),
    }
    gen_kwargs["temperature"] = temperature
    if top_k is not None:
        gen_kwargs["top_k"] = top_k
    if top_p is not None:
        gen_kwargs["top_p"] = top_p

    # Combine model
    model = eqx.combine(params, static)

    # Generate
    click.echo("Generating...")
    needs_key = temperature > 0
    output_ids, _cache, _next_key = mega_generate(
        model,
        prompt_ids,
        max_tokens,
        key=gen_key if needs_key else None,
        **gen_kwargs,
    )

    # Decode output
    output_host = jax.device_get(output_ids)
    output_tokens = [int(x) for x in output_host[0].tolist()]
    gen_tokens = output_tokens[len(prompt_tokens) :]

    generated_text = tokenizer.decode(gen_tokens)

    # Output
    click.echo("\n" + "=" * 60)
    click.echo("Prompt:")
    click.echo(prompt)
    click.echo("-" * 60)
    click.echo("Generated:")
    click.echo(generated_text)
    click.echo("=" * 60)
