"""Generate subcommand for standalone text generation from checkpoints."""

from __future__ import annotations

import click

from chomp.utils.ckpt_paths import load_config_for_checkpoint, resolve_checkpoint_path


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
    type=click.IntRange(min=1),
    default=128,
    help="Maximum number of tokens to generate.",
)
@click.option(
    "--temperature",
    type=click.FloatRange(min=0.0),
    default=1.0,
    help="Sampling temperature. Use 0 for greedy decoding.",
)
@click.option(
    "--top-k",
    type=click.IntRange(min=1),
    default=None,
    help="Top-k sampling (optional).",
)
@click.option(
    "--top-p",
    type=click.FloatRange(min=0.0, max=1.0, min_open=True),
    default=None,
    help="Nucleus sampling threshold (optional, in (0, 1]).",
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
    import jax

    from chomp.ckpt import restore_params_only
    from chomp.data.pipeline import load_tokenizer_snapshot, prepare_tokenizer_and_config
    from chomp.model import build_model, generate_tokens
    from chomp.utils.tree import abstractify_tree

    # Find checkpoint and load config
    try:
        step_dir, run_dir = resolve_checkpoint_path(checkpoint, config_override=config_override)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Loading checkpoint from: {step_dir}")

    try:
        cfg = load_config_for_checkpoint(
            step_dir=step_dir, run_dir=run_dir, config_override=config_override
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    if cfg.model.backend != "megalodon":
        raise click.ClickException(
            "generate only supports model.backend='megalodon'. "
            f"Found {cfg.model.backend!r} in the checkpoint config."
        )

    # Prefer the run-pinned tokenizer so mutable upstream tokenizer revisions
    # cannot reinterpret the restored embedding rows.
    tokenizer = None
    if run_dir is not None and (run_dir / "tokenizer").exists():
        try:
            tokenizer = load_tokenizer_snapshot(run_dir, cfg)
        except Exception as exc:
            raise click.ClickException(str(exc)) from exc
    cfg, tokenizer = prepare_tokenizer_and_config(cfg, tokenizer=tokenizer)

    # Build model skeleton for abstract shapes
    key = jax.random.key(seed)
    model_key, gen_key = jax.random.split(key)
    params, static = build_model(cfg, key=model_key)

    # Restore params from checkpoint
    click.echo("Restoring model parameters...")
    try:
        params = restore_params_only(step_dir, abstractify_tree(params))
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc

    # Tokenize prompt
    prompt_tokens = tokenizer.encode(prompt)
    if not prompt_tokens:
        raise click.ClickException("Prompt tokenized to empty sequence")

    # Generate
    click.echo("Generating...")
    try:
        gen_tokens, _next_key = generate_tokens(
            params,
            static,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_tokens,
            bos_token_id=cfg.model.bos_token_id,
            eos_token_id=cfg.model.eos_token_id,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            key=gen_key,
        )
    except ImportError as exc:
        raise click.ClickException(
            "megalodon_jax is required for generation. Install with: pip install megalodon-jax"
        ) from exc
    except Exception as exc:
        raise click.ClickException(f"Generation failed: {exc}") from exc

    generated_text = tokenizer.decode(gen_tokens)

    # Output
    click.echo("\n" + "=" * 60)
    click.echo("Prompt:")
    click.echo(prompt)
    click.echo("-" * 60)
    click.echo("Generated:")
    click.echo(generated_text)
    click.echo("=" * 60)
