"""Generate subcommand for standalone text generation from checkpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import click

if TYPE_CHECKING:
    from collections.abc import Callable

    import jax

    from chomp.config import Config
    from chomp.data.pipeline import Tokenizer


def _generate_and_print(
    *,
    params: Any,
    static: Any,
    cfg: Config,
    tokenizer: Tokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    gen_key: jax.Array,
    generate_tokens: Callable[..., tuple[list[int], Any]],
) -> None:
    """Tokenize, generate, and print, given a model that is already loaded.

    Shared by both load paths: a training checkpoint restored through Orbax and
    an export directory rebuilt from safetensors arrive here as the same
    ``(params, static)`` pair.

    :param Any params: Model parameters.
    :param Any static: Static model components.
    :param Config cfg: Config supplying the special-token IDs.
    :param Tokenizer tokenizer: Tokenizer bound to these weights.
    :param str prompt: Text prompt for generation.
    :param int max_tokens: Maximum number of tokens to generate.
    :param float temperature: Sampling temperature (0 for greedy).
    :param top_k: Top-k sampling cutoff (optional).
    :param top_p: Nucleus sampling threshold (optional).
    :param gen_key: PRNG key for sampling.
    :param generate_tokens: Backend generation entry point.
    :raises click.ClickException: If the prompt is empty or generation fails.
    """
    prompt_tokens = tokenizer.encode(prompt)
    if not prompt_tokens:
        raise click.ClickException("Prompt tokenized to empty sequence")

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
    except Exception as exc:
        raise click.ClickException(f"Generation failed: {exc}") from exc

    click.echo("\n" + "=" * 60)
    click.echo("Prompt:")
    click.echo(prompt)
    click.echo("-" * 60)
    click.echo("Generated:")
    click.echo(tokenizer.decode(gen_tokens))
    click.echo("=" * 60)


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
    help=(
        "Override config file (defaults to selected checkpoint metadata; "
        "legacy checkpoints use the run config)."
    ),
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

    :param str checkpoint: Path to a run directory, checkpoint step directory,
        or a directory produced by ``chomp export``.
    :param str prompt: Text prompt for generation.
    :param int max_tokens: Maximum number of tokens to generate.
    :param float temperature: Sampling temperature (0 for greedy).
    :param top_k: Top-k sampling cutoff (optional).
    :param top_p: Nucleus sampling threshold (optional).
    :param int seed: Random seed for sampling.
    :param config_override: Path to override config file (optional).
    """
    # Deferred imports keep CLI startup from initializing JAX before arguments are validated.
    import jax

    from chomp.ckpt import restore_params_only
    from chomp.data.pipeline import (
        load_tokenizer_snapshot,
        load_tokenizer_snapshot_for_resume,
        prepare_tokenizer_and_config,
    )
    from chomp.export import is_export_dir, load_export, load_export_tokenizer
    from chomp.model import build_model, generate_tokens
    from chomp.utils.ckpt_paths import (
        load_config_for_checkpoint,
        read_checkpoint_meta,
        resolve_checkpoint_path,
    )
    from chomp.utils.tree import abstractify_tree

    key = jax.random.key(seed)
    model_key, gen_key = jax.random.split(key)

    # An export directory is already the model: upstream rebuilds it from the
    # config in the safetensors header, so none of the run-directory config,
    # tokenizer-identity, or Orbax restore machinery below applies.
    if is_export_dir(checkpoint):
        if config_override:
            # Refusing beats ignoring: the architecture comes from the weights
            # header here, so an override could only disagree with it.
            raise click.ClickException(
                "--config does not apply to an export directory; its config is stored "
                "in the export itself. Generate from the run directory to override it."
            )
        click.echo(f"Loading exported model from: {checkpoint}")
        try:
            loaded = load_export(checkpoint, key=model_key)
            tokenizer = load_export_tokenizer(checkpoint, loaded.config)
        except Exception as exc:
            raise click.ClickException(str(exc)) from exc
        _generate_and_print(
            params=loaded.params,
            static=loaded.static,
            cfg=loaded.config,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            gen_key=gen_key,
            generate_tokens=generate_tokens,
        )
        return

    # Find checkpoint and load config
    try:
        step_dir, run_dir = resolve_checkpoint_path(checkpoint)
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

    try:
        checkpoint_meta = read_checkpoint_meta(step_dir)
    except FileNotFoundError:
        checkpoint_meta = None
    checkpoint_tokenizer_identity = (
        None if checkpoint_meta is None else checkpoint_meta.get("tokenizer_identity")
    )
    if (
        checkpoint_meta is not None
        and checkpoint_meta.get("schema_version") in {2, 3}
        and not isinstance(checkpoint_tokenizer_identity, dict)
    ):
        raise click.ClickException(
            "Checkpoint metadata is missing tokenizer_identity; cannot verify generation tokens."
        )

    # Prefer the run-pinned tokenizer so mutable upstream tokenizer revisions
    # cannot reinterpret the restored embedding rows.
    tokenizer = None
    if isinstance(checkpoint_tokenizer_identity, dict):
        if run_dir is None or not (run_dir / "tokenizer").exists():
            raise click.ClickException(
                "Checkpoint requires its run-pinned tokenizer snapshot for generation."
            )
        try:
            tokenizer, observed_tokenizer_identity = load_tokenizer_snapshot_for_resume(
                run_dir, cfg
            )
        except Exception as exc:
            raise click.ClickException(str(exc)) from exc
        # Token IDs directly index the restored embedding rows, so effective
        # tokenizer drift is never a meaningful generation override.
        if observed_tokenizer_identity != checkpoint_tokenizer_identity:
            raise click.ClickException(
                "Tokenizer identity does not match the selected checkpoint; refusing "
                "generation because token IDs may not match its embedding rows."
            )
    elif (
        cfg.data.tokenizer.kind == "hf" and run_dir is not None and (run_dir / "tokenizer").exists()
    ):
        try:
            tokenizer = load_tokenizer_snapshot(run_dir, cfg)
        except Exception as exc:
            raise click.ClickException(str(exc)) from exc
    cfg, tokenizer = prepare_tokenizer_and_config(cfg, tokenizer=tokenizer)

    # Build model skeleton for abstract shapes
    params, static = build_model(cfg, key=model_key)

    # Restore params from checkpoint
    click.echo("Restoring model parameters...")
    try:
        params = restore_params_only(step_dir, abstractify_tree(params))
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc

    _generate_and_print(
        params=params,
        static=static,
        cfg=cfg,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        gen_key=gen_key,
        generate_tokens=generate_tokens,
    )
