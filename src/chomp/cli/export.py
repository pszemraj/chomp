"""Export subcommand for writing portable safetensors weights."""

from __future__ import annotations

import click


@click.command()
@click.argument("checkpoint", type=click.Path(exists=True))
@click.option(
    "--out",
    "-o",
    "export_dir",
    required=True,
    type=click.Path(),
    help="Destination directory for the exported model.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Replace an existing export in the destination directory.",
)
@click.option(
    "--verify/--no-verify",
    default=True,
    help="Re-read the written weights and compare them to what was exported (default: on).",
)
@click.option(
    "--dtype",
    # Spelled out rather than imported from chomp.export, which pulls in JAX at
    # import time; test_cli_dtype_choices_match_the_exporter keeps them in step.
    type=click.Choice(["float32", "policy"]),
    default="float32",
    help=(
        "Weight dtypes to write. 'float32' is the master weights, exactly as trained. "
        "'policy' re-encodes them at megalodon-jax's bf16 policy dtypes: about half "
        "the size, and inference-equivalent (default: float32)."
    ),
)
@click.option(
    "--config",
    "config_override",
    type=click.Path(exists=True),
    default=None,
    help="Override config file (defaults to the selected checkpoint's own metadata).",
)
def export(
    checkpoint: str,
    export_dir: str,
    overwrite: bool,
    verify: bool,
    dtype: str,
    config_override: str | None,
) -> None:
    """Export a checkpoint's weights as portable safetensors.

    CHECKPOINT is a run directory, a checkpoint root, or a step directory. The
    resulting directory holds the weights, the run's tokenizer, and a manifest,
    and can be loaded without chomp via megalodon_jax.load_checkpoint.

    :param str checkpoint: Path to a run dir, checkpoint root, or step dir.
    :param str export_dir: Destination directory.
    :param bool overwrite: Whether to replace an existing export.
    :param bool verify: Whether to reload the written weights and compare.
    :param str dtype: Weight dtype variant to write.
    :param config_override: Path to override config file (optional).
    """
    # Deferred import keeps CLI startup from initializing JAX before arguments
    # are validated.
    from chomp.export import export_checkpoint

    try:
        result = export_checkpoint(
            checkpoint,
            export_dir,
            config_override=config_override,
            overwrite=overwrite,
            verify=verify,
            dtype=dtype,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    size = (
        f"{result.weights_bytes / 1e9:.2f} GB"
        if result.weights_bytes >= 1e9
        else f"{result.weights_bytes / 1e6:.1f} MB"
    )
    click.echo(f"Exported step {result.step} to {result.export_dir}")
    click.echo(
        f"  {result.weights_path.name}: {result.param_count:,} params, {size}, "
        f"{result.weights_dtype}" + (" (verified)" if result.verified else " (NOT verified)")
    )
    click.echo(f"  {result.config_path.name}: architecture, Hugging Face layout")
    if result.tokenizer_files:
        click.echo(f"  tokenizer: {', '.join(result.tokenizer_files)}")
    else:
        click.echo("  tokenizer: none shipped (byte tokenizer or no run snapshot)")
