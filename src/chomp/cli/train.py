"""Train subcommand.

Ported from scripts/train.py to use Click.
"""

from __future__ import annotations

from dataclasses import replace

import click

from chomp.cli.main import parse_resume, print_banner
from chomp.config import load_config
from chomp.utils.io import setup_python_logging


@click.command()
@click.argument("config", type=click.Path(exists=True))
@click.option(
    "--override",
    "-o",
    "overrides",
    multiple=True,
    help="Dotpath override, e.g. train.steps=1000 (repeatable).",
)
@click.option(
    "--run-dir",
    type=click.Path(),
    default=None,
    help="Override logging.run_dir (required for resume).",
)
@click.option(
    "--resume",
    "resume_raw",
    type=str,
    default="none",
    help="Resume from checkpoint: 'none' (default), 'latest', or an integer step.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate config, build model/data, compile one step, then exit.",
)
def train(
    config: str,
    overrides: tuple[str, ...],
    run_dir: str | None,
    resume_raw: str,
    dry_run: bool,
) -> None:
    """Train a Megalodon model.

    :param str config: Path to YAML config file.
    :param overrides: Config overrides in key=value format.
    :param run_dir: Override for logging.run_dir (optional).
    :param str resume_raw: Resume mode: 'none', 'latest', or step number.
    :param bool dry_run: If True, compile one step then exit.
    """
    print_banner()
    cfg = load_config(config, overrides=list(overrides))

    if run_dir is not None:
        cfg = replace(cfg, logging=replace(cfg.logging, run_dir=run_dir))

    resume = parse_resume(resume_raw)

    # Logging first so subsequent errors are readable
    setup_python_logging(cfg.logging.level, use_rich=cfg.logging.console_use_rich)

    # Deferred import keeps config-only CLI startup from initializing JAX.
    from chomp.train import TrainingPreempted, run

    try:
        run_dir_path = run(  # type: ignore[arg-type]
            cfg, config_path=config, resume=resume, dry_run=dry_run
        )
    except TrainingPreempted as exc:
        click.echo(f"[chomp] run_dir: {exc.run_dir}")
        click.echo(str(exc), err=True)
        raise click.exceptions.Exit(exc.exit_code) from exc
    click.echo(f"[chomp] run_dir: {run_dir_path}")
