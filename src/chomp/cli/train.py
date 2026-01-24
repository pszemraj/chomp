"""Train subcommand.

Ported from scripts/train.py to use Click.
"""

from __future__ import annotations

from dataclasses import replace

import click

from chomp.cli.main import parse_resume
from chomp.config import load_config
from chomp.utils.io import setup_python_logging
from chomp.utils.xla import configure_blackwell_xla_env


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

    CONFIG is the path to a YAML config file.
    """
    cfg = load_config(config, overrides=list(overrides))

    if run_dir is not None:
        cfg = replace(cfg, logging=replace(cfg.logging, run_dir=run_dir))

    resume = parse_resume(resume_raw)

    # Logging first so subsequent errors are readable
    setup_python_logging(cfg.logging.level, use_rich=cfg.logging.console_use_rich)

    # Configure XLA env quirks before JAX backend init.
    configure_blackwell_xla_env()

    from chomp.train import run
    from chomp.utils.devices import validate_default_device

    # Fail fast on CPU unless explicitly allowed
    validate_default_device(allow_cpu=cfg.train.allow_cpu)

    run_dir_path = run(cfg, config_path=config, resume=resume, dry_run=dry_run)  # type: ignore[arg-type]
    click.echo(f"[chomp] run_dir: {run_dir_path}")
