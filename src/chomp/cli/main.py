"""Main CLI entry point.

Defines the Click group and shared utilities.
"""

from __future__ import annotations

import click

from chomp._version import __version__

BANNER = f"""
           __
  _____   / /_   ____    ____ ___     ____
 / ___/  / __ \\ / __ \\  / __ `__ \\   / __ \\
/ /__   / / / // /_/ / / / / / / /  / /_/ /
\\___/  /_/ /_/ \\____/ /_/ /_/ /_/  / .___/
                                  /_/

Version: {__version__}
Repo:    https://github.com/pszemraj/chomp
""".strip("\n")


def print_banner() -> None:
    """Print the chomp banner."""
    click.echo(BANNER)


def parse_resume(raw: str) -> str | int:
    """Parse the resume CLI argument.

    :param str raw: Raw string from --resume argument.
    :raises click.BadParameter: If raw is not a valid resume value.
    :return str | int: "none", "latest", or an integer step number.
    """
    raw = raw.strip().lower()
    if raw in {"none", "no", "false", "0"}:
        return "none"
    if raw in {"latest", "last"}:
        return "latest"
    try:
        step = int(raw)
        if step < 0:
            raise click.BadParameter(f"Resume step must be non-negative, got {step}.")
        return step
    except ValueError as e:
        raise click.BadParameter(
            f"Invalid resume value {raw!r}. Use 'none', 'latest', or an integer step."
        ) from e


@click.group()
@click.version_option(version=__version__, prog_name="chomp")
def cli() -> None:
    """Chomp: minimal, single-GPU JAX/Equinox pretraining harness for Megalodon."""


# Import and register subcommands
from chomp.cli.train import train  # noqa: E402

cli.add_command(train)

from chomp.cli.generate import generate  # noqa: E402

cli.add_command(generate)
