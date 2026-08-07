"""Main CLI entry point.

Defines the Click group and shared utilities.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import traceback
from typing import NoReturn

import click

from chomp import __version__

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


def _exit_status(code: int | str | None) -> int:
    """Normalize a ``SystemExit`` code to an integer process status.

    :param code: Value carried by ``SystemExit``.
    :return int: Integer exit status.
    """
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    print(code, file=sys.stderr)
    return 1


def _hard_exit(code: int) -> NoReturn:
    """Flush chomp-owned output, then exit without interpreter finalization.

    Native threads inside our dependencies can call back into CPython after
    ``Py_FinalizeEx`` has begun and kill the process with
    ``Fatal Python error: PyGILState_Release: thread state ... must be current
    when releasing``. The known offender here is Apache Arrow (arrow#45214),
    reached through Hugging Face streaming's Parquet readers; ``datasets``
    ships a sleep-based workaround and ``HFStreamingTextStream.close`` applies
    it twice, but a sleep narrows a race, it does not close it. A 100,000-step
    run observed the abort on 2026-08-07 after every step was done, the final
    checkpoint was durable, and the closing ``run_dir`` line had printed.

    The failure is therefore harmless to the run and still costs a truthful
    exit status: SIGABRT (128+6 = 134) makes a complete run look crashed to a
    shell, an ``&&`` chain, or a scheduler. Chomp closes what it owns before
    reaching here -- checkpoint manager, data iterator, metrics writer, W&B
    run, logging handlers -- so finalization has no remaining work of ours to
    perform, and ``atexit`` hooks are not load-bearing for us. Skipping both
    buys a deterministic exit code.

    :param int code: Exit status to report.
    """
    # Last-resort flush: nothing downstream could report a failure here, and a
    # closed stdout (piped to `head`, say) must not mask the real exit status.
    with contextlib.suppress(Exception):
        logging.shutdown()
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.flush()
    os._exit(code)


def main() -> NoReturn:
    """Console-script entry point for ``chomp``.

    Wraps the Click group so the process ends through :func:`_hard_exit`.
    Library callers and tests invoke :data:`cli` directly and keep ordinary
    interpreter shutdown, which is why this indirection exists at all rather
    than living inside the group.
    """
    code = 0
    try:
        cli.main(prog_name="chomp")
    except SystemExit as exc:
        code = _exit_status(exc.code)
    except BaseException:
        # Click already renders its own errors; anything reaching here is
        # unexpected and its traceback is the useful part.
        traceback.print_exc()
        code = 1
    _hard_exit(code)
