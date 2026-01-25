"""CLI entrypoints for chomp.

Invoked via ``pyproject.toml`` entrypoints::

    chomp train <config.yaml> ...
    chomp generate <checkpoint> --prompt "Hello"

Keep these modules thin: argument parsing + calling into library code.
"""

from __future__ import annotations

__all__ = ["cli"]

from chomp.cli.main import cli
