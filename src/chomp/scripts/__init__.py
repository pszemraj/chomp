"""CLI entrypoints live here.

They are invoked via `pyproject.toml` entrypoints, e.g.:
  chomp-train <config.yaml> ...

Keep these modules thin: argument parsing + calling into library code.
"""

from __future__ import annotations
