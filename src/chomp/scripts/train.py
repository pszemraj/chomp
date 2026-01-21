"""`chomp-train` CLI.

We intentionally keep this thin:
- parse args
- load config (YAML + overrides)
- apply a couple CLI conveniences (run_dir, resume)
- validate device backend
- call `chomp.train.run`

Modern packaging note:
- CLI is defined in pyproject.toml (`[project.scripts]`).
  Don't use `python -m ...` patterns.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from chomp.config import load_config
from chomp.train import run
from chomp.utils.devices import validate_default_device
from chomp.utils.io import setup_python_logging


def _parse_resume(raw: str) -> str | int:
    raw = raw.strip().lower()
    if raw in {"none", "no", "false", "0"}:
        return "none"
    if raw in {"latest", "last"}:
        return "latest"
    try:
        return int(raw)
    except Exception as e:
        raise ValueError(
            f"Invalid --resume {raw!r}. Use 'none', 'latest', or an integer step."
        ) from e


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="chomp-train")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--override",
        type=str,
        action="append",
        default=[],
        help="Dotpath override, e.g. train.steps=1000 (repeatable)",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Override logging.run_dir (required for resume).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="none",
        help="Resume from checkpoint: 'none' (default), 'latest', or an integer step.",
    )

    args = parser.parse_args(argv)

    cfg = load_config(args.config, overrides=args.override)

    if args.run_dir is not None:
        cfg = replace(cfg, logging=replace(cfg.logging, run_dir=args.run_dir))

    resume = _parse_resume(args.resume)

    # Logging first so subsequent errors are readable
    setup_python_logging(cfg.logging.level)

    # Fail fast on CPU unless explicitly allowed
    validate_default_device(allow_cpu=cfg.train.allow_cpu)

    run_dir = run(cfg, config_path=args.config, resume=resume)  # type: ignore[arg-type]
    print(f"[chomp] run_dir: {run_dir}")


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
