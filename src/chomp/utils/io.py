"""Filesystem + metrics logging utilities.

chomp uses deliberately boring IO:
- a run directory containing config + metrics.jsonl
- JSONL is append-only and resilient (works even if the process crashes)

W&B integration is optional and configured via logging.wandb_*.

Phase 3 addendum:
- If you resume into an existing run_dir, we do not clobber the original
  config snapshot. We write a `config_resume.json` alongside it.
"""

from __future__ import annotations

import contextlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from chomp.config import Config

_NOISY_CONSOLE_PREFIXES = ("orbax", "jax", "jaxlib", "absl")


class _ConsoleNoiseFilter(logging.Filter):
    """Filter that hides noisy third-party INFO logs from the console."""

    def filter(self, record: logging.LogRecord) -> bool:
        for prefix in _NOISY_CONSOLE_PREFIXES:
            if record.name.startswith(prefix):
                return record.levelno >= logging.WARNING
        return True


def _console_handler(level: int, *, use_rich: bool) -> logging.Handler:
    """Build a console handler with optional Rich formatting."""

    if use_rich:
        try:
            from rich.logging import RichHandler

            handler: logging.Handler = RichHandler(
                show_time=True,
                show_level=True,
                show_path=False,
                markup=True,
                rich_tracebacks=False,
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
        except Exception:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

    handler.setLevel(level)
    handler.addFilter(_ConsoleNoiseFilter())
    return handler


def setup_python_logging(level: str, *, use_rich: bool = True) -> None:
    """Configure Python logging with a console handler.

    :param str level: Log level name (DEBUG, INFO, WARNING, ERROR).
    :param bool use_rich: If True, use Rich for nicer console logs when available.
    """
    numeric_level = getattr(logging, level, logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric_level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.addHandler(_console_handler(numeric_level, use_rich=use_rich))


def add_file_logging(path: Path, *, level: str) -> None:
    """Attach a file handler that captures all logs.

    :param Path path: Log file path.
    :param str level: Log level name (DEBUG, INFO, WARNING, ERROR).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == str(path):
            return
    file_handler = logging.FileHandler(path)
    file_handler.setLevel(getattr(logging, level, logging.INFO))
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    root.addHandler(file_handler)


def create_run_dir(
    cfg: Config, *, config_path: str | Path | None, allow_existing: bool = False
) -> Path:
    """Create (or reuse) a run directory.

    - If cfg.logging.run_dir is None: always create a fresh timestamped run dir.
      (Resume is not possible because we don't know which directory to use.)

    - If cfg.logging.run_dir is set:
      - if it doesn't exist: create it
      - if it exists:
         - allow_existing=True  => treat as resume/continue
         - allow_existing=False => error (refuse to clobber)

    We persist config snapshots:
    - fresh run: config_resolved.json + optional config_original.yaml
    - resume:    config_resume.json (so you can see how you invoked resume)

    :param Config cfg: Training configuration.
    :param config_path: Optional path to original YAML config.
    :param bool allow_existing: If True, allow reusing an existing directory.
    :raises RuntimeError: If directory exists and allow_existing=False, or resume without run_dir.
    :return Path: Path to the run directory.
    """

    if cfg.logging.run_dir is not None:
        run_dir = Path(cfg.logging.run_dir)
        if run_dir.exists():
            if not allow_existing:
                raise RuntimeError(
                    f"Run dir already exists: {run_dir}. "
                    "Refusing to clobber. Set logging.run_dir to a new path or pass --resume."
                )
        else:
            run_dir.mkdir(parents=True, exist_ok=False)
    else:
        if allow_existing:
            raise RuntimeError(
                "Resume requested but logging.run_dir is null. "
                "Set logging.run_dir to an existing run directory to resume."
            )
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = "run"
        if config_path is not None:
            name = Path(config_path).stem
        run_dir = Path("runs") / cfg.logging.project / f"{stamp}_{name}"
        run_dir.mkdir(parents=True, exist_ok=False)

    # Save config snapshot (avoid clobbering on resume)
    if (run_dir / "config_resolved.json").exists() and allow_existing:
        (run_dir / "config_resume.json").write_text(
            json.dumps(cfg.to_dict(), indent=2, sort_keys=True)
        )
    else:
        (run_dir / "config_resolved.json").write_text(
            json.dumps(cfg.to_dict(), indent=2, sort_keys=True)
        )

        # Also copy original config file if available
        if config_path is not None:
            src = Path(config_path)
            if src.exists():
                (run_dir / "config_original.yaml").write_text(src.read_text())

    return run_dir


class MetricsWriter:
    """Append-only JSONL metrics writer."""

    def __init__(self, path: str | Path):
        """Initialize the metrics writer.

        :param path: Path to the JSONL file.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("a", buffering=1)

    def write(self, row: dict[str, Any]) -> None:
        """Write a metrics row to the JSONL file.

        :param dict[str, Any] row: Dictionary of metrics to write.
        """
        self._f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        """Close the file handle."""
        with contextlib.suppress(Exception):
            self._f.close()

    def __enter__(self) -> MetricsWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
