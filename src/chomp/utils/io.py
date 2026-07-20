"""Filesystem + metrics logging utilities.

chomp uses deliberately boring IO:
- a run directory containing config + metrics.jsonl
- JSONL is append-only and resilient (works even if the process crashes)

W&B integration is optional and configured via logging.wandb.*.

Resuming does not clobber the original config snapshot.
"""

from __future__ import annotations

import contextlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from chomp.config import Config, resolve_decay_horizon

_NOISY_CONSOLE_PREFIXES = ("orbax", "jax", "jaxlib", "absl")


def resolve_run_dir(cfg: Config, *, config_path: str | Path | None) -> Path:
    """Resolve the run directory before any run-owned artifact is written.

    :param Config cfg: Training configuration.
    :param config_path: Optional source config path used in generated names.
    :return Path: Explicit or timestamp-derived run directory.
    """
    if cfg.logging.run_dir is not None:
        return Path(cfg.logging.run_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = Path(config_path).stem if config_path is not None else "run"
    return Path("runs") / cfg.logging.project / f"{stamp}_{name}"


class _ConsoleNoiseFilter(logging.Filter):
    """Filter that hides noisy third-party INFO logs from the console."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter a log record based on module prefix and level.

        :param logging.LogRecord record: Log record to evaluate.
        :return bool: True if record should be shown on console.
        """
        for prefix in _NOISY_CONSOLE_PREFIXES:
            if record.name.startswith(prefix):
                return record.levelno >= logging.WARNING
        return True


def _console_handler(level: int, *, use_rich: bool) -> logging.Handler:
    """Build a console handler with optional Rich formatting.

    :param int level: Logging level to set on the handler.
    :param bool use_rich: If True, use RichHandler.
    :return logging.Handler: Configured console handler.
    """

    if use_rich:
        from rich.logging import RichHandler

        handler: logging.Handler = RichHandler(
            show_time=True,
            show_level=True,
            show_path=False,
            markup=True,
            rich_tracebacks=False,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

    handler.setLevel(level)
    handler.addFilter(_ConsoleNoiseFilter())
    return handler


def setup_python_logging(level: str, *, use_rich: bool = True) -> None:
    """Configure Python logging with a console handler.

    :param str level: Log level name (DEBUG, INFO, WARNING, ERROR).
    :param bool use_rich: If True, use Rich for nicer console logs.
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

    - If cfg.logging.run_dir is set for a fresh run, create it or refuse to
      clobber an existing directory.
    - If allow_existing=True, require that explicit run directory to exist.

    Fresh runs persist config_resolved.json and optional config_original.yaml.

    :param Config cfg: Training configuration.
    :param config_path: Optional path to original YAML config.
    :param bool allow_existing: If True, allow reusing an existing directory.
    :raises RuntimeError: If the run directory conflicts with the requested fresh/resume mode.
    :return Path: Path to the run directory.
    """

    run_dir = resolve_run_dir(cfg, config_path=config_path)
    if cfg.logging.run_dir is not None:
        if allow_existing and not run_dir.exists():
            raise RuntimeError(f"Resume requested but run directory does not exist: {run_dir}")
        if not allow_existing:
            if run_dir.exists():
                raise RuntimeError(
                    f"Run dir already exists: {run_dir}. "
                    "Refusing to clobber. Set logging.run_dir to a new path or pass --resume."
                )
            run_dir.mkdir(parents=True, exist_ok=False)
    else:
        if allow_existing:
            raise RuntimeError(
                "Resume requested but logging.run_dir is null. "
                "Set logging.run_dir to an existing run directory to resume."
            )
        run_dir.mkdir(parents=True, exist_ok=False)

    # Save config snapshot (avoid clobbering on resume)
    resolved_cfg = cfg.to_dict()
    resolved_cfg["derived"] = {
        "optim": {
            "decay_steps_effective": int(resolve_decay_horizon(cfg)),
        }
    }
    if not allow_existing:
        (run_dir / "config_resolved.json").write_text(
            json.dumps(resolved_cfg, indent=2, sort_keys=True)
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
        # Text-mode line buffering flushes every newline-terminated JSONL row.
        self._f = self.path.open("a", buffering=1)

    def write(self, row: dict[str, Any]) -> None:
        """Write a metrics row to the JSONL file.

        :param dict[str, Any] row: Dictionary of metrics to write.
        """
        self._f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def close(self) -> None:
        """Close the file handle."""
        with contextlib.suppress(Exception):
            self._f.close()

    def __enter__(self) -> MetricsWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
