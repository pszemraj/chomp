"""Filesystem + metrics logging utilities.

chomp uses deliberately boring IO:
- a run directory containing config + metrics.jsonl
- JSONL is append-only and resilient (works even if the process crashes)

If you want wandb/tensorboard later, add it *on top*, not instead.

Phase 3 addendum:
- If you resume into an existing run_dir, we do not clobber the original
  config snapshot. We write a `config_resume.json` alongside it.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from chomp.config import Config


def setup_python_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def create_run_dir(cfg: Config, *, config_path: str | Path | None, allow_existing: bool = False) -> Path:
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
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("a", buffering=1)

    def write(self, row: dict[str, Any]) -> None:
        self._f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass

    def __enter__(self) -> "MetricsWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
