"""Compilation stability smoke test.

Goal: protect the single most important invariant for JAX training repos:

  **fixed shapes => compile once**

We attempt to validate this by running a short training job with compilation logs enabled
and counting occurrences.

Caveat: JAX log formats vary across versions. This test is best-effort and will skip
if it cannot reliably detect `train_step` compilation lines.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(os.name == "nt", reason="subprocess env parsing differs on Windows")
def test_train_step_compiles_once(tmp_path: Path):
    config_src = Path(__file__).resolve().parents[1] / "configs" / "debug_smoke.yaml"
    assert config_src.exists(), "debug_smoke.yaml missing"

    # Run in an isolated working directory so run dirs don't clutter the repo.
    work = tmp_path / "run"
    work.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["JAX_LOG_COMPILES"] = "1"
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

    cmd = [
        sys.executable,
        "-c",
        f"from chomp.scripts.train import main; main(['--config','{config_src}'])",
    ]

    p = subprocess.run(cmd, cwd=str(work), env=env, capture_output=True, text=True)
    # We allow stderr logs; fail only if the process itself failed.
    assert p.returncode == 0, f"process failed\nstdout:\n{p.stdout}\nstderr:\n{p.stderr}"

    logs = p.stderr + "\n" + p.stdout

    # Try to find compilation lines for train_step. Common patterns:
    # - "Compiling train_step" (older)
    # - "Compiling <function train_step" (newer)
    pattern = re.compile(r"Compiling.*train_step")

    hits = set()
    for line in logs.splitlines():
        match = pattern.search(line)
        if match:
            # Normalize away logging prefixes so duplicate loggers don't double-count.
            hits.add(line[line.find("Compiling") :].strip())

    if not hits:
        pytest.skip("Could not detect train_step compilation logs; log format unsupported")

    assert len(hits) == 1, (
        f"Expected exactly 1 unique train_step compilation, saw {len(hits)}\nLogs:\n{logs}"
    )
