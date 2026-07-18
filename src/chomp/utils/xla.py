"""XLA and NVIDIA runtime inspection helpers."""

from __future__ import annotations

import os
import subprocess

_DETERMINISTIC_FLAG_PREFIX = "--xla_gpu_deterministic_ops="


def _query_nvidia_gpu_names() -> list[str]:
    """Best-effort GPU name query via nvidia-smi.

    :return list[str]: GPU names, or an empty list on failure.
    """
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except Exception:
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def deterministic_gpu_ops_setting() -> bool | None:
    """Effective xla_gpu_deterministic_ops value parsed from XLA_FLAGS.

    Recorded in the resume-compat fingerprint so a resumed process can warn
    when kernel determinism drifted across the resume boundary. Last
    occurrence wins, matching XLA's own flag parsing.

    :return bool | None: Parsed boolean, or None when the flag is unset.
    """
    setting: bool | None = None
    for tok in os.environ.get("XLA_FLAGS", "").split():
        if tok.startswith(_DETERMINISTIC_FLAG_PREFIX):
            setting = tok[len(_DETERMINISTIC_FLAG_PREFIX) :].strip().lower() == "true"
    return setting
