"""XLA and NVIDIA runtime inspection helpers."""

from __future__ import annotations

import os

_DETERMINISTIC_FLAG_PREFIX = "--xla_gpu_deterministic_ops="


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
