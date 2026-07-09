"""XLA environment helpers for GPU-specific quirks."""

from __future__ import annotations

import logging
import os
import re
import subprocess

_TRITON_FLAG = "--xla_gpu_enable_triton_gemm=false"
_DETERMINISTIC_FLAG_PREFIX = "--xla_gpu_deterministic_ops="
_DETERMINISTIC_FLAG = _DETERMINISTIC_FLAG_PREFIX + "true"
_PREALLOC_ENV = "XLA_PYTHON_CLIENT_PREALLOCATE"
_RTX_RE = re.compile(r"RTX\s*(\d{4})", re.IGNORECASE)
_CONFIG_DONE = False
_LAST_RESULT: bool | None = None


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


def _is_rtx_50xx(name: str) -> bool:
    """Return True if a GPU name looks like an RTX 50xx (Blackwell) GeForce card.

    :param str name: GPU name string.
    :return bool: True if the name indicates an RTX 50xx GeForce GPU.
    """
    match = _RTX_RE.search(name)
    if not match:
        return False
    try:
        model_num = int(match.group(1))
    except ValueError:
        return False
    if model_num < 5000 or model_num >= 5100:
        return False
    return not (model_num == 5000 and "geforce" not in name.lower())


def _detect_blackwell(names: list[str]) -> list[str]:
    """Filter GPU names for RTX 50xx matches.

    :param list[str] names: GPU name strings.
    :return list[str]: Names that match the RTX 50xx pattern.
    """
    return [name for name in names if _is_rtx_50xx(name)]


def _update_xla_flags(existing: str) -> tuple[str, bool]:
    """Ensure the Triton GEMM flag is present and conflicting entries are removed.

    :param str existing: Existing XLA_FLAGS value.
    :return tuple[str, bool]: (updated_flags, changed)
    """
    tokens = [tok for tok in existing.split() if tok]
    filtered = [tok for tok in tokens if not tok.startswith("--xla_gpu_enable_triton_gemm=")]
    changed = len(filtered) != len(tokens)
    if _TRITON_FLAG not in filtered:
        filtered.append(_TRITON_FLAG)
        changed = True
    return " ".join(filtered).strip(), changed


def _log_prealloc_status(log: logging.Logger, prealloc: str | None) -> None:
    """Log warnings or info about XLA preallocation settings.

    :param logging.Logger log: Logger to emit messages.
    :param str | None prealloc: Value of preallocation env var.
    """
    if prealloc is None or str(prealloc).strip() == "":
        log.warning(
            "%s is not set. Recommended: %s=false to avoid full GPU preallocation.",
            _PREALLOC_ENV,
            _PREALLOC_ENV,
        )
    elif str(prealloc).lower() not in {"false", "0", "no"}:
        log.warning(
            "%s=%s (recommended: false) to avoid full GPU preallocation.",
            _PREALLOC_ENV,
            prealloc,
        )
    else:
        log.info("%s=%s", _PREALLOC_ENV, prealloc)


def deterministic_gpu_ops_setting() -> bool | None:
    """Effective xla_gpu_deterministic_ops value parsed from XLA_FLAGS.

    This is the value the resume-compat fingerprint records: bit-exact resume
    on GPU holds only when the interrupted and resumed processes agree on it.
    Last occurrence wins, matching XLA's own flag parsing.

    :return bool | None: Parsed boolean, or None when the flag is unset.
    """
    setting: bool | None = None
    for tok in os.environ.get("XLA_FLAGS", "").split():
        if tok.startswith(_DETERMINISTIC_FLAG_PREFIX):
            setting = tok[len(_DETERMINISTIC_FLAG_PREFIX) :].strip().lower() == "true"
    return setting


def warn_if_gpu_determinism_unknown(*, logger: logging.Logger | None = None) -> bool:
    """Warn when JAX initialized a GPU backend with no determinism setting.

    Fallback for hosts where nvidia-smi is unavailable (containers without
    the binary, blocked drivers): ensure_deterministic_gpu_ops() cannot see
    the GPU there, so XLA_FLAGS stays untouched and bit-exact resume is not
    guaranteed. Must be called after JAX backend init — by then it is too
    late to set the flag, so this warns instead of fixing.

    :param logger: Optional logger override.
    :return bool: True if the warning fired.
    """
    import jax

    if jax.default_backend() != "gpu" or deterministic_gpu_ops_setting() is not None:
        return False
    log = logger or logging.getLogger(__name__)
    log.warning(
        "JAX is on a GPU backend but %s was not set (nvidia-smi unavailable at "
        "startup?). Bit-exact resume is NOT guaranteed; export XLA_FLAGS=%s "
        "before launch to restore the contract.",
        _DETERMINISTIC_FLAG_PREFIX.rstrip("="),
        _DETERMINISTIC_FLAG,
    )
    return True


def ensure_deterministic_gpu_ops(*, logger: logging.Logger | None = None) -> bool:
    """Default XLA GPU ops to deterministic so bit-exact resume actually holds.

    Without --xla_gpu_deterministic_ops=true, XLA may pick nondeterministic
    GPU kernels (atomic-reduction scatters, algorithm choices sensitive to
    prior GPU state), so an interrupted+resumed run diverges from the
    continuous one in the low bits of optimizer state — silently voiding the
    harness's core resume contract. An explicit user setting in XLA_FLAGS
    (either value) is respected; call before JAX initializes its backend.

    :param logger: Optional logger override.
    :return bool: True if the flag was appended to XLA_FLAGS.
    """
    log = logger or logging.getLogger(__name__)
    if not _query_nvidia_gpu_names():
        return False

    if deterministic_gpu_ops_setting() is not None:
        log.info("XLA_FLAGS already sets xla_gpu_deterministic_ops; leaving it as-is.")
        return False
    existing = os.environ.get("XLA_FLAGS", "")
    os.environ["XLA_FLAGS"] = f"{existing} {_DETERMINISTIC_FLAG}".strip()
    log.info(
        "Setting %s for bit-exact resume. Pre-set %sfalse in XLA_FLAGS to trade "
        "exactness for throughput.",
        _DETERMINISTIC_FLAG,
        _DETERMINISTIC_FLAG_PREFIX,
    )
    return True


def configure_blackwell_xla_env(
    *, logger: logging.Logger | None = None, force: bool = False
) -> bool:
    """Configure XLA env vars for RTX 50xx GPUs.

    Returns True if an RTX 50xx GPU was detected.

    :param logger: Optional logger override.
    :param bool force: If True, re-run even if already configured.
    :return bool: True if an RTX 50xx GPU was detected.
    """
    global _CONFIG_DONE, _LAST_RESULT
    if _CONFIG_DONE and not force:
        return bool(_LAST_RESULT)

    log = logger or logging.getLogger(__name__)
    names = _query_nvidia_gpu_names()
    if not names:
        log.debug("No NVIDIA GPUs detected via nvidia-smi; skipping Blackwell XLA setup.")
        _CONFIG_DONE = True
        _LAST_RESULT = False
        return False
    blackwell = _detect_blackwell(names)
    if not blackwell:
        log.debug("NVIDIA GPU(s) detected but not RTX 50xx: %s", ", ".join(names))
        _CONFIG_DONE = True
        _LAST_RESULT = False
        return False

    log.info("Detected RTX 50xx GPU(s): %s", ", ".join(blackwell))

    existing = os.environ.get("XLA_FLAGS", "")
    updated, changed = _update_xla_flags(existing)
    if changed:
        os.environ["XLA_FLAGS"] = updated
        if existing:
            log.info("Updated XLA_FLAGS to include %s.", _TRITON_FLAG)
        else:
            log.info("Setting XLA_FLAGS=%s", updated)
    else:
        log.info("XLA_FLAGS already contains %s.", _TRITON_FLAG)

    prealloc = os.environ.get(_PREALLOC_ENV)
    _log_prealloc_status(log, prealloc)

    _CONFIG_DONE = True
    _LAST_RESULT = True
    return True
