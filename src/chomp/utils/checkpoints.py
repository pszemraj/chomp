"""Checkpoint path and config resolution utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chomp.config import _from_nested_dict, _resolve_variables, validate_config


def _is_step_dir(path: Path) -> bool:
    """Return True if the path looks like a checkpoint step directory.

    :param Path path: Path to inspect.
    :return bool: True if the directory contains a train_state subdir.
    """
    return path.is_dir() and (path / "train_state").exists()


def _list_step_dirs(root: Path) -> list[Path]:
    """List numeric step directories under a checkpoint root.

    :param Path root: Checkpoint root directory.
    :return list[Path]: Sorted list of numeric step directories.
    """
    steps = [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
    steps.sort(key=lambda p: int(p.name))
    return steps


def _latest_step_dir(root: Path) -> Path | None:
    """Return the latest step dir with train_state, if any.

    :param Path root: Checkpoint root directory.
    :return Path | None: Latest step directory or None if not found.
    """
    for step_dir in reversed(_list_step_dirs(root)):
        if _is_step_dir(step_dir):
            return step_dir
    return None


def _find_run_dir_upwards(start: Path) -> Path | None:
    """Search upwards for a directory containing config_resolved.json.

    :param Path start: Starting path to search from.
    :return Path | None: First parent containing config_resolved.json, if any.
    """
    for parent in (start, *start.parents):
        if (parent / "config_resolved.json").exists():
            return parent
    return None


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file into a dict.

    :param Path path: Path to JSON file.
    :return dict[str, Any]: Parsed JSON mapping.
    """
    with path.open() as f:
        data = json.load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def _read_config_override(config_override: str) -> dict[str, Any]:
    """Read a config override file (YAML or JSON) into a dict.

    :param str config_override: Path to override config file.
    :return dict[str, Any]: Parsed config mapping.
    """
    config_path = Path(config_override)
    if not config_path.exists():
        raise FileNotFoundError(f"Config override not found: {config_path}")
    if config_path.suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "pyyaml is required to load YAML configs. Install with: pip install pyyaml"
            ) from exc
        try:
            with config_path.open() as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in {config_path}: {exc}") from exc
    else:
        try:
            data = _read_json(config_path)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {config_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Config override must be a mapping, got {type(data).__name__}")
    return data


def _read_run_dir_config(run_dir: Path) -> dict[str, Any]:
    """Read config_resolved.json from a run directory.

    :param Path run_dir: Run directory containing config_resolved.json.
    :return dict[str, Any]: Parsed config mapping.
    """
    config_path = run_dir / "config_resolved.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"config_resolved.json not found in {run_dir}. Use --config to provide a config file."
        )
    try:
        return _read_json(config_path)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Corrupted config_resolved.json in {run_dir}: {exc}") from exc


def _read_meta_config(step_dir: Path) -> dict[str, Any] | None:
    """Read the config snapshot from checkpoint metadata, if available.

    :param Path step_dir: Checkpoint step directory.
    :return dict[str, Any] | None: Config mapping if present, else None.
    """
    meta_dir = step_dir / "meta"
    meta_path = meta_dir / "metadata" if meta_dir.is_dir() else meta_dir
    if not meta_path.exists():
        return None
    try:
        meta = _read_json(meta_path)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Corrupted checkpoint metadata in {meta_path}: {exc}") from exc
    cfg = meta.get("config")
    if cfg is None:
        return None
    if not isinstance(cfg, dict):
        raise ValueError(f"Checkpoint metadata config must be a mapping, got {type(cfg).__name__}")
    return cfg


def _config_from_data(data: dict[str, Any]) -> Any:
    """Build and validate a Config from raw data.

    :param dict[str, Any] data: Raw config mapping.
    :return Config: Validated configuration.
    """
    data = _resolve_variables(data)
    cfg = _from_nested_dict(data)
    validate_config(cfg)
    return cfg


def load_config_for_checkpoint(
    *, step_dir: Path, run_dir: Path | None, config_override: str | None
) -> Any:
    """Load config for a checkpoint from override, run_dir, or checkpoint meta.

    :param Path step_dir: Checkpoint step directory.
    :param Path | None run_dir: Run directory if known.
    :param str | None config_override: Optional path to override config file.
    :return Config: Loaded configuration.
    :raises FileNotFoundError: If no config source can be found.
    """
    if config_override:
        data = _read_config_override(config_override)
        return _config_from_data(data)

    if run_dir is not None:
        try:
            data = _read_run_dir_config(run_dir)
            return _config_from_data(data)
        except FileNotFoundError:
            pass

    data = _read_meta_config(step_dir)
    if data is None:
        raise FileNotFoundError(
            f"No config found for checkpoint {step_dir}. "
            "Provide --config or ensure config_resolved.json is available."
        )
    return _config_from_data(data)


def _infer_run_dir_from_meta(step_dir: Path) -> Path | None:
    """Infer run_dir from checkpoint metadata if possible.

    :param Path step_dir: Checkpoint step directory.
    :return Path | None: Run directory from metadata, if present.
    """
    data = _read_meta_config(step_dir)
    if data is None:
        return None
    logging_cfg = data.get("logging")
    if isinstance(logging_cfg, dict):
        run_dir = logging_cfg.get("run_dir")
        if run_dir:
            path = Path(run_dir)
            if path.exists():
                return path
    return None


def resolve_checkpoint_path(
    checkpoint_path: str | Path, *, config_override: str | None = None
) -> tuple[Path, Path | None]:
    """Resolve a checkpoint path to a step directory and optional run_dir.

    :param str | Path checkpoint_path: Path to a run dir, checkpoint root, or step dir.
    :param str | None config_override: Optional config override path.
    :return tuple[Path, Path | None]: (step_dir, run_dir) where run_dir may be None.

    Supports:
    - Direct step directory: /path/to/ckpts/500
    - Checkpoint root directory: /path/to/ckpts
    - Run directory with config_resolved.json
    """
    path = Path(checkpoint_path).resolve()

    if _is_step_dir(path):
        run_dir = _find_run_dir_upwards(path)
        if run_dir is None:
            run_dir = _infer_run_dir_from_meta(path)
        return path, run_dir

    if (path / "config_resolved.json").exists():
        run_dir = path
        cfg = load_config_for_checkpoint(
            step_dir=path, run_dir=run_dir, config_override=config_override
        )
        ckpt_root = Path(cfg.checkpoint.root_dir) if cfg.checkpoint.root_dir else None
        if ckpt_root is None:
            ckpt_root = run_dir / "checkpoints"
        elif not ckpt_root.is_absolute():
            ckpt_root = run_dir / ckpt_root
        step_dir = _latest_step_dir(ckpt_root)
        if step_dir is None:
            raise FileNotFoundError(f"No step directories found in {ckpt_root}")
        return step_dir, run_dir

    step_dir = _latest_step_dir(path)
    if step_dir is not None:
        run_dir = None
        if path.name == "checkpoints":
            run_dir = _find_run_dir_upwards(path.parent)
        else:
            run_dir = _find_run_dir_upwards(path)
        if run_dir is None:
            run_dir = _infer_run_dir_from_meta(step_dir)
        return step_dir, run_dir

    if config_override is not None:
        run_dir = path
        cfg = load_config_for_checkpoint(
            step_dir=path, run_dir=run_dir, config_override=config_override
        )
        ckpt_root = Path(cfg.checkpoint.root_dir) if cfg.checkpoint.root_dir else None
        if ckpt_root is None:
            ckpt_root = run_dir / "checkpoints"
        elif not ckpt_root.is_absolute():
            ckpt_root = run_dir / ckpt_root
        step_dir = _latest_step_dir(ckpt_root)
        if step_dir is None:
            raise FileNotFoundError(f"No step directories found in {ckpt_root}")
        return step_dir, run_dir

    raise FileNotFoundError(
        f"Could not find checkpoint at {path}. Provide a run_dir, checkpoint root, or step dir."
    )
