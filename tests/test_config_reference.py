# SPDX-License-Identifier: Apache-2.0

"""Executable check for the human-maintained annotated config reference."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from chomp.config import Config, build_config, read_config_mapping

REFERENCE_PATH = Path(__file__).parents[1] / "docs" / "config-reference.yaml"


def test_config_reference_is_valid_default_config() -> None:
    """The copyable reference must parse and build the default configuration."""
    reference = read_config_mapping(REFERENCE_PATH)
    assert build_config(reference) == Config()


def _missing_field_paths(node: Any, mapping: dict[str, Any], prefix: str = "") -> list[str]:
    """Collect dotted paths of dataclass fields absent from a nested mapping."""
    missing: list[str] = []
    for field in dataclasses.fields(node):
        path = f"{prefix}{field.name}"
        if field.name not in mapping:
            missing.append(path)
            continue
        value = getattr(node, field.name)
        if dataclasses.is_dataclass(value):
            sub = mapping[field.name]
            if not isinstance(sub, dict):
                missing.append(f"{path} (expected a mapping)")
                continue
            missing.extend(_missing_field_paths(value, sub, prefix=f"{path}."))
    return missing


def test_config_reference_documents_every_field() -> None:
    """Every config field must appear in the reference, even at its default.

    build_config silently fills omitted keys from dataclass defaults, so the
    default-equality test alone cannot catch a new knob that was never added
    to the reference YAML.
    """
    reference = read_config_mapping(REFERENCE_PATH)
    missing = _missing_field_paths(Config(), reference)
    assert not missing, (
        "config fields missing from docs/config-reference.yaml (add them with a "
        f"comment describing the knob): {missing}"
    )
