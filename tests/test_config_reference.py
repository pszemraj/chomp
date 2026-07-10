# SPDX-License-Identifier: Apache-2.0

"""Drift checks for the human-maintained annotated config reference."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from chomp.config import Config, build_config, read_config_mapping

REFERENCE_PATH = Path(__file__).parents[1] / "docs" / "config-reference.yaml"
_KEY_MARKER = re.compile(r"^\s*# Key: (\S+)\s*$", re.MULTILINE)


def _mapping_paths(value: dict[str, Any], prefix: str = "") -> set[str]:
    """Return every mapping key as a dotted path.

    :param dict[str, Any] value: Mapping to traverse.
    :param str prefix: Dotted parent path.
    :return set[str]: Container and leaf paths.
    """
    paths: set[str] = set()
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else key
        paths.add(path)
        if isinstance(child, dict):
            paths.update(_mapping_paths(child, path))
    return paths


def _yaml_compatible(value: Any) -> Any:
    """Normalize tuples and other JSON-compatible containers for YAML comparison.

    :param Any value: Dataclass-derived configuration value.
    :return Any: Value using only YAML/JSON collection types.
    """
    return json.loads(json.dumps(value))


def _read_reference() -> tuple[str, dict[str, Any]]:
    """Read the annotated reference as text and YAML.

    :return tuple[str, dict[str, Any]]: Source text and parsed mapping.
    """
    text = REFERENCE_PATH.read_text()
    return text, read_config_mapping(REFERENCE_PATH)


def _assert_same_typed_tree(actual: Any, expected: Any, path: str = "") -> None:
    """Assert recursive value and exact scalar/container type equality.

    :param Any actual: Parsed reference value.
    :param Any expected: Dataclass-derived default value.
    :param str path: Dotted path used in assertion messages.
    """
    assert type(actual) is type(expected), (
        f"{path or '<root>'} has type {type(actual).__name__}; expected {type(expected).__name__}"
    )
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            child = f"{path}.{key}" if path else key
            _assert_same_typed_tree(actual[key], expected[key], child)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True)):
            _assert_same_typed_tree(actual_item, expected_item, f"{path}[{index}]")
    else:
        assert actual == expected, path


def test_config_reference_is_valid_default_config() -> None:
    """The copyable reference remains valid and preserves every literal default."""
    _, reference = _read_reference()
    assert reference["variables"] == {}
    assert reference["derived"] == {}

    authored = {
        key: value for key, value in reference.items() if key not in {"variables", "derived"}
    }
    expected = _yaml_compatible(Config().to_dict())
    _assert_same_typed_tree(authored, expected)
    assert build_config(reference) == Config()


def test_config_reference_key_inventory_matches_schema() -> None:
    """Every accepted schema key appears exactly once and no stale key survives."""
    text, reference = _read_reference()
    expected_paths = _mapping_paths(
        {"variables": {}, "derived": {}, **_yaml_compatible(Config().to_dict())}
    )
    assert _mapping_paths(reference) == expected_paths

    markers = _KEY_MARKER.findall(text)
    assert len(markers) == len(set(markers)), "duplicate '# Key:' annotation"
    assert set(markers) == expected_paths


def test_config_reference_keys_have_inline_contracts() -> None:
    """Each key keeps the required contract labels adjacent to its YAML value."""
    text, _ = _read_reference()
    lines = text.splitlines()
    blocks: dict[str, list[str]] = {}

    for index, line in enumerate(lines):
        match = _KEY_MARKER.match(line)
        if match is None:
            continue
        block: list[str] = []
        for following in lines[index + 1 :]:
            stripped = following.lstrip()
            if not stripped.startswith("#"):
                break
            block.append(stripped)
        path = match.group(1)
        blocks[path] = block

        value_line = lines[index + len(block) + 1].strip()
        documented_key = value_line.split(":", 1)[0]
        assert documented_key == path.rsplit(".", 1)[-1], (
            f"{path} annotation precedes {documented_key!r}"
        )

    required_labels = ("# Type:", "# Default:", "# Required:", "# Valid:")
    for path, block in blocks.items():
        for label in required_labels:
            assert any(line.startswith(label) for line in block), f"{path} is missing {label}"
        assert any(
            line.startswith(("# Use:", "# Warning:", "# Precedence:", "# Interaction:"))
            for line in block
        ), f"{path} is missing inline usage guidance"
