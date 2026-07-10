# SPDX-License-Identifier: Apache-2.0

"""Executable check for the human-maintained annotated config reference."""

from __future__ import annotations

from pathlib import Path

from chomp.config import Config, build_config, read_config_mapping

REFERENCE_PATH = Path(__file__).parents[1] / "docs" / "config-reference.yaml"


def test_config_reference_is_valid_default_config() -> None:
    """The copyable reference must parse and build the default configuration."""
    reference = read_config_mapping(REFERENCE_PATH)
    assert build_config(reference) == Config()
