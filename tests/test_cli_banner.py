"""Tests for the CLI banner output and main utilities."""

from __future__ import annotations

import click
import pytest

from chomp.cli.main import BANNER, parse_resume, print_banner


def test_print_banner_outputs_expected_text(capsys: object) -> None:
    """print_banner emits the banner once with a trailing newline."""
    print_banner()
    captured = capsys.readouterr()

    assert captured.out.endswith("\n")
    assert captured.out.rstrip("\n") == BANNER


def test_parse_resume_returns_none_for_none_variants() -> None:
    """parse_resume should return 'none' for none/no/false/0."""
    assert parse_resume("none") == "none"
    assert parse_resume("no") == "none"
    assert parse_resume("false") == "none"
    assert parse_resume("0") == "none"
    assert parse_resume("  NONE  ") == "none"


def test_parse_resume_returns_latest_for_latest_variants() -> None:
    """parse_resume should return 'latest' for latest/last."""
    assert parse_resume("latest") == "latest"
    assert parse_resume("last") == "latest"
    assert parse_resume("  LATEST  ") == "latest"


def test_parse_resume_returns_int_for_valid_step() -> None:
    """parse_resume should return an int for valid positive step numbers."""
    assert parse_resume("100") == 100
    assert parse_resume("5000") == 5000
    assert parse_resume("  42  ") == 42


def test_parse_resume_rejects_negative_step() -> None:
    """parse_resume should raise BadParameter for negative step numbers."""
    with pytest.raises(click.BadParameter, match="non-negative"):
        parse_resume("-1")
    with pytest.raises(click.BadParameter, match="non-negative"):
        parse_resume("-100")


def test_parse_resume_rejects_invalid_string() -> None:
    """parse_resume should raise BadParameter for invalid strings."""
    with pytest.raises(click.BadParameter, match="Invalid resume value"):
        parse_resume("invalid")
    with pytest.raises(click.BadParameter, match="Invalid resume value"):
        parse_resume("step100")
