"""Tests for the CLI banner output."""

from __future__ import annotations

from chomp.cli.main import BANNER, print_banner


def test_print_banner_outputs_expected_text(capsys: object) -> None:
    """print_banner emits the banner once with a trailing newline."""
    print_banner()
    captured = capsys.readouterr()

    assert captured.out.endswith("\n")
    assert captured.out.rstrip("\n") == BANNER
