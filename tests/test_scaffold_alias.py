"""Tests for the ``scaffold`` binary alias entry point."""

from __future__ import annotations

from unittest.mock import patch

from agent_scaffold import cli


def _argv_after(argv_in: list[str]) -> list[str]:
    """Run ``scaffold_main`` with the given args and return sys.argv as seen by app()."""
    captured: list[str] = []

    def fake_app() -> None:
        captured.extend(cli.sys.argv)

    with patch.object(cli.sys, "argv", ["scaffold", *argv_in]):
        with patch.object(cli, "app", side_effect=fake_app):
            cli.scaffold_main()
    return captured


def test_bare_invocation_injects_scaffold_subcommand() -> None:
    assert _argv_after([]) == ["scaffold", "scaffold"]


def test_verbose_only_still_opens_repl() -> None:
    assert _argv_after(["-v"]) == ["scaffold", "scaffold", "-v"]


def test_help_passes_through() -> None:
    assert _argv_after(["--help"]) == ["scaffold", "--help"]


def test_version_passes_through() -> None:
    assert _argv_after(["--version"]) == ["scaffold", "--version"]


def test_explicit_subcommand_passes_through() -> None:
    assert _argv_after(["doctor"]) == ["scaffold", "doctor"]
