"""Tests for the ``scaffold`` binary alias entry point."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agent_scaffold import cli


def _argv_after(argv_in: list[str]) -> list[str]:
    """Run ``scaffold_main`` with the given args and return sys.argv as seen by app()."""
    captured: list[str] = []

    def fake_app() -> None:
        captured.extend(cli.sys.argv)

    # scaffold_main reads the registered command/group names off the app to
    # tell subcommands apart from project directories — carry the real
    # registries over so the routing under test matches production.
    fake = MagicMock(side_effect=fake_app)
    fake.registered_commands = cli.app.registered_commands
    fake.registered_groups = cli.app.registered_groups
    with patch.object(cli.sys, "argv", ["scaffold", *argv_in]):
        with patch.object(cli, "app", new=fake):
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


def test_directory_argument_injects_scaffold_subcommand(tmp_path) -> None:
    """`scaffold <existing-dir>` opens the shell attached to that project."""
    project = tmp_path / "my-proj"
    project.mkdir()
    assert _argv_after([str(project)]) == ["scaffold", "scaffold", str(project)]


def test_known_subcommand_wins_over_same_named_directory(tmp_path, monkeypatch) -> None:
    """A ./doctor directory must not shadow the doctor subcommand."""
    (tmp_path / "doctor").mkdir()
    monkeypatch.chdir(tmp_path)
    assert _argv_after(["doctor"]) == ["scaffold", "doctor"]


def test_nondirectory_argument_still_routes_to_app(tmp_path) -> None:
    """A positional that is neither a command nor a directory passes through
    (typer reports the unknown command instead of silently opening the REPL)."""
    assert _argv_after([str(tmp_path / "missing")]) == ["scaffold", str(tmp_path / "missing")]
