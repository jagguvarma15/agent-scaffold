"""The docs reference pages must cover the live command surface.

docs/reference/cli.md and docs/reference/repl.md are hand-written tables
(Typer's generated docs are too flat and can't describe the REPL at all).
These tests walk the actual registries so a command added without a docs
row fails CI instead of drifting silently. They check presence, not
wording — descriptions are free to be edited by hand.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_scaffold import cli
from agent_scaffold.repl.commands import CommandHandler
from agent_scaffold.repl.refine import REFINEMENT_KEYS

_REFERENCE_DIR = Path(__file__).parent.parent / "docs" / "reference"


@pytest.fixture(scope="module")
def cli_reference() -> str:
    return (_REFERENCE_DIR / "cli.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def repl_reference() -> str:
    return (_REFERENCE_DIR / "repl.md").read_text(encoding="utf-8")


def test_cli_reference_covers_every_root_command(cli_reference: str) -> None:
    for command in cli.app.registered_commands:
        name = command.name or command.callback.__name__  # type: ignore[union-attr]
        assert f"agent-scaffold {name}" in cli_reference, (
            f"docs/reference/cli.md is missing root command `{name}`"
        )


def test_cli_reference_covers_every_sub_app_command(cli_reference: str) -> None:
    for group in cli.app.registered_groups:
        assert group.name is not None
        assert f"agent-scaffold {group.name}" in cli_reference, (
            f"docs/reference/cli.md is missing sub-app `{group.name}`"
        )
        for sub in group.typer_instance.registered_commands:  # type: ignore[union-attr]
            sub_name = sub.name or sub.callback.__name__  # type: ignore[union-attr]
            assert f"{group.name} {sub_name}" in cli_reference, (
                f"docs/reference/cli.md is missing `{group.name} {sub_name}`"
            )


def test_repl_reference_covers_every_command(repl_reference: str) -> None:
    handler = CommandHandler(recipes=[])
    for name in handler.commands:
        assert f"/{name}" in repl_reference, f"docs/reference/repl.md is missing command `/{name}`"


def test_repl_reference_covers_every_alias(repl_reference: str) -> None:
    handler = CommandHandler(recipes=[])
    for alias in handler._aliases:
        assert f"/{alias}" in repl_reference, f"docs/reference/repl.md is missing alias `/{alias}`"


def test_repl_reference_covers_every_refinement_key(repl_reference: str) -> None:
    for key in REFINEMENT_KEYS:
        assert f"`{key}`" in repl_reference, (
            f"docs/reference/repl.md is missing refinement key `{key}`"
        )
