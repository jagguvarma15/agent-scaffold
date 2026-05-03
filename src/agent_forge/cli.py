"""Typer CLI entry point for agent-forge.

The full command surface is wired in later build steps. For now this exposes
``--version`` so the skeleton is testable end-to-end.
"""

from __future__ import annotations

import typer

from agent_forge import __version__

app = typer.Typer(
    name="agent-forge",
    help="Generate runnable AI agent projects from markdown specs.",
    add_completion=False,
    invoke_without_command=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"agent-forge {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the agent-forge version and exit.",
    ),
) -> None:
    """agent-forge: generate runnable AI agent projects from markdown specs."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()
