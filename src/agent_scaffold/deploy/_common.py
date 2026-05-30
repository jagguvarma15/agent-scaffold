"""Shared helpers for deploy provider plugins.

The contract every plugin implements (via duck typing — not a strict
Protocol, since provider modules are imported lazily):

.. code-block:: python

    name: str                       # short id, e.g. "vercel"
    cli_binary: str                 # e.g. "vercel"; checked with shutil.which
    dashboard_url: str              # printed in dry-run + result
    install_hint: str               # what to run if cli_binary is missing
    config_file: str | None         # name of the deploy config the project should have

    def deploy(project_dir: Path, dry_run: bool, yes: bool) -> DeployResult: ...

``DeployResult`` is the typed payload returned by every plugin so
``cmd_deploy`` can render uniformly.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class DeployResult:
    """Outcome of one deploy invocation."""

    target: str
    """Plugin name (``"vercel"``, ``"fly"``, ``"railway"``)."""

    cmd_run: list[str] = field(default_factory=list)
    """The full shell command the plugin would run (or did run)."""

    exit_code: int | None = None
    """``None`` when ``dry_run=True``; otherwise the provider CLI's exit code."""

    dashboard_url: str | None = None

    summary: str = ""
    """Human-readable one-liner for the CLI panel."""

    skipped: bool = False
    """``True`` when the plugin returned early (missing CLI, user declined, etc.)."""

    skip_reason: str = ""


class DeployTarget(Protocol):
    """Loose interface every plugin module satisfies."""

    name: str
    cli_binary: str
    dashboard_url: str
    install_hint: str
    config_file: str | None

    def deploy(self, project_dir: Path, dry_run: bool, yes: bool) -> DeployResult: ...


def cli_present(binary: str) -> bool:
    return shutil.which(binary) is not None


def confirm(prompt: str) -> bool:
    """Block-on-stdin yes/no prompt.

    Returns ``True`` only on exact ``"yes"`` (case-insensitive). Anything
    else — including bare Enter — is treated as no. Stricter than a typical
    y/N to discourage muscle-memory ``y\\n`` accidents on a real deploy.
    """
    if not sys.stdin.isatty():
        # Non-interactive context (CI) without ``--yes`` is always no.
        return False
    try:
        reply = input(f"{prompt} [type 'yes' to confirm] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return reply == "yes"


def run_provider_cli(
    cmd: list[str],
    cwd: Path,
    *,
    timeout: float | None = None,
) -> int:
    """Stream a provider CLI subprocess to the user's terminal.

    Returns the exit code. Stdin/stdout/stderr inherit from the parent so
    the user sees provider output (and any interactive prompts) live.
    Provider CLIs handle their own auth — we just forward.
    """
    try:
        completed = subprocess.run(  # noqa: S603 — list-form, shell=False
            cmd,
            cwd=str(cwd),
            check=False,
            timeout=timeout,
        )
    except FileNotFoundError:
        # cli_present() should have caught this earlier; defence in depth.
        return 127
    return completed.returncode


__all__ = [
    "DeployResult",
    "DeployTarget",
    "cli_present",
    "confirm",
    "run_provider_cli",
]
