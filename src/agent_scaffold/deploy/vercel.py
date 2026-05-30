"""Vercel deploy plugin.

Shells out to ``vercel deploy --prod``. Assumes the user has already run
``vercel link`` against the project (which writes ``.vercel/project.json``);
we don't try to drive the interactive linking flow ourselves.
"""

from __future__ import annotations

from pathlib import Path

from agent_scaffold.deploy._common import (
    DeployResult,
    cli_present,
    confirm,
    run_provider_cli,
)

name = "vercel"
cli_binary = "vercel"
dashboard_url = "https://vercel.com/dashboard"
install_hint = "npm i -g vercel"
config_file: str | None = "vercel.json"


def deploy(project_dir: Path, dry_run: bool, yes: bool) -> DeployResult:
    cmd = ["vercel", "deploy", "--prod"]
    if yes:
        cmd.append("--yes")  # skip Vercel's own confirmation

    if not cli_present(cli_binary):
        return DeployResult(
            target=name,
            cmd_run=cmd,
            dashboard_url=dashboard_url,
            summary=f"vercel CLI not found — install with `{install_hint}`",
            skipped=True,
            skip_reason="missing_cli",
        )

    if not (project_dir / ".vercel" / "project.json").is_file():
        return DeployResult(
            target=name,
            cmd_run=cmd,
            dashboard_url=dashboard_url,
            summary=(
                "project not linked to Vercel — run `vercel link` once "
                "(interactive) before deploying"
            ),
            skipped=True,
            skip_reason="not_linked",
        )

    if dry_run:
        return DeployResult(
            target=name,
            cmd_run=cmd,
            dashboard_url=dashboard_url,
            summary=f"DRY-RUN: would run `{' '.join(cmd)}` in {project_dir}",
            skipped=True,
            skip_reason="dry_run",
        )

    if not yes and not confirm(
        f"About to deploy {project_dir.name} to Vercel production."
    ):
        return DeployResult(
            target=name,
            cmd_run=cmd,
            dashboard_url=dashboard_url,
            summary="deploy cancelled by user",
            skipped=True,
            skip_reason="declined",
        )

    exit_code = run_provider_cli(cmd, cwd=project_dir)
    return DeployResult(
        target=name,
        cmd_run=cmd,
        exit_code=exit_code,
        dashboard_url=dashboard_url,
        summary=(
            f"vercel deploy exited {exit_code}"
            if exit_code == 0
            else f"vercel deploy FAILED (exit {exit_code})"
        ),
    )


__all__ = ["deploy"]
