"""Railway deploy plugin.

Shells out to ``railway up``. Project must already be linked
(``railway link`` writes ``.railway/config.json``); we surface unlinked
projects as a skip with a clear instruction rather than driving the link
flow ourselves.
"""

from __future__ import annotations

from pathlib import Path

from agent_scaffold.deploy._common import (
    DeployResult,
    cli_present,
    confirm,
    run_provider_cli,
)

name = "railway"
cli_binary = "railway"
dashboard_url = "https://railway.app/dashboard"
install_hint = "brew install railway  # or npm i -g @railway/cli"
config_file: str | None = "railway.json"


def deploy(project_dir: Path, dry_run: bool, yes: bool) -> DeployResult:
    cmd = ["railway", "up"]
    if not cli_present(cli_binary):
        return DeployResult(
            target=name,
            cmd_run=cmd,
            dashboard_url=dashboard_url,
            summary=f"railway CLI not found — install with `{install_hint}`",
            skipped=True,
            skip_reason="missing_cli",
        )

    if not (project_dir / ".railway" / "config.json").is_file():
        return DeployResult(
            target=name,
            cmd_run=cmd,
            dashboard_url=dashboard_url,
            summary=(
                "project not linked to Railway — run `railway login && railway link` "
                "once (interactive) before deploying"
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
        f"About to deploy {project_dir.name} to Railway."
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
            f"railway up exited {exit_code}"
            if exit_code == 0
            else f"railway up FAILED (exit {exit_code})"
        ),
    )


__all__ = ["deploy"]
