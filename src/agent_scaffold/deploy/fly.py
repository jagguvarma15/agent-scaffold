"""Fly.io deploy plugin.

Shells out to ``fly deploy``. First-time projects need an interactive
``fly launch --no-deploy`` to create the Fly app + write ``fly.toml`` with
the app name; we surface that as a clear skip reason rather than trying to
drive the launch flow.
"""

from __future__ import annotations

from pathlib import Path

from agent_scaffold.deploy._common import (
    DeployResult,
    cli_present,
    confirm,
    run_provider_cli,
)

name = "fly"
cli_binary = "fly"
dashboard_url = "https://fly.io/dashboard"
install_hint = "curl -L https://fly.io/install.sh | sh"
config_file: str | None = "fly.toml"


def deploy(project_dir: Path, dry_run: bool, yes: bool) -> DeployResult:
    cmd = ["fly", "deploy"]
    if yes:
        cmd.append("--yes")  # fly auto-confirms destructive changes

    if not cli_present(cli_binary):
        return DeployResult(
            target=name,
            cmd_run=cmd,
            dashboard_url=dashboard_url,
            summary=f"fly CLI not found — install with `{install_hint}`",
            skipped=True,
            skip_reason="missing_cli",
        )

    if not (project_dir / "fly.toml").is_file():
        return DeployResult(
            target=name,
            cmd_run=cmd,
            dashboard_url=dashboard_url,
            summary=(
                "no fly.toml in project — run `fly launch --no-deploy` "
                "once (interactive) to register the app"
            ),
            skipped=True,
            skip_reason="not_launched",
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

    if not yes and not confirm(f"About to deploy {project_dir.name} to Fly."):
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
            f"fly deploy exited {exit_code}"
            if exit_code == 0
            else f"fly deploy FAILED (exit {exit_code})"
        ),
    )


__all__ = ["deploy"]
