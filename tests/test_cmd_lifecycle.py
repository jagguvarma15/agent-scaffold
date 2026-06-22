"""Tests for the lifecycle CLI verbs: ``deploy`` / ``down`` / ``status`` / ``logs``."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent_scaffold.cli import app
from agent_scaffold.manifest import Manifest, write_manifest

runner = CliRunner()


@pytest.fixture
def project_with_manifest(tmp_path: Path) -> Iterator[Path]:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    manifest = Manifest(
        recipe="demo",
        language="python",
        framework="none",
        model="claude-test",
        generated_at="2026-05-29T00:00:00+00:00",
        capabilities=["host.vercel"],
    )
    write_manifest(project_dir, manifest)
    yield project_dir


def test_deploy_dry_run_default_does_not_invoke_provider(
    project_with_manifest: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invoked: list[list[str]] = []

    def fake_run(cmd: list[str], cwd: Path, timeout: float | None = None) -> int:
        invoked.append(cmd)
        return 0

    monkeypatch.setattr("agent_scaffold.deploy.vercel.cli_present", lambda _: True)
    monkeypatch.setattr("agent_scaffold.deploy.vercel.run_provider_cli", fake_run)
    # Make the deploy plugin think the project is linked.
    (project_with_manifest / ".vercel").mkdir()
    (project_with_manifest / ".vercel" / "project.json").write_text("{}")

    result = runner.invoke(
        app,
        ["deploy", "--target", "vercel", "--cwd", str(project_with_manifest)],
    )
    assert result.exit_code == 0
    assert invoked == []  # default dry-run = no provider invocation
    assert "DRY-RUN" in result.stdout


def test_deploy_no_dry_run_yes_invokes_provider(
    project_with_manifest: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invoked: list[list[str]] = []

    def fake_run(cmd: list[str], cwd: Path, timeout: float | None = None) -> int:
        invoked.append(cmd)
        return 0

    monkeypatch.setattr("agent_scaffold.deploy.vercel.cli_present", lambda _: True)
    monkeypatch.setattr("agent_scaffold.deploy.vercel.run_provider_cli", fake_run)
    (project_with_manifest / ".vercel").mkdir()
    (project_with_manifest / ".vercel" / "project.json").write_text("{}")

    result = runner.invoke(
        app,
        [
            "deploy",
            "--target",
            "vercel",
            "--cwd",
            str(project_with_manifest),
            "--no-dry-run",
            "--yes",
        ],
    )
    assert result.exit_code == 0
    assert invoked == [["vercel", "deploy", "--prod", "--yes"]]


def test_deploy_unknown_target_errors(project_with_manifest: Path) -> None:
    result = runner.invoke(
        app,
        ["deploy", "--target", "aws-lambda", "--cwd", str(project_with_manifest)],
    )
    assert result.exit_code == 1
    assert "Unknown deploy target" in result.stdout or "Unknown deploy target" in result.output


def test_deploy_without_manifest_errors(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["deploy", "--target", "vercel", "--cwd", str(tmp_path)],
    )
    assert result.exit_code == 1


def test_down_without_compose_errors(tmp_path: Path) -> None:
    result = runner.invoke(app, ["down", "--cwd", str(tmp_path), "--yes"])
    assert result.exit_code == 1
    assert "no docker-compose.yml" in result.stdout


def test_down_runs_docker_compose(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "docker-compose.yml").write_text("services: {}\n")
    monkeypatch.setattr(
        "agent_scaffold.cli.shutil.which",
        lambda name: "/usr/bin/docker" if name == "docker" else shutil.which(name),
    )
    calls: list[tuple[list[str], str]] = []

    class FakeProc:
        returncode = 0

    def fake_run(cmd: list[str], cwd: str, check: bool = False) -> FakeProc:
        calls.append((cmd, cwd))
        return FakeProc()

    monkeypatch.setattr("agent_scaffold.cli.subprocess.run", fake_run)
    result = runner.invoke(app, ["down", "--cwd", str(tmp_path), "--yes"])
    assert result.exit_code == 0
    assert calls == [(["docker", "compose", "down"], str(tmp_path))]


def test_down_with_volumes_requires_confirm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "docker-compose.yml").write_text("services: {}\n")
    monkeypatch.setattr("agent_scaffold.cli.shutil.which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr("agent_scaffold.deploy._common.confirm", lambda _msg: False)
    calls: list[list[str]] = []

    class FakeProc:
        returncode = 0

    def fake_run(cmd: list[str], cwd: str, check: bool = False) -> FakeProc:
        calls.append(cmd)
        return FakeProc()

    monkeypatch.setattr("agent_scaffold.cli.subprocess.run", fake_run)
    # User declines confirmation → no docker call, clean exit.
    result = runner.invoke(app, ["down", "-v", "--cwd", str(tmp_path)])
    assert result.exit_code == 0
    assert calls == []
    assert "Aborted" in result.stdout


def test_down_with_volumes_yes_skips_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "docker-compose.yml").write_text("services: {}\n")
    monkeypatch.setattr("agent_scaffold.cli.shutil.which", lambda _: "/usr/bin/docker")
    calls: list[list[str]] = []

    class FakeProc:
        returncode = 0

    def fake_run(cmd: list[str], cwd: str, check: bool = False) -> FakeProc:
        calls.append(cmd)
        return FakeProc()

    monkeypatch.setattr("agent_scaffold.cli.subprocess.run", fake_run)
    result = runner.invoke(app, ["down", "-v", "--cwd", str(tmp_path), "--yes"])
    assert result.exit_code == 0
    assert calls == [["docker", "compose", "down", "-v"]]


def test_resolve_stack_uses_chosen_capabilities_not_recipe_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: the post-gen stack must reflect the manifest's CHOSEN caps, not
    the recipe's declared defaults — so a langsmith run doesn't show phantom
    langfuse. Verifies the add/remove diff handed to resolve_capabilities."""
    from types import SimpleNamespace

    from agent_scaffold import cli
    from agent_scaffold.discovery import Recipe

    recipe = Recipe(
        slug="r",
        title="R",
        path=Path("/r.md"),
        capabilities=["relational.postgres", "obs.langfuse"],  # recipe default = langfuse
    )
    captured: dict[str, object] = {}

    def _fake_resolve(_recipe: object, _catalog: object, *, add_capabilities, remove_capabilities):  # type: ignore[no-untyped-def]
        captured["add"] = add_capabilities
        captured["remove"] = remove_capabilities
        return SimpleNamespace(capabilities=["x"])  # non-empty so it's returned

    monkeypatch.setattr(
        cli,
        "load_config",
        lambda: SimpleNamespace(
            deployments_path=None, deployments_source="auto", cache_dir=Path("/c")
        ),
    )
    monkeypatch.setattr(cli, "resolve_deployments", lambda **_k: SimpleNamespace(path=Path("/dep")))
    monkeypatch.setattr(cli, "load_capabilities", lambda _p: object())
    monkeypatch.setattr(cli, "resolve_capabilities", _fake_resolve)

    # Manifest recorded langsmith (the user's /observability swap).
    cli._resolve_capability_stack_silently(
        recipe, capabilities=["relational.postgres", "obs.langsmith"]
    )
    assert captured["add"] == ["obs.langsmith"]  # the chosen one is added
    assert captured["remove"] == {"obs.langfuse"}  # the recipe default is removed


def test_status_emits_json(project_with_manifest: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # No recipe / capability resolution will succeed in the test env, but
    # the command should still exit cleanly (no FAIL) and emit valid JSON.
    monkeypatch.setattr("agent_scaffold.cli._resolve_recipe_silently", lambda _: None)
    monkeypatch.setattr(
        "agent_scaffold.cli._resolve_capability_stack_silently", lambda _r, **_k: None
    )
    result = runner.invoke(app, ["status", "--cwd", str(project_with_manifest), "--json"])
    assert result.exit_code == 0
    body = json.loads(result.stdout)
    assert "services" in body
    assert "capabilities" in body


def test_logs_without_compose_errors(tmp_path: Path) -> None:
    result = runner.invoke(app, ["logs", "redis", "--cwd", str(tmp_path)])
    assert result.exit_code == 1


def test_logs_invokes_docker_compose(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "docker-compose.yml").write_text("services: {}\n")
    monkeypatch.setattr("agent_scaffold.cli.shutil.which", lambda _: "/usr/bin/docker")
    calls: list[list[str]] = []

    class FakeProc:
        returncode = 0

    def fake_run(cmd: list[str], cwd: str, check: bool = False) -> FakeProc:
        calls.append(cmd)
        return FakeProc()

    monkeypatch.setattr("agent_scaffold.cli.subprocess.run", fake_run)
    result = runner.invoke(
        app, ["logs", "redis", "--cwd", str(tmp_path), "--no-follow", "--tail", "50"]
    )
    assert result.exit_code == 0
    assert calls == [["docker", "compose", "logs", "--tail", "50", "redis"]]
