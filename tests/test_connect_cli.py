"""Tests for ``agent-scaffold connect`` (all provider/network seams monkeypatched)."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from agent_scaffold import integrations
from agent_scaffold.auth import StoredCredential
from agent_scaffold.cli import app
from agent_scaffold.doctor import CheckResult, CheckStatus
from agent_scaffold.integrations import UpstashDatabase, ValidationResult
from agent_scaffold.manifest import Manifest, write_manifest


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def project(tmp_path: Path) -> Path:
    manifest = Manifest(
        recipe="test-recipe",
        language="python",
        framework="none",
        model="claude-test",
        generated_at="2026-07-01T00:00:00+00:00",
        capabilities=["obs.langsmith", "cache.redis"],
    )
    write_manifest(tmp_path, manifest)
    return tmp_path


def _ok_probe(svc: Any, timeout: float, *, env: Any = None) -> CheckResult:
    return CheckResult(
        id=f"service.{svc.id}",
        category="Recipe services",
        status=CheckStatus.OK,
        title=f"{svc.id}: ok",
    )


@pytest.fixture
def seams(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Neutralize every external seam; individual tests override as needed."""
    stored: list[tuple[str, str, str]] = []
    commands: list[tuple[list[str], dict[str, str]]] = []

    def fake_store(namespace: str, env_var: str, value: Any) -> StoredCredential:
        stored.append((namespace, env_var, value.get_secret_value()))
        return StoredCredential(name=env_var, backend="keyring", masked_value="masked")

    def fake_run(cmd: list[str], **kwargs: Any) -> Any:
        commands.append((list(cmd), dict(kwargs.get("env") or {})))

        class _Done:
            returncode = 0

        return _Done()

    env = {"LANGCHAIN_API_KEY": "", "REDIS_URL": ""}
    monkeypatch.setattr(integrations, "store_project_secret", fake_store)
    monkeypatch.setattr(integrations.subprocess, "run", fake_run)
    monkeypatch.setattr(integrations, "build_runtime_env", lambda *_a, **_k: dict(env))
    monkeypatch.setattr(integrations, "browser_available", lambda: False)
    monkeypatch.setattr(integrations.shutil, "which", lambda _name: "/usr/bin/docker")
    monkeypatch.setattr("agent_scaffold.steps.docker_up._compose_app_service", lambda _dir: "app")
    monkeypatch.setattr(
        "agent_scaffold.steps.bootstrap_langsmith.ensure_project",
        lambda _client, name: ("exists", None),
    )
    for key, spec in list(integrations.INTEGRATIONS.items()):
        monkeypatch.setitem(
            integrations.INTEGRATIONS,
            key,
            dataclasses.replace(
                spec,
                validate=lambda _c, _t: ValidationResult(True, "validated"),
                probe=_ok_probe,
            ),
        )
    return {"stored": stored, "commands": commands, "env": env}


def test_unknown_integration_exits_2(runner: CliRunner, project: Path) -> None:
    result = runner.invoke(app, ["connect", "nope", str(project)])
    assert result.exit_code == 2
    assert "Unknown integration" in result.output


def test_capability_absent_exits_1(
    runner: CliRunner, tmp_path: Path, seams: dict[str, Any]
) -> None:
    write_manifest(
        tmp_path,
        Manifest(
            recipe="r",
            language="python",
            framework="none",
            model="m",
            generated_at="2026-07-01T00:00:00+00:00",
            capabilities=[],
        ),
    )
    result = runner.invoke(app, ["connect", "langsmith", str(tmp_path)])
    assert result.exit_code == 1
    assert "obs.langsmith" in result.output


def test_yes_langsmith_without_value_exits_2(
    runner: CliRunner, project: Path, seams: dict[str, Any]
) -> None:
    result = runner.invoke(app, ["connect", "langsmith", str(project), "--yes"])
    assert result.exit_code == 2
    assert "export LANGCHAIN_API_KEY" in result.output
    assert seams["stored"] == []


def test_yes_redis_without_url_exits_2(
    runner: CliRunner, project: Path, seams: dict[str, Any]
) -> None:
    result = runner.invoke(app, ["connect", "redis", str(project), "--yes"])
    assert result.exit_code == 2
    assert "--url" in result.output


def test_yes_langsmith_happy_path(runner: CliRunner, project: Path, seams: dict[str, Any]) -> None:
    (project / "docker-compose.yml").write_text(
        "services:\n  app:\n    build:\n      context: .\n", encoding="utf-8"
    )
    seams["env"]["LANGCHAIN_API_KEY"] = "lsv2_secret_value"
    result = runner.invoke(app, ["connect", "langsmith", str(project), "--yes"])
    assert result.exit_code == 0, result.output
    # stored in the vault, never echoed in output
    assert [(v, s) for _, v, s in seams["stored"]] == [("LANGCHAIN_API_KEY", "lsv2_secret_value")]
    assert "lsv2_secret_value" not in result.output
    # tracing companion env written to the project
    env_local = (project / ".env.local").read_text(encoding="utf-8")
    assert "LANGCHAIN_TRACING_V2=true" in env_local
    assert "LANGCHAIN_PROJECT=test-recipe" in env_local
    # app container recreated with the fresh env
    assert seams["commands"], "docker compose up was not invoked"
    cmd, env = seams["commands"][0]
    assert cmd == ["docker", "compose", "up", "-d", "app"]
    assert env.get("LANGCHAIN_API_KEY") == "lsv2_secret_value"
    assert "smith.langchain.com" in result.output


def test_validation_auth_failure_stores_nothing(
    runner: CliRunner, project: Path, seams: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    seams["env"]["LANGCHAIN_API_KEY"] = "bad-key"
    spec = integrations.INTEGRATIONS["langsmith"]
    monkeypatch.setitem(
        integrations.INTEGRATIONS,
        "langsmith",
        dataclasses.replace(
            spec,
            validate=lambda _c, _t: ValidationResult(False, "401 unauthorized", auth_failure=True),
        ),
    )
    result = runner.invoke(app, ["connect", "langsmith", str(project), "--yes"])
    assert result.exit_code == 1
    assert seams["stored"] == []
    assert "Nothing stored" in result.output


def test_yes_redis_with_url_stores_and_recreates(
    runner: CliRunner, project: Path, seams: dict[str, Any]
) -> None:
    url = "rediss://:secretpw@usw1.upstash.io:6380"
    result = runner.invoke(app, ["connect", "redis", str(project), "--yes", "--url", url])
    assert result.exit_code == 0, result.output
    assert [(v, s) for _, v, s in seams["stored"]] == [("REDIS_URL", url)]
    assert "secretpw" not in result.output
    assert "local compose redis container keeps running" in result.output


def test_interactive_redis_keep_local_is_noop(
    runner: CliRunner, project: Path, seams: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(integrations, "_stdin_isatty", lambda: True)
    monkeypatch.setattr(
        "agent_scaffold.cli_interactive._interactive_select", lambda *a, **k: "local"
    )
    result = runner.invoke(app, ["connect", "redis", str(project)])
    assert result.exit_code == 0
    assert seams["stored"] == []
    assert "nothing to change" in result.output


def test_interactive_redis_upstash_provisioning(
    runner: CliRunner, project: Path, seams: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(integrations, "_stdin_isatty", lambda: True)
    monkeypatch.setattr(
        "agent_scaffold.cli_interactive._interactive_select", lambda *a, **k: "upstash"
    )
    monkeypatch.setattr(
        integrations,
        "provision_upstash_free",
        lambda timeout: UpstashDatabase(
            url="rediss://:tok@fly-y.upstash.io:6379", claim_url="https://u/claim/9"
        ),
    )
    monkeypatch.setattr(integrations.typer, "confirm", lambda *a, **k: True)
    result = runner.invoke(app, ["connect", "redis", str(project)])
    assert result.exit_code == 0, result.output
    assert [(v, s) for _, v, s in seams["stored"]] == [
        ("REDIS_URL", "rediss://:tok@fly-y.upstash.io:6379")
    ]
    assert "https://u/claim/9" in result.output
    assert "72" in result.output


def test_compose_literal_repair_applied_with_yes(
    runner: CliRunner, project: Path, seams: dict[str, Any]
) -> None:
    compose = project / "docker-compose.yml"
    compose.write_text(
        "services:\n"
        "  app:\n"
        "    build:\n"
        "      context: .\n"
        "    environment:\n"
        "      REDIS_URL: redis://redis:6379\n",
        encoding="utf-8",
    )
    url = "rediss://:pw@managed:6380"
    result = runner.invoke(app, ["connect", "redis", str(project), "--yes", "--url", url])
    assert result.exit_code == 0, result.output
    assert "REDIS_URL: ${REDIS_URL:-redis://redis:6379}" in compose.read_text(encoding="utf-8")


def test_probe_failure_exits_1_after_storing(
    runner: CliRunner, project: Path, seams: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_probe(svc: Any, timeout: float, *, env: Any = None) -> CheckResult:
        return CheckResult(
            id=f"service.{svc.id}",
            category="Recipe services",
            status=CheckStatus.FAIL,
            title=f"{svc.id}: connection failed",
        )

    spec = integrations.INTEGRATIONS["redis"]
    monkeypatch.setitem(
        integrations.INTEGRATIONS, "redis", dataclasses.replace(spec, probe=failing_probe)
    )
    result = runner.invoke(
        app, ["connect", "redis", str(project), "--yes", "--url", "rediss://:p@h:1"]
    )
    assert result.exit_code == 1
    assert seams["stored"]  # stored, but verify surfaced the failure
    assert "connection failed" in result.output
