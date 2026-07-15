"""Tests for ``agent-scaffold connect`` (all provider/network seams monkeypatched)."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from agent_scaffold import cli as cli_module
from agent_scaffold import integrations
from agent_scaffold.auth import StoredCredential
from agent_scaffold.cli import app
from agent_scaffold.doctor import CheckResult, CheckStatus
from agent_scaffold.integrations import UpstashDatabase, ValidationResult
from agent_scaffold.manifest import Manifest, write_manifest
from agent_scaffold.stack_options import (
    MODE_CLOUD,
    MODE_INTERNAL_OVERRIDABLE,
    CredentialSpec,
    StackOption,
)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


ALL_OPTIONS = [
    StackOption(
        id="langsmith",
        title="LangSmith",
        capability_ids=frozenset({"obs.langsmith"}),
        kind="obs",
        mode=MODE_CLOUD,
        credentials=(CredentialSpec(var="LANGCHAIN_API_KEY", placeholder="lsv2_..."),),
        managed_vars=(
            "LANGCHAIN_API_KEY",
            "LANGCHAIN_TRACING_V2",
            "LANGCHAIN_PROJECT",
            "LANGCHAIN_ENDPOINT",
        ),
        docker_service=None,
        probe="langsmith_workspace",
        bootstrap_step="bootstrap_langsmith",
        key_page_url="https://smith.langchain.com/settings",
    ),
    StackOption(
        id="redis",
        title="Redis",
        capability_ids=frozenset({"cache.redis", "queue.redis-streams"}),
        kind="cache",
        mode=MODE_INTERNAL_OVERRIDABLE,
        credentials=(
            CredentialSpec(var="REDIS_URL", placeholder="rediss://:<password>@<host>:6380"),
        ),
        managed_vars=("REDIS_URL",),
        docker_service="redis",
        probe="redis_ping",
        bootstrap_step=None,
        key_page_url=None,
    ),
    StackOption(
        id="postgres",
        title="Postgres",
        capability_ids=frozenset({"relational.postgres"}),
        kind="relational",
        mode=MODE_INTERNAL_OVERRIDABLE,
        credentials=(
            CredentialSpec(var="DATABASE_URL", placeholder="postgresql://user:password@host/db"),
        ),
        managed_vars=("DATABASE_URL", "POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"),
        docker_service="postgres",
        probe="postgres_select_one",
        bootstrap_step=None,
        key_page_url="https://console.neon.tech",
    ),
    StackOption(
        id="qdrant",
        title="Qdrant",
        capability_ids=frozenset({"vector_db.qdrant"}),
        kind="vector_db",
        mode=MODE_INTERNAL_OVERRIDABLE,
        credentials=(
            CredentialSpec(var="QDRANT_URL", secret=False, placeholder="https://x.qdrant.io"),
            CredentialSpec(var="QDRANT_API_KEY", optional=True),
        ),
        managed_vars=("QDRANT_URL", "QDRANT_API_KEY"),
        docker_service="qdrant",
        probe="qdrant_collections",
        bootstrap_step="bootstrap_vector_db",
        key_page_url="https://cloud.qdrant.io",
    ),
    StackOption(
        id="langfuse",
        title="Langfuse",
        capability_ids=frozenset({"obs.langfuse"}),
        kind="obs",
        mode=MODE_INTERNAL_OVERRIDABLE,
        credentials=(
            CredentialSpec(var="LANGFUSE_PUBLIC_KEY", secret=False, placeholder="pk-lf-..."),
            CredentialSpec(var="LANGFUSE_SECRET_KEY", placeholder="sk-lf-..."),
            CredentialSpec(var="LANGFUSE_HOST", secret=False, optional=True),
        ),
        managed_vars=("LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"),
        docker_service="langfuse",
        probe="langfuse_health",
        bootstrap_step="bootstrap_langfuse",
        key_page_url="https://cloud.langfuse.com",
    ),
]

_KNOWN_CAPS = {
    "langsmith": ["obs.langsmith"],
    "redis": ["cache.redis"],
    "postgres": ["relational.postgres"],
    "qdrant": ["vector_db.qdrant"],
    "langfuse": ["obs.langfuse"],
}


def _project(tmp_path: Path, capabilities: list[str]) -> Path:
    manifest = Manifest(
        recipe="test-recipe",
        language="python",
        framework="none",
        model="claude-test",
        generated_at="2026-07-01T00:00:00+00:00",
        capabilities=capabilities,
    )
    write_manifest(tmp_path, manifest)
    return tmp_path


@pytest.fixture
def project(tmp_path: Path) -> Path:
    return _project(tmp_path, ["obs.langsmith", "cache.redis"])


def _ok_run_probe(
    svc: Any, *, timeout: float = 5.0, skip: bool = False, env: Any = None
) -> CheckResult:
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

    def fake_load_options(capability_ids: Any) -> list[StackOption]:
        wanted = set(capability_ids or [])
        return [o for o in ALL_OPTIONS if o.capability_ids & wanted]

    monkeypatch.setattr(integrations, "store_project_secret", fake_store)
    monkeypatch.setattr(integrations.subprocess, "run", fake_run)
    monkeypatch.setattr(integrations, "build_runtime_env", lambda *_a, **_k: dict(env))
    monkeypatch.setattr(integrations, "list_project_secret_names", lambda _ns: {})
    monkeypatch.setattr(integrations, "browser_available", lambda: False)
    monkeypatch.setattr(integrations.shutil, "which", lambda _name: "/usr/bin/docker")
    monkeypatch.setattr(integrations, "run_probe", _ok_run_probe)
    monkeypatch.setattr(cli_module, "load_stack_options", fake_load_options)
    monkeypatch.setattr(
        cli_module, "known_provider_capabilities", lambda handle: _KNOWN_CAPS.get(handle, [])
    )
    monkeypatch.setattr(cli_module, "build_runtime_env", lambda *_a, **_k: dict(env))
    monkeypatch.setattr("agent_scaffold.steps.docker_up._compose_app_service", lambda _dir: "app")
    monkeypatch.setattr(
        "agent_scaffold.steps.bootstrap_langsmith.ensure_project",
        lambda _client, name: ("exists", None),
    )
    for key, spec in list(integrations.PROVIDER_EXTRAS.items()):
        monkeypatch.setitem(
            integrations.PROVIDER_EXTRAS,
            key,
            dataclasses.replace(
                spec, validate=lambda _captured, _t: ValidationResult(True, "validated")
            ),
        )
    return {"stored": stored, "commands": commands, "env": env}


def test_unknown_integration_exits_2(
    runner: CliRunner, project: Path, seams: dict[str, Any]
) -> None:
    result = runner.invoke(app, ["connect", "nope", str(project)])
    assert result.exit_code == 2
    assert "Unknown integration" in result.output


def test_capability_absent_exits_1(
    runner: CliRunner, tmp_path: Path, seams: dict[str, Any]
) -> None:
    _project(tmp_path, [])
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


def test_runtime_env_read_once(
    runner: CliRunner, project: Path, seams: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The vault-backed env resolves once per connect (each read can pop a
    keychain consent dialog per stored secret on macOS)."""
    calls = {"n": 0}

    def counting_env(*_a: Any, **_k: Any) -> dict[str, str]:
        calls["n"] += 1
        return dict(seams["env"])

    monkeypatch.setattr(integrations, "build_runtime_env", counting_env)
    (project / "docker-compose.yml").write_text(
        "services:\n  app:\n    build:\n      context: .\n", encoding="utf-8"
    )
    seams["env"]["LANGCHAIN_API_KEY"] = "lsv2_secret_value"
    result = runner.invoke(app, ["connect", "langsmith", str(project), "--yes"])
    assert result.exit_code == 0, result.output
    assert calls["n"] == 1


def test_recreate_env_includes_companion_tracing_vars(
    runner: CliRunner, project: Path, seams: dict[str, Any]
) -> None:
    """The companion writes tracing vars to .env.local after the first env
    read; the recreate must still carry them (guards the .env.local re-read)."""
    (project / "docker-compose.yml").write_text(
        "services:\n  app:\n    build:\n      context: .\n", encoding="utf-8"
    )
    seams["env"]["LANGCHAIN_API_KEY"] = "lsv2_secret_value"
    result = runner.invoke(app, ["connect", "langsmith", str(project), "--yes"])
    assert result.exit_code == 0, result.output
    assert seams["commands"], "docker compose up was not invoked"
    _, recreate_env = seams["commands"][0]
    assert recreate_env.get("LANGCHAIN_TRACING_V2") == "true"


def test_keychain_heads_up_when_secrets_indexed(
    runner: CliRunner, project: Path, seams: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        integrations, "list_project_secret_names", lambda _ns: {"REDIS_URL": "keyring"}
    )
    url = "rediss://:secretpw@usw1.upstash.io:6380"
    result = runner.invoke(app, ["connect", "redis", str(project), "--yes", "--url", url])
    assert result.exit_code == 0, result.output
    assert "not a request to re-enter" in result.output


def test_no_keychain_heads_up_without_indexed_secrets(
    runner: CliRunner, project: Path, seams: dict[str, Any]
) -> None:
    url = "rediss://:secretpw@usw1.upstash.io:6380"
    result = runner.invoke(app, ["connect", "redis", str(project), "--yes", "--url", url])
    assert result.exit_code == 0, result.output
    assert "not a request to re-enter" not in result.output


def test_validation_auth_failure_stores_nothing(
    runner: CliRunner, project: Path, seams: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    seams["env"]["LANGCHAIN_API_KEY"] = "bad-key"
    spec = integrations.PROVIDER_EXTRAS["langsmith"]
    monkeypatch.setitem(
        integrations.PROVIDER_EXTRAS,
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


def _probe_stub(status: CheckStatus, detail: str = "") -> Any:
    def stub(svc: Any, *, timeout: float = 5.0, skip: bool = False, env: Any = None) -> CheckResult:
        return CheckResult(
            id=f"service.{svc.id}",
            category="Recipe services",
            status=status,
            title=f"{svc.id}: probed",
            detail=detail,
        )

    return stub


def test_verify_ok_states_connection_established(
    runner: CliRunner, project: Path, seams: dict[str, Any]
) -> None:
    seams["env"]["LANGCHAIN_API_KEY"] = "lsv2_secret_value"
    result = runner.invoke(app, ["connect", "langsmith", str(project), "--yes"])
    assert result.exit_code == 0, result.output
    assert "connection established" in result.output


def test_verify_skip_renders_not_verified(
    runner: CliRunner, project: Path, seams: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        integrations, "run_probe", _probe_stub(CheckStatus.SKIP, "httpx not available")
    )
    seams["env"]["LANGCHAIN_API_KEY"] = "lsv2_secret_value"
    result = runner.invoke(app, ["connect", "langsmith", str(project), "--yes"])
    assert result.exit_code == 0, result.output
    assert "not verified" in result.output
    assert "connection established" not in result.output


def test_verify_warn_renders_partially_verified(
    runner: CliRunner, project: Path, seams: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(integrations, "run_probe", _probe_stub(CheckStatus.WARN, "TCP-only ok"))
    url = "rediss://:secretpw@usw1.upstash.io:6380"
    result = runner.invoke(app, ["connect", "redis", str(project), "--yes", "--url", url])
    assert result.exit_code == 0, result.output
    assert "partially verified" in result.output
    assert "connection established" not in result.output


def test_unverified_store_says_so(
    runner: CliRunner, project: Path, seams: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A verdict that never reached the provider must not print as validated."""
    spec = integrations.PROVIDER_EXTRAS["langsmith"]
    monkeypatch.setitem(
        integrations.PROVIDER_EXTRAS,
        "langsmith",
        dataclasses.replace(
            spec,
            validate=lambda _c, _t: ValidationResult(
                True, "not validated (no probe declared)", verified=False
            ),
        ),
    )
    seams["env"]["LANGCHAIN_API_KEY"] = "lsv2_secret_value"
    result = runner.invoke(app, ["connect", "langsmith", str(project), "--yes"])
    assert result.exit_code == 0, result.output
    assert "Storing without live verification" in result.output
    assert "Validated:" not in result.output
    assert [(v, s) for _, v, s in seams["stored"]] == [("LANGCHAIN_API_KEY", "lsv2_secret_value")]


def test_yes_redis_with_url_stores_and_recreates(
    runner: CliRunner, project: Path, seams: dict[str, Any]
) -> None:
    url = "rediss://:secretpw@usw1.upstash.io:6380"
    result = runner.invoke(app, ["connect", "redis", str(project), "--yes", "--url", url])
    assert result.exit_code == 0, result.output
    assert [(v, s) for _, v, s in seams["stored"]] == [("REDIS_URL", url)]
    assert "secretpw" not in result.output
    assert "local compose redis container keeps running" in result.output


def test_interactive_redis_keep_local_ensures_the_container(
    runner: CliRunner, project: Path, seams: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    (project / "docker-compose.yml").write_text(
        "services:\n  redis:\n    image: redis:7-alpine\n", encoding="utf-8"
    )
    monkeypatch.setattr(integrations, "_stdin_isatty", lambda: True)
    monkeypatch.setattr(
        "agent_scaffold.cli_interactive._interactive_select", lambda *a, **k: "local"
    )
    result = runner.invoke(app, ["connect", "redis", str(project)])
    assert result.exit_code == 0, result.output
    assert seams["stored"] == []
    assert "nothing to change" in result.output
    # keep-local now ensures the container is running
    assert seams["commands"], "docker compose up was not invoked"
    assert seams["commands"][0][0] == ["docker", "compose", "up", "-d", "redis"]


def test_interactive_redis_upstash_provisioning(
    runner: CliRunner, project: Path, seams: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(integrations, "_stdin_isatty", lambda: True)
    monkeypatch.setattr(
        "agent_scaffold.cli_interactive._interactive_select", lambda *a, **k: "provision"
    )
    spec = integrations.PROVIDER_EXTRAS["redis"]
    monkeypatch.setitem(
        integrations.PROVIDER_EXTRAS,
        "redis",
        dataclasses.replace(
            spec,
            provision=lambda timeout: UpstashDatabase(
                url="rediss://:tok@fly-y.upstash.io:6379", claim_url="https://u/claim/9"
            ),
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
    def failing_probe(
        svc: Any, *, timeout: float = 5.0, skip: bool = False, env: Any = None
    ) -> CheckResult:
        return CheckResult(
            id=f"service.{svc.id}",
            category="Recipe services",
            status=CheckStatus.FAIL,
            title=f"{svc.id}: connection failed",
        )

    monkeypatch.setattr(integrations, "run_probe", failing_probe)
    result = runner.invoke(
        app, ["connect", "redis", str(project), "--yes", "--url", "rediss://:p@h:1"]
    )
    assert result.exit_code == 1
    assert seams["stored"]  # stored, but verify surfaced the failure
    assert "connection failed" in result.output


def test_yes_postgres_with_url_stores_managed_database(
    runner: CliRunner, tmp_path: Path, seams: dict[str, Any]
) -> None:
    project = _project(tmp_path, ["relational.postgres"])
    url = "postgresql://agent:dbpassword@ep.neon.tech:5432/agent_db"
    result = runner.invoke(app, ["connect", "postgres", str(project), "--yes", "--url", url])
    assert result.exit_code == 0, result.output
    assert [(v, s) for _, v, s in seams["stored"]] == [("DATABASE_URL", url)]
    assert "dbpassword" not in result.output
    assert "local compose postgres container keeps running" in result.output


def test_yes_langfuse_env_supplied_keys_store_in_order(
    runner: CliRunner, tmp_path: Path, seams: dict[str, Any]
) -> None:
    project = _project(tmp_path, ["obs.langfuse"])
    seams["env"]["LANGFUSE_PUBLIC_KEY"] = "pk-lf-public"
    seams["env"]["LANGFUSE_SECRET_KEY"] = "sk-lf-secret"
    result = runner.invoke(app, ["connect", "langfuse", str(project), "--yes"])
    assert result.exit_code == 0, result.output
    # only the secret key goes to the vault; the public key lands in .env.local
    assert [(v, s) for _, v, s in seams["stored"]] == [("LANGFUSE_SECRET_KEY", "sk-lf-secret")]
    assert "sk-lf-secret" not in result.output
    env_local = (project / ".env.local").read_text(encoding="utf-8")
    assert "LANGFUSE_PUBLIC_KEY=pk-lf-public" in env_local


def test_interactive_qdrant_paste_captures_in_spec_order(
    runner: CliRunner, tmp_path: Path, seams: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(tmp_path, ["vector_db.qdrant"])
    monkeypatch.setattr(integrations, "_stdin_isatty", lambda: True)
    monkeypatch.setattr(
        "agent_scaffold.cli_interactive._interactive_select", lambda *a, **k: "paste"
    )
    monkeypatch.setattr(
        integrations.typer, "prompt", lambda *a, **k: "https://x.cloud.qdrant.io:6333"
    )
    monkeypatch.setattr(integrations.getpass, "getpass", lambda _p: "qd-api-key")
    result = runner.invoke(app, ["connect", "qdrant", str(project)])
    assert result.exit_code == 0, result.output
    assert [(v, s) for _, v, s in seams["stored"]] == [("QDRANT_API_KEY", "qd-api-key")]
    assert "qd-api-key" not in result.output
    env_local = (project / ".env.local").read_text(encoding="utf-8")
    assert "QDRANT_URL=https://x.cloud.qdrant.io:6333" in env_local


def test_dashboard_lists_options_with_next_commands(
    runner: CliRunner, project: Path, seams: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_probe_all(services: Any, *, timeout: float = 5.0, env: Any = None, **_k: Any) -> list:
        results = []
        for svc in services:
            status = CheckStatus.SKIP if svc.id == "langsmith" else CheckStatus.OK
            results.append(
                CheckResult(
                    id=f"service.{svc.id}",
                    category="Recipe services",
                    status=status,
                    title=f"{svc.id}: probed",
                )
            )
        return results

    monkeypatch.setattr("agent_scaffold.probes.probe_external_services", fake_probe_all)
    monkeypatch.setenv("COLUMNS", "200")
    result = runner.invoke(app, ["connect", str(project)])
    assert result.exit_code == 0, result.output
    assert "LangSmith" in result.output
    assert "Redis" in result.output
    assert "connect langsmith" in result.output


def test_dashboard_exit_1_on_probe_failure(
    runner: CliRunner, project: Path, seams: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_probe_all(services: Any, *, timeout: float = 5.0, env: Any = None, **_k: Any) -> list:
        return [
            CheckResult(
                id=f"service.{svc.id}",
                category="Recipe services",
                status=CheckStatus.FAIL,
                title=f"{svc.id}: down",
            )
            for svc in services
        ]

    monkeypatch.setattr("agent_scaffold.probes.probe_external_services", fake_probe_all)
    result = runner.invoke(app, ["connect", str(project)])
    assert result.exit_code == 1
    assert "down" in result.output
