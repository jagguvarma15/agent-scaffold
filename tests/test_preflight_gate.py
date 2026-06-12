"""Tests for agent_scaffold.preflight — the env + service gate before the LLM call."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from agent_scaffold import envfile as envfile_mod
from agent_scaffold import preflight as preflight_mod
from agent_scaffold.catalog import EnvContractEntry, RecipeEntry
from agent_scaffold.discovery import ExternalService, Recipe
from agent_scaffold.doctor import CheckResult, CheckStatus
from agent_scaffold.preflight import (
    PreflightReport,
    collect_env_requirements,
    fill_missing,
    persist_filled,
    render_service_panel,
    run_preflight,
)


def _recipe(services: list[ExternalService] | None = None) -> Recipe:
    return Recipe(
        slug="docs-rag-qa",
        title="Recipe: docs-rag-qa",
        path=Path("docs/recipes/docs-rag-qa.md"),
        external_services=services or [],
    )


def _catalog_entry(env_contract: list[dict[str, Any]] | None = None) -> RecipeEntry:
    return RecipeEntry.model_validate(
        {
            "slug": "docs-rag-qa",
            "path": "docs/recipes/docs-rag-qa.md",
            "title": "Recipe: docs-rag-qa",
            "env_contract": env_contract or [],
        }
    )


class _Cap:
    def __init__(self, cap_id: str, env_vars: list[str]) -> None:
        self.id = cap_id
        self.env_vars = env_vars


class _Stack:
    def __init__(self, caps: list[_Cap]) -> None:
        self.capabilities = caps


def _console() -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    return Console(file=buf, force_terminal=False, width=120), buf


@pytest.fixture(autouse=True)
def _no_keyring(monkeypatch: pytest.MonkeyPatch) -> None:
    """Presence checks must not consult the developer's real auth backends."""
    monkeypatch.setattr(envfile_mod, "load_key", lambda: None)


def test_collect_unions_three_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    recipe = _recipe([ExternalService(id="redis", env_vars=["REDIS_URL"], required=True)])
    entry = _catalog_entry(
        [
            {"name": "LANGFUSE_HOST", "source_capability": "obs.langfuse"},
            {"name": "APP_PORT", "source_capability": "recipe", "default": 8000},
        ]
    )
    stack = _Stack([_Cap("vector_db.qdrant", ["QDRANT_URL"])])

    reqs = collect_env_requirements(recipe, entry, stack, tmp_path)
    by_name = {r.name: r for r in reqs}

    assert set(by_name) == {"REDIS_URL", "LANGFUSE_HOST", "APP_PORT", "QDRANT_URL"}
    assert by_name["REDIS_URL"].source == "redis"
    assert by_name["QDRANT_URL"].source == "vector_db.qdrant"
    # default counts as satisfied without user input
    assert by_name["APP_PORT"].satisfied and by_name["APP_PORT"].has_default
    assert not by_name["LANGFUSE_HOST"].satisfied


def test_collect_merge_rules_first_source_wins_required_ored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("REDIS_URL", raising=False)
    recipe = _recipe([ExternalService(id="redis", env_vars=["REDIS_URL"], required=False)])
    stack = _Stack([_Cap("cache.redis", ["REDIS_URL"])])

    reqs = collect_env_requirements(recipe, None, stack, tmp_path)
    assert len(reqs) == 1
    req = reqs[0]
    assert req.source == "redis"  # first-seen wins
    assert req.required  # capability declaration ORs in required=True


def test_collect_sees_env_and_env_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
    monkeypatch.delenv("QDRANT_URL", raising=False)
    (tmp_path / ".env.local").write_text("QDRANT_URL=http://localhost:6333\n", encoding="utf-8")
    recipe = _recipe(
        [
            ExternalService(id="redis", env_vars=["REDIS_URL"]),
            ExternalService(id="qdrant", env_vars=["QDRANT_URL"]),
        ]
    )
    reqs = collect_env_requirements(recipe, None, None, tmp_path)
    assert all(r.satisfied for r in reqs)


def test_non_interactive_prints_missing_to_stderr_and_skips_probes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("REDIS_URL", raising=False)
    recipe = _recipe([ExternalService(id="redis", env_vars=["REDIS_URL"], required=True)])
    probes_called: list[Any] = []
    console, buf = _console()

    report = run_preflight(
        recipe=recipe,
        catalog_entry=None,
        resolved_stack=None,
        project_dir=tmp_path,
        console=console,
        interactive=False,
        probe=lambda svcs: probes_called.append(svcs) or [],
    )

    err = capsys.readouterr().err
    assert "REDIS_URL" in err
    assert probes_called == []  # never probe in non-interactive mode
    assert buf.getvalue() == ""  # nothing rendered to the console either
    assert [r.name for r in report.missing_required] == ["REDIS_URL"]


def test_interactive_renders_panels_and_reuses_probe_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
    svc = ExternalService(id="redis", env_vars=["REDIS_URL"], probe="redis_ping")
    recipe = _recipe([svc])
    fake_result = CheckResult(
        id="redis", category="service", status=CheckStatus.OK, title="redis: PING ok"
    )
    console, buf = _console()

    report = run_preflight(
        recipe=recipe,
        catalog_entry=None,
        resolved_stack=None,
        project_dir=tmp_path,
        console=console,
        interactive=True,
        probe=lambda _svcs: [fake_result],
        confirm=lambda _p: False,
    )

    output = buf.getvalue()
    assert "Pre-flight: environment" in output
    assert "Pre-flight: services" in output
    assert "redis: PING ok" in output
    assert report.probe_results == [fake_result]


def test_fill_missing_exports_env_and_queues_persistence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("REDIS_URL", raising=False)
    recipe = _recipe([ExternalService(id="redis", env_vars=["REDIS_URL"], required=True)])
    console, _buf = _console()

    report = run_preflight(
        recipe=recipe,
        catalog_entry=None,
        resolved_stack=None,
        project_dir=tmp_path,
        console=console,
        interactive=True,
        probe=lambda _svcs: [],
        confirm=lambda _p: True,
        ask=lambda _p: "redis://filled:6379",
    )

    import os

    assert os.environ["REDIS_URL"] == "redis://filled:6379"
    assert "REDIS_URL" in report.filled
    assert report.missing == []  # requirement re-marked satisfied
    monkeypatch.delenv("REDIS_URL", raising=False)


def test_fill_missing_routes_anthropic_to_auth_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    stored: dict[str, Any] = {}

    def fake_store(name: str, value: Any, backend: str = "keyring") -> None:
        stored["name"] = name
        stored["backend"] = backend

    monkeypatch.setattr(preflight_mod, "store_key", fake_store)
    console, _buf = _console()
    report = PreflightReport(
        requirements=collect_env_requirements(
            _recipe([ExternalService(id="anthropic", env_vars=["ANTHROPIC_API_KEY"])]),
            None,
            None,
            tmp_path,
        )
    )

    fill_missing(report, console, ask=lambda _p: "sk-ant-test123456")

    assert stored == {"name": "anthropic", "backend": "keyring"}
    assert "ANTHROPIC_API_KEY" not in report.filled  # never queued for .env.local
    assert report.missing == []


def test_fill_missing_empty_input_skips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REDIS_URL", raising=False)
    console, _buf = _console()
    report = PreflightReport(
        requirements=collect_env_requirements(
            _recipe([ExternalService(id="redis", env_vars=["REDIS_URL"])]),
            None,
            None,
            tmp_path,
        )
    )
    fill_missing(report, console, ask=lambda _p: "")
    assert report.filled == {}
    assert [r.name for r in report.missing] == ["REDIS_URL"]


def test_persist_filled_writes_env_local_after_dir_exists(tmp_path: Path) -> None:
    from pydantic import SecretStr

    project = tmp_path / "proj"
    filled = {"REDIS_URL": SecretStr("redis://filled:6379")}

    # Before the project dir exists (gate time): nothing persisted, no error.
    assert persist_filled(project, filled) == []

    project.mkdir()
    assert persist_filled(project, filled) == ["REDIS_URL"]
    text = (project / ".env.local").read_text(encoding="utf-8")
    assert "REDIS_URL=redis://filled:6379" in text


def test_service_panel_softens_docker_backed_failures() -> None:
    svc = ExternalService(id="qdrant", env_vars=["QDRANT_URL"], docker_service="qdrant")
    fail = CheckResult(
        id="qdrant",
        category="service",
        status=CheckStatus.FAIL,
        title="qdrant: connection refused",
    )
    console, buf = _console()
    console.print(render_service_panel([fail], [svc]))
    output = buf.getvalue()
    assert "docker compose" in output
    assert "✗" not in output  # softened, not alarming
