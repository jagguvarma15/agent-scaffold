"""Tests for the REPL readiness layer: `/config`, `/status`, and the generate gate.

`config_requirements` / `required_gaps` decide what "configured" means; the three
surfaces (config fill, status display, generate block) all agree because they
route through them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_scaffold.config import Config
from agent_scaffold.discovery import ExternalService, Recipe
from agent_scaffold.repl import readiness
from agent_scaffold.repl.commands import CommandHandler, CommandResult
from agent_scaffold.repl.readiness import config_requirements, required_gaps
from agent_scaffold.repl.session import SessionState
from agent_scaffold.sources import DEPLOYMENTS_SPEC, ResolvedSource

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_ambient_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """No Anthropic key resolvable by default (env cleared; keyring is the
    in-memory autouse fake). Tests opt in by setting the env var."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture
def base_state(tmp_path: Path) -> SessionState:
    # `anthropic_api_key` on Config is unrelated to `resolve_active` (which reads
    # env > keyring > file); the readiness key-check ignores Config entirely.
    cfg = Config(
        anthropic_api_key="cfg-placeholder",
        cache_dir=tmp_path / "cache",
        failures_dir=tmp_path / "cache" / "failures",
    )
    src = ResolvedSource(
        spec=DEPLOYMENTS_SPEC,
        path=tmp_path / "deployments",
        label="test",
        kind="explicit-path",
        commit_sha=None,
    )
    # dest isolates the env_local / vault presence reads to a clean tmp dir.
    return SessionState(cfg=cfg, deployments=src, blueprints=src, dest=tmp_path / "proj")


def _recipe(tmp_path: Path, *, services: list[ExternalService] | None = None) -> Recipe:
    md = tmp_path / "r.md"
    md.write_text("# R\n", encoding="utf-8")
    return Recipe(slug="r", title="R", path=md, external_services=services or [])


# ---------------------------------------------------------------------------
# config_requirements / required_gaps
# ---------------------------------------------------------------------------


def test_anthropic_key_is_always_required_and_gates_when_absent(base_state: SessionState) -> None:
    reqs = config_requirements(base_state)
    key = next(r for r in reqs if r.name == "ANTHROPIC_API_KEY")
    assert key.required and not key.satisfied
    assert required_gaps(base_state) == ["ANTHROPIC_API_KEY"]


def test_anthropic_key_satisfied_from_env(
    base_state: SessionState, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-0001")
    assert required_gaps(base_state) == []


def test_external_service_credential_gates(
    base_state: SessionState, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-0001")  # isolate the service check
    # A non-docker external service (a cloud search API) with a required key.
    svc = ExternalService(id="tavily", env_vars=["TAVILY_API_KEY"], required=True)
    state = base_state
    state.recipe = _recipe(tmp_path, services=[svc])
    assert "TAVILY_API_KEY" in required_gaps(state)


def test_docker_provided_var_does_not_gate(
    base_state: SessionState, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-0001")
    # A docker-backed service: `up` wires DATABASE_URL, so it must NOT gate even
    # though it's declared required and isn't set in the environment.
    svc = ExternalService(
        id="postgres", env_vars=["DATABASE_URL"], required=True, docker_service="postgres"
    )
    state = base_state
    state.recipe = _recipe(tmp_path, services=[svc])
    gaps = required_gaps(state)
    assert "DATABASE_URL" not in gaps
    assert gaps == []  # only the (now-satisfied) key would gate, and it's set


# ---------------------------------------------------------------------------
# /config and /status commands
# ---------------------------------------------------------------------------


def test_cmd_config_defers_to_shell_for_interactive_fill(base_state: SessionState) -> None:
    result = CommandHandler(recipes=[]).dispatch("/config", base_state)
    assert result.next_action == "config"  # shell owns the getpass I/O


def test_cmd_status_reports_missing_key_and_docker(
    base_state: SessionState, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Deterministic docker line regardless of the host.
    monkeypatch.setattr(readiness, "docker_status", lambda **_k: (False, "not installed"))
    result = CommandHandler(recipes=[]).dispatch("/status", base_state)
    text = _messages_text(result)
    assert "ANTHROPIC_API_KEY" in text
    assert "Docker" in text and "not installed" in text
    assert "/config" in text  # nudges the user to the fix


def test_cmd_status_ready_when_key_present(
    base_state: SessionState, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-0001")
    monkeypatch.setattr(readiness, "docker_status", lambda **_k: (True, "ok"))
    text = _messages_text(CommandHandler(recipes=[]).dispatch("/status", base_state))
    assert "Ready to generate" in text


# ---------------------------------------------------------------------------
# the blocking generate gate
# ---------------------------------------------------------------------------


def test_generate_gate_blocks_and_never_spends_when_unconfigured(
    base_state: SessionState, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rich.console import Console

    from agent_scaffold.repl import shell

    # The gate reads `required_gaps` from the readiness module at call time.
    monkeypatch.setattr(readiness, "required_gaps", lambda _s: ["ANTHROPIC_API_KEY"])

    def _boom(*_a: object, **_k: object) -> object:
        raise AssertionError("run_generation must not be called when the gate blocks")

    monkeypatch.setattr(shell, "run_generation", _boom)
    console = Console(record=True, color_system=None, width=100)
    shell._run_generation_and_render(base_state, console)
    text = console.export_text()
    assert "Not configured yet" in text and "/config" in text


def _messages_text(result: CommandResult) -> str:
    from rich.console import Console

    console = Console(record=True, color_system=None, width=120)
    for msg in result.messages:
        console.print(msg)
    return console.export_text()
