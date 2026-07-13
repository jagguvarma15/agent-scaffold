"""Tests for the ``agent-scaffold up`` subcommand."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from agent_scaffold.cli import app
from agent_scaffold.discovery import ExternalService, Recipe
from agent_scaffold.manifest import Manifest, write_manifest
from agent_scaffold.orchestrator import (
    DetectionResult,
    StepContext,
    StepResult,
    StepStatus,
    compute_fingerprint,
)


@pytest.fixture
def runner() -> CliRunner:
    # mix_stderr is removed in newer Click — older signature in case it's needed.
    return CliRunner()


@pytest.fixture
def generated_project(tmp_path: Path) -> Path:
    """A directory with a valid ``.scaffold/manifest.json``."""
    manifest = Manifest(
        recipe="test-recipe",
        language="python",
        framework="none",
        model="claude-test",
        generated_at="2026-05-24T00:00:00+00:00",
    )
    write_manifest(tmp_path, manifest)
    return tmp_path


# Tiny step doubles -----------------------------------------------------


@dataclass
class _StubStep:
    id: str
    description: str = "stub"
    depends_on: tuple[str, ...] = ()
    optional: bool = True
    detect_status: StepStatus = StepStatus.PENDING
    apply_status: StepStatus = StepStatus.DONE
    apply_error: str | None = None
    apply_calls: int = field(default=0, init=False)

    def detect(self, ctx: StepContext) -> DetectionResult:
        return DetectionResult(self.detect_status, reason="stub")

    def apply(self, ctx: StepContext) -> StepResult:
        self.apply_calls += 1
        return StepResult(self.apply_status, detail="stub done", error=self.apply_error)

    def fingerprint(self, ctx: StepContext) -> str:
        return compute_fingerprint({"id": self.id})


def _install_steps(monkeypatch: pytest.MonkeyPatch, steps: list[Any]) -> None:
    from agent_scaffold import cli as cli_mod

    monkeypatch.setattr(cli_mod, "default_steps_for", lambda *a, **kw: list(steps))


def _stub_recipe(
    monkeypatch: pytest.MonkeyPatch, services: list[ExternalService] | None = None
) -> None:
    """Make ``_resolve_recipe_silently`` return a controlled Recipe."""
    from agent_scaffold import cli as cli_mod

    recipe = Recipe(
        slug="test-recipe",
        title="Test",
        path=Path("/nonexistent.md"),
        external_services=services or [],
    )
    monkeypatch.setattr(cli_mod, "_resolve_recipe_silently", lambda _slug: recipe)


# Tests ------------------------------------------------------------------


def test_up_plan_only_exits_without_running(
    runner: CliRunner, generated_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    step = _StubStep(id="s1")
    _install_steps(monkeypatch, [step])
    _stub_recipe(monkeypatch)
    result = runner.invoke(app, ["up", str(generated_project), "--plan"])
    assert result.exit_code == 0, result.output
    assert step.apply_calls == 0
    assert "Provisioning plan" in result.output


def test_up_yes_skips_confirmation_and_runs(
    runner: CliRunner, generated_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    step = _StubStep(id="s1")
    _install_steps(monkeypatch, [step])
    _stub_recipe(monkeypatch)
    result = runner.invoke(app, ["up", str(generated_project), "--yes"])
    assert result.exit_code == 0, result.output
    assert step.apply_calls == 1
    assert "Run summary" in result.output


def test_up_failed_step_renders_failure_panel(
    runner: CliRunner, generated_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = _StubStep(
        id="docker_up", apply_status=StepStatus.FAILED, apply_error="port already in use"
    )
    _install_steps(monkeypatch, [bad])
    _stub_recipe(monkeypatch)
    result = runner.invoke(app, ["up", str(generated_project), "--yes"])
    assert result.exit_code == 1
    assert "docker_up failed" in result.output
    assert "port already in use" in result.output


def test_up_optional_step_failure_does_not_block_independent_steps(
    runner: CliRunner, generated_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A best-effort (optional) step failing must NOT halt independent steps —
    # this is what lets the servers come up even when e.g. an eval fails.
    bad = _StubStep(id="first", apply_status=StepStatus.FAILED, apply_error="boom")
    after = _StubStep(id="second")  # depends_on=() → independent
    _install_steps(monkeypatch, [bad, after])
    _stub_recipe(monkeypatch)
    result = runner.invoke(app, ["up", str(generated_project), "--yes"])
    assert result.exit_code == 1  # a failure still surfaces as non-zero
    assert bad.apply_calls == 1
    assert after.apply_calls == 1  # but the independent step still ran
    state = generated_project / ".scaffold" / "state.json"
    assert "failed" in state.read_text(encoding="utf-8")


def test_up_essential_step_failure_halts(
    runner: CliRunner, generated_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An essential step (optional=False, e.g. install_deps) failing halts the run.
    bad = _StubStep(id="first", apply_status=StepStatus.FAILED, apply_error="boom", optional=False)
    after = _StubStep(id="second")
    _install_steps(monkeypatch, [bad, after])
    _stub_recipe(monkeypatch)
    result = runner.invoke(app, ["up", str(generated_project), "--yes"])
    assert result.exit_code == 1
    assert bad.apply_calls == 1
    assert after.apply_calls == 0  # halted


def test_up_failed_step_blocks_only_its_dependents(
    runner: CliRunner, generated_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Dependency-aware skip: a failure blocks its dependents but not independents.
    a = _StubStep(id="a", apply_status=StepStatus.FAILED, apply_error="boom")
    b = _StubStep(id="b", depends_on=("a",))  # depends on the failure → blocked
    c = _StubStep(id="c")  # independent → still runs
    _install_steps(monkeypatch, [a, b, c])
    _stub_recipe(monkeypatch)
    result = runner.invoke(app, ["up", str(generated_project), "--yes"])
    assert result.exit_code == 1
    assert a.apply_calls == 1
    assert b.apply_calls == 0  # blocked by failed 'a'
    assert c.apply_calls == 1  # independent → ran


def test_up_resume_skips_done_steps(
    runner: CliRunner, generated_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    step = _StubStep(id="only_one")
    _install_steps(monkeypatch, [step])
    _stub_recipe(monkeypatch)
    # First run completes.
    assert runner.invoke(app, ["up", str(generated_project), "--yes"]).exit_code == 0
    # Second run with --resume sees state DONE and skips.
    second = runner.invoke(app, ["up", str(generated_project), "--yes", "--resume"])
    assert second.exit_code == 0
    assert step.apply_calls == 1  # not re-run


def test_up_only_runs_named_step_plus_deps(
    runner: CliRunner, generated_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a = _StubStep(id="a")
    b = _StubStep(id="b", depends_on=("a",))
    c = _StubStep(id="c")
    _install_steps(monkeypatch, [a, b, c])
    _stub_recipe(monkeypatch)
    result = runner.invoke(app, ["up", str(generated_project), "--yes", "--only", "b"])
    assert result.exit_code == 0
    assert a.apply_calls == 1  # pulled in as a dep of b
    assert b.apply_calls == 1
    assert c.apply_calls == 0


def test_up_missing_manifest_exits_with_helpful_error(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(app, ["up", str(tmp_path), "--yes"])
    assert result.exit_code == 1
    assert "manifest" in result.output.lower()


def test_up_skip_flag_marks_step_skipped(
    runner: CliRunner, generated_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a = _StubStep(id="a")
    b = _StubStep(id="b")
    _install_steps(monkeypatch, [a, b])
    _stub_recipe(monkeypatch)
    result = runner.invoke(app, ["up", str(generated_project), "--yes", "--skip", "b"])
    assert result.exit_code == 0
    assert a.apply_calls == 1
    assert b.apply_calls == 0


# ---------------------------------------------------------------------------
# _resolve_use_docker — opt-in docker mode
# ---------------------------------------------------------------------------


def _flags(**overrides: Any) -> Any:
    from agent_scaffold.cli import StepFlags

    base: dict[str, Any] = dict(
        only=[], skip=[], force=[], retry=[], resume=False, plan_only=False, yes=False, debug=False
    )
    base.update(overrides)
    return StepFlags(**base)


def test_resolve_use_docker_explicit_no_docker(tmp_path: Path) -> None:
    from agent_scaffold.cli import _resolve_use_docker

    # --no-docker → local, without even probing docker.
    assert _resolve_use_docker(_flags(use_docker=False), True, tmp_path) is False


def test_resolve_use_docker_explicit_docker_when_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_scaffold.cli import _resolve_use_docker

    monkeypatch.setattr(
        "agent_scaffold.steps.docker_up.docker_available", lambda **_k: (True, "ok")
    )
    assert _resolve_use_docker(_flags(use_docker=True), False, tmp_path) is True


def test_resolve_use_docker_falls_back_to_local_when_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_scaffold.cli import _resolve_use_docker

    monkeypatch.setattr(
        "agent_scaffold.steps.docker_up.docker_available", lambda **_k: (False, "not installed")
    )
    assert _resolve_use_docker(_flags(use_docker=True), False, tmp_path) is False


def test_resolve_use_docker_prompts_when_interactive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_scaffold import cli as cli_mod
    from agent_scaffold.cli import _resolve_use_docker

    (tmp_path / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")
    monkeypatch.setattr(cli_mod, "_interactive_select", lambda *_a, **_k: "docker")
    monkeypatch.setattr(
        "agent_scaffold.steps.docker_up.docker_available", lambda **_k: (True, "ok")
    )
    assert _resolve_use_docker(_flags(use_docker=None), True, tmp_path) is True


def test_resolve_use_docker_non_interactive_defaults_local(tmp_path: Path) -> None:
    from agent_scaffold.cli import _resolve_use_docker

    (tmp_path / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")
    # None intent + non-interactive → local (never prompts).
    assert _resolve_use_docker(_flags(use_docker=None), False, tmp_path) is False


# Port-conflict pre-flight + recovery ------------------------------------

BIND_ERROR = "Bind for 0.0.0.0:6379 failed: port is already allocated"


def _write_compose(project_dir: Path) -> None:
    (project_dir / "docker-compose.yml").write_text(
        'services:\n  redis:\n    image: redis:7\n    ports:\n      - "6379:6379"\n',
        encoding="utf-8",
    )


def _docker_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agent_scaffold.steps.docker_up.docker_available", lambda **_k: (True, "ok")
    )


def _canned_conflict(port: int = 6379) -> Any:
    from agent_scaffold import ports

    owner = ports.PortOwner(kind="process", port=port, pid=4242, command="redis-server")
    return ports.PortConflict(port=port, owner=owner)


def test_up_preflight_port_conflict_yes_fails_fast(
    runner: CliRunner, generated_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_scaffold import cli as cli_mod
    from agent_scaffold import ports

    step = _StubStep(id="docker_up")
    _install_steps(monkeypatch, [step])
    _stub_recipe(monkeypatch)
    _write_compose(generated_project)
    _docker_ok(monkeypatch)
    monkeypatch.setattr(ports, "port_in_use", lambda _p: True)
    monkeypatch.setattr(ports, "scan_conflicts", lambda busy, **_k: [_canned_conflict()])
    ran: list[list[str]] = []
    monkeypatch.setattr(
        cli_mod, "_run_remediation_command", lambda argv, **_k: ran.append(argv) or True
    )

    result = runner.invoke(app, ["up", str(generated_project), "--docker", "--yes"])

    assert result.exit_code == 1
    assert step.apply_calls == 0  # aborted before the orchestrator ran
    assert "6379" in result.output
    assert "lsof -nP -iTCP:6379 -sTCP:LISTEN" in result.output
    assert ran == []  # --yes never runs a kill


def test_up_preflight_interactive_remediation_success_proceeds(
    runner: CliRunner, generated_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_scaffold import cli as cli_mod
    from agent_scaffold import ports

    step = _StubStep(id="docker_up")
    _install_steps(monkeypatch, [step])
    _stub_recipe(monkeypatch)
    _write_compose(generated_project)
    _docker_ok(monkeypatch)
    monkeypatch.setattr(cli_mod, "_interactive_select", lambda *_a, **_k: "yes")
    monkeypatch.setattr(ports, "port_in_use", lambda _p: True)
    monkeypatch.setattr(ports, "scan_conflicts", lambda busy, **_k: [_canned_conflict()])
    remediations: list[Any] = []
    monkeypatch.setattr(
        cli_mod,
        "_remediate_port_conflicts",
        lambda conflicts, **_k: remediations.append(conflicts) or True,
    )

    result = runner.invoke(app, ["up", str(generated_project), "--docker"])

    assert result.exit_code == 0, result.output
    assert step.apply_calls == 1
    assert len(remediations) == 1


def test_up_preflight_interactive_remediation_declined_aborts(
    runner: CliRunner, generated_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_scaffold import cli as cli_mod
    from agent_scaffold import ports

    step = _StubStep(id="docker_up")
    _install_steps(monkeypatch, [step])
    _stub_recipe(monkeypatch)
    _write_compose(generated_project)
    _docker_ok(monkeypatch)
    monkeypatch.setattr(cli_mod, "_interactive_select", lambda *_a, **_k: "yes")
    monkeypatch.setattr(ports, "port_in_use", lambda _p: True)
    monkeypatch.setattr(ports, "scan_conflicts", lambda busy, **_k: [_canned_conflict()])
    monkeypatch.setattr(cli_mod, "_remediate_port_conflicts", lambda *_a, **_k: False)

    result = runner.invoke(app, ["up", str(generated_project), "--docker"])

    assert result.exit_code == 1
    assert step.apply_calls == 0
    assert "aborting before docker compose up" in result.output


def test_up_preflight_no_conflict_runs_normally(
    runner: CliRunner, generated_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_scaffold import ports

    step = _StubStep(id="docker_up")
    _install_steps(monkeypatch, [step])
    _stub_recipe(monkeypatch)
    _write_compose(generated_project)
    _docker_ok(monkeypatch)
    monkeypatch.setattr(ports, "port_in_use", lambda _p: False)

    result = runner.invoke(app, ["up", str(generated_project), "--docker", "--yes"])

    assert result.exit_code == 0, result.output
    assert step.apply_calls == 1
    assert "Port conflicts" not in result.output


def test_up_port_conflict_recovery_reruns_failed_step(
    runner: CliRunner, generated_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_scaffold import cli as cli_mod
    from agent_scaffold import ports
    from agent_scaffold.orchestrator import Orchestrator

    bad = _StubStep(id="docker_up", apply_status=StepStatus.FAILED, apply_error=BIND_ERROR)
    _install_steps(monkeypatch, [bad])
    _stub_recipe(monkeypatch)
    monkeypatch.setattr(cli_mod, "_interactive_select", lambda *_a, **_k: "yes")
    monkeypatch.setattr(ports, "port_in_use", lambda _p: False)  # freed by check time
    monkeypatch.setattr("typer.confirm", lambda *_a, **_k: True)

    run_calls: list[dict[str, Any]] = []
    orig_run = Orchestrator.run

    def spy_run(self: Orchestrator, **kwargs: Any) -> Any:
        run_calls.append(kwargs)
        return orig_run(self, **kwargs)

    monkeypatch.setattr(Orchestrator, "run", spy_run)

    result = runner.invoke(app, ["up", str(generated_project)])

    assert result.exit_code == 1  # the stub fails again on retry
    assert bad.apply_calls == 2
    assert "are free now" in result.output
    assert len(run_calls) == 2
    assert run_calls[1]["retry"] == ["docker_up"]
    assert run_calls[1]["resume"] is True


def test_up_port_conflict_yes_prints_manual_commands(
    runner: CliRunner, generated_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = _StubStep(id="docker_up", apply_status=StepStatus.FAILED, apply_error=BIND_ERROR)
    _install_steps(monkeypatch, [bad])
    _stub_recipe(monkeypatch)

    result = runner.invoke(app, ["up", str(generated_project), "--yes"])

    assert result.exit_code == 1
    assert bad.apply_calls == 1  # no retry without prompts
    assert "lsof -nP -iTCP:6379 -sTCP:LISTEN" in result.output


def test_remediate_port_conflicts_failed_command_returns_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_scaffold import cli as cli_mod
    from agent_scaffold import ports

    monkeypatch.setattr(cli_mod, "_confirm_remediation_command", lambda *_a, **_k: True)
    monkeypatch.setattr(cli_mod, "_run_remediation_command", lambda *_a, **_k: False)
    monkeypatch.setattr(ports, "port_in_use", lambda _p: True)

    ok = cli_mod._remediate_port_conflicts(
        [_canned_conflict()], project_dir=tmp_path, compose_dir=None
    )
    assert ok is False


def test_remediate_port_conflicts_groups_ports_by_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_scaffold import cli as cli_mod
    from agent_scaffold import ports

    owner_a = ports.PortOwner(kind="docker", port=6379, container_id="abc", container_name="redis")
    conflicts = [
        ports.PortConflict(port=6379, owner=owner_a),
        ports.PortConflict(
            port=6380,
            owner=ports.PortOwner(
                kind="docker", port=6380, container_id="abc", container_name="redis"
            ),
        ),
    ]
    confirms: list[list[str]] = []
    monkeypatch.setattr(
        cli_mod,
        "_confirm_remediation_command",
        lambda argv, **_k: confirms.append(argv) or True,
    )
    monkeypatch.setattr(cli_mod, "_run_remediation_command", lambda *_a, **_k: True)
    monkeypatch.setattr(ports, "port_in_use", lambda _p: False)
    monkeypatch.setattr(ports, "wait_port_free", lambda _p, **_k: True)

    ok = cli_mod._remediate_port_conflicts(conflicts, project_dir=tmp_path, compose_dir=None)
    assert ok is True
    assert confirms == [["docker", "stop", "redis"]]  # one confirm for both ports
