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


def test_up_failed_step_halts_subsequent_steps(
    runner: CliRunner, generated_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = _StubStep(id="first", apply_status=StepStatus.FAILED, apply_error="boom")
    after = _StubStep(id="second")
    _install_steps(monkeypatch, [bad, after])
    _stub_recipe(monkeypatch)
    result = runner.invoke(app, ["up", str(generated_project), "--yes"])
    assert result.exit_code == 1
    assert bad.apply_calls == 1
    assert after.apply_calls == 0
    # State file recorded the failure.
    state = generated_project / ".scaffold" / "state.json"
    assert state.is_file()
    assert "failed" in state.read_text(encoding="utf-8")


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
