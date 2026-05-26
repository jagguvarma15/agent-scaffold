"""Integration tests for the full ``agent-scaffold up`` step pipeline.

Hangs together Q6 + Q7: a stub-step list mirroring the real ``default_steps_for``
order is fed to ``cmd_up`` so we can assert ordering, halting, resume, and the
opt-in nature of ``commit_push``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from agent_scaffold.cli import app
from agent_scaffold.discovery import Recipe
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
    return CliRunner()


@pytest.fixture
def generated_project(tmp_path: Path) -> Path:
    write_manifest(
        tmp_path,
        Manifest(
            recipe="test-recipe",
            language="python",
            framework="none",
            model="claude-test",
            generated_at="2026-05-26T00:00:00+00:00",
        ),
    )
    return tmp_path


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
        return StepResult(self.apply_status, detail="stub", error=self.apply_error)

    def fingerprint(self, ctx: StepContext) -> str:
        return compute_fingerprint({"id": self.id, "calls": self.apply_calls})


def _install_steps(monkeypatch: pytest.MonkeyPatch, steps: list[Any]) -> None:
    from agent_scaffold import cli as cli_mod

    monkeypatch.setattr(cli_mod, "default_steps_for", lambda *a, **kw: list(steps))


def _stub_recipe(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_scaffold import cli as cli_mod

    recipe = Recipe(
        slug="test-recipe", title="Test", path=Path("/nonexistent.md"), external_services=[]
    )
    monkeypatch.setattr(cli_mod, "_resolve_recipe_silently", lambda _slug: recipe)


def _full_six() -> list[_StubStep]:
    """A stub list mirroring the default 6-step pipeline order."""
    return [
        _StubStep(id="install_deps"),
        _StubStep(id="docker_up", depends_on=("install_deps",)),
        _StubStep(id="wire_credentials"),
        _StubStep(id="migrations", depends_on=("docker_up", "wire_credentials")),
        _StubStep(id="seed", depends_on=("migrations",)),
        _StubStep(id="smoke_test", depends_on=("seed",)),
    ]


def test_full_happy_path_runs_all_six_in_dag_order(
    runner: CliRunner, generated_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    steps = _full_six()
    _install_steps(monkeypatch, steps)
    _stub_recipe(monkeypatch)
    result = runner.invoke(app, ["up", str(generated_project), "--yes"])
    assert result.exit_code == 0, result.output
    for s in steps:
        assert s.apply_calls == 1, f"{s.id} not run"


def test_failed_migrations_halts_seed_and_smoke(
    runner: CliRunner, generated_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    steps = _full_six()
    migrations = steps[3]
    migrations.apply_status = StepStatus.FAILED
    migrations.apply_error = "could not connect to server"
    _install_steps(monkeypatch, steps)
    _stub_recipe(monkeypatch)
    result = runner.invoke(app, ["up", str(generated_project), "--yes"])
    assert result.exit_code == 1
    by_id = {s.id: s for s in steps}
    assert by_id["install_deps"].apply_calls == 1
    assert by_id["docker_up"].apply_calls == 1
    assert by_id["wire_credentials"].apply_calls == 1
    assert by_id["migrations"].apply_calls == 1
    assert by_id["seed"].apply_calls == 0, "seed must not run after migrations failed"
    assert by_id["smoke_test"].apply_calls == 0
    # Failure panel rendered with the matched troubleshoot hint.
    assert "migrations failed" in result.output


def test_resume_after_failed_migrations_reruns_only_failed_and_downstream(
    runner: CliRunner, generated_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # First run: migrations fails.
    steps_first = _full_six()
    steps_first[3].apply_status = StepStatus.FAILED
    steps_first[3].apply_error = "boom"
    _install_steps(monkeypatch, steps_first)
    _stub_recipe(monkeypatch)
    first = runner.invoke(app, ["up", str(generated_project), "--yes"])
    assert first.exit_code == 1

    # Second run with --resume + fresh stubs: install_deps etc. should skip.
    steps_second = _full_six()
    _install_steps(monkeypatch, steps_second)
    second = runner.invoke(app, ["up", str(generated_project), "--yes", "--resume"])
    assert second.exit_code == 0
    by_id = {s.id: s for s in steps_second}
    # DONE-on-disk steps don't re-run on --resume.
    assert by_id["install_deps"].apply_calls == 0
    assert by_id["docker_up"].apply_calls == 0
    assert by_id["wire_credentials"].apply_calls == 0
    # migrations was FAILED on disk → re-run; seed/smoke follow.
    assert by_id["migrations"].apply_calls == 1
    assert by_id["seed"].apply_calls == 1
    assert by_id["smoke_test"].apply_calls == 1


def test_commit_push_not_included_by_default(
    runner: CliRunner, generated_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``commit_push`` is opt-in via recipe; default plan should not list it."""
    from agent_scaffold import cli as cli_mod
    from agent_scaffold.steps import default_steps_for as real_factory

    _stub_recipe(monkeypatch)
    monkeypatch.setattr(cli_mod, "default_steps_for", real_factory)
    result = runner.invoke(app, ["up", str(generated_project), "--plan"])
    assert result.exit_code == 0, result.output
    assert "commit_push" not in result.output, "commit_push must be opt-in"
