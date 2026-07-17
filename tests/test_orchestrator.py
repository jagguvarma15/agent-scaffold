"""Tests for ``agent_scaffold.orchestrator`` — the keystone framework.

Cover state persistence, fingerprinting, topology, flag semantics, crash
recovery, and progress events. Concrete production steps land in Q6/Q7;
here we exercise only the abstract framework via ``tests/fixtures/steps``.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from agent_scaffold.manifest import Manifest
from agent_scaffold.orchestrator import (
    CycleError,
    DetectionResult,
    MissingDependencyError,
    Orchestrator,
    OrchestratorError,
    OrchestratorState,
    PlanRow,
    StepContext,
    StepFinished,
    StepResult,
    StepStarted,
    StepState,
    StepStatus,
    compute_fingerprint,
    read_state,
    render_plan_table,
    state_path,
    write_state,
)
from tests.fixtures.steps import (
    AlreadyDoneStep,
    DependentStep,
    DriftingStep,
    FailingStep,
    FlakyStep,
    NoopStep,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _manifest() -> Manifest:
    return Manifest(
        recipe="test-recipe",
        language="python",
        framework="none",
        model="claude-test",
        generated_at="2026-05-24T00:00:00+00:00",
    )


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    return tmp_path


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------


def test_read_state_missing_file_returns_fresh(project_dir: Path) -> None:
    state = read_state(project_dir)
    assert state.steps == {}
    assert state.schema_version >= 1


def test_state_round_trip(project_dir: Path) -> None:
    state = OrchestratorState(
        started_at="2026-05-24T01:00:00+00:00",
        last_run_at="2026-05-24T02:00:00+00:00",
        steps={
            "alpha": StepState(status=StepStatus.DONE, fingerprint="sha256:abc", attempt=1),
            "beta": StepState(status=StepStatus.FAILED, error="boom"),
        },
    )
    write_state(project_dir, state)
    loaded = read_state(project_dir)
    assert loaded.started_at == state.started_at
    assert loaded.steps["alpha"].status == StepStatus.DONE
    assert loaded.steps["alpha"].fingerprint == "sha256:abc"
    assert loaded.steps["alpha"].attempt == 1
    assert loaded.steps["beta"].status == StepStatus.FAILED
    assert loaded.steps["beta"].error == "boom"


def test_state_file_is_mode_0644(project_dir: Path) -> None:
    write_state(project_dir, OrchestratorState())
    path = state_path(project_dir)
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o644


def test_state_atomic_no_partial_file_on_failure(
    project_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If ``os.replace`` fails mid-write, the prior state.json must be intact."""
    write_state(project_dir, OrchestratorState(started_at="original"))
    original = state_path(project_dir).read_text()

    def boom(src: str, dst: str | Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr("agent_scaffold.orchestrator.os.replace", boom)
    with pytest.raises(OSError):
        write_state(project_dir, OrchestratorState(started_at="new"))
    # The original is untouched; no .tmp leak in the directory.
    assert state_path(project_dir).read_text() == original
    leaked = list(state_path(project_dir).parent.glob("state.json.*"))
    assert leaked == []


def test_read_state_invalid_json_raises(project_dir: Path) -> None:
    target = state_path(project_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("not json")
    with pytest.raises(OrchestratorError):
        read_state(project_dir)


def test_read_state_drops_malformed_steps(project_dir: Path) -> None:
    target = state_path(project_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "started_at": "",
                "last_run_at": "",
                "steps": {
                    "good": {"status": "done"},
                    "bad-status": {"status": "not-a-real-status"},
                    "not-an-object": "string",
                },
            }
        )
    )
    state = read_state(project_dir)
    assert state.steps["good"].status == StepStatus.DONE
    # Bad status is reset to PENDING; non-object is dropped.
    assert state.steps["bad-status"].status == StepStatus.PENDING
    assert "not-an-object" not in state.steps


# ---------------------------------------------------------------------------
# compute_fingerprint
# ---------------------------------------------------------------------------


def test_fingerprint_deterministic() -> None:
    a = compute_fingerprint({"a": 1, "b": 2})
    b = compute_fingerprint({"a": 1, "b": 2})
    assert a == b


def test_fingerprint_key_order_does_not_matter() -> None:
    a = compute_fingerprint({"a": 1, "b": 2})
    b = compute_fingerprint({"b": 2, "a": 1})
    assert a == b


def test_fingerprint_different_inputs_differ() -> None:
    a = compute_fingerprint({"a": 1})
    b = compute_fingerprint({"a": 2})
    assert a != b


def test_fingerprint_accepts_paths_via_default_str() -> None:
    fp = compute_fingerprint({"path": Path("/tmp/foo")})
    assert fp.startswith("sha256:")


def test_fingerprint_rejects_non_serializable() -> None:
    class _NotSerializable:
        pass

    # Use default=str so this is actually serializable; but a set with
    # non-stringable cycles is not. Use an object() instance which str() works
    # on but produces inconsistent results — we just want to ensure the helper
    # tolerates default=str. So instead test a JSON-incompatible cycle:
    cycle: list[Any] = []
    cycle.append(cycle)
    with pytest.raises(ValueError):
        compute_fingerprint({"x": cycle})


# ---------------------------------------------------------------------------
# Topological sort
# ---------------------------------------------------------------------------


def test_topo_linear_order(project_dir: Path) -> None:
    steps = [
        DependentStep(id="c", depends_on=("b",)),
        DependentStep(id="b", depends_on=("a",)),
        DependentStep(id="a"),
    ]
    orch = Orchestrator(steps=steps, project_dir=project_dir, manifest=_manifest())
    assert orch._order == ["a", "b", "c"]


def test_topo_diamond_order(project_dir: Path) -> None:
    a = DependentStep(id="a")
    b = DependentStep(id="b", depends_on=("a",))
    c = DependentStep(id="c", depends_on=("a",))
    d = DependentStep(id="d", depends_on=("b", "c"))
    orch = Orchestrator(steps=[a, b, c, d], project_dir=project_dir, manifest=_manifest())
    order = orch._order
    # a first, d last; b/c in between.
    assert order[0] == "a"
    assert order[-1] == "d"
    assert set(order[1:3]) == {"b", "c"}


def test_topo_cycle_raises(project_dir: Path) -> None:
    a = DependentStep(id="a", depends_on=("b",))
    b = DependentStep(id="b", depends_on=("a",))
    with pytest.raises(CycleError) as exc:
        Orchestrator(steps=[a, b], project_dir=project_dir, manifest=_manifest())
    assert set(exc.value.cycle) == {"a", "b"}


def test_topo_missing_dep_raises(project_dir: Path) -> None:
    a = DependentStep(id="a", depends_on=("ghost",))
    with pytest.raises(MissingDependencyError) as exc:
        Orchestrator(steps=[a], project_dir=project_dir, manifest=_manifest())
    assert exc.value.missing == "ghost"


def test_duplicate_step_ids_raise(project_dir: Path) -> None:
    a1 = DependentStep(id="dup")
    a2 = DependentStep(id="dup")
    with pytest.raises(OrchestratorError):
        Orchestrator(steps=[a1, a2], project_dir=project_dir, manifest=_manifest())


# ---------------------------------------------------------------------------
# Plan + run — fresh state
# ---------------------------------------------------------------------------


def test_run_empty_steps(project_dir: Path) -> None:
    orch = Orchestrator(steps=[], project_dir=project_dir, manifest=_manifest())
    result = orch.run()
    assert result.statuses == {}
    assert result.exit_code == 0


def test_run_single_step_fresh(project_dir: Path) -> None:
    step = NoopStep()
    orch = Orchestrator(steps=[step], project_dir=project_dir, manifest=_manifest())
    result = orch.run()
    assert result.statuses == {"noop": StepStatus.DONE}
    assert step.apply_calls == 1
    state = read_state(project_dir)
    assert state.steps["noop"].status == StepStatus.DONE
    assert state.steps["noop"].fingerprint is not None
    assert result.exit_code == 0


def test_plan_returns_row_per_step(project_dir: Path) -> None:
    steps = [
        NoopStep(id="alpha"),
        AlreadyDoneStep(id="beta"),
    ]
    orch = Orchestrator(steps=steps, project_dir=project_dir, manifest=_manifest())
    rows = orch.plan()
    by_id = {r.step_id: r for r in rows}
    assert by_id["alpha"].action == "run"
    # AlreadyDoneStep advertises DONE in detect() → plan skips it.
    assert by_id["beta"].action == "skip"
    assert by_id["beta"].detected == StepStatus.DONE


# ---------------------------------------------------------------------------
# Flag semantics
# ---------------------------------------------------------------------------


def test_resume_skips_done_steps(project_dir: Path) -> None:
    step = AlreadyDoneStep()
    # Pre-seed state as DONE.
    write_state(
        project_dir,
        OrchestratorState(steps={"already-done": StepState(status=StepStatus.DONE)}),
    )
    orch = Orchestrator(steps=[step], project_dir=project_dir, manifest=_manifest())
    result = orch.run(resume=True)
    assert result.statuses == {"already-done": StepStatus.DONE}
    assert step.apply_calls == 0


def test_no_resume_re_detects_done(project_dir: Path) -> None:
    step = AlreadyDoneStep()
    write_state(
        project_dir,
        OrchestratorState(steps={"already-done": StepState(status=StepStatus.DONE)}),
    )
    orch = Orchestrator(steps=[step], project_dir=project_dir, manifest=_manifest())
    result = orch.run()  # no resume
    # detect() still says DONE → skip.
    assert step.apply_calls == 0
    assert result.statuses["already-done"] == StepStatus.DONE


def test_drift_detected_when_state_done_but_detect_pending(project_dir: Path) -> None:
    """If stored=DONE but detect() reports PENDING, the step must re-run."""
    step = NoopStep()
    write_state(project_dir, OrchestratorState(steps={"noop": StepState(status=StepStatus.DONE)}))
    orch = Orchestrator(steps=[step], project_dir=project_dir, manifest=_manifest())
    orch.run()
    assert step.apply_calls == 1


def _seed_done(project_dir: Path, step_id: str, fingerprint: str | None) -> None:
    write_state(
        project_dir,
        OrchestratorState(
            steps={step_id: StepState(status=StepStatus.DONE, fingerprint=fingerprint)}
        ),
    )


def test_done_step_reruns_on_fingerprint_drift(project_dir: Path) -> None:
    """Stored DONE + detect DONE, but the inputs changed → re-run.

    This is the regenerate-into-existing-destination case: containers from
    the previous generation are still running (detect says DONE) while the
    compose file on disk changed (fingerprint differs)."""
    step = DriftingStep(fingerprint_value="sha256:new")
    _seed_done(project_dir, "drifting", "sha256:old")
    orch = Orchestrator(steps=[step], project_dir=project_dir, manifest=_manifest())
    result = orch.run()
    assert step.apply_calls == 1
    assert result.statuses["drifting"] == StepStatus.DONE
    # The re-run stores the current fingerprint, so the next run skips again.
    assert read_state(project_dir).steps["drifting"].fingerprint == "sha256:new"


def test_done_step_skips_when_fingerprint_matches(project_dir: Path) -> None:
    step = DriftingStep(fingerprint_value="sha256:same")
    _seed_done(project_dir, "drifting", "sha256:same")
    orch = Orchestrator(steps=[step], project_dir=project_dir, manifest=_manifest())
    result = orch.run()
    assert step.apply_calls == 0
    assert result.statuses["drifting"] == StepStatus.DONE


def test_done_step_skips_when_stored_fingerprint_missing(project_dir: Path) -> None:
    """Pre-fingerprint state files keep the detect-only behavior."""
    step = DriftingStep(fingerprint_value="sha256:new")
    _seed_done(project_dir, "drifting", None)
    orch = Orchestrator(steps=[step], project_dir=project_dir, manifest=_manifest())
    orch.run()
    assert step.apply_calls == 0


def test_resume_ignores_fingerprint_drift(project_dir: Path) -> None:
    step = DriftingStep(fingerprint_value="sha256:new")
    _seed_done(project_dir, "drifting", "sha256:old")
    orch = Orchestrator(steps=[step], project_dir=project_dir, manifest=_manifest())
    orch.run(resume=True)
    assert step.apply_calls == 0


def test_fingerprint_exception_during_decide_falls_back_to_detect(project_dir: Path) -> None:
    """A failing fingerprint() never forces a re-run — detect DONE wins."""
    step = DriftingStep(raise_in_fingerprint=True)
    _seed_done(project_dir, "drifting", "sha256:old")
    orch = Orchestrator(steps=[step], project_dir=project_dir, manifest=_manifest())
    result = orch.run()
    assert step.apply_calls == 0
    assert result.statuses["drifting"] == StepStatus.DONE


def test_plan_shows_run_on_fingerprint_drift(project_dir: Path) -> None:
    """The plan table must match what run() will do — no silent rebuilds."""
    step = DriftingStep(fingerprint_value="sha256:new")
    _seed_done(project_dir, "drifting", "sha256:old")
    orch = Orchestrator(steps=[step], project_dir=project_dir, manifest=_manifest())
    rows = {r.step_id: r for r in orch.plan()}
    assert rows["drifting"].action == "run"
    assert "fingerprint drift" in rows["drifting"].reason
    # And with matching fingerprints the plan still skips.
    _seed_done(project_dir, "drifting", "sha256:new")
    rows = {r.step_id: r for r in orch.plan()}
    assert rows["drifting"].action == "skip"


def test_force_clears_state_and_runs(project_dir: Path) -> None:
    step = AlreadyDoneStep()
    write_state(
        project_dir,
        OrchestratorState(steps={"already-done": StepState(status=StepStatus.DONE)}),
    )
    orch = Orchestrator(steps=[step], project_dir=project_dir, manifest=_manifest())
    orch.run(force=["already-done"])
    assert step.apply_calls == 1


def test_skip_marks_step_skipped(project_dir: Path) -> None:
    step = NoopStep()
    orch = Orchestrator(steps=[step], project_dir=project_dir, manifest=_manifest())
    result = orch.run(skip=["noop"])
    assert result.statuses == {"noop": StepStatus.SKIPPED}
    assert step.apply_calls == 0


def test_only_restricts_to_target_plus_deps(project_dir: Path) -> None:
    a = DependentStep(id="a")
    b = DependentStep(id="b", depends_on=("a",))
    c = DependentStep(id="c")  # sibling not in dep chain
    orch = Orchestrator(steps=[a, b, c], project_dir=project_dir, manifest=_manifest())
    result = orch.run(only=["b"])
    # a + b should run, c should not even appear.
    assert set(result.statuses) == {"a", "b"}
    assert a.apply_calls == 1
    assert b.apply_calls == 1
    assert c.apply_calls == 0


def test_retry_reruns_failed_only(project_dir: Path) -> None:
    step = FlakyStep(fail_first=1)
    orch = Orchestrator(steps=[step], project_dir=project_dir, manifest=_manifest())
    first = orch.run()
    assert first.statuses["flaky"] == StepStatus.FAILED
    # --retry on a failed step re-runs it.
    second = orch.run(retry=["flaky"])
    assert second.statuses["flaky"] == StepStatus.DONE
    assert step.apply_calls == 2


def test_retry_does_not_rerun_done_step(project_dir: Path) -> None:
    step = NoopStep()
    orch = Orchestrator(steps=[step], project_dir=project_dir, manifest=_manifest())
    orch.run()  # → DONE
    assert step.apply_calls == 1
    orch.run(retry=["noop"])
    # --retry on a DONE step is a no-op.
    assert step.apply_calls == 1


def test_unknown_flag_target_raises(project_dir: Path) -> None:
    step = NoopStep()
    orch = Orchestrator(steps=[step], project_dir=project_dir, manifest=_manifest())
    with pytest.raises(OrchestratorError):
        orch.run(only=["does-not-exist"])


# ---------------------------------------------------------------------------
# Crash recovery
# ---------------------------------------------------------------------------


def test_apply_exception_marks_failed(project_dir: Path) -> None:
    step = FailingStep(raise_in_apply=True)
    orch = Orchestrator(steps=[step], project_dir=project_dir, manifest=_manifest())
    result = orch.run()
    assert result.statuses["failing"] == StepStatus.FAILED
    state = read_state(project_dir)
    assert state.steps["failing"].error is not None
    assert "RuntimeError" in state.steps["failing"].error


def test_running_state_is_recoverable(project_dir: Path) -> None:
    """A step left in RUNNING (process killed) re-runs on the next invocation."""
    write_state(
        project_dir, OrchestratorState(steps={"noop": StepState(status=StepStatus.RUNNING)})
    )
    step = NoopStep()
    orch = Orchestrator(steps=[step], project_dir=project_dir, manifest=_manifest())
    orch.run()
    assert step.apply_calls == 1


def test_failure_halts_downstream(project_dir: Path) -> None:
    a = FailingStep(id="a")
    b = NoopStep(id="b", depends_on=("a",))
    orch = Orchestrator(steps=[a, b], project_dir=project_dir, manifest=_manifest())
    result = orch.run()
    assert result.statuses["a"] == StepStatus.FAILED
    # b never runs after a fails.
    assert b.apply_calls == 0
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# Progress events
# ---------------------------------------------------------------------------


def test_callback_receives_started_and_finished(project_dir: Path) -> None:
    events: list[Any] = []
    step = NoopStep()
    orch = Orchestrator(
        steps=[step], project_dir=project_dir, manifest=_manifest(), callback=events.append
    )
    orch.run()
    kinds = [type(e).__name__ for e in events]
    assert kinds == ["StepStarted", "StepFinished"]
    assert isinstance(events[0], StepStarted)
    finished = events[1]
    assert isinstance(finished, StepFinished)
    assert finished.result is not None and finished.result.status == StepStatus.DONE


def test_callback_none_does_not_crash(project_dir: Path) -> None:
    orch = Orchestrator(
        steps=[NoopStep()], project_dir=project_dir, manifest=_manifest(), callback=None
    )
    assert orch.run().exit_code == 0


# ---------------------------------------------------------------------------
# Plan-table rendering
# ---------------------------------------------------------------------------


def test_render_plan_table_returns_panel() -> None:
    rows = [
        PlanRow(
            step_id="install_deps",
            description="install deps",
            detected=StepStatus.PENDING,
            action="run",
        ),
        PlanRow(
            step_id="docker_up",
            description="start services",
            detected=StepStatus.PARTIAL,
            action="run (resume)",
            reason="redis missing",
        ),
    ]
    panel = render_plan_table(rows)
    # Render to a string for substring assertions.
    import io

    from rich.console import Console

    buf = io.StringIO()
    Console(file=buf, force_terminal=True, width=120).print(panel)
    out = buf.getvalue()
    assert "Provisioning plan" in out
    assert "install_deps" in out
    assert "docker_up" in out
    assert "redis missing" in out


# ---------------------------------------------------------------------------
# CLI flag callback
# ---------------------------------------------------------------------------


def test_step_flags_callback_parses(runner: CliRunner = CliRunner()) -> None:
    import typer

    from agent_scaffold.cli import StepFlags, step_flags_callback

    captured: dict[str, StepFlags] = {}
    app = typer.Typer()

    @app.command()
    def harness(flags: StepFlags = typer.Option(default_factory=step_flags_callback)) -> None:  # type: ignore[assignment]
        captured["flags"] = flags

    # Typer's Option(default_factory=...) is awkward for callbacks that
    # themselves declare Typer Options. Easier: invoke the callback directly
    # by re-using the underlying signature with click.testing-friendly args.
    # For the smoke test we just call step_flags_callback positionally with
    # explicit values — the function is a thin dataclass constructor.
    flags = step_flags_callback(
        only=["a"],
        skip=["b"],
        force=["c"],
        retry=["d"],
        resume=True,
        plan_only=False,
        yes=True,
        debug=False,
    )
    assert flags.only == ["a"]
    assert flags.skip == ["b"]
    assert flags.force == ["c"]
    assert flags.retry == ["d"]
    assert flags.resume is True
    assert flags.yes is True


# ---------------------------------------------------------------------------
# Step context behaviour
# ---------------------------------------------------------------------------


def test_step_context_emit_dispatches() -> None:
    received: list[Any] = []
    ctx = StepContext(
        project_dir=Path("/tmp"),
        manifest=_manifest(),
        state=OrchestratorState(),
        callback=received.append,
    )
    ctx.emit(StepStarted(step_id="x"))
    assert len(received) == 1
    assert received[0].step_id == "x"


def test_step_context_emit_without_callback_is_noop() -> None:
    ctx = StepContext(
        project_dir=Path("/tmp"),
        manifest=_manifest(),
        state=OrchestratorState(),
    )
    # Should not raise.
    ctx.emit(StepStarted(step_id="x"))


# ---------------------------------------------------------------------------
# DetectionResult / StepResult are inert dataclasses — sanity defaults
# ---------------------------------------------------------------------------


def test_detection_result_defaults() -> None:
    r = DetectionResult(status=StepStatus.PENDING)
    assert r.reason == ""


def test_step_result_defaults() -> None:
    r = StepResult(status=StepStatus.DONE)
    assert r.detail == ""
    assert r.error is None
