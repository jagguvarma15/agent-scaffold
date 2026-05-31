"""Tests for the ``agent-scaffold eval`` CLI verb."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import agent_scaffold.cli as cli_mod
from agent_scaffold.cli import app
from agent_scaffold.eval._common import EvalCase, EvalResult
from agent_scaffold.manifest import Manifest, read_manifest, write_manifest


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _project_with_manifest(
    tmp_path: Path,
    *,
    capabilities: list[str] | None = None,
    baseline: str | None = None,
) -> Path:
    answers = {"eval_baseline": baseline} if baseline is not None else {}
    manifest = Manifest(
        recipe="restaurant-rebooking",
        language="python",
        framework="langgraph",
        model="claude-test",
        generated_at="2026-05-30T00:00:00+00:00",
        answers=answers,
        capabilities=capabilities or [],
    )
    write_manifest(tmp_path, manifest)
    return tmp_path


def _stub_plugin(monkeypatch: pytest.MonkeyPatch, *, result: EvalResult) -> dict[str, Any]:
    calls: dict[str, Any] = {"runs": 0, "baseline": None, "project_dir": None}

    class _Stub:
        name = "promptfoo"

        def run(self, project_dir: Path, baseline_total: float | None) -> EvalResult:
            calls["runs"] += 1
            calls["baseline"] = baseline_total
            calls["project_dir"] = project_dir
            return result

    monkeypatch.setattr("agent_scaffold.eval.get_plugin", lambda _t: _Stub())
    return calls


# ---------------------------------------------------------------------------
# No eval capability → exit 0 with friendly message
# ---------------------------------------------------------------------------


def test_eval_exits_zero_when_no_eval_capability(runner: CliRunner, tmp_path: Path) -> None:
    _project_with_manifest(tmp_path, capabilities=["obs.langfuse"])
    result = runner.invoke(app, ["eval", "--cwd", str(tmp_path)])
    assert result.exit_code == 0
    assert "No eval capability" in result.output


def test_eval_exits_one_when_no_manifest(runner: CliRunner, tmp_path: Path) -> None:
    """No manifest → exit 1; the user is in the wrong directory."""
    result = runner.invoke(app, ["eval", "--cwd", str(tmp_path)])
    assert result.exit_code == 1
    assert "manifest" in result.output.lower()


# ---------------------------------------------------------------------------
# Happy path — baseline read, plugin called, result rendered
# ---------------------------------------------------------------------------


def test_eval_happy_path_reads_baseline_and_calls_plugin(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _project_with_manifest(tmp_path, capabilities=["eval.promptfoo"], baseline="0.90")
    calls = _stub_plugin(
        monkeypatch,
        result=EvalResult(
            target="promptfoo",
            cases=[EvalCase(name="x", score=0.91, passed=True)],
            total=0.91,
            baseline_total=0.90,
            delta=0.01,
        ),
    )

    result = runner.invoke(app, ["eval", "--cwd", str(tmp_path)])
    assert result.exit_code == 0
    assert calls["runs"] == 1
    assert calls["baseline"] == pytest.approx(0.90)
    assert "0.91" in result.output
    assert "Total" in result.output or "total" in result.output.lower()


# ---------------------------------------------------------------------------
# Regression detection
# ---------------------------------------------------------------------------


def test_eval_exits_one_on_regression(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _project_with_manifest(tmp_path, capabilities=["eval.promptfoo"], baseline="0.90")
    _stub_plugin(
        monkeypatch,
        result=EvalResult(
            target="promptfoo",
            cases=[EvalCase(name="x", score=0.85, passed=False)],
            total=0.85,
            baseline_total=0.90,
            delta=-0.05,
        ),
    )

    result = runner.invoke(app, ["eval", "--cwd", str(tmp_path)])
    assert result.exit_code == 1
    assert "regression" in result.output.lower()


def test_eval_within_noise_floor_exits_zero(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-case score deltas of ±1% shouldn't fail; that's sampling noise."""
    _project_with_manifest(tmp_path, capabilities=["eval.promptfoo"], baseline="0.90")
    _stub_plugin(
        monkeypatch,
        result=EvalResult(
            target="promptfoo",
            cases=[EvalCase(name="x", score=0.895, passed=True)],
            total=0.895,
            baseline_total=0.90,
            delta=-0.005,
        ),
    )
    result = runner.invoke(app, ["eval", "--cwd", str(tmp_path)])
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# --update-baseline
# ---------------------------------------------------------------------------


def test_eval_update_baseline_persists_total_and_exits_zero(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _project_with_manifest(tmp_path, capabilities=["eval.promptfoo"], baseline="0.90")
    _stub_plugin(
        monkeypatch,
        result=EvalResult(
            target="promptfoo",
            cases=[EvalCase(name="x", score=0.85, passed=False)],
            total=0.85,
            baseline_total=0.90,
            delta=-0.05,
        ),
    )

    # Without --update-baseline this would exit 1 (regression). With it: 0 + persist.
    result = runner.invoke(app, ["eval", "--cwd", str(tmp_path), "--update-baseline"])
    assert result.exit_code == 0
    persisted = read_manifest(tmp_path)
    assert persisted.answers["eval_baseline"] == "0.8500"


def test_eval_update_baseline_skipped_when_plugin_skipped(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the plugin couldn't run, don't blow away an existing baseline."""
    _project_with_manifest(tmp_path, capabilities=["eval.promptfoo"], baseline="0.90")
    _stub_plugin(
        monkeypatch,
        result=EvalResult(target="promptfoo", skipped=True, skip_reason="npx missing"),
    )
    result = runner.invoke(app, ["eval", "--cwd", str(tmp_path), "--update-baseline"])
    # Skipped result is exit 0, but the baseline must not change.
    assert result.exit_code == 0
    persisted = read_manifest(tmp_path)
    assert persisted.answers["eval_baseline"] == "0.90"


# ---------------------------------------------------------------------------
# --json
# ---------------------------------------------------------------------------


def test_eval_json_emits_parseable_payload(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _project_with_manifest(tmp_path, capabilities=["eval.promptfoo"], baseline="0.90")
    _stub_plugin(
        monkeypatch,
        result=EvalResult(
            target="promptfoo",
            cases=[
                EvalCase(name="a", score=0.9, passed=True),
                EvalCase(name="b", score=0.8, passed=True),
            ],
            total=0.85,
            baseline_total=0.90,
            delta=-0.05,
            cmd_run=["npx", "promptfoo", "eval"],
        ),
    )

    result = runner.invoke(app, ["eval", "--cwd", str(tmp_path), "--json"])
    assert result.exit_code == 1  # regression
    payload = json.loads(result.output)
    assert payload["target"] == "promptfoo"
    assert payload["total"] == pytest.approx(0.85)
    assert payload["baseline_total"] == pytest.approx(0.90)
    assert payload["delta"] == pytest.approx(-0.05)
    assert payload["is_regression"] is True
    assert payload["cmd_run"] == ["npx", "promptfoo", "eval"]
    assert len(payload["cases"]) == 2
    assert payload["cases"][0]["name"] == "a"


def test_eval_json_no_baseline_emits_null(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _project_with_manifest(tmp_path, capabilities=["eval.promptfoo"])  # no baseline
    _stub_plugin(
        monkeypatch,
        result=EvalResult(
            target="promptfoo",
            cases=[EvalCase(name="x", score=1.0, passed=True)],
            total=1.0,
        ),
    )

    result = runner.invoke(app, ["eval", "--cwd", str(tmp_path), "--json"])
    payload = json.loads(result.output)
    assert payload["baseline_total"] is None
    assert payload["delta"] is None
    assert payload["is_regression"] is False
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Unknown target
# ---------------------------------------------------------------------------


def test_eval_unknown_target_exits_one(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _project_with_manifest(tmp_path, capabilities=["eval.promptfoo"])

    def boom(_target: str) -> Any:
        raise KeyError(_target)

    monkeypatch.setattr("agent_scaffold.eval.get_plugin", boom)
    result = runner.invoke(app, ["eval", "--cwd", str(tmp_path), "--target", "deepeval"])
    assert result.exit_code == 1
    assert "Unknown eval target" in result.output or "unknown" in result.output.lower()


# ---------------------------------------------------------------------------
# _read_eval_baseline helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0.94", 0.94),
        ("0", 0.0),
        ("", None),
        ("not-a-float", None),
    ],
)
def test_read_eval_baseline_parses_or_returns_none(
    raw: str, expected: float | None, tmp_path: Path
) -> None:
    manifest = Manifest(
        recipe="r",
        language="python",
        framework="f",
        model="m",
        generated_at="2026-05-30T00:00:00+00:00",
        answers={"eval_baseline": raw},
    )
    assert cli_mod._read_eval_baseline(manifest) == expected


# ---------------------------------------------------------------------------
# Help text mentions the new flags
# ---------------------------------------------------------------------------


def test_eval_command_registers_new_flags(runner: CliRunner) -> None:
    """The new flags are registered as Click params on the eval subcommand.

    Asserts against Click's introspected param list instead of ``--help`` text
    so CI's narrow terminal can't truncate option names and break the test.
    """
    import typer.main

    click_app = typer.main.get_command(app)
    eval_cmd = click_app.commands["eval"]  # type: ignore[attr-defined]
    param_names = {p.name for p in eval_cmd.params}
    assert "target" in param_names
    assert "update_baseline" in param_names
    assert "json_output" in param_names
    # And the verb is invocable:
    result = runner.invoke(app, ["eval", "--help"])
    assert result.exit_code == 0
