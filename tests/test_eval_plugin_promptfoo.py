"""Tests for ``agent_scaffold.eval.promptfoo``."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

import agent_scaffold.eval.promptfoo as pf_mod
from agent_scaffold.eval import get_plugin
from agent_scaffold.eval._common import EvalResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _seed_project(tmp_path: Path, with_config: bool = True) -> Path:
    """Create a project dir with an ``evals/promptfooconfig.yaml`` stub."""
    if with_config:
        (tmp_path / "evals").mkdir()
        (tmp_path / "evals" / "promptfooconfig.yaml").write_text(
            "providers: [anthropic]\ntests: []\n", encoding="utf-8"
        )
    return tmp_path


def _write_output_json(project_dir: Path, payload: dict[str, Any]) -> None:
    out = project_dir / "evals" / "last-run.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload), encoding="utf-8")


def _fake_completed(returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout="", stderr=stderr)


# ---------------------------------------------------------------------------
# Plugin registry exposes promptfoo
# ---------------------------------------------------------------------------


def test_get_plugin_returns_promptfoo_module() -> None:
    plugin = get_plugin("promptfoo")
    assert plugin is pf_mod


def test_get_plugin_unknown_target_raises() -> None:
    with pytest.raises(KeyError):
        get_plugin("does-not-exist")


# ---------------------------------------------------------------------------
# cli_present + missing-config short-circuits
# ---------------------------------------------------------------------------


def test_run_skipped_when_npx_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_project(tmp_path)
    monkeypatch.setattr(pf_mod, "cli_present", lambda _b: False)

    result = pf_mod.run(tmp_path, baseline_total=None)
    assert result.skipped is True
    assert "PATH" in result.skip_reason or "install" in result.skip_reason.lower()
    assert result.cmd_run[0] == "npx"


def test_run_skipped_when_config_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # No evals/ tree.
    monkeypatch.setattr(pf_mod, "cli_present", lambda _b: True)
    result = pf_mod.run(tmp_path, baseline_total=None)
    assert result.skipped is True
    assert "promptfooconfig" in result.skip_reason


# ---------------------------------------------------------------------------
# Subprocess: parses valid JSON, never actually runs npx
# ---------------------------------------------------------------------------


def test_run_parses_canonical_promptfoo_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _seed_project(tmp_path)
    _write_output_json(
        project,
        {
            "results": {
                "results": [
                    {"description": "greets", "score": 1.0, "success": True},
                    {"description": "refuses", "score": 0.95, "success": True},
                    {"description": "tool", "score": 0.6, "success": False},
                ]
            }
        },
    )

    monkeypatch.setattr(pf_mod, "cli_present", lambda _b: True)
    captured: dict[str, Any] = {}

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        captured["cwd"] = kwargs.get("cwd")
        return _fake_completed()

    monkeypatch.setattr(pf_mod.subprocess, "run", fake_run)

    result = pf_mod.run(project, baseline_total=None)
    assert result.skipped is False
    assert result.error is None
    assert [c.name for c in result.cases] == ["greets", "refuses", "tool"]
    assert pytest.approx(result.total, abs=1e-6) == (1.0 + 0.95 + 0.6) / 3
    # Subprocess invoked with the canonical promptfoo command and the project cwd.
    assert captured["cmd"][:3] == ["npx", "promptfoo", "eval"]
    assert captured["cwd"] == str(project)


def test_run_parses_flatter_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Older Promptfoo JSON with ``results`` at the top level."""
    project = _seed_project(tmp_path)
    _write_output_json(
        project,
        {
            "results": [
                {"description": "a", "score": 0.8, "success": True},
                {"description": "b", "score": 0.4, "success": False},
            ]
        },
    )
    monkeypatch.setattr(pf_mod, "cli_present", lambda _b: True)
    monkeypatch.setattr(pf_mod.subprocess, "run", lambda *_a, **_kw: _fake_completed())

    result = pf_mod.run(project, baseline_total=None)
    assert len(result.cases) == 2
    assert pytest.approx(result.total, abs=1e-6) == 0.6


def test_run_clamps_runaway_scores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM-judged scores can exceed 1.0; the plugin clamps to [0, 1]."""
    project = _seed_project(tmp_path)
    _write_output_json(
        project,
        {"results": {"results": [{"description": "x", "score": 1.7, "success": True}]}},
    )
    monkeypatch.setattr(pf_mod, "cli_present", lambda _b: True)
    monkeypatch.setattr(pf_mod.subprocess, "run", lambda *_a, **_kw: _fake_completed())

    result = pf_mod.run(project, baseline_total=None)
    assert result.cases[0].score == 1.0


def test_run_derives_score_from_success_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _seed_project(tmp_path)
    _write_output_json(
        project,
        {
            "results": {
                "results": [
                    {"description": "a", "success": True},  # no score field
                    {"description": "b", "success": False},
                ]
            }
        },
    )
    monkeypatch.setattr(pf_mod, "cli_present", lambda _b: True)
    monkeypatch.setattr(pf_mod.subprocess, "run", lambda *_a, **_kw: _fake_completed())

    result = pf_mod.run(project, baseline_total=None)
    assert [c.score for c in result.cases] == [1.0, 0.0]


# ---------------------------------------------------------------------------
# Baseline + delta
# ---------------------------------------------------------------------------


def test_run_computes_delta_when_baseline_provided(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _seed_project(tmp_path)
    _write_output_json(
        project,
        {"results": {"results": [{"description": "x", "score": 0.8, "success": True}]}},
    )
    monkeypatch.setattr(pf_mod, "cli_present", lambda _b: True)
    monkeypatch.setattr(pf_mod.subprocess, "run", lambda *_a, **_kw: _fake_completed())

    result = pf_mod.run(project, baseline_total=0.9)
    assert result.baseline_total == 0.9
    assert pytest.approx(result.delta, abs=1e-6) == -0.1
    assert result.is_regression is True  # delta < -0.01


def test_run_no_baseline_yields_none_delta(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _seed_project(tmp_path)
    _write_output_json(
        project,
        {"results": {"results": [{"description": "x", "score": 0.9, "success": True}]}},
    )
    monkeypatch.setattr(pf_mod, "cli_present", lambda _b: True)
    monkeypatch.setattr(pf_mod.subprocess, "run", lambda *_a, **_kw: _fake_completed())

    result = pf_mod.run(project, baseline_total=None)
    assert result.delta is None
    assert result.is_regression is False


def test_run_small_delta_within_noise_floor_not_regression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _seed_project(tmp_path)
    _write_output_json(
        project,
        {"results": {"results": [{"description": "x", "score": 0.895, "success": True}]}},
    )
    monkeypatch.setattr(pf_mod, "cli_present", lambda _b: True)
    monkeypatch.setattr(pf_mod.subprocess, "run", lambda *_a, **_kw: _fake_completed())

    result = pf_mod.run(project, baseline_total=0.9)
    assert result.is_regression is False  # delta ~ -0.005, within ±0.01 floor


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_run_missing_output_file_returns_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _seed_project(tmp_path)  # config present but we never write last-run.json
    monkeypatch.setattr(pf_mod, "cli_present", lambda _b: True)
    monkeypatch.setattr(
        pf_mod.subprocess, "run", lambda *_a, **_kw: _fake_completed(returncode=2, stderr="boom")
    )

    result = pf_mod.run(project, baseline_total=None)
    assert result.skipped is False
    assert result.error is not None
    assert "last-run.json" in result.error


def test_run_invalid_json_in_output_returns_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _seed_project(tmp_path)
    (project / "evals" / "last-run.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(pf_mod, "cli_present", lambda _b: True)
    monkeypatch.setattr(pf_mod.subprocess, "run", lambda *_a, **_kw: _fake_completed())

    result = pf_mod.run(project, baseline_total=None)
    assert result.error is not None
    assert "JSON" in result.error or "valid JSON" in result.error


def test_run_subprocess_timeout_returns_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _seed_project(tmp_path)
    monkeypatch.setattr(pf_mod, "cli_present", lambda _b: True)

    def boom(*_a: Any, **_kw: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=["npx"], timeout=1.0)

    monkeypatch.setattr(pf_mod.subprocess, "run", boom)
    result = pf_mod.run(project, baseline_total=None)
    assert result.error is not None
    assert "timed out" in result.error


def test_run_subprocess_oserror_treated_as_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _seed_project(tmp_path)
    monkeypatch.setattr(pf_mod, "cli_present", lambda _b: True)

    def boom(*_a: Any, **_kw: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("npx is gone after which() said it was here")

    monkeypatch.setattr(pf_mod.subprocess, "run", boom)
    result = pf_mod.run(project, baseline_total=None)
    assert result.skipped is True


# ---------------------------------------------------------------------------
# EvalResult shape sanity
# ---------------------------------------------------------------------------


def test_eval_result_is_frozen() -> None:
    r = EvalResult(target="t")
    with pytest.raises(Exception):  # noqa: B017,PT011 — frozen dataclass
        r.target = "u"  # type: ignore[misc]
