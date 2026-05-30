"""Tests for the post-gen formatter's frontend/ extension."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from agent_scaffold import pipeline


def test_python_project_with_frontend_runs_prettier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Python primary + frontend/ present + prettier on PATH → prettier called on frontend/."""
    (tmp_path / "frontend").mkdir()
    (tmp_path / "frontend" / "app.tsx").write_text("export default function() {}\n")

    def fake_which(name: str) -> str | None:
        return "/usr/bin/prettier" if name == "prettier" else "/usr/bin/ruff"

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], on_event: object) -> int:
        calls.append(cmd)
        return 0

    monkeypatch.setattr(shutil, "which", fake_which)
    monkeypatch.setattr(pipeline.shutil, "which", fake_which)
    monkeypatch.setattr(pipeline, "_run_subprocess_with_events", fake_run)

    pipeline.run_post_gen_formatter(tmp_path, "python")
    # Two ruff invocations + one prettier on frontend/
    assert any("prettier" in cmd[0] and str(tmp_path / "frontend") in cmd for cmd in calls)


def test_python_project_without_frontend_skips_prettier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No frontend/ → no prettier invocation."""

    def fake_which(name: str) -> str | None:
        return "/usr/bin/prettier" if name == "prettier" else "/usr/bin/ruff"

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], on_event: object) -> int:
        calls.append(cmd)
        return 0

    monkeypatch.setattr(pipeline.shutil, "which", fake_which)
    monkeypatch.setattr(pipeline, "_run_subprocess_with_events", fake_run)

    pipeline.run_post_gen_formatter(tmp_path, "python")
    assert not any(cmd and cmd[0] == "prettier" for cmd in calls)


def test_typescript_project_only_runs_prettier_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TypeScript primary already runs prettier on dest — frontend/ doesn't double up."""
    (tmp_path / "frontend").mkdir()

    def fake_which(name: str) -> str | None:
        return "/usr/bin/prettier" if name == "prettier" else None

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], on_event: object) -> int:
        calls.append(cmd)
        return 0

    monkeypatch.setattr(pipeline.shutil, "which", fake_which)
    monkeypatch.setattr(pipeline, "_run_subprocess_with_events", fake_run)

    pipeline.run_post_gen_formatter(tmp_path, "typescript")
    prettier_calls = [cmd for cmd in calls if cmd and cmd[0] == "prettier"]
    assert len(prettier_calls) == 1
    assert str(tmp_path) in prettier_calls[0]
    # NOT also called on frontend/ specifically — the single dest-level call covers it.
    assert str(tmp_path / "frontend") not in prettier_calls[0]


def test_prettier_missing_skips_silently(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """frontend/ present + prettier missing → no exception, no calls."""
    (tmp_path / "frontend").mkdir()

    monkeypatch.setattr(
        pipeline.shutil, "which", lambda name: "/usr/bin/ruff" if name == "ruff" else None
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(
        pipeline, "_run_subprocess_with_events", lambda cmd, on_event: calls.append(cmd) or 0
    )

    pipeline.run_post_gen_formatter(tmp_path, "python")
    assert not any(cmd and cmd[0] == "prettier" for cmd in calls)
