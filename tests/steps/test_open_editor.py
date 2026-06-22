"""Tests for ``agent_scaffold.steps.open_editor``."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from agent_scaffold.orchestrator import StepContext, StepStatus
from agent_scaffold.steps import open_editor as oe_mod
from agent_scaffold.steps.open_editor import OpenEditorStep


def _seed_readme(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# project\n", encoding="utf-8")


def _recording_popen(record: list[dict[str, Any]]) -> Callable[..., object]:
    """A fake ``subprocess.Popen`` that records the call and never blocks."""

    def _popen(cmd: list[str], **kwargs: Any) -> object:
        record.append({"cmd": cmd, "kwargs": kwargs})
        return object()

    return _popen


def test_detect_skipped_in_yes_mode(
    tmp_path: Path, ctx_factory: Callable[..., StepContext]
) -> None:
    _seed_readme(tmp_path)
    result = OpenEditorStep(yes=True).detect(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.SKIPPED
    assert "--yes" in result.reason


def test_detect_skipped_when_no_editor_resolvable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
) -> None:
    _seed_readme(tmp_path)
    monkeypatch.delenv("EDITOR", raising=False)
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.setattr(oe_mod.shutil, "which", lambda _name: None)
    result = OpenEditorStep().detect(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.SKIPPED


def test_detect_skipped_when_no_readme(
    tmp_path: Path, ctx_factory: Callable[..., StepContext]
) -> None:
    # GUI editor so we exercise the no-README branch, not the terminal-editor one.
    result = OpenEditorStep(editor_override="code").detect(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.SKIPPED
    assert "README" in result.reason


def test_detect_skips_terminal_editor(
    tmp_path: Path, ctx_factory: Callable[..., StepContext]
) -> None:
    _seed_readme(tmp_path)
    result = OpenEditorStep(editor_override="vim").detect(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.SKIPPED
    assert "terminal editor" in result.reason


def test_apply_launches_gui_editor_detached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
) -> None:
    _seed_readme(tmp_path)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(oe_mod.subprocess, "Popen", _recording_popen(calls))
    result = OpenEditorStep(editor_override="code").apply(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.DONE
    assert len(calls) == 1
    assert calls[0]["cmd"] == ["code", str(tmp_path / "README.md")]
    # Detached + non-blocking: new session, stdio to /dev/null, never waited on.
    assert calls[0]["kwargs"]["start_new_session"] is True
    assert calls[0]["kwargs"]["stdin"] == oe_mod.subprocess.DEVNULL


def test_apply_handles_editor_with_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
) -> None:
    """``EDITOR='code -n'`` should split into ``['code', '-n', readme]``."""
    _seed_readme(tmp_path)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(oe_mod.subprocess, "Popen", _recording_popen(calls))
    OpenEditorStep(editor_override="code -n").apply(ctx_factory(project_dir=tmp_path))
    assert calls[0]["cmd"] == ["code", "-n", str(tmp_path / "README.md")]


@pytest.mark.parametrize("editor", ["vim", "nano", "/usr/local/bin/nvim", "emacs"])
def test_apply_skips_terminal_editor_without_launching(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
    editor: str,
) -> None:
    """A terminal editor must be skipped — never spawned, so it can never block."""
    _seed_readme(tmp_path)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(oe_mod.subprocess, "Popen", _recording_popen(calls))
    result = OpenEditorStep(editor_override=editor).apply(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.SKIPPED
    assert calls == []  # the crux: nothing was launched


def test_apply_skipped_in_yes_mode(tmp_path: Path, ctx_factory: Callable[..., StepContext]) -> None:
    _seed_readme(tmp_path)
    result = OpenEditorStep(yes=True).apply(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.SKIPPED


def test_apply_launch_failure_is_non_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
) -> None:
    _seed_readme(tmp_path)

    def _boom(*_a: Any, **_k: Any) -> object:
        raise OSError("launch failed")

    monkeypatch.setattr(oe_mod.subprocess, "Popen", _boom)
    result = OpenEditorStep(editor_override="code").apply(ctx_factory(project_dir=tmp_path))
    # A cosmetic step must never FAIL the run, even when the editor won't launch.
    assert result.status is StepStatus.SKIPPED


def test_resolve_editor_falls_back_to_gui_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EDITOR", raising=False)
    monkeypatch.delenv("VISUAL", raising=False)
    seen: list[str] = []

    def fake_which(name: str) -> str | None:
        seen.append(name)
        return "/usr/bin/cursor" if name == "cursor" else None

    monkeypatch.setattr(oe_mod.shutil, "which", fake_which)
    assert OpenEditorStep()._resolve_editor() == "cursor"
    # GUI fallbacks, probed in order; terminal editors are never in the list.
    assert seen[:2] == ["code", "cursor"]
    assert "vim" not in seen
    assert "nano" not in seen
