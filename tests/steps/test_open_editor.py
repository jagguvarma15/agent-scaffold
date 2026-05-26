"""Tests for ``agent_scaffold.steps.open_editor``."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from agent_scaffold.orchestrator import StepContext, StepStatus
from agent_scaffold.steps import open_editor as oe_mod
from agent_scaffold.steps.open_editor import OpenEditorStep


def _seed_readme(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# project\n", encoding="utf-8")


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
    result = OpenEditorStep(editor_override="vim").detect(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.SKIPPED


def test_apply_invokes_editor_with_readme_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
) -> None:
    _seed_readme(tmp_path)
    invoked: list[list[str]] = []

    def fake_run(
        cmd: list[str], *, check: bool = False, shell: bool = False
    ) -> subprocess.CompletedProcess[str]:
        del check, shell
        invoked.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(oe_mod.subprocess, "run", fake_run)
    step = OpenEditorStep(editor_override="vim")
    result = step.apply(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.DONE
    assert invoked == [["vim", str(tmp_path / "README.md")]]


def test_apply_handles_editor_with_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
) -> None:
    """``EDITOR='code -n'`` should split into ``['code', '-n', readme]``."""
    _seed_readme(tmp_path)
    invoked: list[list[str]] = []

    def fake_run(
        cmd: list[str], *, check: bool = False, shell: bool = False
    ) -> subprocess.CompletedProcess[str]:
        del check, shell
        invoked.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(oe_mod.subprocess, "run", fake_run)
    step = OpenEditorStep(editor_override="code -n")
    step.apply(ctx_factory(project_dir=tmp_path))
    assert invoked == [["code", "-n", str(tmp_path / "README.md")]]


def test_apply_skipped_in_yes_mode(tmp_path: Path, ctx_factory: Callable[..., StepContext]) -> None:
    _seed_readme(tmp_path)
    result = OpenEditorStep(yes=True).apply(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.SKIPPED


def test_resolve_editor_falls_back_to_path_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EDITOR", raising=False)
    monkeypatch.delenv("VISUAL", raising=False)
    seen: list[str] = []

    def fake_which(name: str) -> str | None:
        seen.append(name)
        return "/usr/bin/nano" if name == "nano" else None

    monkeypatch.setattr(oe_mod.shutil, "which", fake_which)
    assert OpenEditorStep()._resolve_editor() == "nano"
    # We probed the fallback list in order; nano won.
    assert seen[:3] == ["code", "cursor", "nano"]
