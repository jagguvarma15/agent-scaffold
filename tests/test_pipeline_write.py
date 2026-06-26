"""Tests for the pipeline write phase: deadlock-free confirm, names-only
overwrite summary, and 3-way merge into an existing project.

The historic bug these lock down: the diff/overwrite confirmation used to run
*inside* the live progress display (Rich ``Live`` + a termios-muted terminal),
so the prompt never received a completed line — generation hung at "0 files
written". The fix moved the confirm into the pipeline, gated on
``display.suspend()``. These tests assert the confirm only ever runs while the
display is suspended, and that merge preserves user edits.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from rich.console import Console

from agent_scaffold import pipeline
from agent_scaffold.contract import GeneratedFile, GenerationResult
from agent_scaffold.manifest import Manifest, build_file_entries, write_manifest
from agent_scaffold.pipeline import PipelineError, _merge_into_existing, _write_phase
from agent_scaffold.progress import ProgressEvent
from agent_scaffold.template_snapshot import save_generation_snapshot
from agent_scaffold.writer import WriteMode


class _FakeDisplay:
    """Minimal display stand-in recording suspend state + emitted events."""

    def __init__(self, *, interactive: bool) -> None:
        self._interactive = interactive
        self.console = Console(file=io.StringIO(), force_terminal=False)
        self.suspended = False
        self.suspend_calls = 0
        self.events: list[ProgressEvent] = []

    @property
    def interactive(self) -> bool:
        return self._interactive

    @contextlib.contextmanager
    def suspend(self) -> Any:
        self.suspended = True
        self.suspend_calls += 1
        try:
            yield
        finally:
            self.suspended = False

    def on_event(self, event: ProgressEvent) -> None:
        self.events.append(event)


def _result(files: dict[str, str]) -> GenerationResult:
    return GenerationResult(
        project_name="demo",
        language="python",
        files=[GeneratedFile(path=p, content=c) for p, c in files.items()],
        smoke_check="echo ok",
    )


def _inputs(dest: Path, *, mode: WriteMode, deployments: Path | None = None) -> Any:
    # _write_phase / _merge_into_existing only read dest, deployments, write_mode.
    return SimpleNamespace(dest=dest, deployments=deployments or dest.parent, write_mode=mode)


def _make_project(
    project_dir: Path, files: dict[str, str], *, sha: str, snapshot: bool = True
) -> None:
    """Write files + a v2 manifest + (optionally) snapshot them as the base."""
    for rel, content in files.items():
        target = project_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    manifest = Manifest(
        recipe="demo",
        language="python",
        framework="none",
        model="claude-test",
        generated_at="2026-06-26T00:00:00+00:00",
        files=build_file_entries(project_dir, list(files)),
        template_snapshot_sha=sha,
        answers={"project_name": "demo"},
    )
    write_manifest(project_dir, manifest)
    if snapshot:
        save_generation_snapshot(project_dir, sha, files)


# ---------------------------------------------------------------------------
# Deadlock regression: the overwrite confirm runs only while suspended
# ---------------------------------------------------------------------------


def test_overwrite_confirm_runs_while_display_suspended(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = tmp_path / "proj"
    dest.mkdir()
    (dest / "README.md").write_text("OLD\n", encoding="utf-8")
    result = _result({"README.md": "NEW\n"})
    display = _FakeDisplay(interactive=True)

    seen: dict[str, bool] = {}

    def fake_confirm(change: Any, console: Any, dst: Path) -> bool:
        # The whole point of the fix: the prompt must run with the live
        # display suspended, never while it owns stdin.
        seen["suspended"] = display.suspended
        return False  # decline → cancel

    monkeypatch.setattr(pipeline, "confirm_change_summary", fake_confirm)

    with pytest.raises(PipelineError, match="cancelled"):
        _write_phase(result, _inputs(dest, mode=WriteMode.overwrite), display)

    assert seen["suspended"] is True, "confirm ran while the display was NOT suspended"
    assert display.suspend_calls == 1
    # Declined → the existing file is untouched.
    assert (dest / "README.md").read_text() == "OLD\n"


def test_overwrite_confirm_approved_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dest = tmp_path / "proj"
    dest.mkdir()
    (dest / "README.md").write_text("OLD\n", encoding="utf-8")
    result = _result({"README.md": "NEW\n", "new.py": "x = 1\n"})
    display = _FakeDisplay(interactive=True)
    monkeypatch.setattr(pipeline, "confirm_change_summary", lambda *a, **k: True)

    report = _write_phase(result, _inputs(dest, mode=WriteMode.overwrite), display)
    assert (dest / "README.md").read_text() == "NEW\n"
    assert "README.md" in report.overwritten
    assert "new.py" in report.written


def test_overwrite_non_interactive_skips_confirm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-interactive display (CI / pipes) must never prompt — it just writes."""
    dest = tmp_path / "proj"
    dest.mkdir()
    (dest / "README.md").write_text("OLD\n", encoding="utf-8")
    result = _result({"README.md": "NEW\n"})
    display = _FakeDisplay(interactive=False)

    def boom(*_a: Any, **_k: Any) -> bool:
        raise AssertionError("confirm must not run for a non-interactive display")

    monkeypatch.setattr(pipeline, "confirm_change_summary", boom)
    report = _write_phase(result, _inputs(dest, mode=WriteMode.overwrite), display)
    assert (dest / "README.md").read_text() == "NEW\n"
    assert "README.md" in report.overwritten
    assert display.suspend_calls == 0


# ---------------------------------------------------------------------------
# 3-way merge into an existing project
# ---------------------------------------------------------------------------


def test_merge_preserves_user_edits_on_nonoverlapping_change(tmp_path: Path) -> None:
    dest = tmp_path / "proj"
    dest.mkdir()
    # Base = the snapshot of the first generation.
    _make_project(dest, {"app/main.py": "a\nb\nc\n"}, sha="base-sha")
    # The user edited the last line on disk (ours).
    (dest / "app" / "main.py").write_text("a\nb\nUSER\n", encoding="utf-8")
    # The regeneration changed the FIRST line (theirs) — no overlap.
    result = _result({"app/main.py": "TEMPLATE\nb\nc\n"})

    report = _merge_into_existing(
        result, _inputs(dest, mode=WriteMode.merge), _FakeDisplay(interactive=False)
    )

    assert report is not None
    # Both edits survive — the 3-way merge auto-resolved them.
    assert (dest / "app" / "main.py").read_text() == "TEMPLATE\nb\nUSER\n"
    assert "app/main.py" in report.overwritten
    assert not (dest / ".scaffold" / "update.in-progress.json").exists()


def test_merge_writes_conflict_markers_and_resume_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = tmp_path / "proj"
    dest.mkdir()
    _make_project(dest, {"app/main.py": "a\nb\nc\n"}, sha="base-sha")
    (dest / "app" / "main.py").write_text("X\nb\nc\n", encoding="utf-8")  # ours: line1
    result = _result({"app/main.py": "Y\nb\nc\n"})  # theirs: line1 → conflict

    # compute_template_sha is imported lazily inside _merge_into_existing.
    monkeypatch.setattr(
        "agent_scaffold.template_snapshot.compute_template_sha", lambda _p: "new-sha"
    )

    with pytest.raises(PipelineError, match="conflict"):
        _merge_into_existing(
            result, _inputs(dest, mode=WriteMode.merge), _FakeDisplay(interactive=False)
        )

    merged = (dest / "app" / "main.py").read_text()
    assert "<<<<<<<" in merged and ">>>>>>>" in merged
    in_progress = dest / ".scaffold" / "update.in-progress.json"
    assert in_progress.is_file()
    assert "app/main.py" in in_progress.read_text()


def test_merge_without_snapshot_returns_none(tmp_path: Path) -> None:
    """No manifest/snapshot to merge against → caller falls back to overwrite."""
    dest = tmp_path / "proj"
    dest.mkdir()
    (dest / "README.md").write_text("hi\n", encoding="utf-8")
    result = _result({"README.md": "bye\n"})

    assert (
        _merge_into_existing(
            result, _inputs(dest, mode=WriteMode.merge), _FakeDisplay(interactive=False)
        )
        is None
    )


def test_write_phase_merge_falls_back_to_overwrite_without_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = tmp_path / "proj"
    dest.mkdir()
    (dest / "README.md").write_text("OLD\n", encoding="utf-8")
    result = _result({"README.md": "NEW\n"})
    # Non-interactive so the fallback overwrite proceeds without a prompt.
    report = _write_phase(
        result, _inputs(dest, mode=WriteMode.merge), _FakeDisplay(interactive=False)
    )
    assert (dest / "README.md").read_text() == "NEW\n"
    assert "README.md" in report.overwritten
