"""Tests for agent_scaffold.writer."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from agent_scaffold.contract import GeneratedFile, GenerationResult
from agent_scaffold.progress import ProgressEvent
from agent_scaffold.writer import (
    DestinationExistsError,
    DiffPreviewCancelled,
    FileDiff,
    WriteMode,
    preview_diffs,
    write_project,
)


def _result(files: list[tuple[str, str]]) -> GenerationResult:
    return GenerationResult(
        project_name="demo",
        language="python",
        files=[GeneratedFile(path=p, content=c) for p, c in files],
        smoke_check="echo ok",
    )


def test_writes_to_empty_destination(tmp_path: Path) -> None:
    dest = tmp_path / "demo"
    result = _result(
        [
            ("README.md", "# demo\n"),
            ("src/demo/main.py", "x = 1\n"),
            (".env.example", "K=\n"),
        ]
    )
    report = write_project(result, dest, WriteMode.overwrite)
    assert (dest / "README.md").read_text() == "# demo\n"
    assert (dest / "src/demo/main.py").read_text() == "x = 1\n"
    assert sorted(report.written) == sorted(["README.md", "src/demo/main.py", ".env.example"])
    assert report.overwritten == []


def test_abort_on_non_empty_destination(tmp_path: Path) -> None:
    dest = tmp_path / "demo"
    dest.mkdir()
    (dest / "existing.txt").write_text("hi")
    result = _result([("README.md", "# demo\n")])
    with pytest.raises(DestinationExistsError):
        write_project(result, dest, WriteMode.abort)
    # Existing file untouched.
    assert (dest / "existing.txt").read_text() == "hi"
    assert not (dest / "README.md").exists()


def test_skip_preserves_existing(tmp_path: Path) -> None:
    dest = tmp_path / "demo"
    dest.mkdir()
    (dest / "README.md").write_text("OLD\n")
    result = _result(
        [
            ("README.md", "NEW\n"),
            ("src/demo/main.py", "x = 1\n"),
        ]
    )
    report = write_project(result, dest, WriteMode.skip)
    assert (dest / "README.md").read_text() == "OLD\n"
    assert (dest / "src/demo/main.py").read_text() == "x = 1\n"
    assert "src/demo/main.py" in report.written
    assert "README.md" in report.skipped


def test_overwrite_replaces_existing(tmp_path: Path) -> None:
    dest = tmp_path / "demo"
    dest.mkdir()
    (dest / "README.md").write_text("OLD\n")
    result = _result([("README.md", "NEW\n")])
    report = write_project(result, dest, WriteMode.overwrite)
    assert (dest / "README.md").read_text() == "NEW\n"
    assert "README.md" in report.overwritten


def test_atomic_on_mid_write_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dest = tmp_path / "demo"
    dest.mkdir()
    (dest / "preexisting.txt").write_text("safe\n")

    result = _result(
        [
            ("a.txt", "1\n"),
            ("b.txt", "BOOM"),
            ("c.txt", "3\n"),
        ]
    )

    real_write_text = Path.write_text

    def flaky_write_text(self: Path, data: str, *args, **kwargs) -> int:  # type: ignore[no-untyped-def]
        if data == "BOOM":
            raise OSError("simulated mid-write failure")
        return real_write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", flaky_write_text)

    with pytest.raises(OSError, match="simulated"):
        write_project(result, dest, WriteMode.overwrite)

    # Destination contents should be unchanged: preexisting still there,
    # none of the new files written.
    assert (dest / "preexisting.txt").read_text() == "safe\n"
    assert not (dest / "a.txt").exists()
    assert not (dest / "b.txt").exists()
    assert not (dest / "c.txt").exists()


def test_executable_bit_on_sh_files(tmp_path: Path) -> None:
    dest = tmp_path / "demo"
    result = _result(
        [
            ("scripts/smoke.sh", "#!/usr/bin/env bash\necho ok\n"),
            ("README.md", "# demo\n"),
        ]
    )
    write_project(result, dest, WriteMode.overwrite)
    sh_mode = (dest / "scripts/smoke.sh").stat().st_mode
    md_mode = (dest / "README.md").stat().st_mode
    assert sh_mode & stat.S_IXUSR, "smoke.sh should be executable"
    assert not md_mode & stat.S_IXUSR, "README.md should not be executable"


def test_diff_mode_writes_when_confirmed(tmp_path: Path) -> None:
    dest = tmp_path / "demo"
    dest.mkdir()
    (dest / "README.md").write_text("OLD\n")
    result = _result([("README.md", "NEW\n"), ("a.txt", "a\n")])
    answers = {"README.md": True}

    def confirm(rel: str, _diff: str) -> bool:
        return answers.get(rel, False)

    report = write_project(result, dest, WriteMode.diff, confirm_diff=confirm)
    assert (dest / "README.md").read_text() == "NEW\n"
    assert "README.md" in report.overwritten
    assert "a.txt" in report.written


def test_emits_file_written_events_with_mode(tmp_path: Path) -> None:
    """P1: writer fires file_written per file with the correct mode tag."""
    dest = tmp_path / "demo"
    dest.mkdir()
    (dest / "README.md").write_text("OLD\n")
    result = _result(
        [
            ("README.md", "NEW\n"),  # overwrite
            ("src/main.py", "x = 1\n"),  # new
        ]
    )
    events: list[ProgressEvent] = []
    write_project(result, dest, WriteMode.overwrite, on_event=events.append)
    kinds = [e.kind for e in events]
    assert kinds == ["file_written", "file_written"]
    payloads = {e.payload["path"]: e.payload for e in events}
    assert payloads["README.md"]["mode"] == "overwrite"
    assert payloads["src/main.py"]["mode"] == "new"
    assert payloads["src/main.py"]["bytes"] == len("x = 1\n")


def test_emits_file_written_event_for_skipped_files(tmp_path: Path) -> None:
    dest = tmp_path / "demo"
    dest.mkdir()
    (dest / "README.md").write_text("OLD\n")
    result = _result([("README.md", "NEW\n"), ("src/main.py", "x = 1\n")])
    events: list[ProgressEvent] = []
    write_project(result, dest, WriteMode.skip, on_event=events.append)
    by_path = {e.payload["path"]: e.payload["mode"] for e in events}
    # README.md was skipped (already existed), main.py created new.
    assert by_path["README.md"] == "skip"
    assert by_path["src/main.py"] == "new"


def test_diff_mode_keeps_when_declined(tmp_path: Path) -> None:
    dest = tmp_path / "demo"
    dest.mkdir()
    (dest / "README.md").write_text("OLD\n")
    result = _result([("README.md", "NEW\n")])

    def confirm(_rel: str, _diff: str) -> bool:
        return False

    report = write_project(result, dest, WriteMode.diff, confirm_diff=confirm)
    assert (dest / "README.md").read_text() == "OLD\n"
    assert "README.md" in report.skipped


# ---------------------------------------------------------------------------
# preview_diffs / pre_confirm batch gate
# ---------------------------------------------------------------------------


def test_preview_diffs_marks_new_modified_unchanged(tmp_path: Path) -> None:
    dest = tmp_path / "demo"
    dest.mkdir()
    (dest / "README.md").write_text("OLD\n")
    (dest / "unchanged.txt").write_text("same\n")
    result = _result(
        [
            ("README.md", "NEW\n"),
            ("unchanged.txt", "same\n"),
            ("src/main.py", "print('hi')\n"),
        ],
    )

    diffs = preview_diffs(result, dest)
    by_path = {d.path: d for d in diffs}
    assert by_path["README.md"].status == "modified"
    assert by_path["README.md"].diff_text  # has the unified-diff body
    assert by_path["unchanged.txt"].status == "unchanged"
    assert by_path["src/main.py"].status == "new"
    # Existing files were NOT mutated by preview_diffs.
    assert (dest / "README.md").read_text() == "OLD\n"


def test_pre_confirm_true_overwrites_without_per_file_prompt(tmp_path: Path) -> None:
    dest = tmp_path / "demo"
    dest.mkdir()
    (dest / "README.md").write_text("OLD\n")
    result = _result([("README.md", "NEW\n")])

    seen: list[list[FileDiff]] = []

    def pre(diffs: list[FileDiff]) -> bool:
        seen.append(diffs)
        return True

    def per_file(_rel: str, _diff: str) -> bool:
        pytest.fail("per-file confirm should not run when pre_confirm returns True")

    report = write_project(
        result,
        dest,
        WriteMode.diff,
        confirm_diff=per_file,
        pre_confirm=pre,
    )
    assert (dest / "README.md").read_text() == "NEW\n"
    assert "README.md" in report.overwritten
    assert len(seen) == 1
    assert {d.path for d in seen[0]} == {"README.md"}


def test_pre_confirm_false_raises_diff_preview_cancelled(tmp_path: Path) -> None:
    dest = tmp_path / "demo"
    dest.mkdir()
    (dest / "README.md").write_text("OLD\n")
    result = _result([("README.md", "NEW\n")])

    def pre(_diffs: list[FileDiff]) -> bool:
        return False

    with pytest.raises(DiffPreviewCancelled):
        write_project(result, dest, WriteMode.diff, pre_confirm=pre)
    # Existing file is untouched.
    assert (dest / "README.md").read_text() == "OLD\n"


def test_pre_confirm_ignored_for_non_diff_modes(tmp_path: Path) -> None:
    """``pre_confirm`` is a diff-mode-only hook; setting it for skip/overwrite
    is a no-op (the writer never calls it)."""
    dest = tmp_path / "demo"
    dest.mkdir()
    (dest / "README.md").write_text("OLD\n")
    result = _result([("README.md", "NEW\n")])
    calls: list[int] = []

    def pre(_diffs: list[FileDiff]) -> bool:
        calls.append(1)
        return False

    write_project(result, dest, WriteMode.overwrite, pre_confirm=pre)
    assert (dest / "README.md").read_text() == "NEW\n"
    assert calls == []  # never invoked for non-diff modes
