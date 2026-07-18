"""Tests for agent_scaffold.writer."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from agent_scaffold.contract import GeneratedFile, GenerationResult
from agent_scaffold.progress import ProgressEvent
from agent_scaffold.writer import (
    DestinationExistsError,
    WriteMode,
    preview_diffs,
    summarize_diffs,
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


def test_rollback_restores_overwritten_files_on_mid_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure after some files were replaced restores the originals."""
    dest = tmp_path / "demo"
    dest.mkdir()
    (dest / "a.txt").write_text("original a\n")
    (dest / "b.txt").write_text("original b\n")

    result = _result(
        [
            ("a.txt", "new a\n"),
            ("b.txt", "new b\n"),
            ("c.txt", "new c\n"),
        ]
    )

    real_replace = os.replace

    def failing_replace(src, dst, *args, **kwargs):  # type: ignore[no-untyped-def]
        # Fail only the staged-to-final move of the last file, after the
        # first two overwrites have already landed.
        if ".agent-scaffold-stage-" in str(src) and str(dst).endswith("c.txt"):
            raise OSError("simulated disk full")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", failing_replace)

    with pytest.raises(OSError, match="disk full"):
        write_project(result, dest, WriteMode.overwrite)

    # Overwritten files restored byte-for-byte; the aborted create absent.
    assert (dest / "a.txt").read_text() == "original a\n"
    assert (dest / "b.txt").read_text() == "original b\n"
    assert not (dest / "c.txt").exists()


def test_rollback_removes_created_files_on_mid_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure into a fresh destination removes every file created so far."""
    dest = tmp_path / "demo"

    result = _result(
        [
            ("README.md", "# demo\n"),
            ("src/demo/main.py", "x = 1\n"),
            ("src/demo/tail.py", "y = 2\n"),
        ]
    )

    real_replace = os.replace

    def failing_replace(src, dst, *args, **kwargs):  # type: ignore[no-untyped-def]
        if str(dst).endswith("tail.py"):
            raise OSError("simulated failure")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", failing_replace)

    with pytest.raises(OSError, match="simulated"):
        write_project(result, dest, WriteMode.overwrite)

    assert not (dest / "README.md").exists()
    assert not (dest / "src/demo/main.py").exists()
    assert not (dest / "src/demo/tail.py").exists()


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


# ---------------------------------------------------------------------------
# preview_diffs / summarize_diffs (names-only change summary)
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


def test_summarize_diffs_buckets_by_status(tmp_path: Path) -> None:
    dest = tmp_path / "demo"
    dest.mkdir()
    (dest / "README.md").write_text("OLD\n")
    (dest / "unchanged.txt").write_text("same\n")
    result = _result(
        [
            ("README.md", "NEW\n"),  # modified
            ("unchanged.txt", "same\n"),  # unchanged
            ("src/main.py", "print('hi')\n"),  # new
        ],
    )
    summary = summarize_diffs(preview_diffs(result, dest))
    assert summary.modified == ["README.md"]
    assert summary.unchanged == ["unchanged.txt"]
    assert summary.new == ["src/main.py"]
    assert summary.touches_existing is True


def test_summarize_diffs_no_overlap_into_fresh_dir(tmp_path: Path) -> None:
    dest = tmp_path / "demo"  # does not exist yet
    result = _result([("README.md", "# demo\n"), ("a.txt", "a\n")])
    summary = summarize_diffs(preview_diffs(result, dest))
    assert sorted(summary.new) == ["README.md", "a.txt"]
    assert summary.modified == []
    assert summary.touches_existing is False


def test_overwrite_is_non_interactive(tmp_path: Path) -> None:
    """``write_project`` takes no confirm callbacks — overwrite just writes.

    The overwrite confirmation now lives in the pipeline (with the live
    display suspended); the writer itself must never block on stdin.
    """
    dest = tmp_path / "demo"
    dest.mkdir()
    (dest / "README.md").write_text("OLD\n")
    result = _result([("README.md", "NEW\n")])
    report = write_project(result, dest, WriteMode.overwrite)
    assert (dest / "README.md").read_text() == "NEW\n"
    assert "README.md" in report.overwritten
