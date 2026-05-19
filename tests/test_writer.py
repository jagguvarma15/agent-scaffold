"""Tests for agent_scaffold.writer."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from agent_scaffold.contract import GeneratedFile, GenerationResult
from agent_scaffold.writer import (
    DestinationExistsError,
    WriteMode,
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
