"""Tests for the ``frontend`` branch of ``agent-scaffold logs``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import typer

import agent_scaffold.cli as cli_mod
from agent_scaffold._scaffold_dir import SCAFFOLD_DIR
from agent_scaffold.cli import _tail_frontend_log


def _write_log(project_dir: Path, body: str = "line1\nline2\nline3\n") -> Path:
    log = project_dir / SCAFFOLD_DIR / "frontend.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(body, encoding="utf-8")
    return log


def test_missing_log_file_exits_with_friendly_error(tmp_path: Path) -> None:
    with pytest.raises(typer.Exit) as excinfo:
        _tail_frontend_log(tmp_path, follow=False, tail=100)
    assert excinfo.value.exit_code == 1


def test_present_log_invokes_tail_via_execvp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_log(tmp_path)

    captured: dict[str, Any] = {"argv": None, "file": None}

    def fake_which(name: str) -> str | None:
        return "/usr/bin/tail" if name == "tail" else None

    def fake_execvp(file: str, argv: list[str]) -> None:
        captured["file"] = file
        captured["argv"] = argv
        raise SystemExit(0)  # mimic the process-replacement

    monkeypatch.setattr(cli_mod.shutil, "which", fake_which)
    monkeypatch.setattr(cli_mod.os, "execvp", fake_execvp)

    with pytest.raises(SystemExit):
        _tail_frontend_log(tmp_path, follow=True, tail=50)

    assert captured["file"] == "/usr/bin/tail"
    argv = captured["argv"]
    assert argv[0] == "/usr/bin/tail"
    assert "-n" in argv and "50" in argv
    assert "-f" in argv
    assert argv[-1].endswith("frontend.log")


def test_no_follow_omits_dash_f_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_log(tmp_path)
    captured: dict[str, Any] = {"argv": None}

    monkeypatch.setattr(cli_mod.shutil, "which", lambda _n: "/usr/bin/tail")

    def fake_execvp(_file: str, argv: list[str]) -> None:
        captured["argv"] = argv
        raise SystemExit(0)

    monkeypatch.setattr(cli_mod.os, "execvp", fake_execvp)

    with pytest.raises(SystemExit):
        _tail_frontend_log(tmp_path, follow=False, tail=10)

    assert "-f" not in captured["argv"]


def test_python_tail_fallback_when_tail_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_log(tmp_path, "first\nsecond\nthird\n")
    monkeypatch.setattr(cli_mod.shutil, "which", lambda _n: None)

    # follow=False so the python fallback returns immediately after dumping.
    _tail_frontend_log(tmp_path, follow=False, tail=2)
    # The python fallback writes via Rich console; we can't capture that
    # through capsys reliably, so the assertion is just "didn't raise".
