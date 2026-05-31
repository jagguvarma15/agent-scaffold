"""Tests for ``_stop_frontend`` invoked by ``agent-scaffold down``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import agent_scaffold.cli as cli_mod
from agent_scaffold._scaffold_dir import SCAFFOLD_DIR
from agent_scaffold.cli import _stop_frontend
from agent_scaffold.orchestrator import OrchestratorState, StepState, StepStatus, write_state


def _write_pid_file(project_dir: Path, *, pid: int = 4321, port: int = 3000) -> Path:
    path = project_dir / SCAFFOLD_DIR / "frontend.pid"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"pid": pid, "port": port, "started_at": "2026-05-30T00:00:00+00:00"}),
        encoding="utf-8",
    )
    return path


def _seed_step_state(project_dir: Path, step_id: str, status: StepStatus) -> None:
    state = OrchestratorState(steps={step_id: StepState(status=status, fingerprint="x")})
    write_state(project_dir, state)


def test_stop_frontend_sigterms_pid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_pid_file(tmp_path, pid=12345)
    signals: list[tuple[int, int]] = []

    def fake_killpg(pgid: int, sig: int) -> None:
        signals.append((pgid, sig))

    def fake_getpgid(pid: int) -> int:
        return pid  # return the pid as its own group; what os.killpg expects

    monkeypatch.setattr(cli_mod.os, "killpg", fake_killpg)
    monkeypatch.setattr(cli_mod.os, "getpgid", fake_getpgid)

    _stop_frontend(tmp_path)

    assert signals and signals[0][0] == 12345


def test_stop_frontend_removes_pid_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pid_file = _write_pid_file(tmp_path)
    monkeypatch.setattr(cli_mod.os, "killpg", lambda _pg, _sig: None)
    monkeypatch.setattr(cli_mod.os, "getpgid", lambda pid: pid)

    _stop_frontend(tmp_path)
    assert not pid_file.is_file()


def test_stop_frontend_resets_step_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_pid_file(tmp_path)
    _seed_step_state(tmp_path, "launch_frontend", StepStatus.DONE)
    monkeypatch.setattr(cli_mod.os, "killpg", lambda _pg, _sig: None)
    monkeypatch.setattr(cli_mod.os, "getpgid", lambda pid: pid)

    _stop_frontend(tmp_path)

    from agent_scaffold.orchestrator import read_state

    state = read_state(tmp_path)
    assert state.steps["launch_frontend"].status is StepStatus.PENDING


def test_stop_frontend_noop_when_no_pid_file(tmp_path: Path) -> None:
    # Must not raise even when there's nothing to do.
    _stop_frontend(tmp_path)


def test_stop_frontend_cleans_up_malformed_pid_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid_file = tmp_path / SCAFFOLD_DIR / "frontend.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text("{not-json", encoding="utf-8")

    killed: list[int] = []

    def fake_killpg(pg: int, _sig: int) -> None:
        killed.append(pg)

    monkeypatch.setattr(cli_mod.os, "killpg", fake_killpg)
    monkeypatch.setattr(cli_mod.os, "getpgid", lambda pid: pid)

    _stop_frontend(tmp_path)

    assert not pid_file.is_file()
    assert killed == []  # nothing to kill


def test_stop_frontend_tolerates_processlookuperror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_pid_file(tmp_path, pid=999999)

    def boom(_pg: int, _sig: int) -> None:
        raise ProcessLookupError

    fallback_calls: list[int] = []

    def fallback_kill(pid: int, _sig: int) -> None:
        fallback_calls.append(pid)

    monkeypatch.setattr(cli_mod.os, "killpg", boom)
    monkeypatch.setattr(cli_mod.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(cli_mod.os, "kill", fallback_kill)

    # Should still complete cleanly + try the direct-kill fallback.
    _stop_frontend(tmp_path)
    assert fallback_calls == [999999]
    assert not (tmp_path / SCAFFOLD_DIR / "frontend.pid").is_file()
