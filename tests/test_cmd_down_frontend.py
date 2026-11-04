"""Tests for ``_stop_frontend`` invoked by ``agent-scaffold down``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import agent_scaffold.cli as cli_mod
import agent_scaffold.steps._subprocess as sub_mod
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


def _fake_stop(monkeypatch: pytest.MonkeyPatch, *, result: bool = True) -> list[int]:
    stopped: list[int] = []

    def fake(pid: int, *, grace: float = 5.0) -> bool:
        stopped.append(pid)
        return result

    monkeypatch.setattr(sub_mod, "stop_process_group", fake)
    return stopped


def test_stop_frontend_stops_the_recorded_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_pid_file(tmp_path, pid=12345, port=3000)
    stopped = _fake_stop(monkeypatch)

    assert _stop_frontend(tmp_path) == (True, 3000)
    assert stopped == [12345]


def test_stop_frontend_removes_pid_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pid_file = _write_pid_file(tmp_path)
    _fake_stop(monkeypatch)

    _stop_frontend(tmp_path)
    assert not pid_file.is_file()


def test_stop_frontend_resets_step_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_pid_file(tmp_path)
    _seed_step_state(tmp_path, "launch_frontend", StepStatus.DONE)
    _fake_stop(monkeypatch)

    _stop_frontend(tmp_path)

    from agent_scaffold.orchestrator import read_state

    state = read_state(tmp_path)
    assert state.steps["launch_frontend"].status is StepStatus.PENDING


def test_stop_frontend_noop_when_no_pid_file(tmp_path: Path) -> None:
    # Must not raise even when there's nothing to do.
    assert _stop_frontend(tmp_path) == (True, None)


def test_stop_frontend_cleans_up_malformed_pid_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid_file = tmp_path / SCAFFOLD_DIR / "frontend.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text("{not-json", encoding="utf-8")
    stopped = _fake_stop(monkeypatch)

    _stop_frontend(tmp_path)

    assert not pid_file.is_file()
    assert stopped == []  # nothing to kill


def test_stop_frontend_keeps_pid_file_when_server_survives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid_file = _write_pid_file(tmp_path, pid=12345, port=3000)
    _fake_stop(monkeypatch, result=False)
    monkeypatch.setattr(cli_mod, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(cli_mod, "_is_group_leader", lambda _pid: True)

    assert _stop_frontend(tmp_path) == (False, 3000)
    # Kept so a retried `down` still knows which process to stop.
    assert pid_file.is_file()


def test_stop_frontend_drops_pid_file_reused_by_another_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid_file = _write_pid_file(tmp_path, pid=12345)
    _fake_stop(monkeypatch, result=False)
    monkeypatch.setattr(cli_mod, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(cli_mod, "_is_group_leader", lambda _pid: False)

    assert _stop_frontend(tmp_path)[0] is True
    assert not pid_file.is_file()
