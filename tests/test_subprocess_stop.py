"""Tests for ``stop_process_group`` — how ``down`` stops the servers we spawned."""

from __future__ import annotations

import os
import signal
import subprocess
import sys

import pytest

import agent_scaffold.steps._subprocess as sub_mod
from agent_scaffold.steps._subprocess import stop_process_group


class _FakeProc:
    """Scripted process: dies after the signals listed in ``dies_on``."""

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, *, pgid: int, dies_on: set[int], alive: bool = True
    ) -> None:
        self.alive = alive
        self.sent: list[int] = []
        self.dies_on = dies_on
        monkeypatch.setattr(sub_mod, "_reap_or_alive", lambda _pid: self.alive)
        monkeypatch.setattr(sub_mod.os, "getpgid", lambda _pid: pgid)
        monkeypatch.setattr(sub_mod.os, "killpg", self._killpg)
        monkeypatch.setattr(sub_mod, "_STOP_POLL_INTERVAL", 0.0)
        monkeypatch.setattr(sub_mod, "_KILL_WAIT", 0.0)

    def _killpg(self, _pgid: int, sig: int) -> None:
        self.sent.append(sig)
        if sig in self.dies_on:
            self.alive = False


def test_already_dead_sends_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    proc = _FakeProc(monkeypatch, pgid=4321, dies_on=set(), alive=False)
    assert stop_process_group(4321) is True
    assert proc.sent == []


def test_exits_on_sigterm(monkeypatch: pytest.MonkeyPatch) -> None:
    proc = _FakeProc(monkeypatch, pgid=4321, dies_on={signal.SIGTERM})
    assert stop_process_group(4321, grace=0.0) is True
    assert proc.sent == [signal.SIGTERM]


def test_escalates_to_sigkill_when_sigterm_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    # uvicorn waiting on an open browser connection behaves like this.
    proc = _FakeProc(monkeypatch, pgid=4321, dies_on={signal.SIGKILL})
    assert stop_process_group(4321, grace=0.0) is True
    assert proc.sent == [signal.SIGTERM, signal.SIGKILL]


def test_reused_pid_is_never_signalled(monkeypatch: pytest.MonkeyPatch) -> None:
    # Our servers lead their own session; a pid in someone else's group was recycled.
    proc = _FakeProc(monkeypatch, pgid=1, dies_on={signal.SIGTERM, signal.SIGKILL})
    assert stop_process_group(4321) is False
    assert proc.sent == []


def test_survivor_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    proc = _FakeProc(monkeypatch, pgid=4321, dies_on=set())
    assert stop_process_group(4321, grace=0.0) is False
    assert proc.sent == [signal.SIGTERM, signal.SIGKILL]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_stops_a_real_sigterm_ignoring_child() -> None:
    """End to end: a spawned child that ignores SIGTERM is killed and reaped."""
    code = (
        "import signal, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "print('ready', flush=True)\n"
        "time.sleep(60)\n"
    )
    proc = subprocess.Popen(  # noqa: S603
        [sys.executable, "-c", code], stdout=subprocess.PIPE, start_new_session=True
    )
    assert proc.stdout is not None
    proc.stdout.readline()  # handler installed
    try:
        assert stop_process_group(proc.pid, grace=0.3) is True
        with pytest.raises(ProcessLookupError):
            os.kill(proc.pid, 0)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.stdout.close()
