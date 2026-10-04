"""The ``--free-ports`` executor in cli.py.

Everything is faked at the seams (``os.kill``, ``ports.lsof_port_pids``, the
remediation runner, the clock-free ``wait_port_free``): no test signals a real
process or runs docker. The pure planning and the protected list are covered
in test_ports.py; these tests cover what the executor does with a plan.
"""

from __future__ import annotations

import io
import signal
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

import agent_scaffold.cli as cli_mod
from agent_scaffold import ports


@pytest.fixture(autouse=True)
def _linux(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the platform so these tests mean the same on every CI OS."""
    monkeypatch.setattr(ports.sys, "platform", "linux")


@pytest.fixture
def out(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    """A wide, plain console so table rows never wrap mid-assertion."""
    buffer = io.StringIO()
    monkeypatch.setattr(
        cli_mod, "console", Console(file=buffer, width=240, force_terminal=False, color_system=None)
    )
    return buffer


def _permissive_ctx(*, comm: str = "/usr/bin/python3") -> ports.ProtectionContext:
    return ports.ProtectionContext(
        own_pids=frozenset({500}),
        euid=501,
        inspect=lambda _pid: ports.ProcInfo(uid=501, ppid=100, comm=comm),
    )


class _Recorder:
    """Spies for everything the executor can run or signal."""

    def __init__(self) -> None:
        self.commands: list[tuple[list[str], Path | None]] = []
        self.kills: list[tuple[int, int]] = []

    def run(self, argv: list[str], *, cwd: Path | None = None) -> bool:
        self.commands.append((list(argv), cwd))
        return True

    def kill(self, pid: int, sig: int) -> None:
        self.kills.append((pid, sig))


@pytest.fixture
def rec(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    recorder = _Recorder()
    monkeypatch.setattr(cli_mod, "_run_remediation_command", recorder.run)
    monkeypatch.setattr(cli_mod.os, "kill", recorder.kill)
    monkeypatch.setattr(ports, "build_protection_context", lambda **_kw: _permissive_ctx())
    # Nothing here may ever ask: the flag is the consent.
    monkeypatch.setattr(
        cli_mod.typer, "confirm", lambda *_a, **_k: pytest.fail("free-ports must not prompt")
    )
    monkeypatch.setattr(
        cli_mod,
        "_confirm_remediation_command",
        lambda *_a, **_k: pytest.fail("free-ports must not prompt"),
    )
    return recorder


def _free(monkeypatch: pytest.MonkeyPatch, *, busy_after: bool = False) -> None:
    """Make every port read as free (or still busy) after the actions ran."""
    monkeypatch.setattr(ports, "wait_port_free", lambda *_a, **_k: not busy_after)
    monkeypatch.setattr(ports, "port_in_use", lambda *_a, **_k: busy_after)


def _docker(port: int, name: str, project: str = "other") -> ports.PortConflict:
    owner = ports.PortOwner(
        kind="docker",
        port=port,
        container_id=f"id-{name}",
        container_name=name,
        compose_project=project,
    )
    return ports.PortConflict(port=port, owner=owner)


def _process(port: int, pid: int, command: str = "python3") -> ports.PortConflict:
    owner = ports.PortOwner(kind="process", port=port, pid=pid, command=command)
    return ports.PortConflict(port=port, owner=owner)


def _run(conflicts: list[ports.PortConflict], tmp_path: Path, **kwargs: Any) -> bool:
    return cli_mod._free_conflicting_ports(
        conflicts,
        project_dir=tmp_path / "myproj",
        compose_dir=kwargs.pop("compose_dir", tmp_path),
        **kwargs,
    )


# ---- containers ------------------------------------------------------------------


def test_stops_a_foreign_container_without_prompting(
    rec: _Recorder, out: io.StringIO, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _free(monkeypatch)
    assert _run([_docker(6379, "stale-redis")], tmp_path) is True
    assert rec.commands == [(["docker", "stop", "stale-redis"], None)]
    report = out.getvalue()
    assert "free-ports is on" in report
    assert "stale-redis" in report
    assert "freed" in report


def test_own_stack_is_taken_down_with_one_compose_down(
    rec: _Recorder, out: io.StringIO, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _free(monkeypatch)
    conflicts = [
        _docker(6379, "myproj-redis-1", project="myproj"),
        _docker(5432, "myproj-pg-1", project="myproj"),
    ]
    assert _run(conflicts, tmp_path) is True
    assert rec.commands == [(["docker", "compose", "down"], tmp_path)]
    assert "this project's previous stack" in out.getvalue()


def test_own_stack_then_foreign_container_in_order(
    rec: _Recorder, out: io.StringIO, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _free(monkeypatch)
    conflicts = [
        _docker(6333, "qdrant"),
        _docker(6379, "myproj-redis-1", project="myproj"),
    ]
    assert _run(conflicts, tmp_path) is True
    assert [argv for argv, _cwd in rec.commands] == [
        ["docker", "compose", "down"],
        ["docker", "stop", "qdrant"],
    ]


# ---- host processes --------------------------------------------------------------


def test_kills_a_foreign_process_with_sigterm(
    rec: _Recorder, out: io.StringIO, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(ports, "lsof_port_pids", lambda _port, **_kw: [(4242, "python3")])
    _free(monkeypatch)
    assert _run([_process(8000, 4242)], tmp_path) is True
    assert rec.kills == [(4242, signal.SIGTERM)]
    assert rec.commands == []  # the external `kill` binary is never spawned


def test_escalates_to_sigkill_when_sigterm_is_ignored(
    rec: _Recorder, out: io.StringIO, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(ports, "lsof_port_pids", lambda _port, **_kw: [(4242, "python3")])
    waits = iter([False, True, True])  # busy after SIGTERM, free after SIGKILL
    monkeypatch.setattr(ports, "wait_port_free", lambda *_a, **_k: next(waits))
    monkeypatch.setattr(ports, "port_in_use", lambda *_a, **_k: False)
    assert _run([_process(8000, 4242)], tmp_path) is True
    assert rec.kills == [(4242, signal.SIGTERM), (4242, signal.SIGKILL)]


def test_never_signals_when_the_holder_changed_since_planning(
    rec: _Recorder, out: io.StringIO, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """pid reuse / a restart between plan and kill: the fresh lookup no longer
    contains the planned pid, so nothing is signalled."""
    monkeypatch.setattr(ports, "lsof_port_pids", lambda _port, **_kw: [(9999, "other")])
    _free(monkeypatch, busy_after=True)
    assert _run([_process(8000, 4242)], tmp_path) is False
    assert rec.kills == []
    assert "holder changed" in out.getvalue()


def test_a_protected_co_listener_vetoes_the_whole_kill(
    rec: _Recorder, out: io.StringIO, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Several listeners on one port: one protected pid means NO pid is killed."""
    monkeypatch.setattr(
        ports, "lsof_port_pids", lambda _port, **_kw: [(4242, "python3"), (77, "com.docke")]
    )

    def inspect(pid: int) -> ports.ProcInfo:
        comm = "/Applications/Docker.app/Contents/MacOS/com.docker.backend" if pid == 77 else "py"
        return ports.ProcInfo(uid=501, ppid=100, comm=comm)

    monkeypatch.setattr(
        ports,
        "build_protection_context",
        lambda **_kw: ports.ProtectionContext(own_pids=frozenset({500}), euid=501, inspect=inspect),
    )
    _free(monkeypatch, busy_after=True)
    assert _run([_process(8000, 4242)], tmp_path) is False
    assert rec.kills == []
    assert "refused pid 77" in out.getvalue()


def test_permission_denied_is_reported_not_raised(
    rec: _Recorder, out: io.StringIO, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(ports, "lsof_port_pids", lambda _port, **_kw: [(4242, "python3")])

    def deny(pid: int, sig: int) -> None:
        raise PermissionError

    monkeypatch.setattr(cli_mod.os, "kill", deny)
    _free(monkeypatch, busy_after=True)
    assert _run([_process(8000, 4242)], tmp_path) is False
    assert "permission denied" in out.getvalue()


def test_windows_never_kills(
    rec: _Recorder, out: io.StringIO, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ports.sys, "platform", "win32")
    owner = ports.PortOwner(kind="process", port=8000, pid=4242, command="python.exe")
    ok, why = cli_mod._terminate_port_holder(8000, owner, _permissive_ctx())
    assert ok is False
    assert "Windows" in why
    assert rec.kills == []


# ---- refusals and unfreed ports ---------------------------------------------------


def test_protected_and_unknown_holders_are_reported_and_fail_the_run(
    rec: _Recorder, out: io.StringIO, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        ports,
        "build_protection_context",
        lambda **_kw: ports.ProtectionContext(
            own_pids=frozenset({500}),
            euid=501,
            inspect=lambda _pid: ports.ProcInfo(
                501, 100, "/Applications/Docker.app/Contents/MacOS/com.docker.backend"
            ),
        ),
    )
    _free(monkeypatch, busy_after=True)
    conflicts = [
        _process(8000, 78948, "com.docke"),
        ports.PortConflict(port=9000, owner=ports.PortOwner(kind="unknown", port=9000)),
    ]
    assert _run(conflicts, tmp_path) is False
    assert rec.kills == []
    assert rec.commands == []
    report = out.getvalue()
    assert "left alone" in report
    assert "container runtime" in report
    assert "could not be identified" in report
    assert "find and stop the owner manually" in report  # manual commands for the unfreed


def test_a_port_still_in_use_after_the_stop_fails_the_run(
    rec: _Recorder, out: io.StringIO, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _free(monkeypatch, busy_after=True)
    assert _run([_docker(6379, "respawner")], tmp_path) is False
    assert "still in use" in out.getvalue()


def test_output_is_redacted(
    rec: _Recorder, out: io.StringIO, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    secret = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz012345"
    monkeypatch.setattr(ports, "lsof_port_pids", lambda _port, **_kw: [(4242, secret)])
    _free(monkeypatch)
    _run([_process(8000, 4242, secret)], tmp_path)
    assert secret not in out.getvalue()


# ---- the auto switch on the existing entry point ------------------------------------


def test_remediate_auto_delegates_without_prompting_or_the_conflict_table(
    rec: _Recorder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[list[ports.PortConflict]] = []

    def fake_free(conflicts: list[ports.PortConflict], **_kw: Any) -> bool:
        seen.append(conflicts)
        return True

    monkeypatch.setattr(cli_mod, "_free_conflicting_ports", fake_free)
    monkeypatch.setattr(
        cli_mod,
        "_render_port_conflict_table",
        lambda *_a, **_k: pytest.fail("auto mode reports through the free table only"),
    )
    conflicts = [_docker(6379, "stale-redis")]
    ok = cli_mod._remediate_port_conflicts(
        conflicts, project_dir=tmp_path, compose_dir=tmp_path, auto=True
    )
    assert ok is True
    assert seen == [conflicts]


def test_remediate_without_auto_still_asks_per_command(
    monkeypatch: pytest.MonkeyPatch, out: io.StringIO, tmp_path: Path
) -> None:
    asked: list[list[str]] = []

    def decline(argv: list[str], **_kw: Any) -> bool:
        asked.append(argv)
        return False

    monkeypatch.setattr(cli_mod, "_confirm_remediation_command", decline)
    monkeypatch.setattr(
        cli_mod,
        "_run_remediation_command",
        lambda *_a, **_k: pytest.fail("a declined command must not run"),
    )
    monkeypatch.setattr(ports, "port_in_use", lambda *_a, **_k: True)
    ok = cli_mod._remediate_port_conflicts(
        [_docker(6379, "stale-redis")], project_dir=tmp_path, compose_dir=tmp_path
    )
    assert ok is False
    assert asked == [["docker", "stop", "stale-redis"]]


# ---- small helpers ----------------------------------------------------------------


def test_split_own_stack_separates_by_compose_project() -> None:
    own_conflict = _docker(6379, "myproj-redis-1", project="myproj")
    foreign = _docker(6333, "qdrant", project="other")
    process = _process(8000, 4242)
    own, others = cli_mod._split_own_stack([own_conflict, foreign, process], "myproj")
    assert own == [own_conflict]
    assert others == [foreign, process]


def test_split_own_stack_blank_project_matches_nothing() -> None:
    loose = _docker(6379, "loose", project="")
    own, others = cli_mod._split_own_stack([loose], "")
    assert own == []
    assert others == [loose]


def test_owner_label_shapes() -> None:
    docker = _docker(6379, "stale-redis", project="other").owner
    assert cli_mod._owner_label(docker) == "stale-redis (compose project: other)"
    process = _process(8000, 4242, "python3").owner
    assert cli_mod._owner_label(process) == "pid 4242: python3"
    unknown = ports.PortOwner(kind="unknown", port=9000)
    assert cli_mod._owner_label(unknown) == "owner lookup unavailable"
