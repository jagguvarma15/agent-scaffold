"""Tests for ``_down_inline`` — the teardown behind ``agent-scaffold down`` and ``/down``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import agent_scaffold.cli as cli_mod
from agent_scaffold import ports
from agent_scaffold._scaffold_dir import SCAFFOLD_DIR


class _FakeProc:
    returncode = 0


@pytest.fixture
def compose_calls(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], cwd: str, check: bool = False) -> _FakeProc:
        calls.append(cmd)
        return _FakeProc()

    monkeypatch.setattr(cli_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(cli_mod.shutil, "which", lambda _name: "/usr/bin/docker")
    return calls


def _stub_servers(
    monkeypatch: pytest.MonkeyPatch,
    *,
    frontend: tuple[bool, int | None] = (True, None),
    backend: tuple[bool, int | None] = (True, None),
) -> None:
    monkeypatch.setattr(cli_mod, "_stop_frontend", lambda _d: frontend)
    monkeypatch.setattr(cli_mod, "_stop_backend", lambda _d: backend)


def _held(monkeypatch: pytest.MonkeyPatch, held: set[int]) -> list[int]:
    scanned: list[int] = []
    monkeypatch.setattr(ports, "wait_port_free", lambda port, **_kw: port not in held)

    def fake_scan(port_list: list[int], **_kw: Any) -> list[ports.PortConflict]:
        scanned.extend(port_list)
        return [
            ports.PortConflict(
                port=p, owner=ports.PortOwner(kind="process", port=p, pid=999, command="python")
            )
            for p in port_list
        ]

    monkeypatch.setattr(ports, "scan_conflicts", fake_scan)
    return scanned


def test_local_project_without_compose_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / SCAFFOLD_DIR).mkdir()
    _stub_servers(monkeypatch, backend=(True, 18765))
    _held(monkeypatch, set())

    assert cli_mod._down_inline(tmp_path, yes=True) == 0


def test_local_project_reports_a_server_that_survived(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / SCAFFOLD_DIR).mkdir()
    _stub_servers(monkeypatch, backend=(False, 18765))
    _held(monkeypatch, set())

    assert cli_mod._down_inline(tmp_path, yes=True) == 1


def test_not_a_project_still_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_servers(monkeypatch)
    assert cli_mod._down_inline(tmp_path, yes=True) == 1


def test_compose_without_docker_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "docker-compose.yml").write_text("services: {}\n")
    _stub_servers(monkeypatch)
    monkeypatch.setattr(cli_mod.shutil, "which", lambda _name: None)

    assert cli_mod._down_inline(tmp_path, yes=True) == 1


def test_compose_down_removes_orphans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, compose_calls: list[list[str]]
) -> None:
    (tmp_path / "docker-compose.yml").write_text("services: {}\n")
    _stub_servers(monkeypatch)
    _held(monkeypatch, set())

    assert cli_mod._down_inline(tmp_path, yes=True) == 0
    assert compose_calls == [["docker", "compose", "down", "--remove-orphans"]]


def test_still_held_port_is_reported_and_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compose_calls: list[list[str]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "docker-compose.yml").write_text(
        "services:\n  app:\n    ports: ['18080:8000']\n  db:\n    ports: ['15432:5432']\n"
    )
    _stub_servers(monkeypatch, frontend=(True, 15173))
    scanned = _held(monkeypatch, {18080})

    assert cli_mod._down_inline(tmp_path, yes=True) == 1
    assert scanned == [18080]
    assert "still held" in capsys.readouterr().out
