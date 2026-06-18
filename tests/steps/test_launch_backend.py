"""Tests for ``agent_scaffold.steps.launch_backend``."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from agent_scaffold._scaffold_dir import SCAFFOLD_DIR
from agent_scaffold.manifest import Manifest
from agent_scaffold.orchestrator import StepContext, StepStatus
from agent_scaffold.steps import launch_backend as lb_mod
from agent_scaffold.steps.launch_backend import (
    LaunchBackendStep,
    _entry_is_server,
    _module_for,
)

_SERVER_MAIN = (
    "import uvicorn\n"
    "from fastapi import FastAPI\n"
    "app = FastAPI()\n"
    "def main() -> None:\n"
    "    uvicorn.run(app)\n"
    "if __name__ == '__main__':\n"
    "    main()\n"
)
_AGENT_ONLY_MAIN = "from .agent import build\n\nagent = build()\n"


def _seed_backend(tmp_path: Path, *, pkg: str = "demo_app", body: str = _SERVER_MAIN) -> None:
    main = tmp_path / "src" / pkg / "main.py"
    main.parent.mkdir(parents=True, exist_ok=True)
    main.write_text(body, encoding="utf-8")


def _ctx(
    ctx_factory: Callable[..., StepContext],
    manifest_factory: Callable[..., Manifest],
    tmp_path: Path,
    *,
    language: str = "python",
) -> StepContext:
    return ctx_factory(project_dir=tmp_path, manifest=manifest_factory(language=language))


# ---- pure helpers ---------------------------------------------------------


def test_entry_is_server_detects_uvicorn() -> None:
    assert _entry_is_server(_SERVER_MAIN) is True


def test_entry_is_server_rejects_agent_only_module() -> None:
    assert _entry_is_server(_AGENT_ONLY_MAIN) is False


def test_module_for_derives_dotted_module() -> None:
    assert _module_for(Path("/proj/src/research_assistant/main.py")) == "research_assistant.main"


# ---- detection ------------------------------------------------------------


def test_detect_skips_non_python(
    tmp_path: Path,
    ctx_factory: Callable[..., StepContext],
    manifest_factory: Callable[..., Manifest],
) -> None:
    _seed_backend(tmp_path)  # has a server, but language is TS
    result = LaunchBackendStep().detect(
        _ctx(ctx_factory, manifest_factory, tmp_path, language="typescript")
    )
    assert result.status is StepStatus.SKIPPED
    assert "Python" in result.reason


def test_detect_skips_when_no_entry(
    tmp_path: Path,
    ctx_factory: Callable[..., StepContext],
    manifest_factory: Callable[..., Manifest],
) -> None:
    result = LaunchBackendStep().detect(_ctx(ctx_factory, manifest_factory, tmp_path))
    assert result.status is StepStatus.SKIPPED
    assert "no src" in result.reason


def test_detect_skips_agent_only_module(
    tmp_path: Path,
    ctx_factory: Callable[..., StepContext],
    manifest_factory: Callable[..., Manifest],
) -> None:
    _seed_backend(tmp_path, body=_AGENT_ONLY_MAIN)
    result = LaunchBackendStep().detect(_ctx(ctx_factory, manifest_factory, tmp_path))
    assert result.status is StepStatus.SKIPPED
    assert "agent module" in result.reason


def test_detect_pending_then_done(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
    manifest_factory: Callable[..., Manifest],
) -> None:
    _seed_backend(tmp_path)
    ctx = _ctx(ctx_factory, manifest_factory, tmp_path)
    # No PID file yet → PENDING.
    assert LaunchBackendStep().detect(ctx).status is StepStatus.PENDING
    # Live PID file → DONE.
    pid_file = tmp_path / SCAFFOLD_DIR / "backend.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(json.dumps({"pid": 4321, "port": 8000}), encoding="utf-8")
    monkeypatch.setattr(lb_mod, "_is_alive", lambda pid: True)
    assert LaunchBackendStep().detect(ctx).status is StepStatus.DONE


# ---- apply ----------------------------------------------------------------


def test_apply_launches_server_detached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
    manifest_factory: Callable[..., Manifest],
) -> None:
    _seed_backend(tmp_path, pkg="demo_app")
    calls: list[dict[str, Any]] = []

    class _Proc:
        pid = 4321

    def _fake_popen(cmd: list[str], **kwargs: Any) -> _Proc:
        calls.append({"cmd": cmd, "kwargs": kwargs})
        return _Proc()

    monkeypatch.setattr(lb_mod.shutil, "which", lambda _name: "/usr/bin/uv")
    monkeypatch.setattr(lb_mod.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(lb_mod, "_port_reachable", lambda *_a, **_k: True)

    result = LaunchBackendStep().apply(_ctx(ctx_factory, manifest_factory, tmp_path))
    assert result.status is StepStatus.DONE
    # Runs the project's own entry as a module, detached, with PORT exported.
    assert calls[0]["cmd"] == ["uv", "run", "python", "-m", "demo_app.main"]
    assert calls[0]["kwargs"]["start_new_session"] is True
    assert calls[0]["kwargs"]["env"]["PORT"] == "8000"
    # PID file written for down/logs.
    pid_file = tmp_path / SCAFFOLD_DIR / "backend.pid"
    assert json.loads(pid_file.read_text())["pid"] == 4321


def test_apply_failed_when_port_never_opens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
    manifest_factory: Callable[..., Manifest],
) -> None:
    _seed_backend(tmp_path)

    class _Proc:
        pid = 4321

    monkeypatch.setattr(lb_mod.shutil, "which", lambda _name: "/usr/bin/uv")
    monkeypatch.setattr(lb_mod.subprocess, "Popen", lambda cmd, **kw: _Proc())
    monkeypatch.setattr(lb_mod, "_terminate", lambda pid: None)
    # ready_timeout=0 → the readiness loop never iterates → immediate failure.
    result = LaunchBackendStep(ready_timeout=0.0).apply(
        _ctx(ctx_factory, manifest_factory, tmp_path)
    )
    assert result.status is StepStatus.FAILED
    assert "didn't start listening" in (result.error or "")


def test_apply_skips_non_python(
    tmp_path: Path,
    ctx_factory: Callable[..., StepContext],
    manifest_factory: Callable[..., Manifest],
) -> None:
    _seed_backend(tmp_path)
    result = LaunchBackendStep().apply(
        _ctx(ctx_factory, manifest_factory, tmp_path, language="typescript")
    )
    assert result.status is StepStatus.SKIPPED


def test_apply_skips_when_served_by_docker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
    manifest_factory: Callable[..., Manifest],
) -> None:
    # Docker mode + a root Dockerfile → the backend is the compose `app`
    # container, so don't also launch it locally (would clash on the port).
    _seed_backend(tmp_path)
    (tmp_path / "Dockerfile").write_text("FROM python:3.12-slim\n", encoding="utf-8")
    spawned: list[Any] = []
    monkeypatch.setattr(lb_mod.subprocess, "Popen", lambda *a, **k: spawned.append(a))
    result = LaunchBackendStep(served_by_docker=True).apply(
        _ctx(ctx_factory, manifest_factory, tmp_path)
    )
    assert result.status is StepStatus.SKIPPED
    assert "docker container" in (result.detail or "")
    assert spawned == []  # never launched locally


def test_apply_launches_locally_in_docker_mode_without_dockerfile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
    manifest_factory: Callable[..., Manifest],
) -> None:
    # served_by_docker=True but no Dockerfile (backend isn't containerized) →
    # still launch locally.
    _seed_backend(tmp_path, pkg="demo_app")
    calls: list[list[str]] = []

    class _Proc:
        pid = 4321

    def _fake_popen(cmd: list[str], **kwargs: Any) -> _Proc:
        calls.append(cmd)
        return _Proc()

    monkeypatch.setattr(lb_mod.shutil, "which", lambda _name: "/usr/bin/uv")
    monkeypatch.setattr(lb_mod.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(lb_mod, "_port_reachable", lambda *_a, **_k: True)
    result = LaunchBackendStep(served_by_docker=True).apply(
        _ctx(ctx_factory, manifest_factory, tmp_path)
    )
    assert result.status is StepStatus.DONE
    assert calls and calls[0][:4] == ["uv", "run", "python", "-m"]
