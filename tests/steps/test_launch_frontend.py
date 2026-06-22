"""Tests for ``agent_scaffold.steps.launch_frontend``."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from agent_scaffold._scaffold_dir import SCAFFOLD_DIR
from agent_scaffold.orchestrator import StepContext, StepStatus
from agent_scaffold.steps import launch_frontend as lf_mod
from agent_scaffold.steps.launch_frontend import LaunchFrontendStep

# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


def _seed_frontend(tmp_path: Path, package_json: str = '{"name":"f","version":"0"}') -> Path:
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "package.json").write_text(package_json, encoding="utf-8")
    return frontend


def _write_pid_file(project_dir: Path, *, pid: int, port: int = 3000) -> Path:
    pid_file = project_dir / SCAFFOLD_DIR / "frontend.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(
        json.dumps({"pid": pid, "port": port, "started_at": "2026-05-30T00:00:00+00:00"}),
        encoding="utf-8",
    )
    return pid_file


# ---------------------------------------------------------------------------
# detect()
# ---------------------------------------------------------------------------


def test_detect_skipped_when_no_package_json(
    tmp_path: Path, ctx_factory: Callable[..., StepContext]
) -> None:
    result = LaunchFrontendStep().detect(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.SKIPPED
    assert "package.json" in result.reason


def test_skips_local_launch_when_served_by_docker(
    tmp_path: Path, ctx_factory: Callable[..., StepContext]
) -> None:
    # Docker mode + a frontend Dockerfile → the frontend is the compose container,
    # so don't also run pnpm dev locally (would clash on the port).
    frontend = _seed_frontend(tmp_path)
    (frontend / "Dockerfile").write_text("FROM node:20-alpine\n", encoding="utf-8")
    step = LaunchFrontendStep(served_by_docker=True)
    assert step.detect(ctx_factory(project_dir=tmp_path)).status is StepStatus.SKIPPED
    result = step.apply(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.SKIPPED
    assert "docker container" in (result.detail or "")


def test_launches_locally_in_docker_mode_without_frontend_dockerfile(
    tmp_path: Path, ctx_factory: Callable[..., StepContext]
) -> None:
    # served_by_docker=True but no frontend/Dockerfile (frontend isn't
    # containerized) → still launch locally (PENDING, not skipped).
    _seed_frontend(tmp_path)
    result = LaunchFrontendStep(served_by_docker=True).detect(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.PENDING


def test_detect_done_when_pid_file_present_and_alive(
    tmp_path: Path,
    ctx_factory: Callable[..., StepContext],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_frontend(tmp_path)
    _write_pid_file(tmp_path, pid=4242, port=3000)
    monkeypatch.setattr(lf_mod, "_is_alive", lambda _pid: True)
    result = LaunchFrontendStep().detect(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.DONE
    assert "4242" in result.reason


def test_detect_pending_when_pid_file_present_but_dead(
    tmp_path: Path,
    ctx_factory: Callable[..., StepContext],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_frontend(tmp_path)
    _write_pid_file(tmp_path, pid=99999, port=3000)
    monkeypatch.setattr(lf_mod, "_is_alive", lambda _pid: False)
    result = LaunchFrontendStep().detect(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.PENDING
    assert "respawn" in result.reason.lower()


def test_detect_pending_when_no_pid_file(
    tmp_path: Path, ctx_factory: Callable[..., StepContext]
) -> None:
    _seed_frontend(tmp_path)
    result = LaunchFrontendStep().detect(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.PENDING
    assert "no PID file" in result.reason


def test_detect_pending_when_pid_file_malformed(
    tmp_path: Path, ctx_factory: Callable[..., StepContext]
) -> None:
    _seed_frontend(tmp_path)
    pid_file = tmp_path / SCAFFOLD_DIR / "frontend.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text("{not json", encoding="utf-8")
    result = LaunchFrontendStep().detect(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.PENDING


# ---------------------------------------------------------------------------
# apply()
# ---------------------------------------------------------------------------


class _FakeProc:
    def __init__(self, pid: int = 4321) -> None:
        self.pid = pid


def _install_apply_doubles(
    monkeypatch: pytest.MonkeyPatch,
    *,
    has_pnpm: bool = True,
    has_node_modules: bool = False,
    ready: bool = True,
    spawn_raises: BaseException | None = None,
) -> dict[str, Any]:
    """Patch the I/O surface of apply() and return observable side-effect log."""
    calls: dict[str, Any] = {"popen_args": None, "popen_kwargs": None, "terminated": []}

    def fake_which(name: str) -> str | None:
        if name == "pnpm":
            return "/usr/bin/pnpm" if has_pnpm else None
        return f"/usr/bin/{name}"

    monkeypatch.setattr(lf_mod.shutil, "which", fake_which)

    class _PopenStub:
        def __init__(self, args: list[str], **kwargs: Any) -> None:
            calls["popen_args"] = args
            calls["popen_kwargs"] = kwargs
            if spawn_raises is not None:
                raise spawn_raises
            self.pid = 4321

    monkeypatch.setattr(lf_mod.subprocess, "Popen", _PopenStub)

    def fake_terminate(pid: int) -> None:
        calls["terminated"].append(pid)

    monkeypatch.setattr(lf_mod, "_terminate", fake_terminate)

    def fake_wait_for_ready(self: LaunchFrontendStep, log_file: Path) -> tuple[bool, str]:
        return (True, "") if ready else (False, "boom\noom")

    monkeypatch.setattr(LaunchFrontendStep, "_wait_for_ready", fake_wait_for_ready)

    return calls


def test_apply_skipped_when_no_package_json(
    tmp_path: Path, ctx_factory: Callable[..., StepContext]
) -> None:
    result = LaunchFrontendStep().apply(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.SKIPPED


def test_apply_skipped_when_pnpm_missing(
    tmp_path: Path,
    ctx_factory: Callable[..., StepContext],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_frontend(tmp_path)
    _install_apply_doubles(monkeypatch, has_pnpm=False)
    result = LaunchFrontendStep().apply(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.SKIPPED
    assert "pnpm" in result.detail


def test_apply_done_writes_pid_file(
    tmp_path: Path,
    ctx_factory: Callable[..., StepContext],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frontend = _seed_frontend(tmp_path)
    # Pre-create node_modules so we skip the install fast-path.
    (frontend / "node_modules").mkdir()
    _install_apply_doubles(monkeypatch)

    step = LaunchFrontendStep(port=3001)
    result = step.apply(ctx_factory(project_dir=tmp_path))

    assert result.status is StepStatus.DONE
    assert "3001" in result.detail

    pid_file = tmp_path / SCAFFOLD_DIR / "frontend.pid"
    data = json.loads(pid_file.read_text(encoding="utf-8"))
    assert data["pid"] == 4321
    assert data["port"] == 3001
    assert "started_at" in data


def test_apply_failed_when_ready_marker_never_appears(
    tmp_path: Path,
    ctx_factory: Callable[..., StepContext],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frontend = _seed_frontend(tmp_path)
    (frontend / "node_modules").mkdir()
    calls = _install_apply_doubles(monkeypatch, ready=False)

    result = LaunchFrontendStep(ready_timeout=0.01).apply(ctx_factory(project_dir=tmp_path))

    assert result.status is StepStatus.FAILED
    assert "ready marker" in (result.error or "")
    # Failure tears the child down so we don't leak a process.
    assert calls["terminated"] == [4321]
    # And cleans up so the next run doesn't think the dev server is up.
    assert not (tmp_path / SCAFFOLD_DIR / "frontend.pid").is_file()


def test_apply_runs_pnpm_install_when_node_modules_missing(
    tmp_path: Path,
    ctx_factory: Callable[..., StepContext],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_frontend(tmp_path)
    _install_apply_doubles(monkeypatch)

    install_called: dict[str, Any] = {"count": 0}

    def fake_install(self: LaunchFrontendStep, ctx: StepContext, frontend: Path) -> Any:
        install_called["count"] += 1
        return None  # success

    monkeypatch.setattr(LaunchFrontendStep, "_run_pnpm_install", fake_install)

    result = LaunchFrontendStep().apply(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.DONE
    assert install_called["count"] == 1


def test_apply_skips_pnpm_install_when_node_modules_exists(
    tmp_path: Path,
    ctx_factory: Callable[..., StepContext],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frontend = _seed_frontend(tmp_path)
    (frontend / "node_modules").mkdir()
    _install_apply_doubles(monkeypatch)

    def fail(*_a: Any, **_kw: Any) -> Any:
        raise AssertionError("pnpm install must not run when node_modules exists")

    monkeypatch.setattr(LaunchFrontendStep, "_run_pnpm_install", fail)
    result = LaunchFrontendStep().apply(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.DONE


# ---------------------------------------------------------------------------
# fingerprint()
# ---------------------------------------------------------------------------


def test_fingerprint_changes_when_package_json_changes(
    tmp_path: Path, ctx_factory: Callable[..., StepContext]
) -> None:
    _seed_frontend(tmp_path, package_json='{"name":"a","version":"0"}')
    step = LaunchFrontendStep()
    ctx = ctx_factory(project_dir=tmp_path)
    fp_a = step.fingerprint(ctx)

    (tmp_path / "frontend" / "package.json").write_text(
        '{"name":"b","version":"1"}', encoding="utf-8"
    )
    fp_b = step.fingerprint(ctx)

    assert fp_a != fp_b


def test_fingerprint_stable_when_inputs_unchanged(
    tmp_path: Path, ctx_factory: Callable[..., StepContext]
) -> None:
    _seed_frontend(tmp_path)
    step = LaunchFrontendStep()
    ctx = ctx_factory(project_dir=tmp_path)
    assert step.fingerprint(ctx) == step.fingerprint(ctx)


def test_fingerprint_changes_when_port_changes(
    tmp_path: Path, ctx_factory: Callable[..., StepContext]
) -> None:
    _seed_frontend(tmp_path)
    ctx = ctx_factory(project_dir=tmp_path)
    fp_default = LaunchFrontendStep().fingerprint(ctx)
    fp_alt = LaunchFrontendStep(port=4000).fingerprint(ctx)
    assert fp_default != fp_alt


# ---------------------------------------------------------------------------
# _is_alive
# ---------------------------------------------------------------------------


def test_is_alive_handles_invalid_pid() -> None:
    assert lf_mod._is_alive(0) is False
    assert lf_mod._is_alive(-1) is False


# ---------------------------------------------------------------------------
# backend-URL wiring
# ---------------------------------------------------------------------------


def _frontend_stack(cap_id: str, env_vars: list[str]) -> object:
    from agent_scaffold.capabilities import Capability, ResolvedStack

    return ResolvedStack(
        capabilities=[Capability(id=cap_id, kind="frontend", path=Path("/x.md"), env_vars=env_vars)]
    )


def _apply_with_stack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
    stack: object | None,
) -> dict[str, Any]:
    frontend = _seed_frontend(tmp_path)
    (frontend / "node_modules").mkdir()  # skip pnpm install
    calls = _install_apply_doubles(monkeypatch)
    result = LaunchFrontendStep().apply(ctx_factory(project_dir=tmp_path, resolved_stack=stack))
    assert result.status is StepStatus.DONE
    return calls


def test_dev_server_gets_backend_url_for_frontend_capability(
    tmp_path: Path,
    ctx_factory: Callable[..., StepContext],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NEXT_PUBLIC_AGENT_URL", raising=False)
    stack = _frontend_stack("frontend.nextjs-chat", ["NEXT_PUBLIC_AGENT_URL"])
    calls = _apply_with_stack(tmp_path, monkeypatch, ctx_factory, stack)
    env = calls["popen_kwargs"]["env"]
    assert env["NEXT_PUBLIC_AGENT_URL"] == "http://localhost:8000"


def test_dev_server_backend_url_for_streamlit_var(
    tmp_path: Path,
    ctx_factory: Callable[..., StepContext],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AGENT_URL", raising=False)
    stack = _frontend_stack("frontend.streamlit", ["AGENT_URL"])
    calls = _apply_with_stack(tmp_path, monkeypatch, ctx_factory, stack)
    assert calls["popen_kwargs"]["env"]["AGENT_URL"] == "http://localhost:8000"


def test_dev_server_user_backend_url_override_wins(
    tmp_path: Path,
    ctx_factory: Callable[..., StepContext],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEXT_PUBLIC_AGENT_URL", "https://prod.example.com")
    stack = _frontend_stack("frontend.nextjs-chat", ["NEXT_PUBLIC_AGENT_URL"])
    calls = _apply_with_stack(tmp_path, monkeypatch, ctx_factory, stack)
    assert calls["popen_kwargs"]["env"]["NEXT_PUBLIC_AGENT_URL"] == "https://prod.example.com"


def test_dev_server_no_backend_url_without_frontend_capability(
    tmp_path: Path,
    ctx_factory: Callable[..., StepContext],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NEXT_PUBLIC_AGENT_URL", raising=False)
    calls = _apply_with_stack(tmp_path, monkeypatch, ctx_factory, None)
    assert "NEXT_PUBLIC_AGENT_URL" not in calls["popen_kwargs"]["env"]
