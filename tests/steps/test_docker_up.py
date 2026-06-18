"""Tests for ``agent_scaffold.steps.docker_up``."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from agent_scaffold.discovery import ExternalService
from agent_scaffold.doctor import CheckResult, CheckStatus
from agent_scaffold.orchestrator import StepContext, StepStatus
from agent_scaffold.steps import docker_up as du_mod
from agent_scaffold.steps._subprocess import SubprocessResult
from agent_scaffold.steps.docker_up import DockerUpStep


def _redis_svc() -> ExternalService:
    return ExternalService(
        id="redis",
        env_vars=["REDIS_URL"],
        docker_service="redis",
        probe="redis_ping",
        default_local="localhost:6379",
    )


def test_detect_skipped_when_no_compose_file(
    tmp_path: Path, ctx_factory: Callable[..., StepContext]
) -> None:
    result = DockerUpStep().detect(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.SKIPPED


def test_detect_skipped_when_no_docker_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
    recipe_factory: Callable[..., Any],
    patch_load_recipe: Callable[[Any], None],
) -> None:
    (tmp_path / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")
    svc = ExternalService(id="x")  # no docker_service field
    patch_load_recipe(recipe_factory(external_services=[svc]))
    monkeypatch.setattr(du_mod.shutil, "which", lambda _name: "/usr/bin/docker")
    result = DockerUpStep().detect(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.SKIPPED


def test_detect_skipped_when_docker_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
    recipe_factory: Callable[..., Any],
    patch_load_recipe: Callable[[Any], None],
) -> None:
    (tmp_path / "docker-compose.yml").write_text(
        "services: {redis: {image: redis:7}}\n", encoding="utf-8"
    )
    patch_load_recipe(recipe_factory(external_services=[_redis_svc()]))
    monkeypatch.setattr(du_mod.shutil, "which", lambda _name: None)
    result = DockerUpStep().detect(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.SKIPPED
    assert "docker" in result.reason.lower()


def test_detect_done_when_all_services_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
    recipe_factory: Callable[..., Any],
    patch_load_recipe: Callable[[Any], None],
) -> None:
    (tmp_path / "docker-compose.yml").write_text(
        "services: {redis: {image: redis:7}}\n", encoding="utf-8"
    )
    patch_load_recipe(recipe_factory(external_services=[_redis_svc()]))
    monkeypatch.setattr(du_mod.shutil, "which", lambda _name: "/usr/bin/docker")
    monkeypatch.setattr(
        du_mod,
        "_capture_stdout",
        lambda _cmd, **_kw: "redis\n",
    )
    # _running_services calls stream_subprocess once for exit-code check
    monkeypatch.setattr(
        du_mod,
        "stream_subprocess",
        lambda *_args, **_kw: SubprocessResult(0, "", False, 0.0),
    )
    result = DockerUpStep().detect(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.DONE


def test_detect_pending_when_services_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
    recipe_factory: Callable[..., Any],
    patch_load_recipe: Callable[[Any], None],
) -> None:
    (tmp_path / "docker-compose.yml").write_text(
        "services: {redis: {image: redis:7}}\n", encoding="utf-8"
    )
    patch_load_recipe(recipe_factory(external_services=[_redis_svc()]))
    monkeypatch.setattr(du_mod.shutil, "which", lambda _name: "/usr/bin/docker")
    monkeypatch.setattr(du_mod, "_capture_stdout", lambda _cmd, **_kw: "")
    monkeypatch.setattr(
        du_mod,
        "stream_subprocess",
        lambda *_args, **_kw: SubprocessResult(0, "", False, 0.0),
    )
    result = DockerUpStep().detect(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.PENDING
    assert "redis" in result.reason


def test_apply_skipped_when_daemon_down(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
    recipe_factory: Callable[..., Any],
    patch_load_recipe: Callable[[Any], None],
) -> None:
    (tmp_path / "docker-compose.yml").write_text(
        "services: {redis: {image: redis:7}}\n", encoding="utf-8"
    )
    patch_load_recipe(recipe_factory(external_services=[_redis_svc()]))
    monkeypatch.setattr(du_mod.shutil, "which", lambda _name: "/usr/bin/docker")
    monkeypatch.setattr(
        du_mod,
        "stream_subprocess",
        lambda *_args, **_kw: SubprocessResult(
            exit_code=1,
            stderr_tail="Cannot connect to the Docker daemon",
            timed_out=False,
            duration=0.1,
        ),
    )
    result = DockerUpStep().apply(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.SKIPPED
    assert "daemon" in (result.detail or "").lower()


def test_apply_calls_compose_up_with_service_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
    recipe_factory: Callable[..., Any],
    patch_load_recipe: Callable[[Any], None],
) -> None:
    (tmp_path / "docker-compose.yml").write_text(
        "services: {redis: {image: redis:7}}\n", encoding="utf-8"
    )
    patch_load_recipe(recipe_factory(external_services=[_redis_svc()]))
    monkeypatch.setattr(du_mod.shutil, "which", lambda _name: "/usr/bin/docker")
    calls: list[list[str]] = []

    def fake_stream(cmd: list[str], **_kw: Any) -> SubprocessResult:
        calls.append(cmd)
        return SubprocessResult(0, "", False, 0.1)

    monkeypatch.setattr(du_mod, "stream_subprocess", fake_stream)
    # Healthcheck wait returns immediately as OK.
    monkeypatch.setattr(
        "agent_scaffold.probes.run_probe",
        lambda svc, timeout=5.0: CheckResult(
            id="x", category="x", status=CheckStatus.OK, title="ok"
        ),
    )
    result = DockerUpStep().apply(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.DONE
    assert any(cmd[:4] == ["docker", "compose", "up", "-d"] for cmd in calls)
    assert any("redis" in cmd for cmd in calls if "up" in cmd)


def test_detect_pending_whole_stack_when_no_docker_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
    recipe_factory: Callable[..., Any],
    patch_load_recipe: Callable[[Any], None],
) -> None:
    # Compose file with real services but bare-string external_services (no
    # docker_service) → whole-stack mode, not a skip.
    (tmp_path / "docker-compose.yml").write_text(
        "services:\n  postgres: {}\n  redis: {}\n", encoding="utf-8"
    )
    patch_load_recipe(
        recipe_factory(
            external_services=[ExternalService(id="postgres"), ExternalService(id="redis")]
        )
    )
    monkeypatch.setattr(du_mod.shutil, "which", lambda _name: "/usr/bin/docker")
    # `config --services` lists the stack; `ps` (running) lists nothing.
    monkeypatch.setattr(
        du_mod, "_capture_stdout", lambda cmd, **_kw: "postgres\nredis\n" if "config" in cmd else ""
    )
    monkeypatch.setattr(
        du_mod, "stream_subprocess", lambda *_a, **_kw: SubprocessResult(0, "", False, 0.0)
    )
    result = DockerUpStep().detect(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.PENDING
    assert "compose stack" in result.reason


def test_apply_whole_stack_uses_compose_up_wait(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
    recipe_factory: Callable[..., Any],
    patch_load_recipe: Callable[[Any], None],
) -> None:
    (tmp_path / "docker-compose.yml").write_text(
        "services:\n  postgres: {}\n  redis: {}\n", encoding="utf-8"
    )
    patch_load_recipe(
        recipe_factory(
            external_services=[ExternalService(id="postgres"), ExternalService(id="redis")]
        )
    )
    monkeypatch.setattr(du_mod.shutil, "which", lambda _name: "/usr/bin/docker")
    monkeypatch.setattr(
        du_mod, "_capture_stdout", lambda cmd, **_kw: "postgres\nredis\n" if "config" in cmd else ""
    )
    calls: list[list[str]] = []

    def fake_stream(cmd: list[str], **_kw: Any) -> SubprocessResult:
        calls.append(cmd)
        return SubprocessResult(0, "", False, 0.1)

    monkeypatch.setattr(du_mod, "stream_subprocess", fake_stream)
    result = DockerUpStep().apply(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.DONE
    # Whole stack up with native healthcheck waiting — no per-service names
    # appended (that's the declared path).
    up_cmds = [cmd for cmd in calls if "up" in cmd]
    assert up_cmds and all("--wait" in cmd for cmd in up_cmds)
    assert not any("postgres" in cmd or "redis" in cmd for cmd in up_cmds)


def test_detect_and_apply_skip_when_docker_mode_off(
    tmp_path: Path, ctx_factory: Callable[..., StepContext]
) -> None:
    # Docker is opt-in: enabled=False (the default local mode) → always skip.
    (tmp_path / "docker-compose.yml").write_text("services:\n  redis: {}\n", encoding="utf-8")
    step = DockerUpStep(enabled=False)
    detect = step.detect(ctx_factory(project_dir=tmp_path))
    assert detect.status is StepStatus.SKIPPED
    assert "docker mode off" in detect.reason
    assert step.apply(ctx_factory(project_dir=tmp_path)).status is StepStatus.SKIPPED


def test_docker_available_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(du_mod.shutil, "which", lambda _n: None)
    ok, reason = du_mod.docker_available()
    assert ok is False
    assert reason == "not installed"


def test_docker_available_daemon_down(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(du_mod.shutil, "which", lambda _n: "/usr/bin/docker")
    monkeypatch.setattr(
        du_mod,
        "stream_subprocess",
        lambda *_a, **_kw: SubprocessResult(1, "Cannot connect to the Docker daemon", False, 0.0),
    )
    ok, reason = du_mod.docker_available()
    assert ok is False
    assert "daemon" in reason


def test_docker_available_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(du_mod.shutil, "which", lambda _n: "/usr/bin/docker")
    monkeypatch.setattr(
        du_mod, "stream_subprocess", lambda *_a, **_kw: SubprocessResult(0, "", False, 0.0)
    )
    ok, _reason = du_mod.docker_available()
    assert ok is True


def test_default_steps_for_docker_mode_toggles_steps() -> None:
    from agent_scaffold.manifest import Manifest
    from agent_scaffold.steps import default_steps_for

    m = Manifest(
        recipe="r",
        language="python",
        framework="none",
        model="x",
        generated_at="2026-06-17T00:00:00Z",
    )
    local = {s.id: s for s in default_steps_for(m, None, use_docker=False)}
    docker = {s.id: s for s in default_steps_for(m, None, use_docker=True)}
    # docker_up runs only in docker mode; launch_backend defers to the container only there.
    assert local["docker_up"].enabled is False
    assert docker["docker_up"].enabled is True
    assert local["launch_backend"].served_by_docker is False
    assert docker["launch_backend"].served_by_docker is True


def test_apply_failed_when_healthcheck_times_out(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
    recipe_factory: Callable[..., Any],
    patch_load_recipe: Callable[[Any], None],
) -> None:
    (tmp_path / "docker-compose.yml").write_text(
        "services: {redis: {image: redis:7}}\n", encoding="utf-8"
    )
    patch_load_recipe(recipe_factory(external_services=[_redis_svc()]))
    monkeypatch.setattr(du_mod.shutil, "which", lambda _name: "/usr/bin/docker")
    monkeypatch.setattr(
        du_mod,
        "stream_subprocess",
        lambda *_args, **_kw: SubprocessResult(0, "", False, 0.1),
    )
    # Probe always fails — the wait loop bails on the deadline.
    monkeypatch.setattr(
        "agent_scaffold.probes.run_probe",
        lambda svc, timeout=5.0: CheckResult(
            id="x", category="x", status=CheckStatus.FAIL, title="down"
        ),
    )
    monkeypatch.setattr(du_mod.time, "sleep", lambda _s: None)
    step = DockerUpStep(healthcheck_timeout=0.01)
    result = step.apply(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.FAILED
    assert "redis" in (result.error or "")
