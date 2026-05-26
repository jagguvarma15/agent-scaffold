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
