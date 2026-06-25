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
    # --build rebuilds the app/frontend images every run so a regenerated backend
    # (e.g. one that just gained POST /chat) isn't served from a stale image.
    assert all("--build" in cmd for cmd in up_cmds)
    assert not any("postgres" in cmd or "redis" in cmd for cmd in up_cmds)


_APP_STACK_COMPOSE = """\
services:
  app:
    build:
      context: .
    ports:
      - "8000:8000"
  postgres:
    image: postgres:16-alpine
  redis:
    image: redis:7-alpine
"""


def _whole_stack_ctx(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recipe_factory: Callable[..., Any],
    patch_load_recipe: Callable[[Any], None],
    *,
    app_state: str,
) -> None:
    """Wire a whole-stack `up` where ``docker compose ps`` reports ``app_state``."""
    (tmp_path / "docker-compose.yml").write_text(_APP_STACK_COMPOSE, encoding="utf-8")
    patch_load_recipe(
        recipe_factory(
            external_services=[ExternalService(id="postgres"), ExternalService(id="redis")]
        )
    )
    monkeypatch.setattr(du_mod.shutil, "which", lambda _name: "/usr/bin/docker")
    monkeypatch.setattr(
        du_mod, "stream_subprocess", lambda *_a, **_kw: SubprocessResult(0, "", False, 0.1)
    )

    def fake_capture(cmd: list[str], **_kw: Any) -> str:
        if "config" in cmd:
            return "app\npostgres\nredis\n"
        if "ps" in cmd:
            return f'[{{"Service": "app", "State": "{app_state}", "ExitCode": 1}}]'
        if "logs" in cmd:
            return (
                'TypeError: "Could not resolve authentication method. Expected one '
                'of api_key, auth_token, or credentials to be set."'
            )
        return ""

    monkeypatch.setattr(du_mod, "_capture_stdout", fake_capture)


def test_apply_whole_stack_fails_when_app_container_exits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
    recipe_factory: Callable[..., Any],
    patch_load_recipe: Callable[[Any], None],
) -> None:
    # The stack comes up (--wait succeeds) but the app container exited on boot.
    _whole_stack_ctx(tmp_path, monkeypatch, recipe_factory, patch_load_recipe, app_state="exited")
    result = DockerUpStep().apply(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.FAILED
    assert "exited during startup" in (result.error or "")
    assert "Could not resolve authentication" in (result.stderr_tail or "")
    # The auth signature maps to the docker-mode suggested fix.
    tail_low = (result.stderr_tail or "").lower()
    matched = [h for needle, h in DockerUpStep().troubleshoot.items() if needle.lower() in tail_low]
    assert any("auth login" in h for h in matched)


def test_apply_whole_stack_done_when_app_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
    recipe_factory: Callable[..., Any],
    patch_load_recipe: Callable[[Any], None],
) -> None:
    _whole_stack_ctx(tmp_path, monkeypatch, recipe_factory, patch_load_recipe, app_state="running")
    result = DockerUpStep().apply(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.DONE


def test_app_service_name_prefers_build_then_conventional(
    tmp_path: Path, ctx_factory: Callable[..., StepContext]
) -> None:
    (tmp_path / "docker-compose.yml").write_text(_APP_STACK_COMPOSE, encoding="utf-8")
    assert DockerUpStep()._app_service_name(ctx_factory(project_dir=tmp_path)) == "app"
    # No build service → fall back to a conventionally named one.
    (tmp_path / "docker-compose.yml").write_text(
        "services:\n  api:\n    image: ghcr.io/acme/api:1\n  redis:\n    image: redis:7\n",
        encoding="utf-8",
    )
    assert DockerUpStep()._app_service_name(ctx_factory(project_dir=tmp_path)) == "api"
    # Only infra images → no app.
    (tmp_path / "docker-compose.yml").write_text(
        "services:\n  postgres:\n    image: postgres:16\n", encoding="utf-8"
    )
    assert DockerUpStep()._app_service_name(ctx_factory(project_dir=tmp_path)) is None


def test_parse_ps_states_handles_array_jsonl_and_garbage() -> None:
    assert du_mod._parse_ps_states('[{"State": "running"}, {"State": "Exited"}]') == {
        "running",
        "exited",
    }
    assert du_mod._parse_ps_states('{"State": "running"}\n{"State": "exited"}\n') == {
        "running",
        "exited",
    }
    assert du_mod._parse_ps_states("") == set()
    assert du_mod._parse_ps_states("not json at all") == set()


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


# --- bring_up (shared by `up` and the --deep-validate docker_up tier) --------


def _app_compose(tmp_path: Path) -> None:
    (tmp_path / "docker-compose.yml").write_text(
        "services:\n  app:\n    build: .\n  redis:\n    image: redis:7\n", encoding="utf-8"
    )


def test_bring_up_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _app_compose(tmp_path)
    monkeypatch.setattr(
        du_mod, "stream_subprocess", lambda *_a, **_kw: SubprocessResult(0, "", False, 0.1)
    )
    monkeypatch.setattr(
        du_mod,
        "_capture_stdout",
        lambda cmd, **_kw: '[{"Service": "app", "State": "running"}]' if "ps" in cmd else "",
    )
    ok, output = du_mod.bring_up(tmp_path)
    assert ok is True
    assert "healthy" in output


def test_bring_up_compose_up_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _app_compose(tmp_path)
    monkeypatch.setattr(
        du_mod,
        "stream_subprocess",
        lambda *_a, **_kw: SubprocessResult(1, "Error: port is already allocated", False, 0.1),
    )
    ok, output = du_mod.bring_up(tmp_path)
    assert ok is False
    assert "failed (exit 1)" in output
    assert "port is already allocated" in output


def test_bring_up_app_crash_on_boot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _app_compose(tmp_path)
    monkeypatch.setattr(
        du_mod, "stream_subprocess", lambda *_a, **_kw: SubprocessResult(0, "", False, 0.1)
    )

    def fake_capture(cmd: list[str], **_kw: Any) -> str:
        if "ps" in cmd:
            return '[{"Service": "app", "State": "exited", "ExitCode": 1}]'
        if "logs" in cmd:
            return "Traceback (most recent call last):\nKeyError: ANTHROPIC_API_KEY"
        return ""

    monkeypatch.setattr(du_mod, "_capture_stdout", fake_capture)
    ok, output = du_mod.bring_up(tmp_path)
    assert ok is False
    assert "exited during startup" in output
    assert "ANTHROPIC_API_KEY" in output
