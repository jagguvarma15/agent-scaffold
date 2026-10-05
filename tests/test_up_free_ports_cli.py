"""`agent-scaffold up --free-ports` end to end through the CLI.

The consent model is the point: the flag (or the persisted setting) frees
ports with no prompt, `--yes` alone never does, and a protected holder is
never killed whatever the flags say. Every side effect is faked - no test
runs docker or signals a real process.
"""

from __future__ import annotations

import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import agent_scaffold.cli as cli_mod
from agent_scaffold import ports
from agent_scaffold.cli import app
from agent_scaffold.discovery import Recipe
from agent_scaffold.manifest import Manifest, write_manifest
from agent_scaffold.orchestrator import (
    DetectionResult,
    StepContext,
    StepResult,
    StepStatus,
    compute_fingerprint,
)

BIND_ERROR = "Bind for 0.0.0.0:6379 failed: port is already allocated"


@pytest.fixture(autouse=True)
def _isolated_setting(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """The persisted free_ports default must come from the test, never from the
    developer's real ~/.config/agent-scaffold/config.toml or shell env."""
    monkeypatch.delenv("AGENT_SCAFFOLD_FREE_PORTS", raising=False)
    missing = tmp_path_factory.mktemp("cfg") / "config.toml"
    monkeypatch.setenv("AGENT_SCAFFOLD_CONFIG_PATH", str(missing))
    monkeypatch.setattr(ports.sys, "platform", "linux")


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def project(tmp_path: Path) -> Path:
    write_manifest(
        tmp_path,
        Manifest(
            recipe="test-recipe",
            language="python",
            framework="none",
            model="claude-test",
            generated_at="2026-10-04T00:00:00+00:00",
        ),
    )
    (tmp_path / "docker-compose.yml").write_text(
        'services:\n  redis:\n    image: redis:7\n    ports:\n      - "6379:6379"\n',
        encoding="utf-8",
    )
    return tmp_path


@dataclass
class _Step:
    id: str
    description: str = "stub"
    depends_on: tuple[str, ...] = ()
    fail_first: bool = False
    on_fail: Any = None
    apply_calls: int = field(default=0, init=False)

    def detect(self, ctx: StepContext) -> DetectionResult:
        return DetectionResult(StepStatus.PENDING, reason="stub")

    def apply(self, ctx: StepContext) -> StepResult:
        self.apply_calls += 1
        if self.fail_first and self.apply_calls == 1:
            if self.on_fail is not None:
                self.on_fail()
            return StepResult(StepStatus.FAILED, error="compose failed", stderr_tail=BIND_ERROR)
        return StepResult(StepStatus.DONE, detail="ok")

    def fingerprint(self, ctx: StepContext) -> str:
        return compute_fingerprint({"id": self.id})


class _World:
    """A port that stays busy until something frees it, and records how."""

    def __init__(self) -> None:
        self.busy = True
        self.ran: list[list[str]] = []
        self.kills: list[tuple[int, int]] = []

    def run(self, argv: list[str], *, cwd: Path | None = None) -> bool:
        self.ran.append(list(argv))
        self.busy = False
        return True

    def kill(self, pid: int, sig: int) -> None:
        self.kills.append((pid, sig))
        self.busy = False


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> _World:
    state = _World()
    monkeypatch.setattr(ports, "port_in_use", lambda *_a, **_k: state.busy)
    monkeypatch.setattr(ports, "wait_port_free", lambda *_a, **_k: not state.busy)
    monkeypatch.setattr(cli_mod, "_run_remediation_command", state.run)
    monkeypatch.setattr(cli_mod.os, "kill", state.kill)
    monkeypatch.setattr(
        "agent_scaffold.steps.docker_up.docker_available", lambda **_k: (True, "ok")
    )
    monkeypatch.setattr(
        cli_mod,
        "_resolve_recipe_silently",
        lambda _slug: Recipe(slug="test-recipe", title="T", path=Path("/x.md")),
    )
    # Nothing here may ever ask: the flag is the consent.
    monkeypatch.setattr(cli_mod.typer, "confirm", lambda *_a, **_k: pytest.fail("must not prompt"))
    monkeypatch.setattr(
        cli_mod, "_confirm_remediation_command", lambda *_a, **_k: pytest.fail("must not prompt")
    )
    return state


def _install(monkeypatch: pytest.MonkeyPatch, *steps: Any) -> None:
    monkeypatch.setattr(cli_mod, "default_steps_for", lambda *_a, **_k: list(steps))


def _foreign_container() -> ports.PortConflict:
    owner = ports.PortOwner(
        kind="docker",
        port=6379,
        container_id="id1",
        container_name="stale-redis",
        compose_project="other",
    )
    return ports.PortConflict(port=6379, owner=owner)


def _foreign_process(pid: int = 4242, command: str = "redis-server") -> ports.PortConflict:
    return ports.PortConflict(
        port=6379, owner=ports.PortOwner(kind="process", port=6379, pid=pid, command=command)
    )


def _scan(monkeypatch: pytest.MonkeyPatch, *conflicts: ports.PortConflict) -> None:
    monkeypatch.setattr(ports, "scan_conflicts", lambda _ports, **_k: list(conflicts))


def _permissive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ports,
        "build_protection_context",
        lambda **_k: ports.ProtectionContext(
            own_pids=frozenset({500}),
            euid=501,
            inspect=lambda _pid: ports.ProcInfo(uid=501, ppid=100, comm="/usr/bin/redis-server"),
        ),
    )


def _up(runner: CliRunner, project: Path, *extra: str, env: dict[str, str] | None = None) -> Any:
    return runner.invoke(app, ["up", str(project), "--docker", "--yes", *extra], env=env)


# ---- the flag is the consent ----------------------------------------------------------


def test_free_ports_stops_a_foreign_container_and_runs_the_plan(
    runner: CliRunner, project: Path, world: _World, monkeypatch: pytest.MonkeyPatch
) -> None:
    step = _Step("docker_up")
    _install(monkeypatch, step)
    _scan(monkeypatch, _foreign_container())

    result = _up(runner, project, "--free-ports")

    assert result.exit_code == 0, result.output
    assert world.ran == [["docker", "stop", "stale-redis"]]
    assert step.apply_calls == 1  # the plan ran once the port was free
    assert "free-ports is on" in result.output


def test_free_ports_kills_a_foreign_process_with_sigterm(
    runner: CliRunner, project: Path, world: _World, monkeypatch: pytest.MonkeyPatch
) -> None:
    step = _Step("docker_up")
    _install(monkeypatch, step)
    _scan(monkeypatch, _foreign_process())
    _permissive(monkeypatch)
    monkeypatch.setattr(ports, "lsof_port_pids", lambda _p, **_k: [(4242, "redis-server")])

    result = _up(runner, project, "--free-ports")

    assert result.exit_code == 0, result.output
    assert world.kills == [(4242, signal.SIGTERM)]
    assert step.apply_calls == 1


def test_a_protected_holder_is_never_killed_even_with_free_ports(
    runner: CliRunner, project: Path, world: _World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Docker Desktop's proxy owns a docker-published port on macOS; killing
    it takes Docker down. The flag must not be able to authorize that."""
    step = _Step("docker_up")
    _install(monkeypatch, step)
    _scan(monkeypatch, _foreign_process(78948, "com.docke"))
    monkeypatch.setattr(
        ports,
        "build_protection_context",
        lambda **_k: ports.ProtectionContext(
            own_pids=frozenset({500}),
            euid=501,
            inspect=lambda _pid: ports.ProcInfo(
                uid=501, ppid=1, comm="/Applications/Docker.app/Contents/MacOS/com.docker.backend"
            ),
        ),
    )

    result = _up(runner, project, "--free-ports")

    assert result.exit_code == 1
    assert world.kills == []
    assert world.ran == []
    assert step.apply_calls == 0  # aborted before provisioning
    assert "left alone" in result.output
    assert "container runtime" in result.output


def test_yes_alone_still_never_frees_anything(
    runner: CliRunner, project: Path, world: _World, monkeypatch: pytest.MonkeyPatch
) -> None:
    step = _Step("docker_up")
    _install(monkeypatch, step)
    _scan(monkeypatch, _foreign_container())

    result = _up(runner, project)

    assert result.exit_code == 1
    assert world.ran == []
    assert world.kills == []
    assert step.apply_calls == 0
    assert "--free-ports" in result.output


# ---- the persisted setting ---------------------------------------------------------------


def test_the_env_setting_acts_as_consent_without_the_flag(
    runner: CliRunner, project: Path, world: _World, monkeypatch: pytest.MonkeyPatch
) -> None:
    step = _Step("docker_up")
    _install(monkeypatch, step)
    _scan(monkeypatch, _foreign_container())

    result = _up(runner, project, env={"AGENT_SCAFFOLD_FREE_PORTS": "1"})

    assert result.exit_code == 0, result.output
    assert world.ran == [["docker", "stop", "stale-redis"]]


def test_no_free_ports_beats_the_persisted_setting(
    runner: CliRunner, project: Path, world: _World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A shell can always switch the destructive default off for one run."""
    step = _Step("docker_up")
    _install(monkeypatch, step)
    _scan(monkeypatch, _foreign_container())

    result = _up(runner, project, "--no-free-ports", env={"AGENT_SCAFFOLD_FREE_PORTS": "1"})

    assert result.exit_code == 1
    assert world.ran == []


def test_a_malformed_setting_warns_and_means_off(
    runner: CliRunner, project: Path, world: _World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A typo must never silently arm a mode that kills processes."""
    step = _Step("docker_up")
    _install(monkeypatch, step)
    _scan(monkeypatch, _foreign_container())

    result = _up(runner, project, env={"AGENT_SCAFFOLD_FREE_PORTS": "maybe"})

    assert result.exit_code == 1
    assert world.ran == []
    assert "Ignoring the free_ports setting" in result.output


# ---- just-in-time recovery -----------------------------------------------------------------


def test_a_port_taken_after_the_preflight_is_freed_and_retried_once(
    runner: CliRunner, project: Path, world: _World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pre-flight saw a free port; something grabbed it before the bind.
    The step fails with docker's bind error, free-ports clears it, one retry."""
    world.busy = False  # free at pre-flight time
    step = _Step("docker_up", fail_first=True, on_fail=lambda: setattr(world, "busy", True))
    _install(monkeypatch, step)
    _scan(monkeypatch, _foreign_container())

    result = _up(runner, project, "--free-ports")

    assert result.exit_code == 0, result.output
    assert step.apply_calls == 2  # failed once, retried once
    assert world.ran == [["docker", "stop", "stale-redis"]]


def test_without_the_flag_the_failure_hint_points_at_free_ports(
    runner: CliRunner, project: Path, world: _World, monkeypatch: pytest.MonkeyPatch
) -> None:
    world.busy = False
    step = _Step("docker_up", fail_first=True, on_fail=lambda: setattr(world, "busy", True))
    _install(monkeypatch, step)
    _scan(monkeypatch, _foreign_container())

    result = _up(runner, project)

    assert result.exit_code == 1
    assert step.apply_calls == 1  # no automatic retry without consent
    assert world.ran == []
    assert "--free-ports" in result.output


# ---- flag resolution --------------------------------------------------------------------------


def test_resolve_free_ports_flag_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    assert cli_mod._resolve_free_ports_flag(True) is True
    assert cli_mod._resolve_free_ports_flag(False, default=True) is False
    assert cli_mod._resolve_free_ports_flag(None, default=True) is True
    assert cli_mod._resolve_free_ports_flag(None, default=False) is False
    monkeypatch.setenv("AGENT_SCAFFOLD_FREE_PORTS", "1")
    assert cli_mod._resolve_free_ports_flag(None) is True
    monkeypatch.setenv("AGENT_SCAFFOLD_FREE_PORTS", "0")
    assert cli_mod._resolve_free_ports_flag(None) is False


def test_autorun_after_new_carries_free_ports_into_the_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_manifest(
        tmp_path,
        Manifest(
            recipe="test-recipe",
            language="python",
            framework="none",
            model="claude-test",
            generated_at="2026-10-04T00:00:00+00:00",
        ),
    )
    captured: dict[str, Any] = {}

    def fake_run_up_inline(**kwargs: Any) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(cli_mod, "_run_up_inline", fake_run_up_inline)
    cli_mod._autorun_after_new(
        project_dir=tmp_path, recipe=None, resolved_stack=None, open_browser=False, free_ports=True
    )
    assert captured["flags"].free_ports is True
    cli_mod._autorun_after_new(
        project_dir=tmp_path, recipe=None, resolved_stack=None, open_browser=False
    )
    assert captured["flags"].free_ports is False
