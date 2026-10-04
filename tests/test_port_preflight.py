"""Which ports a run will bind, and what the pre-flight does about conflicts.

`_planned_host_ports` decides WHICH ports are worth freeing; getting it wrong
means killing a process for a server that was never going to start. The
pre-flight tests pin the consent model: `--free-ports` acts without asking,
`--yes` alone never does, and a project's own running server is never a
"foreign" conflict.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

import agent_scaffold.cli as cli_mod
from agent_scaffold import ports
from agent_scaffold._scaffold_dir import SCAFFOLD_DIR
from agent_scaffold.cli import StepFlags
from agent_scaffold.manifest import Manifest, write_manifest
from agent_scaffold.orchestrator import (
    DetectionResult,
    Orchestrator,
    StepContext,
    StepResult,
    StepStatus,
    compute_fingerprint,
)

_COMPOSE = "services:\n  redis:\n    ports:\n      - '6379:6379'\n"


@dataclass
class _PortStep:
    id: str
    port: int | None = 8000
    detect_status: StepStatus = StepStatus.PENDING
    description: str = "stub"
    depends_on: tuple[str, ...] = ()
    raises: bool = False

    def detect(self, ctx: StepContext) -> DetectionResult:
        return DetectionResult(self.detect_status, reason="stub")

    def apply(self, ctx: StepContext) -> StepResult:
        return StepResult(StepStatus.DONE, detail="ok")

    def fingerprint(self, ctx: StepContext) -> str:
        return compute_fingerprint({"id": self.id})

    def planned_port(self, ctx: StepContext) -> int | None:
        if self.raises:
            raise RuntimeError("unreadable project state")
        return self.port


@dataclass
class _PlainStep:
    """A step with no planned_port at all (every existing step but two)."""

    id: str
    description: str = "stub"
    depends_on: tuple[str, ...] = ()

    def detect(self, ctx: StepContext) -> DetectionResult:
        return DetectionResult(StepStatus.PENDING, reason="stub")

    def apply(self, ctx: StepContext) -> StepResult:
        return StepResult(StepStatus.DONE, detail="ok")

    def fingerprint(self, ctx: StepContext) -> str:
        return compute_fingerprint({"id": self.id})


def _flags(**overrides: Any) -> StepFlags:
    base: dict[str, Any] = {
        "only": [],
        "skip": [],
        "force": [],
        "retry": [],
        "resume": False,
        "plan_only": False,
        "yes": True,
        "debug": False,
    }
    base.update(overrides)
    return StepFlags(**base)


def _manifest(project_dir: Path) -> Manifest:
    manifest = Manifest(
        recipe="test-recipe",
        language="python",
        framework="none",
        model="claude-test",
        generated_at="2026-10-04T00:00:00+00:00",
    )
    write_manifest(project_dir, manifest)
    return manifest


def _planned(
    project_dir: Path, steps: list[Any], flags: StepFlags, *, use_docker: bool = False
) -> list[tuple[int, str]]:
    manifest = _manifest(project_dir)
    orch = Orchestrator(steps, project_dir, manifest)
    result = cli_mod._planned_host_ports(
        project_dir,
        flags,
        use_docker=use_docker,
        steps=steps,
        rows=orch.plan(),
        orch=orch,
        manifest=manifest,
        resolved_stack=None,
        runtime_env=None,
    )
    return [(item.port, item.source) for item in result]


# ---- which local steps count -----------------------------------------------------


def test_a_pending_backend_and_frontend_are_planned(tmp_path: Path) -> None:
    steps = [_PortStep("launch_backend", 8000), _PortStep("launch_frontend", 3000)]
    assert _planned(tmp_path, steps, _flags()) == [(8000, "backend"), (3000, "frontend")]


def test_steps_without_planned_port_contribute_nothing(tmp_path: Path) -> None:
    assert _planned(tmp_path, [_PlainStep("install_deps")], _flags()) == []


def test_a_done_step_is_not_planned_unless_forced(tmp_path: Path) -> None:
    steps = [_PortStep("launch_backend", 8000, detect_status=StepStatus.DONE)]
    assert _planned(tmp_path, steps, _flags()) == []
    assert _planned(tmp_path, steps, _flags(force=["launch_backend"])) == [(8000, "backend")]


def test_a_skipped_detection_is_not_planned(tmp_path: Path) -> None:
    """plan() labels a SKIPPED detection action='run'; detected is what counts."""
    steps = [_PortStep("launch_backend", 8000, detect_status=StepStatus.SKIPPED)]
    assert _planned(tmp_path, steps, _flags()) == []


def test_skip_and_only_exclude_a_step(tmp_path: Path) -> None:
    steps = [_PortStep("launch_backend", 8000), _PortStep("launch_frontend", 3000)]
    assert _planned(tmp_path, steps, _flags(skip=["launch_backend"])) == [(3000, "frontend")]
    assert _planned(tmp_path, steps, _flags(only=["launch_frontend"])) == [(3000, "frontend")]


def test_a_step_that_will_not_run_binds_nothing(tmp_path: Path) -> None:
    """planned_port None (served by docker, runner missing...): never free a
    port for a server that was never going to start."""
    assert _planned(tmp_path, [_PortStep("launch_backend", None)], _flags()) == []


def test_a_planned_port_failure_never_crashes_the_run(tmp_path: Path) -> None:
    steps = [_PortStep("launch_backend", raises=True), _PortStep("launch_frontend", 3000)]
    assert _planned(tmp_path, steps, _flags()) == [(3000, "frontend")]


def test_an_all_unknown_only_plans_nothing(tmp_path: Path) -> None:
    """A typo'd --only must not free anything before run() rejects it."""
    steps = [_PortStep("launch_backend", 8000)]
    assert _planned(tmp_path, steps, _flags(only=["typo"])) == []


# ---- compose ports -----------------------------------------------------------------


def test_compose_ports_count_only_in_docker_mode(tmp_path: Path) -> None:
    (tmp_path / "docker-compose.yml").write_text(_COMPOSE, encoding="utf-8")
    assert _planned(tmp_path, [], _flags(), use_docker=False) == []
    assert _planned(tmp_path, [], _flags(), use_docker=True) == [(6379, "docker compose")]


def test_compose_ports_dropped_when_docker_up_is_skipped_or_excluded(tmp_path: Path) -> None:
    (tmp_path / "docker-compose.yml").write_text(_COMPOSE, encoding="utf-8")
    steps = [_PortStep("launch_backend", 8000)]
    assert _planned(tmp_path, steps, _flags(skip=["docker_up"]), use_docker=True) == [
        (8000, "backend")
    ]
    assert _planned(tmp_path, steps, _flags(only=["launch_backend"]), use_docker=True) == [
        (8000, "backend")
    ]


def test_a_port_in_both_compose_and_local_is_listed_once_as_compose(tmp_path: Path) -> None:
    (tmp_path / "docker-compose.yml").write_text(
        "services:\n  app:\n    ports:\n      - '8000:8000'\n", encoding="utf-8"
    )
    steps = [_PortStep("launch_backend", 8000)]
    assert _planned(tmp_path, steps, _flags(), use_docker=True) == [(8000, "docker compose")]


# ---- the pre-flight's consent model ---------------------------------------------------


@pytest.fixture
def out(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    buffer = io.StringIO()
    monkeypatch.setattr(
        cli_mod, "console", Console(file=buffer, width=240, force_terminal=False, color_system=None)
    )
    return buffer


class _Spy:
    def __init__(self) -> None:
        self.remediations: list[dict[str, Any]] = []
        self.result = True

    def __call__(self, conflicts: list[ports.PortConflict], **kwargs: Any) -> bool:
        self.remediations.append({"conflicts": conflicts, **kwargs})
        return self.result


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> _Spy:
    recorder = _Spy()
    monkeypatch.setattr(cli_mod, "_remediate_port_conflicts", recorder)
    monkeypatch.setattr(ports, "port_in_use", lambda *_a, **_k: True)
    return recorder


def _scan_as(monkeypatch: pytest.MonkeyPatch, *owners: ports.PortOwner) -> None:
    monkeypatch.setattr(
        ports,
        "scan_conflicts",
        lambda _ports, **_kw: [ports.PortConflict(o.port, o) for o in owners],
    )


def _foreign_container(port: int = 6379) -> ports.PortOwner:
    return ports.PortOwner(
        kind="docker",
        port=port,
        container_id="id1",
        container_name="stale-redis",
        compose_project="other",
    )


def _check(
    tmp_path: Path,
    flags: StepFlags,
    planned: list[cli_mod._PlannedPort],
    *,
    interactive: bool = False,
) -> int | None:
    return cli_mod._preflight_port_check(
        tmp_path, flags, interactive=interactive, runtime_env=None, planned=planned
    )


def test_free_ports_remediates_automatically_and_proceeds(
    tmp_path: Path, spy: _Spy, out: io.StringIO, monkeypatch: pytest.MonkeyPatch
) -> None:
    _scan_as(monkeypatch, _foreign_container())
    monkeypatch.setattr(
        cli_mod.typer, "confirm", lambda *_a, **_k: pytest.fail("free-ports must not prompt")
    )
    planned = [cli_mod._PlannedPort(6379, "docker compose")]
    assert _check(tmp_path, _flags(free_ports=True), planned) is None
    assert len(spy.remediations) == 1
    assert spy.remediations[0]["auto"] is True
    assert "freed - continuing" in out.getvalue()


def test_free_ports_failure_aborts_with_a_clear_message(
    tmp_path: Path, spy: _Spy, out: io.StringIO, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy.result = False
    _scan_as(monkeypatch, _foreign_container())
    planned = [cli_mod._PlannedPort(6379, "docker compose")]
    assert _check(tmp_path, _flags(free_ports=True), planned) == 1
    assert "aborting before docker compose up" in out.getvalue()


def test_yes_alone_never_frees_a_compose_conflict(
    tmp_path: Path, spy: _Spy, out: io.StringIO, monkeypatch: pytest.MonkeyPatch
) -> None:
    _scan_as(monkeypatch, _foreign_container())
    planned = [cli_mod._PlannedPort(6379, "docker compose")]
    assert _check(tmp_path, _flags(yes=True), planned) == 1
    assert spy.remediations == []
    assert "--free-ports" in out.getvalue()


def test_interactive_without_the_flag_gets_guided_remediation(
    tmp_path: Path, spy: _Spy, out: io.StringIO, monkeypatch: pytest.MonkeyPatch
) -> None:
    _scan_as(monkeypatch, _foreign_container())
    planned = [cli_mod._PlannedPort(6379, "docker compose")]
    assert _check(tmp_path, _flags(yes=False), planned, interactive=True) is None
    assert spy.remediations[0].get("auto", False) is False


def test_local_only_conflict_without_the_flag_is_left_to_the_step(
    tmp_path: Path, spy: _Spy, out: io.StringIO, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Aborting a whole `up` before install_deps for a non-essential local
    server would be a regression: the step's own guard reports it instead."""
    owner = ports.PortOwner(kind="process", port=8000, pid=4242, command="python3")
    _scan_as(monkeypatch, owner)
    planned = [cli_mod._PlannedPort(8000, "backend")]
    assert _check(tmp_path, _flags(yes=True), planned) is None
    assert spy.remediations == []


def test_local_conflict_is_offered_guided_remediation_interactively(
    tmp_path: Path, spy: _Spy, out: io.StringIO, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = ports.PortOwner(kind="process", port=8000, pid=4242, command="python3")
    _scan_as(monkeypatch, owner)
    planned = [cli_mod._PlannedPort(8000, "backend")]
    assert _check(tmp_path, _flags(yes=False), planned, interactive=True) is None
    assert len(spy.remediations) == 1


def test_local_conflict_is_freed_under_free_ports_and_names_provisioning_on_failure(
    tmp_path: Path, spy: _Spy, out: io.StringIO, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy.result = False
    owner = ports.PortOwner(kind="process", port=8000, pid=4242, command="python3")
    _scan_as(monkeypatch, owner)
    planned = [cli_mod._PlannedPort(8000, "backend")]
    assert _check(tmp_path, _flags(free_ports=True), planned) == 1
    assert "aborting before provisioning" in out.getvalue()


def test_nothing_busy_means_no_remediation(
    tmp_path: Path, spy: _Spy, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ports, "port_in_use", lambda *_a, **_k: False)
    planned = [cli_mod._PlannedPort(8000, "backend")]
    assert _check(tmp_path, _flags(free_ports=True), planned) is None
    assert spy.remediations == []


# ---- the project's own servers are never "foreign" ---------------------------------------


def _write_pid(project_dir: Path, name: str, pid: int) -> None:
    pid_file = project_dir / SCAFFOLD_DIR / name
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(f'{{"pid": {pid}, "port": 8000}}', encoding="utf-8")


def test_the_projects_own_live_backend_is_not_a_conflict(
    tmp_path: Path, spy: _Spy, out: io.StringIO, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-running `up` must not flag (and --free-ports must not kill) the
    user's own healthy server."""
    _write_pid(tmp_path, "backend.pid", os.getpid())
    owner = ports.PortOwner(kind="process", port=8000, pid=os.getpid(), command="python3")
    _scan_as(monkeypatch, owner)
    planned = [cli_mod._PlannedPort(8000, "backend")]
    assert _check(tmp_path, _flags(free_ports=True), planned) is None
    assert spy.remediations == []
    assert "own running server" in out.getvalue()


def test_a_child_of_the_projects_own_server_is_not_a_conflict(
    tmp_path: Path, spy: _Spy, out: io.StringIO, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`uv run` starts the real listener as a child; the spawn made the
    recorded pid its process-group leader."""
    _write_pid(tmp_path, "backend.pid", os.getpid())
    child = ports.PortOwner(kind="process", port=8000, pid=987654, command="python3")
    _scan_as(monkeypatch, child)
    monkeypatch.setattr(cli_mod.os, "getpgid", lambda _pid: os.getpid())
    planned = [cli_mod._PlannedPort(8000, "backend")]
    assert _check(tmp_path, _flags(free_ports=True), planned) is None
    assert spy.remediations == []


def test_a_dead_pid_file_does_not_protect_a_foreign_holder(
    tmp_path: Path, spy: _Spy, out: io.StringIO, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_pid(tmp_path, "backend.pid", 2**22 + 12345)  # not a live pid
    owner = ports.PortOwner(kind="process", port=8000, pid=4242, command="python3")
    _scan_as(monkeypatch, owner)
    planned = [cli_mod._PlannedPort(8000, "backend")]
    assert _check(tmp_path, _flags(free_ports=True), planned) is None
    assert len(spy.remediations) == 1


def test_the_projects_own_compose_stack_is_not_a_conflict(
    tmp_path: Path, spy: _Spy, out: io.StringIO, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = ports.compose_project_name(tmp_path, None)
    own = ports.PortOwner(
        kind="docker",
        port=6379,
        container_id="id1",
        container_name=f"{project}-redis-1",
        compose_project=project,
    )
    _scan_as(monkeypatch, own)
    planned = [cli_mod._PlannedPort(6379, "docker compose")]
    assert _check(tmp_path, _flags(free_ports=True), planned) is None
    assert spy.remediations == []
    assert "compose will reconcile" in out.getvalue()


def test_is_own_service_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    process = ports.PortOwner(kind="process", port=8000, pid=4242, command="x")
    assert cli_mod._is_own_service(process, {4242}) is True
    assert cli_mod._is_own_service(process, set()) is False
    docker = ports.PortOwner(kind="docker", port=8000, container_id="a")
    assert cli_mod._is_own_service(docker, {4242}) is False
    monkeypatch.setattr(cli_mod.os, "getpgid", lambda _pid: 111)
    assert cli_mod._is_own_service(process, {111}) is True
    assert cli_mod._is_own_service(process, {222}) is False
    monkeypatch.setattr(cli_mod.sys, "platform", "win32")
    assert cli_mod._is_own_service(process, {111}) is False
