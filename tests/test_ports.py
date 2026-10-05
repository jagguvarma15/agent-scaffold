"""Tests for the port-conflict detection primitives in ``ports.py``."""

from __future__ import annotations

import os
import shutil
import socket
from collections.abc import Sequence
from pathlib import Path

import pytest

from agent_scaffold import ports

DOCKER_BIND_ERROR = (
    "Error response from daemon: failed to set up container networking: "
    "driver failed programming external connectivity on endpoint "
    "research-assistant-redis-1 (d304be87fa96): "
    "Bind for 0.0.0.0:6379 failed: port is already allocated"
)


# ---- is_port_conflict / parse_conflict_ports ----------------------------


def test_is_port_conflict_matches_known_signatures() -> None:
    assert ports.is_port_conflict(DOCKER_BIND_ERROR)
    assert ports.is_port_conflict("[Errno 48] Address already in use")
    assert ports.is_port_conflict("Error: listen EADDRINUSE: address already in use :::3000")
    assert ports.is_port_conflict("Ports are not available: exposing port TCP 0.0.0.0:6379")


def test_is_port_conflict_negative() -> None:
    assert not ports.is_port_conflict("Cannot connect to the Docker daemon")
    assert not ports.is_port_conflict("")


def test_parse_conflict_ports_docker_bind_error() -> None:
    assert ports.parse_conflict_ports(DOCKER_BIND_ERROR) == [6379]


def test_parse_conflict_ports_node_eaddrinuse() -> None:
    text = "Error: listen EADDRINUSE: address already in use :::3000"
    assert ports.parse_conflict_ports(text) == [3000]


def test_parse_conflict_ports_uvicorn_errno_has_no_port() -> None:
    assert ports.parse_conflict_ports("[Errno 48] Address already in use") == []


def test_parse_conflict_ports_multiple_lines_dedupes_in_order() -> None:
    text = (
        "Bind for 0.0.0.0:6379 failed: port is already allocated\n"
        "some unrelated line mentioning :9999\n"
        "Bind for 0.0.0.0:5432 failed: port is already allocated\n"
        "Bind for 0.0.0.0:6379 failed: port is already allocated"
    )
    assert ports.parse_conflict_ports(text) == [6379, 5432]


def test_parse_conflict_ports_ignores_lines_without_signature() -> None:
    assert ports.parse_conflict_ports("connecting to redis at localhost:6379") == []


# ---- parse_host_port_entry ----------------------------------------------


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        ("8000:8000", 8000),
        ("6379:6379", 6379),
        ("127.0.0.1:6379:6379", 6379),
        ("127.0.0.1:6379:6379/tcp", 6379),
        ('"3000:3000"', 3000),
        ({"published": 8000, "target": 8000}, 8000),
        ({"published": "8000"}, 8000),
        ({"target": 8000}, None),
        ("8000", None),  # container-only: docker picks the host port
        (8000, None),
        ("${PORT:-8000}:8000", None),
        ("not-a-port:8000", None),
        ("0:8000", None),
        ("70000:8000", None),
        (None, None),
    ],
)
def test_parse_host_port_entry(entry: object, expected: int | None) -> None:
    assert ports.parse_host_port_entry(entry) == expected


# ---- compose_host_ports --------------------------------------------------


def test_compose_host_ports_short_and_long_syntax(tmp_path: Path) -> None:
    compose = tmp_path / "docker-compose.yml"
    compose.write_text(
        """
services:
  redis:
    image: redis:7
    ports:
      - "6379:6379"
  postgres:
    image: postgres:16
    ports:
      - published: 5432
        target: 5432
  app:
    build: .
    ports:
      - "8000:8000"
      - "6379:6379"
  worker:
    build: .
""",
        encoding="utf-8",
    )
    assert ports.compose_host_ports(compose) == [6379, 5432, 8000]


def test_compose_host_ports_malformed_yaml(tmp_path: Path) -> None:
    compose = tmp_path / "docker-compose.yml"
    compose.write_text("services: [unclosed", encoding="utf-8")
    assert ports.compose_host_ports(compose) == []


def test_compose_host_ports_missing_file(tmp_path: Path) -> None:
    assert ports.compose_host_ports(tmp_path / "nope.yml") == []


def test_compose_host_ports_non_mapping_document(tmp_path: Path) -> None:
    compose = tmp_path / "docker-compose.yml"
    compose.write_text("- just\n- a\n- list\n", encoding="utf-8")
    assert ports.compose_host_ports(compose) == []


# ---- lsof / docker ps parsers --------------------------------------------


def test_parse_lsof_fpc_pairs_pid_with_command() -> None:
    out = "p842\ncredis-server\np1337\ncnode\n"
    assert ports.parse_lsof_fpc(out) == [(842, "redis-server"), (1337, "node")]


def test_parse_lsof_fpc_tolerates_garbage() -> None:
    assert ports.parse_lsof_fpc("") == []
    assert ports.parse_lsof_fpc("garbage\npnotanumber\ncorphan") == []


def test_parse_docker_ps_lines() -> None:
    out = "d304be87fa96\tresearch-assistant-redis-1\tresearch-assistant\nabc123\tlone-redis\t\n"
    assert ports.parse_docker_ps_lines(out) == [
        ("d304be87fa96", "research-assistant-redis-1", "research-assistant"),
        ("abc123", "lone-redis", ""),
    ]


def test_parse_docker_ps_lines_skips_malformed() -> None:
    assert ports.parse_docker_ps_lines("only-one-field\n\n") == []


# ---- compose_project_name -------------------------------------------------


def test_compose_project_name_normalizes_basename() -> None:
    assert ports.compose_project_name(Path("/tmp/Research Assistant!")) == "researchassistant"
    assert ports.compose_project_name(Path("/tmp/research-assistant")) == "research-assistant"
    assert ports.compose_project_name(Path("/tmp/_leading")) == "leading"


def test_compose_project_name_env_override() -> None:
    name = ports.compose_project_name(Path("/tmp/whatever"), env={"COMPOSE_PROJECT_NAME": "Custom"})
    assert name == "custom"


# ---- remediation_argv / manual_commands ------------------------------------


def test_remediation_argv_docker_prefers_name() -> None:
    owner = ports.PortOwner(
        kind="docker", port=6379, container_id="abc", container_name="stale-redis"
    )
    assert ports.remediation_argv(owner) == ["docker", "stop", "stale-redis"]


def test_remediation_argv_docker_falls_back_to_id() -> None:
    owner = ports.PortOwner(kind="docker", port=6379, container_id="abc")
    assert ports.remediation_argv(owner) == ["docker", "stop", "abc"]


def test_remediation_argv_process() -> None:
    owner = ports.PortOwner(kind="process", port=6379, pid=842, command="redis-server")
    assert ports.remediation_argv(owner) == ["kill", "842"]


def test_remediation_argv_unknown() -> None:
    assert ports.remediation_argv(ports.PortOwner(kind="unknown", port=6379)) is None


def test_manual_commands_mention_port() -> None:
    cmds = ports.manual_commands(6379)
    assert "lsof -nP -iTCP:6379 -sTCP:LISTEN" in cmds
    assert "docker ps --filter publish=6379" in cmds


# ---- port_in_use / wait_port_free ------------------------------------------


def test_port_in_use_against_real_listener() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        assert ports.port_in_use(port) is True
    assert ports.port_in_use(port) is False


def test_wait_port_free_returns_immediately_when_free() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    assert ports.wait_port_free(port, timeout=0.5) is True


def test_wait_port_free_times_out_while_held() -> None:
    # Wildcard bind so the bind-probe detects the listener even after its
    # tiny accept backlog fills up and loopback connects start timing out.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        assert ports.wait_port_free(port, timeout=0.3, interval=0.05) is False


# ---- identify_owner ordering ------------------------------------------------


def _fake_runner(outputs: dict[str, str]) -> ports.Runner:
    def run(cmd: Sequence[str]) -> str:
        return outputs.get(cmd[0], "")

    return run


def test_identify_owner_prefers_docker_over_lsof() -> None:
    run = _fake_runner(
        {
            "docker": "abc123\tstale-redis\totherproject\n",
            "lsof": "p842\nccom.docker.backend\n",
        }
    )
    owner = ports.identify_owner(6379, run=run)
    assert owner.kind == "docker"
    assert owner.container_name == "stale-redis"
    assert owner.compose_project == "otherproject"


def test_identify_owner_falls_back_to_lsof() -> None:
    run = _fake_runner({"lsof": "p842\ncredis-server\n"})
    owner = ports.identify_owner(6379, run=run)
    assert owner.kind == "process"
    assert owner.pid == 842
    assert owner.command == "redis-server"


def test_identify_owner_unknown_when_lookups_empty() -> None:
    owner = ports.identify_owner(6379, run=_fake_runner({}))
    assert owner.kind == "unknown"
    assert owner.port == 6379


def test_identify_owner_unknown_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ports.sys, "platform", "win32")
    called: list[str] = []

    def run(cmd: Sequence[str]) -> str:
        called.append(cmd[0])
        return "should not be used"

    owner = ports.identify_owner(6379, run=run)
    assert owner.kind == "unknown"
    assert called == []


def test_scan_conflicts_one_per_port() -> None:
    run = _fake_runner({"lsof": "p842\ncredis-server\n"})
    conflicts = ports.scan_conflicts([6379, 5432], run=run)
    assert [c.port for c in conflicts] == [6379, 5432]
    assert all(c.owner.kind == "process" for c in conflicts)


# ---- docker Ports-column verification -----------------------------------------


@pytest.mark.parametrize(
    ("ports_text", "port", "expected"),
    [
        ("0.0.0.0:6379->6379/tcp, [::]:6379->6379/tcp", 6379, True),
        ("127.0.0.1:8000->8000/tcp", 8000, True),
        ("0.0.0.0:8000-8002->8000-8002/tcp", 8001, True),
        ("0.0.0.0:8000-8002->8000-8002/tcp", 8003, False),
        # `-p 6380:6379`: the CONTAINER port is 6379, the host port is 6380.
        ("0.0.0.0:6380->6379/tcp", 6379, False),
        ("0.0.0.0:6380->6379/tcp", 6380, True),
        # Exposed only: nothing bound on the host.
        ("6379/tcp", 6379, False),
        ("", 6379, False),
        ("garbage->->", 6379, False),
    ],
)
def test_docker_ports_bind_host_port(ports_text: str, port: int, expected: bool) -> None:
    assert ports.docker_ports_bind_host_port(ports_text, port) is expected


def test_docker_port_owner_ignores_container_that_only_maps_container_port() -> None:
    """The wrong-victim case: publish=6379 can return `-p 6380:6379`."""
    run = _fake_runner({"docker": "abc123\tother-redis\tproj\t0.0.0.0:6380->6379/tcp\n"})
    assert ports.docker_port_owner(6379, run=run) is None
    owner = ports.docker_port_owner(6380, run=run)
    assert owner is not None
    assert owner.container_name == "other-redis"


def test_docker_port_owner_picks_the_row_that_binds_the_host_port() -> None:
    run = _fake_runner(
        {
            "docker": (
                "aaa\tdecoy\tproj\t0.0.0.0:6380->6379/tcp\n"
                "bbb\treal\tproj\t0.0.0.0:6379->6379/tcp\n"
            )
        }
    )
    owner = ports.docker_port_owner(6379, run=run)
    assert owner is not None
    assert owner.container_name == "real"


def test_docker_port_owner_trusts_legacy_three_column_rows() -> None:
    run = _fake_runner({"docker": "abc123\tstale-redis\tproj\n"})
    owner = ports.docker_port_owner(6379, run=run)
    assert owner is not None
    assert owner.container_name == "stale-redis"


def test_docker_port_owner_requests_the_ports_column() -> None:
    seen: list[Sequence[str]] = []

    def run(cmd: Sequence[str]) -> str:
        seen.append(cmd)
        return ""

    ports.docker_port_owner(6379, run=run)
    assert any("{{.Ports}}" in part for part in seen[0])


# ---- every listener, not just the first ----------------------------------------


def test_lsof_port_pids_returns_all_listeners() -> None:
    run = _fake_runner({"lsof": "p10\ncgunicorn\np11\ncgunicorn\np12\ncgunicorn\n"})
    assert ports.lsof_port_pids(8000, run=run) == [
        (10, "gunicorn"),
        (11, "gunicorn"),
        (12, "gunicorn"),
    ]


def test_lsof_port_owner_still_reports_only_the_first() -> None:
    run = _fake_runner({"lsof": "p10\ncgunicorn\np11\ncgunicorn\n"})
    owner = ports.lsof_port_owner(8000, run=run)
    assert owner is not None
    assert owner.pid == 10


@pytest.mark.skipif(shutil.which("lsof") is None, reason="lsof not installed")
def test_lsof_port_pids_finds_a_real_listener() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        found = ports.lsof_port_pids(port)
    assert os.getpid() in {pid for pid, _command in found}


# ---- ps-based process inspection ------------------------------------------------


def _ps_runner(table: dict[int, str]) -> ports.Runner:
    def run(cmd: Sequence[str]) -> str:
        assert cmd[0] == "ps"
        return table.get(int(cmd[-1]), "")

    return run


def test_proc_info_parses_uid_ppid_and_command() -> None:
    run = _ps_runner({4242: "  501   300 /usr/bin/python3\n"})
    assert ports.proc_info(4242, run=run) == ports.ProcInfo(501, 300, "/usr/bin/python3")


def test_proc_info_keeps_paths_containing_spaces() -> None:
    comm = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    run = _ps_runner({7: f"501 1 {comm}\n"})
    info = ports.proc_info(7, run=run)
    assert info is not None
    assert info.comm == comm


@pytest.mark.parametrize("output", ["", "garbage", "abc def ghi", "501 300"])
def test_proc_info_none_on_bad_output(output: str) -> None:
    assert ports.proc_info(4242, run=_ps_runner({4242: output})) is None


def test_ancestor_pids_walks_up_to_init(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ports.os, "getpid", lambda: 300)
    monkeypatch.setattr(ports.os, "getppid", lambda: 200)
    run = _ps_runner({200: "501 150 zsh", 150: "501 100 Terminal", 100: "501 1 launchd"})
    assert ports.ancestor_pids(run=run) == {300, 200, 150, 100}


def test_ancestor_pids_includes_parent_even_when_ps_is_broken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ports.os, "getpid", lambda: 300)
    monkeypatch.setattr(ports.os, "getppid", lambda: 200)
    assert ports.ancestor_pids(run=_ps_runner({})) == {300, 200}


def test_ancestor_pids_terminates_on_a_parent_cycle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ports.os, "getpid", lambda: 300)
    monkeypatch.setattr(ports.os, "getppid", lambda: 200)
    run = _ps_runner({200: "501 150 a", 150: "501 200 b"})
    assert ports.ancestor_pids(run=run) == {300, 200, 150}


# ---- the protected list ------------------------------------------------------------


@pytest.fixture
def posix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the platform so these tests mean the same on every CI OS."""
    monkeypatch.setattr(ports.sys, "platform", "linux")


def _ctx(
    table: dict[int, ports.ProcInfo] | None = None,
    *,
    own: frozenset[int] = frozenset({500, 600}),
    euid: int = 501,
) -> ports.ProtectionContext:
    procs = table or {}
    return ports.ProtectionContext(own_pids=own, euid=euid, inspect=lambda pid: procs.get(pid))


def _fair(
    pid: int = 4242, comm: str = "/usr/bin/python3", uid: int = 501
) -> dict[int, ports.ProcInfo]:
    return {pid: ports.ProcInfo(uid=uid, ppid=100, comm=comm)}


def test_ordinary_user_process_is_fair_game(posix: None) -> None:
    ctx = _ctx(_fair())
    assert ports.process_protection_reason(4242, "python3", ctx) is None


@pytest.mark.parametrize("pid", [None, 0, 1])
def test_init_and_missing_pids_are_protected(posix: None, pid: int | None) -> None:
    assert ports.process_protection_reason(pid, "launchd", _ctx()) is not None


def test_this_process_is_protected(posix: None) -> None:
    reason = ports.process_protection_reason(os.getpid(), "python3", _ctx(_fair(os.getpid())))
    assert reason is not None
    assert "this agent-scaffold process" in reason


def test_ancestors_are_protected(posix: None) -> None:
    ctx = _ctx(_fair(600))
    reason = ports.process_protection_reason(600, "zsh", ctx)
    assert reason is not None
    assert "ancestor" in reason


def test_other_users_processes_are_protected(posix: None) -> None:
    reason = ports.process_protection_reason(4242, "nginx", _ctx(_fair(uid=0)))
    assert reason is not None
    assert "uid 0" in reason


def test_nothing_is_killed_when_running_as_root(posix: None) -> None:
    reason = ports.process_protection_reason(4242, "python3", _ctx(_fair(uid=0), euid=0))
    assert reason is not None
    assert "root" in reason


def test_uninspectable_process_is_refused_not_guessed(posix: None) -> None:
    reason = ports.process_protection_reason(4242, "python3", _ctx({}))
    assert reason is not None
    assert "refusing to guess" in reason


def test_windows_never_kills(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ports.sys, "platform", "win32")
    reason = ports.process_protection_reason(4242, "python.exe", _ctx(_fair()))
    assert reason is not None
    assert "Windows" in reason


def test_docker_desktop_proxy_is_protected_by_lsof_truncated_name(posix: None) -> None:
    """lsof truncates the command to 9 chars: Docker Desktop shows as com.docke."""
    ctx = _ctx(_fair(78948, comm="/Applications/Docker.app/Contents/MacOS/com.docker.backend"))
    reason = ports.process_protection_reason(78948, "com.docke", ctx)
    assert reason is not None
    assert "container runtime" in reason


def test_docker_app_path_alone_is_protected(posix: None) -> None:
    """Even a helper whose own name looks innocent is protected by its path."""
    ctx = _ctx(_fair(900, comm="/Applications/Docker.app/Contents/Resources/bin/helper"))
    assert ports.process_protection_reason(900, "helper", ctx) is not None


@pytest.mark.parametrize(
    "comm",
    [
        "/usr/local/bin/vpnkit-bridge",
        "/usr/bin/containerd",
        "/opt/homebrew/bin/colima",
        "/opt/homebrew/bin/limactl",
        "/Applications/OrbStack.app/Contents/MacOS/orbstack",
        "/usr/bin/podman",
        "/usr/bin/gvproxy",
        "/usr/bin/qemu-system-aarch64",
        "/usr/sbin/sshd",
        "/sbin/launchd",
        "systemd",
    ],
)
def test_container_runtimes_and_system_daemons_are_protected(posix: None, comm: str) -> None:
    ctx = _ctx(_fair(2000, comm=comm))
    assert ports.process_protection_reason(2000, os.path.basename(comm), ctx) is not None


def test_is_protected_delegates_by_owner_kind(posix: None) -> None:
    ctx = _ctx(_fair())
    docker = ports.PortOwner(kind="docker", port=6379, container_name="redis")
    process = ports.PortOwner(kind="process", port=8000, pid=4242, command="python3")
    unknown = ports.PortOwner(kind="unknown", port=8000)
    assert ports.is_protected(docker, ctx) is None
    assert ports.is_protected(process, ctx) is None
    assert ports.is_protected(unknown, ctx) is not None


# ---- plan_free_ports (pure) --------------------------------------------------------


def _conflict(port: int, **owner: object) -> ports.PortConflict:
    kind = str(owner.pop("kind", "docker"))
    return ports.PortConflict(
        port=port,
        owner=ports.PortOwner(kind=kind, port=port, **owner),  # type: ignore[arg-type]
    )


def _plan(
    conflicts: list[ports.PortConflict],
    *,
    own_project: str = "myproj",
    compose_down: bool = True,
    ctx: ports.ProtectionContext | None = None,
) -> ports.FreePlan:
    return ports.plan_free_ports(
        conflicts,
        own_project=own_project,
        compose_down_available=compose_down,
        protection=ctx or _ctx(_fair()),
    )


def test_plan_empty_is_empty() -> None:
    plan = _plan([])
    assert plan.actions == ()
    assert plan.refusals == ()


def test_plan_collapses_own_stack_into_one_compose_down(posix: None) -> None:
    plan = _plan(
        [
            _conflict(
                6379, container_id="a1", container_name="myproj-redis-1", compose_project="myproj"
            ),
            _conflict(
                5432, container_id="b2", container_name="myproj-pg-1", compose_project="myproj"
            ),
        ]
    )
    assert [a.kind for a in plan.actions] == ["compose_down"]
    assert plan.actions[0].ports == (6379, 5432)
    assert plan.actions[0].argv == ("docker", "compose", "down")


def test_plan_falls_back_to_docker_stop_when_compose_down_unavailable(posix: None) -> None:
    plan = _plan(
        [
            _conflict(
                6379, container_id="a1", container_name="myproj-redis-1", compose_project="myproj"
            )
        ],
        compose_down=False,
    )
    assert [a.kind for a in plan.actions] == ["docker_stop"]
    assert plan.actions[0].argv == ("docker", "stop", "myproj-redis-1")


def test_plan_stops_a_foreign_container_once_for_all_its_ports(posix: None) -> None:
    plan = _plan(
        [
            _conflict(6333, container_id="q1", container_name="qdrant", compose_project="other"),
            _conflict(6334, container_id="q1", container_name="qdrant", compose_project="other"),
        ]
    )
    assert len(plan.actions) == 1
    assert plan.actions[0].kind == "docker_stop"
    assert plan.actions[0].ports == (6333, 6334)


def test_plan_kills_a_fair_process(posix: None) -> None:
    plan = _plan([_conflict(8000, kind="process", pid=4242, command="python3")])
    assert [a.kind for a in plan.actions] == ["kill_process"]
    assert plan.actions[0].ports == (8000,)
    assert plan.actions[0].owner is not None
    assert plan.actions[0].owner.pid == 4242


def test_plan_refuses_protected_and_unknown_owners(posix: None) -> None:
    plan = _plan(
        [
            _conflict(8000, kind="process", pid=1, command="launchd"),
            _conflict(9000, kind="unknown"),
        ]
    )
    assert plan.actions == ()
    assert [r.port for r in plan.refusals] == [8000, 9000]
    assert all(r.reason for r in plan.refusals)


@pytest.mark.parametrize("name", ["--privileged", "-rf", "a b", "a;b", "$(x)"])
def test_plan_refuses_unsafe_container_targets(posix: None, name: str) -> None:
    plan = _plan([_conflict(6379, container_id="", container_name=name, compose_project="other")])
    assert plan.actions == ()
    assert "not safe" in plan.refusals[0].reason


def test_plan_refuses_a_container_with_no_name_or_id(posix: None) -> None:
    plan = _plan([_conflict(6379, container_id="", container_name="")])
    assert plan.actions == ()
    assert len(plan.refusals) == 1


def test_plan_orders_compose_down_then_stop_then_kill(posix: None) -> None:
    plan = _plan(
        [
            _conflict(8000, kind="process", pid=4242, command="python3"),
            _conflict(6333, container_id="q1", container_name="qdrant", compose_project="other"),
            _conflict(
                6379, container_id="a1", container_name="myproj-redis-1", compose_project="myproj"
            ),
        ]
    )
    assert [a.kind for a in plan.actions] == ["compose_down", "docker_stop", "kill_process"]


def test_plan_does_not_treat_blank_own_project_as_a_match(posix: None) -> None:
    """A container with no compose label must not match an empty project name."""
    plan = _plan(
        [_conflict(6379, container_id="a1", container_name="loose", compose_project="")],
        own_project="",
    )
    assert [a.kind for a in plan.actions] == ["docker_stop"]
