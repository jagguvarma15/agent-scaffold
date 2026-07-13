"""Tests for the port-conflict detection primitives in ``ports.py``."""

from __future__ import annotations

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
