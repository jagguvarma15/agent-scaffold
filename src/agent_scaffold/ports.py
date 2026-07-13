"""Port-conflict detection primitives for ``agent-scaffold up``.

Two flows consume this module:

- pre-flight: before ``docker compose up``, enumerate the host ports the
  compose file will bind, check which are already in use, and identify
  their owners;
- recovery: after a failed step, parse the conflicting port(s) out of the
  captured stderr and identify their owners.

The module never prints and never kills anything - it only observes and
proposes. cli.py owns console output, redaction, per-command confirmation,
and execution of any remediation command the user approves.

Owner lookup asks docker first, then lsof: on macOS the lsof owner of a
container-published port is Docker Desktop's proxy process, and killing
that would take down Docker itself rather than the conflicting container.
"""

from __future__ import annotations

import re
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

__all__ = [
    "PORT_CONFLICT_NEEDLES",
    "PortConflict",
    "PortOwner",
    "Runner",
    "compose_host_ports",
    "compose_project_name",
    "docker_port_owner",
    "identify_owner",
    "is_port_conflict",
    "lsof_port_owner",
    "manual_commands",
    "parse_conflict_ports",
    "parse_docker_ps_lines",
    "parse_host_port_entry",
    "parse_lsof_fpc",
    "port_in_use",
    "remediation_argv",
    "scan_conflicts",
    "wait_port_free",
]

# Injectable subprocess seam: takes an argv, returns stdout ("" on any
# failure). Mirrors docker_up._capture_stdout so tests never spawn docker
# or lsof.
Runner = Callable[[Sequence[str]], str]

# Error signatures that mean "a host port is taken": docker engine, Docker
# Desktop, node (EADDRINUSE), and the generic POSIX message uvicorn emits.
PORT_CONFLICT_NEEDLES: tuple[str, ...] = (
    "port is already allocated",
    "address already in use",
    "eaddrinuse",
    "ports are not available",
)

_PORT_RE = re.compile(r":(\d{1,5})\b")
_LOOKUP_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class PortOwner:
    """Who holds a port: a docker container, a host process, or unknown."""

    kind: Literal["docker", "process", "unknown"]
    port: int
    container_id: str = ""
    container_name: str = ""
    # com.docker.compose.project label, "" when the container is not
    # compose-managed. Lets callers spot this project's own stale stack.
    compose_project: str = ""
    pid: int | None = None
    command: str = ""


@dataclass(frozen=True)
class PortConflict:
    port: int
    owner: PortOwner


# ---- pure parsers (no subprocess) --------------------------------------


def is_port_conflict(text: str) -> bool:
    lowered = text.lower()
    return any(needle in lowered for needle in PORT_CONFLICT_NEEDLES)


def parse_conflict_ports(text: str) -> list[int]:
    """Extract port numbers from lines carrying a conflict signature.

    Covers docker's ``Bind for 0.0.0.0:6379 failed``, Docker Desktop's
    ``Ports are not available: ... 0.0.0.0:6379`` and node's
    ``EADDRINUSE: ... :::3000``. uvicorn's ``[Errno 48] Address already in
    use`` carries no port, so callers must supply their own fallback.
    """
    ports: list[int] = []
    for line in text.splitlines():
        if not is_port_conflict(line):
            continue
        for match in _PORT_RE.finditer(line):
            port = int(match.group(1))
            if 0 < port <= 65535 and port not in ports:
                ports.append(port)
    return ports


def parse_host_port_entry(entry: object) -> int | None:
    """Host port of one compose ``ports:`` entry, or None if it binds none.

    Handles short syntax ("8000:8000", "127.0.0.1:6379:6379/tcp") and long
    syntax ({"published": 8000, ...}). Container-only entries ("8000") and
    unresolved interpolations ("${PORT}:8000") return None.
    """
    if isinstance(entry, dict):
        published = entry.get("published")
        if published is None:
            return None
        return _to_port(str(published))
    if isinstance(entry, int):
        return None  # bare container port, docker picks the host side
    if isinstance(entry, str):
        text = entry.strip().strip("'\"")
        if "${" in text:
            return None
        text = text.split("/", 1)[0]
        parts = text.split(":")
        if len(parts) == 2:
            return _to_port(parts[0])
        if len(parts) == 3:
            return _to_port(parts[1])
        return None
    return None


def _to_port(raw: str) -> int | None:
    try:
        port = int(raw)
    except ValueError:
        return None
    return port if 0 < port <= 65535 else None


def compose_host_ports(compose_path: Path) -> list[int]:
    """All host ports the compose file publishes, deduped, [] on any error."""
    try:
        data = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return []
    if not isinstance(data, dict):
        return []
    services = data.get("services")
    if not isinstance(services, dict):
        return []
    ports: list[int] = []
    for service in services.values():
        if not isinstance(service, dict):
            continue
        entries = service.get("ports")
        if not isinstance(entries, list):
            continue
        for entry in entries:
            port = parse_host_port_entry(entry)
            if port is not None and port not in ports:
                ports.append(port)
    return ports


def parse_lsof_fpc(text: str) -> list[tuple[int, str]]:
    """(pid, command) pairs from ``lsof -Fpc`` machine-format output."""
    owners: list[tuple[int, str]] = []
    pid: int | None = None
    for line in text.splitlines():
        if not line:
            continue
        tag, value = line[0], line[1:]
        if tag == "p":
            try:
                pid = int(value)
            except ValueError:
                pid = None
        elif tag == "c" and pid is not None:
            if (pid, value) not in owners:
                owners.append((pid, value))
            pid = None
    return owners


def parse_docker_ps_lines(text: str) -> list[tuple[str, str, str]]:
    """(id, name, compose_project) rows from tab-separated ``docker ps``."""
    rows: list[tuple[str, str, str]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        container_id = parts[0].strip()
        name = parts[1].strip()
        project = parts[2].strip() if len(parts) > 2 else ""
        if container_id:
            rows.append((container_id, name, project))
    return rows


def compose_project_name(project_dir: Path, env: Mapping[str, str] | None = None) -> str:
    """The compose project name docker would use for this directory.

    COMPOSE_PROJECT_NAME wins when set; otherwise docker's default is the
    directory basename lowercased with invalid characters stripped.
    """
    if env:
        override = env.get("COMPOSE_PROJECT_NAME", "").strip()
        if override:
            return override.lower()
    name = re.sub(r"[^a-z0-9_-]", "", project_dir.name.lower())
    return name.lstrip("_-")


def remediation_argv(owner: PortOwner) -> list[str] | None:
    """The stop/kill command to propose for an owner, None when unknown."""
    if owner.kind == "docker":
        target = owner.container_name or owner.container_id
        return ["docker", "stop", target] if target else None
    if owner.kind == "process" and owner.pid is not None:
        return ["kill", str(owner.pid)]
    return None


def manual_commands(port: int) -> list[str]:
    """Commands the user can run themselves to find and stop a port owner."""
    return [
        f"lsof -nP -iTCP:{port} -sTCP:LISTEN",
        f"docker ps --filter publish={port}",
        "kill <PID>  (or: docker stop <container>)",
    ]


# ---- probes and subprocess seams ----------------------------------------


def port_in_use(port: int) -> bool:
    """True when an active listener holds the port.

    Two probes are needed: a loopback connect catches listeners bound to
    127.0.0.1 (which a SO_REUSEADDR wildcard bind would coexist with on
    macOS), and a wildcard bind-test catches listeners bound to 0.0.0.0.
    SO_REUSEADDR keeps TIME_WAIT sockets from reading as conflicts.
    """
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.25):
            return True
    except OSError:
        pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("", port))
        except OSError:
            return True
    return False


def wait_port_free(port: int, *, timeout: float = 5.0, interval: float = 0.25) -> bool:
    """Poll until the port frees up (True) or the timeout lapses (False)."""
    deadline = time.monotonic() + timeout
    while True:
        if not port_in_use(port):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(interval)


def _capture(cmd: Sequence[str]) -> str:
    """Read-only lookup runner: stdout on success, "" on any failure.

    lsof exits 1 when nothing matches, so stdout is used regardless of the
    exit code. A missing binary (FileNotFoundError) degrades to "".
    """
    try:
        result = subprocess.run(
            list(cmd),
            capture_output=True,
            text=True,
            timeout=_LOOKUP_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout or ""


def docker_port_owner(port: int, *, run: Runner = _capture) -> PortOwner | None:
    out = run(
        [
            "docker",
            "ps",
            "--filter",
            f"publish={port}",
            "--format",
            '{{.ID}}\t{{.Names}}\t{{.Label "com.docker.compose.project"}}',
        ]
    )
    rows = parse_docker_ps_lines(out)
    if not rows:
        return None
    container_id, name, project = rows[0]
    return PortOwner(
        kind="docker",
        port=port,
        container_id=container_id,
        container_name=name,
        compose_project=project,
    )


def lsof_port_owner(port: int, *, run: Runner = _capture) -> PortOwner | None:
    out = run(["lsof", "-nP", "-Fpc", f"-iTCP:{port}", "-sTCP:LISTEN"])
    owners = parse_lsof_fpc(out)
    if not owners:
        return None
    pid, command = owners[0]
    return PortOwner(kind="process", port=port, pid=pid, command=command)


def identify_owner(port: int, *, run: Runner = _capture) -> PortOwner:
    """Best-effort owner lookup: docker first (macOS docker-proxy safety),
    then lsof; kind="unknown" when neither answers or on Windows."""
    if sys.platform == "win32":
        return PortOwner(kind="unknown", port=port)
    owner = docker_port_owner(port, run=run)
    if owner is not None:
        return owner
    owner = lsof_port_owner(port, run=run)
    if owner is not None:
        return owner
    return PortOwner(kind="unknown", port=port)


def scan_conflicts(ports: Iterable[int], *, run: Runner = _capture) -> list[PortConflict]:
    return [PortConflict(port=port, owner=identify_owner(port, run=run)) for port in ports]
