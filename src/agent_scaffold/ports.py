"""Port-conflict detection primitives for ``agent-scaffold up``.

Two flows consume this module:

- pre-flight: before ``docker compose up``, enumerate the host ports the
  compose file will bind, check which are already in use, and identify
  their owners;
- recovery: after a failed step, parse the conflicting port(s) out of the
  captured stderr and identify their owners.

The module never prints and never kills anything - it only observes and
proposes. cli.py owns console output, redaction, confirmation, and execution
of any remediation command, whether approved per command by the user or
authorized up front by ``--free-ports``.

Owner lookup asks docker first, then lsof: on macOS the lsof owner of a
container-published port is Docker Desktop's proxy process, and killing
that would take down Docker itself rather than the conflicting container.
For the same reason ``plan_free_ports`` runs every host process through a
protected list (:func:`process_protection_reason`) before it will propose a
kill: when anything is uncertain, it refuses.
"""

from __future__ import annotations

import os
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
    "FreeAction",
    "FreePlan",
    "PortConflict",
    "PortOwner",
    "ProcInfo",
    "ProtectionContext",
    "Refusal",
    "Runner",
    "ancestor_pids",
    "build_protection_context",
    "compose_host_ports",
    "compose_project_name",
    "docker_port_owner",
    "docker_ports_bind_host_port",
    "identify_owner",
    "is_port_conflict",
    "is_protected",
    "lsof_port_owner",
    "lsof_port_pids",
    "manual_commands",
    "parse_conflict_ports",
    "parse_docker_ps_lines",
    "parse_host_port_entry",
    "parse_lsof_fpc",
    "plan_free_ports",
    "port_in_use",
    "proc_info",
    "process_protection_reason",
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


def _parse_docker_ps_rows(text: str) -> list[tuple[str, str, str, str | None]]:
    """(id, name, compose_project, ports) rows; ports is None when absent.

    ``None`` (a legacy three-column row) means "no Ports text to verify
    against", distinct from an empty string a container with no mappings
    would produce.
    """
    rows: list[tuple[str, str, str, str | None]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        container_id = parts[0].strip()
        name = parts[1].strip()
        project = parts[2].strip() if len(parts) > 2 else ""
        ports_text = parts[3].strip() if len(parts) > 3 else None
        if container_id:
            rows.append((container_id, name, project, ports_text))
    return rows


def parse_docker_ps_lines(text: str) -> list[tuple[str, str, str]]:
    """(id, name, compose_project) rows from tab-separated ``docker ps``."""
    return [(cid, name, project) for cid, name, project, _ports in _parse_docker_ps_rows(text)]


def docker_ports_bind_host_port(ports_text: str, port: int) -> bool:
    """True when a ``docker ps`` Ports column binds ``port`` on the HOST.

    ``docker ps --filter publish=<p>`` is documented against the published
    port but matches on the container side in practice for some versions, so
    a container run with ``-p 6380:6379`` can be returned for ``publish=6379``.
    Acting on that row would stop an unrelated container, so the Ports column
    is checked directly. Handles ``0.0.0.0:6379->6379/tcp``, ``[::]:6379->``,
    ``127.0.0.1:8000->8000/tcp`` and ranges (``0.0.0.0:8000-8002->8000-8002/tcp``).
    An exposed-only entry (``6379/tcp``) binds nothing on the host.
    """
    for mapping in ports_text.split(","):
        mapping = mapping.strip()
        if "->" not in mapping:
            continue
        host_side = mapping.split("->", 1)[0]
        # "0.0.0.0:8000-8002", "[::]:6379", "127.0.0.1:8000" -> the part after
        # the last colon is the port or range.
        spec = host_side.rsplit(":", 1)[-1]
        low_text, _, high_text = spec.partition("-")
        try:
            low = int(low_text)
            high = int(high_text) if high_text else low
        except ValueError:
            continue
        if low <= port <= high:
            return True
    return False


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
            '{{.ID}}\t{{.Names}}\t{{.Label "com.docker.compose.project"}}\t{{.Ports}}',
        ]
    )
    # Keep only containers whose Ports column really binds this HOST port; a
    # row without a Ports column (legacy output) is trusted as the filter's.
    rows = [
        (cid, name, project)
        for cid, name, project, ports_text in _parse_docker_ps_rows(out)
        if ports_text is None or docker_ports_bind_host_port(ports_text, port)
    ]
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


def lsof_port_pids(port: int, *, run: Runner = _capture) -> list[tuple[int, str]]:
    """EVERY (pid, command) listening on ``port``.

    ``lsof_port_owner`` reports only the first, but a pre-forking server
    (gunicorn workers) or SO_REUSEPORT can leave several listeners; freeing
    the port means dealing with all of them, and each must be vetted.
    """
    return parse_lsof_fpc(run(["lsof", "-nP", "-Fpc", f"-iTCP:{port}", "-sTCP:LISTEN"]))


def lsof_port_owner(port: int, *, run: Runner = _capture) -> PortOwner | None:
    owners = lsof_port_pids(port, run=run)
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


# ---- freeing ports: protection layer and pure plan ------------------------
#
# ``--free-ports`` authorizes stopping containers and killing host processes
# with no per-item prompt, so the safety lives here, observable and testable
# without spawning anything: every host process must clear
# :func:`process_protection_reason` before a kill is even proposed, and every
# doubt resolves to a refusal.


@dataclass(frozen=True)
class ProcInfo:
    """What ``ps`` says about one process."""

    uid: int
    ppid: int
    # macOS reports the full executable path; Linux the bare (truncated) name.
    comm: str


@dataclass(frozen=True)
class ProtectionContext:
    """Everything the protection check needs, injectable so tests never run ps."""

    own_pids: frozenset[int]
    euid: int
    inspect: Callable[[int], ProcInfo | None]


@dataclass(frozen=True)
class FreeAction:
    """One step of a plan: compose down, stop a container, or kill a process."""

    kind: Literal["compose_down", "docker_stop", "kill_process"]
    ports: tuple[int, ...]
    owner: PortOwner | None = None
    argv: tuple[str, ...] = ()


@dataclass(frozen=True)
class Refusal:
    """A holder the plan will not touch, and why."""

    port: int
    owner: PortOwner
    reason: str


@dataclass(frozen=True)
class FreePlan:
    actions: tuple[FreeAction, ...]
    refusals: tuple[Refusal, ...]


def proc_info(pid: int, *, run: Runner = _capture) -> ProcInfo | None:
    """uid / parent pid / command of ``pid`` via ``ps`` (macOS and Linux), or None."""
    out = run(["ps", "-o", "uid=,ppid=,comm=", "-p", str(pid)]).strip()
    parts = out.split(None, 2)
    if len(parts) < 3:
        return None
    try:
        return ProcInfo(uid=int(parts[0]), ppid=int(parts[1]), comm=parts[2].strip())
    except ValueError:
        return None


def ancestor_pids(*, run: Runner = _capture, limit: int = 64) -> frozenset[int]:
    """This process, its parent, and every ancestor above it.

    The parent is always included even if ``ps`` is unusable, so the user's
    shell (or the REPL hosting this run) is never a kill candidate. The walk
    is bounded and stops at pid 1.
    """
    pids = {os.getpid(), os.getppid()}
    current = os.getppid()
    for _ in range(limit):
        if current <= 1:
            break
        info = proc_info(current, run=run)
        if info is None:
            break
        parent = info.ppid
        # pid <= 1 is refused on its own in process_protection_reason, so the
        # walk stops before init rather than recording it as an ancestor.
        if parent <= 1 or parent in pids:
            break
        pids.add(parent)
        current = parent
    return frozenset(pids)


def build_protection_context(*, run: Runner = _capture) -> ProtectionContext:
    euid = os.geteuid() if sys.platform != "win32" else -1
    return ProtectionContext(
        own_pids=ancestor_pids(run=run),
        euid=euid,
        inspect=lambda pid: proc_info(pid, run=run),
    )


# Never killed, whatever the user asked for. Substring markers over-match on
# purpose (``docke`` also catches lsof's 9-character truncation ``com.docke``
# of Docker Desktop's proxy): refusing an innocent process is the safe error.
_PROTECTED_EXACT = frozenset({"init", "launchd", "systemd", "kernel_task", "sshd", "loginwindow"})
_PROTECTED_MARKERS = (
    "docke",
    "vpnkit",
    "containerd",
    "colima",
    "limactl",
    "orbstack",
    "podman",
    "gvproxy",
    "rancher",
    "qemu-system",
)
_PROTECTED_PATH_MARKERS = ("/docker.app/", "/.colima/", "/.lima/", "/orbstack")


def process_protection_reason(pid: int | None, command: str, ctx: ProtectionContext) -> str | None:
    """Why this process must not be killed, or None when it is fair game.

    Order matters only for the message. Anything uncertain (cannot inspect,
    running as root, Windows) refuses: a skipped kill costs one manual
    command, a wrong kill can take down Docker or the user's shell.
    """
    if pid is None or pid <= 1:
        return "init/system process (pid <= 1)"
    if pid == os.getpid():
        return "this agent-scaffold process"
    if pid in ctx.own_pids:
        return "an ancestor of this process (your shell or the REPL)"
    if sys.platform == "win32":
        return "automatic process kills are not supported on Windows"
    if ctx.euid == 0:
        return "running as root: automatic process kills are disabled"
    info = ctx.inspect(pid)
    if info is None:
        return "could not inspect the process (ps unavailable); refusing to guess"
    if info.uid != ctx.euid:
        return f"owned by uid {info.uid} (you are uid {ctx.euid})"
    name = os.path.basename(info.comm).lower()
    command_lower = command.lower()
    if (
        name in _PROTECTED_EXACT
        or any(marker in name or marker in command_lower for marker in _PROTECTED_MARKERS)
        or any(marker in info.comm.lower() for marker in _PROTECTED_PATH_MARKERS)
    ):
        return "container runtime or system process (Docker Desktop proxy, vpnkit, containerd, ...)"
    return None


def is_protected(owner: PortOwner, ctx: ProtectionContext) -> str | None:
    """Refusal reason for any owner kind; docker containers are vetted in the plan."""
    if owner.kind == "process":
        return process_protection_reason(owner.pid, owner.command, ctx)
    if owner.kind == "unknown":
        return "owner could not be identified (lookup unavailable)"
    return None


# docker allows [a-zA-Z0-9][a-zA-Z0-9_.-]* for names and hex for ids; refusing
# anything else (notably a leading dash) keeps a hostile name out of argv.
_SAFE_CONTAINER_TARGET = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def plan_free_ports(
    conflicts: Sequence[PortConflict],
    *,
    own_project: str,
    compose_down_available: bool,
    protection: ProtectionContext,
) -> FreePlan:
    """Turn conflicts into ordered actions plus refusals. Pure; never raises.

    The project's own compose-project containers collapse into one
    ``docker compose down`` (it clears every port a previous run still
    holds). Other containers are ``docker stop``ped once each however many
    ports they hold. Host processes are killed only when they clear the
    protection check; everything else becomes a :class:`Refusal` carrying the
    reason, so the caller can tell the user exactly what it left alone.
    """
    compose_ports: list[int] = []
    stops: dict[str, tuple[PortOwner, list[int]]] = {}
    kills: dict[int, tuple[PortOwner, list[int]]] = {}
    refusals: list[Refusal] = []

    for conflict in conflicts:
        owner, port = conflict.owner, conflict.port
        if owner.kind == "docker":
            if own_project and owner.compose_project == own_project and compose_down_available:
                if port not in compose_ports:
                    compose_ports.append(port)
                continue
            target = owner.container_name or owner.container_id
            if not target or not _SAFE_CONTAINER_TARGET.match(target):
                refusals.append(
                    Refusal(port, owner, "container name is not safe to pass to docker")
                )
                continue
            key = owner.container_id or target
            stops.setdefault(key, (owner, []))[1].append(port)
            continue
        reason = is_protected(owner, protection)
        if reason is not None:
            refusals.append(Refusal(port, owner, reason))
            continue
        # is_protected cleared a process owner, so pid is set (None is refused).
        assert owner.pid is not None
        kills.setdefault(owner.pid, (owner, []))[1].append(port)

    actions: list[FreeAction] = []
    if compose_ports:
        actions.append(
            FreeAction("compose_down", tuple(compose_ports), None, ("docker", "compose", "down"))
        )
    for owner, owner_ports in stops.values():
        target = owner.container_name or owner.container_id
        actions.append(
            FreeAction("docker_stop", tuple(owner_ports), owner, ("docker", "stop", target))
        )
    for owner, owner_ports in kills.values():
        actions.append(FreeAction("kill_process", tuple(owner_ports), owner))
    return FreePlan(actions=tuple(actions), refusals=tuple(refusals))
