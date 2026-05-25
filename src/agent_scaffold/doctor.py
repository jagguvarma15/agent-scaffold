"""Read-only environment audit subcommand.

``agent-scaffold doctor`` reports on local tools, (later) auth, and (later)
recipe-declared external services. It never mutates: remediation lives in
``up``. The ``Check`` Protocol + ``CheckResult`` dataclass keep Q2 (auth) and
Q3 (service probes) additive — new checks plug into ``run_checks`` without
touching the runner or the renderer.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

_VERSION_RE = re.compile(r"(\d+)\.(\d+)(?:\.(\d+))?")
_SUBPROCESS_TIMEOUT = 5


class CheckStatus(str, Enum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass(frozen=True)
class CheckResult:
    id: str
    category: str
    status: CheckStatus
    title: str
    detail: str = ""
    fix_hint: str = ""
    explain_topic: str | None = None


class Check(Protocol):
    id: str
    category: str

    def run(self) -> CheckResult: ...


@dataclass
class DoctorReport:
    results: list[CheckResult] = field(default_factory=list)

    @property
    def summary(self) -> dict[str, int]:
        counts = {s.value: 0 for s in CheckStatus}
        for r in self.results:
            counts[r.status.value] += 1
        return counts

    @property
    def exit_code(self) -> int:
        return 1 if any(r.status == CheckStatus.FAIL for r in self.results) else 0


def run_checks(checks: list[Check]) -> DoctorReport:
    return DoctorReport(results=[c.run() for c in checks])


def _parse_version(text: str) -> tuple[int, int, int] | None:
    match = _VERSION_RE.search(text)
    if match is None:
        return None
    major, minor, patch = match.group(1), match.group(2), match.group(3) or "0"
    try:
        return int(major), int(minor), int(patch)
    except ValueError:
        return None


def _run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        shell=False,
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT,
    )


def _py_version() -> tuple[int, int, int]:
    """Indirection so tests can substitute a version without touching ``sys``.

    Monkeypatching ``sys.version_info`` is unsafe: pytest itself reads it
    during teardown and chokes on non-tuple stand-ins.
    """
    v = sys.version_info
    return v.major, v.minor, v.micro


@dataclass
class PythonCheck:
    id: str = "tool.python"
    category: str = "Tools"

    def run(self) -> CheckResult:
        major, minor, micro = _py_version()
        title = f"python {major}.{minor}.{micro}"
        if (major, minor) >= (3, 11):
            return CheckResult(
                id=self.id,
                category=self.category,
                status=CheckStatus.OK,
                title=title,
                detail="need >=3.11",
                explain_topic="python",
            )
        return CheckResult(
            id=self.id,
            category=self.category,
            status=CheckStatus.FAIL,
            title=title,
            detail="python 3.11 or newer is required",
            fix_hint="install Python 3.11+ via pyenv / asdf / brew",
            explain_topic="python",
        )


@dataclass
class _BinaryVersionCheck:
    """Shared scaffolding for `which <bin> && <bin> --version`-style checks.

    Subclasses set the binary name, the version subcommand, the required
    `(major, minor)` floor, the human-readable fix hint, and the
    ``explain_topic`` slug. Concrete classes below configure each tool.
    """

    id: str
    category: str
    binary: str
    version_args: tuple[str, ...]
    min_version: tuple[int, int]
    fix_hint: str
    explain_topic: str

    def _missing(self) -> CheckResult:
        return CheckResult(
            id=self.id,
            category=self.category,
            status=CheckStatus.FAIL,
            title=f"{self.binary} missing",
            detail=f"{self.binary} not found on PATH",
            fix_hint=self.fix_hint,
            explain_topic=self.explain_topic,
        )

    def _from_completed(self, proc: subprocess.CompletedProcess[str]) -> CheckResult:
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode != 0:
            return CheckResult(
                id=self.id,
                category=self.category,
                status=CheckStatus.FAIL,
                title=f"{self.binary} present but `{' '.join(self.version_args)}` failed",
                detail=combined.strip()[:200] or f"exit code {proc.returncode}",
                fix_hint=self.fix_hint,
                explain_topic=self.explain_topic,
            )
        parsed = _parse_version(combined)
        if parsed is None:
            return CheckResult(
                id=self.id,
                category=self.category,
                status=CheckStatus.FAIL,
                title=f"{self.binary} present (version unparseable)",
                detail=combined.strip()[:200],
                fix_hint=self.fix_hint,
                explain_topic=self.explain_topic,
            )
        if parsed[:2] < self.min_version:
            need = f"{self.min_version[0]}.{self.min_version[1]}"
            return CheckResult(
                id=self.id,
                category=self.category,
                status=CheckStatus.FAIL,
                title=f"{self.binary} {parsed[0]}.{parsed[1]}.{parsed[2]}",
                detail=f"need >={need}",
                fix_hint=self.fix_hint,
                explain_topic=self.explain_topic,
            )
        need = f"{self.min_version[0]}.{self.min_version[1]}"
        return CheckResult(
            id=self.id,
            category=self.category,
            status=CheckStatus.OK,
            title=f"{self.binary} {parsed[0]}.{parsed[1]}.{parsed[2]}",
            detail=f"need >={need}",
            explain_topic=self.explain_topic,
        )

    def run(self) -> CheckResult:
        if shutil.which(self.binary) is None:
            return self._missing()
        try:
            proc = _run_cmd([self.binary, *self.version_args])
        except FileNotFoundError:
            return self._missing()
        except subprocess.TimeoutExpired:
            return CheckResult(
                id=self.id,
                category=self.category,
                status=CheckStatus.FAIL,
                title=f"{self.binary} timed out",
                detail=f"`{self.binary} {' '.join(self.version_args)}` exceeded "
                f"{_SUBPROCESS_TIMEOUT}s",
                fix_hint=self.fix_hint,
                explain_topic=self.explain_topic,
            )
        except OSError as exc:
            return CheckResult(
                id=self.id,
                category=self.category,
                status=CheckStatus.FAIL,
                title=f"{self.binary} failed to invoke",
                detail=str(exc),
                fix_hint=self.fix_hint,
                explain_topic=self.explain_topic,
            )
        return self._from_completed(proc)


@dataclass
class UvCheck(_BinaryVersionCheck):
    id: str = "tool.uv"
    category: str = "Tools"
    binary: str = "uv"
    version_args: tuple[str, ...] = ("--version",)
    min_version: tuple[int, int] = (0, 4)
    fix_hint: str = "curl -LsSf https://astral.sh/uv/install.sh | sh"
    explain_topic: str = "uv"


@dataclass
class RuffCheck(_BinaryVersionCheck):
    id: str = "tool.ruff"
    category: str = "Tools"
    binary: str = "ruff"
    version_args: tuple[str, ...] = ("--version",)
    min_version: tuple[int, int] = (0, 6)
    fix_hint: str = "uv tool install ruff"
    explain_topic: str = "ruff"


@dataclass
class DockerCheck:
    """Docker is two-step: client present AND daemon reachable.

    The daemon-not-running case is the #1 source of friction; we surface it
    as ``WARN`` so the user sees that Docker is installed but needs to be
    started, separate from the "not installed at all" ``FAIL``.
    """

    id: str = "tool.docker"
    category: str = "Tools"
    fix_hint_missing: str = (
        "install Docker Desktop or Colima; run `docker info` to verify the daemon"
    )
    fix_hint_daemon: str = "start Docker Desktop or `colima start`"
    explain_topic: str = "docker"
    min_server_version: tuple[int, int] = (24, 0)

    def _missing(self) -> CheckResult:
        return CheckResult(
            id=self.id,
            category=self.category,
            status=CheckStatus.FAIL,
            title="docker missing",
            detail="docker not found on PATH",
            fix_hint=self.fix_hint_missing,
            explain_topic=self.explain_topic,
        )

    def run(self) -> CheckResult:
        if shutil.which("docker") is None:
            return self._missing()
        try:
            proc = _run_cmd(
                ["docker", "version", "--format", "{{.Server.Version}}"],
            )
        except FileNotFoundError:
            return self._missing()
        except subprocess.TimeoutExpired:
            return CheckResult(
                id=self.id,
                category=self.category,
                status=CheckStatus.WARN,
                title="docker installed",
                detail=f"`docker version` exceeded {_SUBPROCESS_TIMEOUT}s; "
                "daemon likely not running",
                fix_hint=self.fix_hint_daemon,
                explain_topic=self.explain_topic,
            )
        except OSError as exc:
            return CheckResult(
                id=self.id,
                category=self.category,
                status=CheckStatus.FAIL,
                title="docker failed to invoke",
                detail=str(exc),
                fix_hint=self.fix_hint_missing,
                explain_topic=self.explain_topic,
            )

        server_output = (proc.stdout or "").strip()
        # `docker version --format '{{.Server.Version}}'` exits non-zero
        # (and writes to stderr) when the daemon is unreachable but the
        # client is fine — surface that as the daemon-down WARN.
        if proc.returncode != 0 or not server_output:
            return CheckResult(
                id=self.id,
                category=self.category,
                status=CheckStatus.WARN,
                title="docker installed",
                detail="daemon not running",
                fix_hint=self.fix_hint_daemon,
                explain_topic=self.explain_topic,
            )

        parsed = _parse_version(server_output)
        if parsed is None:
            return CheckResult(
                id=self.id,
                category=self.category,
                status=CheckStatus.FAIL,
                title="docker present (server version unparseable)",
                detail=server_output[:200],
                fix_hint=self.fix_hint_missing,
                explain_topic=self.explain_topic,
            )
        if parsed[:2] < self.min_server_version:
            need = f"{self.min_server_version[0]}.{self.min_server_version[1]}"
            return CheckResult(
                id=self.id,
                category=self.category,
                status=CheckStatus.FAIL,
                title=f"docker server {parsed[0]}.{parsed[1]}.{parsed[2]}",
                detail=f"need >={need}",
                fix_hint=self.fix_hint_missing,
                explain_topic=self.explain_topic,
            )
        need = f"{self.min_server_version[0]}.{self.min_server_version[1]}"
        return CheckResult(
            id=self.id,
            category=self.category,
            status=CheckStatus.OK,
            title=f"docker server {parsed[0]}.{parsed[1]}.{parsed[2]}",
            detail=f"need >={need}",
            explain_topic=self.explain_topic,
        )


def baseline_checks() -> list[Check]:
    """Return the Q1 baseline check set: python, uv, docker, ruff."""
    return [PythonCheck(), UvCheck(), DockerCheck(), RuffCheck()]


__all__ = [
    "Check",
    "CheckResult",
    "CheckStatus",
    "DockerCheck",
    "DoctorReport",
    "PythonCheck",
    "RuffCheck",
    "UvCheck",
    "baseline_checks",
    "run_checks",
]
