"""``launch_backend`` step: spawn the backend HTTP server in the background.

Sibling to :mod:`agent_scaffold.steps.launch_frontend`. After ``install_deps``
this starts the project's own server entry point detached, writing the PID +
port to ``<project>/.scaffold/backend.pid`` so ``cmd_down`` / ``cmd_logs`` can
manage it, and waits until the port is actually accepting connections.

We don't guess a uvicorn invocation — we run the project's *own* entry the way
its ``main()`` does (``uv run python -m <pkg>.main``), so whatever host/port/
reload the generated code configured is honoured. ``PORT`` is exported in case
the app reads it.

Detection (all SKIP cleanly — a missing server never fails ``up``):

- Non-Python project → SKIPPED (only Python/uvicorn backends are wired today).
- No ``src/<pkg>/main.py`` entry → SKIPPED.
- Entry is an agent-only module (no ``uvicorn``/server markers) → SKIPPED.
- PID file present + process alive → DONE; dead/absent → PENDING.

"Doesn't need config immediately": this runs off ``install_deps`` only, not
``docker_up``/``wire_credentials`` — the HTTP server comes up regardless of
whether backing services are running. If the app's startup *does* require a
service that's down, the readiness wait times out and we surface the log tail.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_scaffold._scaffold_dir import SCAFFOLD_DIR
from agent_scaffold.language_hints import UnknownLanguageError, load_language_hints
from agent_scaffold.orchestrator import (
    DetectionResult,
    StepContext,
    StepLog,
    StepResult,
    StepStatus,
    compute_fingerprint,
)
from agent_scaffold.steps.launch_frontend import (
    _is_alive,
    _iso_now,
    _read_pid_file,
    _safe_read_text,
    _terminate,
)

_DEFAULT_PORT = 8000
_DEFAULT_TIMEOUT = 60.0
_READY_TIMEOUT = 20.0
_READY_POLL_INTERVAL = 0.3
_LOG_TAIL_LINES = 20

# Markers that say the entry module actually serves HTTP, as opposed to being an
# agent-only module (which exports ``agent`` but no server). Keep broad.
_SERVER_MARKERS = ("uvicorn", "hypercorn", "gunicorn", "granian", "fastapi", "flask", "starlette")


def _pid_file_path(project_dir: Path) -> Path:
    return project_dir / SCAFFOLD_DIR / "backend.pid"


def _log_file_path(project_dir: Path) -> Path:
    return project_dir / SCAFFOLD_DIR / "backend.log"


def _backend_entry(project_dir: Path) -> Path | None:
    """The src-layout backend entry module, or ``None``.

    Convention from ``languages/python.yaml`` ``entry_point``:
    ``src/<pkg>/main.py``. Returns the first match (sorted) so behaviour is
    deterministic when a project somehow ships more than one.
    """
    matches = sorted(project_dir.glob("src/*/main.py"))
    return matches[0] if matches else None


def _entry_is_server(text: str) -> bool:
    """True if the entry module looks like it serves HTTP (not an agent-only module)."""
    low = text.lower()
    if any(marker in low for marker in _SERVER_MARKERS):
        return True
    # Generic runnable server: a __main__ block that calls something `.run(`.
    return "__main__" in text and (".run(" in text or "serve(" in low)


def _module_for(entry: Path) -> str:
    """``src/<pkg>/main.py`` → ``<pkg>.main`` (importable after ``uv sync``)."""
    return f"{entry.parent.name}.main"


def _default_port(language: str) -> int:
    try:
        hints = load_language_hints(language)
    except UnknownLanguageError:
        return _DEFAULT_PORT
    raw = hints.get("default_port")
    return raw if isinstance(raw, int) and raw > 0 else _DEFAULT_PORT


def _port_reachable(port: int, *, timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


@dataclass
class LaunchBackendStep:
    """Spawn the backend HTTP server as a detached background process."""

    id: str = "launch_backend"
    description: str = "Start backend server in the background"
    depends_on: tuple[str, ...] = ("install_deps",)
    # In docker mode the backend runs as the compose `app` container, so we skip
    # the local launch (set by default_steps_for when --docker is chosen).
    served_by_docker: bool = False
    timeout: float = _DEFAULT_TIMEOUT
    ready_timeout: float = _READY_TIMEOUT
    troubleshoot: dict[str, str] = field(
        default_factory=lambda: {
            "Address already in use": (
                "the backend port is taken — stop the process on it "
                "(`lsof -i :<port>`) or `agent-scaffold down`, then retry"
            ),
            "ModuleNotFoundError": (
                "deps not synced — run `agent-scaffold up --retry install_deps` first"
            ),
        }
    )

    # ---- detection ----------------------------------------------------

    def detect(self, ctx: StepContext) -> DetectionResult:
        skip = self._skip_reason(ctx)
        if skip is not None:
            return DetectionResult(StepStatus.SKIPPED, reason=skip)
        data = _read_pid_file(_pid_file_path(ctx.project_dir))
        if data is None:
            return DetectionResult(StepStatus.PENDING, reason="no PID file — will start the server")
        pid = int(data["pid"])
        if _is_alive(pid):
            port = data.get("port", _default_port(ctx.manifest.language))
            return DetectionResult(StepStatus.DONE, reason=f"backend live (pid={pid}, port={port})")
        return DetectionResult(
            StepStatus.PENDING, reason=f"PID {pid} from stale file is dead — will respawn"
        )

    # ---- apply --------------------------------------------------------

    def apply(self, ctx: StepContext) -> StepResult:
        skip = self._skip_reason(ctx)
        if skip is not None:
            return StepResult(StepStatus.SKIPPED, detail=skip)
        if shutil.which("uv") is None:
            return StepResult(StepStatus.SKIPPED, detail="uv not on PATH — can't run the backend")

        project_dir = ctx.project_dir
        entry = _backend_entry(project_dir)
        assert entry is not None  # guaranteed by _skip_reason
        module = _module_for(entry)
        port = _default_port(ctx.manifest.language)

        pid_file = _pid_file_path(project_dir)
        stale = _read_pid_file(pid_file)
        if stale is not None and not _is_alive(int(stale["pid"])):
            pid_file.unlink(missing_ok=True)

        log_file = _log_file_path(project_dir)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text("", encoding="utf-8")

        spawn = self._spawn(project_dir, module, port, log_file, runtime_env=ctx.runtime_env)
        if isinstance(spawn, StepResult):
            return spawn
        pid, started_at = spawn

        if not self._wait_for_port(port):
            tail = "\n".join(_safe_read_text(log_file).splitlines()[-_LOG_TAIL_LINES:])
            _terminate(pid)
            log_file.unlink(missing_ok=True)
            return StepResult(
                StepStatus.FAILED,
                error=(
                    f"backend didn't start listening on port {port} within "
                    f"{self.ready_timeout:.0f}s (run `agent-scaffold up` for backing services "
                    "if it needs them)"
                ),
                stderr_tail=tail,
            )

        self._write_pid_file(pid_file, pid=pid, port=port, started_at=started_at)
        ctx.emit(
            StepLog(
                step_id=self.id,
                line=f"backend ready at http://localhost:{port} (pid={pid})",
                stream="stdout",
            )
        )
        return StepResult(StepStatus.DONE, detail=f"server live at http://localhost:{port}")

    # ---- fingerprint --------------------------------------------------

    def fingerprint(self, ctx: StepContext) -> str:
        entry = _backend_entry(ctx.project_dir)
        entry_sha = (
            hashlib.sha256(entry.read_bytes()).hexdigest() if entry and entry.is_file() else None
        )
        return compute_fingerprint(
            {
                "entry_sha": entry_sha,
                "module": _module_for(entry) if entry else None,
                "port": _default_port(ctx.manifest.language),
            }
        )

    # ---- internals ----------------------------------------------------

    def _skip_reason(self, ctx: StepContext) -> str | None:
        """Return a SKIP reason, or ``None`` if the backend should launch."""
        # Docker mode: the backend is the compose `app` container (built from the
        # root Dockerfile), so don't also start it locally — that would clash on
        # the port. A docker mode with no Dockerfile still launches locally.
        if self.served_by_docker and (ctx.project_dir / "Dockerfile").is_file():
            return "backend runs in the docker container (docker mode)"
        if ctx.manifest.language != "python":
            return (
                f"backend auto-start supports Python/uvicorn for now (not {ctx.manifest.language})"
            )
        entry = _backend_entry(ctx.project_dir)
        if entry is None:
            return "no src/<pkg>/main.py backend entry"
        if not _entry_is_server(_safe_read_text(entry)):
            return "backend entry is an agent module — no HTTP server to start"
        return None

    def _spawn(
        self,
        project_dir: Path,
        module: str,
        port: int,
        log_file: Path,
        runtime_env: dict[str, str] | None = None,
    ) -> tuple[int, str] | StepResult:
        """Spawn ``uv run python -m <module>`` detached. Returns ``(pid, iso)`` or FAILED."""
        try:
            log_fh = log_file.open("a", encoding="utf-8")
        except OSError as exc:
            return StepResult(StepStatus.FAILED, error=f"could not open backend.log: {exc}")
        try:
            base_env = runtime_env if runtime_env is not None else dict(os.environ)
            popen_kwargs: dict[str, Any] = {
                "cwd": str(project_dir),
                "stdout": log_fh,
                "stderr": subprocess.STDOUT,
                "stdin": subprocess.DEVNULL,
                "env": {**base_env, "PORT": str(port)},
            }
            if os.name == "nt":
                popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            else:
                popen_kwargs["start_new_session"] = True
            proc = subprocess.Popen(  # noqa: S603 — list-form, shell=False
                ["uv", "run", "python", "-m", module],
                **popen_kwargs,
            )
        except (OSError, FileNotFoundError) as exc:
            log_fh.close()
            return StepResult(StepStatus.FAILED, error=f"could not spawn backend: {exc}")
        finally:
            try:
                log_fh.close()
            except OSError:
                pass
        return proc.pid, _iso_now()

    def _wait_for_port(self, port: int) -> bool:
        """Poll until the port accepts connections or ``ready_timeout`` elapses."""
        deadline = time.monotonic() + self.ready_timeout
        while time.monotonic() < deadline:
            if _port_reachable(port, timeout=_READY_POLL_INTERVAL):
                return True
            time.sleep(_READY_POLL_INTERVAL)
        return False

    def _write_pid_file(self, path: Path, *, pid: int, port: int, started_at: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps({"pid": pid, "port": port, "started_at": started_at}, indent=2)
        path.write_text(body + "\n", encoding="utf-8")
        try:
            os.chmod(path, 0o644)
        except OSError:
            pass


__all__ = ["LaunchBackendStep"]
