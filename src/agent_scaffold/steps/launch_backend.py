"""``launch_backend`` step: spawn the backend HTTP server in the background.

Sibling to :mod:`agent_scaffold.steps.launch_frontend`. After ``install_deps``
this starts the project's own server entry point detached, writing the PID +
port to ``<project>/.scaffold/backend.pid`` so ``cmd_down`` / ``cmd_logs`` can
manage it, and waits until the port is actually accepting connections.

We find the project's server entry across the conventional files
(``main.py`` / ``app.py`` / ``server.py`` …). A runnable module (a ``__main__``
block that starts the server) is launched as ``uv run python -m <pkg>.<entry>``,
honouring its own host/port/reload; an exported ASGI app with no runner
(``app = FastAPI()`` in ``app.py``) is launched as ``uv run uvicorn
<pkg>.<entry>:app``. ``PORT`` is exported in case the app reads it.

Detection (all SKIP cleanly — a missing server never fails ``up``):

- Non-Python project → SKIPPED (only Python/uvicorn backends are wired today).
- No ``src/<pkg>/`` or top-level ``app/`` ``{main,app,server}.py`` entry → SKIPPED.
- A ``main.py`` that's an agent-only module (no server markers) → SKIPPED.
- PID file present + process alive → DONE; dead/absent → PENDING.

"Doesn't need config immediately": this runs off ``install_deps`` only, not
``docker_up``/``wire_credentials`` — the HTTP server comes up regardless of
whether backing services are running (the resolved ``ANTHROPIC_API_KEY`` is
threaded in via the runtime env, so an agent that builds its client at startup
boots even before ``wire_credentials``). Two failure shapes are distinguished:
the process *crashes* during startup (e.g. a missing key) → reported with its
exit code and the log tail immediately; or it stays up but never binds the port
(a backing service it needs is down) → the readiness wait times out.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
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

# Conventional server entry filenames under a package dir, in priority order.
# A FastAPI ``app`` commonly lives in ``app.py`` next to an agent ``main.py``,
# so we can't just look at ``main.py``.
_ENTRY_CANDIDATES = ("main.py", "app.py", "server.py", "api.py", "asgi.py")

# Top-level package dirs a recipe may use instead of the src layout (the
# research-assistant recipe ships an ``app/`` package, not ``src/<pkg>/``).
_TOP_LEVEL_PACKAGES = ("app", "api", "backend", "server")

# Top-level ASGI/WSGI app assignment, e.g. ``app = FastAPI(...)``.
_ASGI_APP_RE = re.compile(r"^(\w+)\s*=\s*(?:FastAPI|Starlette|Flask|Quart)\b", re.MULTILINE)


def _pid_file_path(project_dir: Path) -> Path:
    return project_dir / SCAFFOLD_DIR / "backend.pid"


def _log_file_path(project_dir: Path) -> Path:
    return project_dir / SCAFFOLD_DIR / "backend.log"


def _candidate_package_dirs(project_dir: Path) -> list[Path]:
    """Importable package dirs that may hold the server entry, priority order.

    Covers both the src layout (``src/<pkg>/``) and a top-level package the
    recipe may ship instead (``app/`` — e.g. the research-assistant recipe).
    src packages come first so a project carrying both keeps prior behaviour.
    """
    # ``p.name.isidentifier()`` skips non-importable dirs like ``research-assistant``
    # (a stray hyphenated sibling of the real ``research_assistant`` package).
    dirs = sorted(p for p in project_dir.glob("src/*") if p.is_dir() and p.name.isidentifier())
    # A real top-level package (``app/__init__.py``) is importable as ``app.<mod>``
    # because ``uv run`` runs with the project root on ``sys.path``.
    for name in _TOP_LEVEL_PACKAGES:
        pkg = project_dir / name
        if pkg.is_dir() and (pkg / "__init__.py").is_file():
            dirs.append(pkg)
    return dirs


def _backend_entry(project_dir: Path) -> Path | None:
    """The backend entry module that serves HTTP, or ``None``.

    Scans the conventional entry files under each candidate package —
    ``src/<pkg>/`` and top-level ``app/`` — and returns the first that looks
    like an HTTP server. Finds an ``app.py``-style layout (a FastAPI ``app``
    separate from an agent ``main.py``), not just ``main.py``.
    """
    for pkg_dir in _candidate_package_dirs(project_dir):
        for name in _ENTRY_CANDIDATES:
            candidate = pkg_dir / name
            if candidate.is_file() and _entry_is_server(_safe_read_text(candidate)):
                return candidate
    return None


def _entry_is_server(text: str) -> bool:
    """True if the entry module looks like it serves HTTP (not an agent-only module)."""
    low = text.lower()
    if any(marker in low for marker in _SERVER_MARKERS):
        return True
    # Generic runnable server: a __main__ block that calls something `.run(`.
    return "__main__" in text and (".run(" in text or "serve(" in low)


def _module_for(entry: Path) -> str:
    """``src/<pkg>/<name>.py`` → ``<pkg>.<name>`` (importable after ``uv sync``)."""
    return f"{entry.parent.name}.{entry.stem}"


def _asgi_app_var(text: str) -> str:
    """The ASGI/WSGI app variable name (``app = FastAPI()`` → ``app``); default ``app``."""
    match = _ASGI_APP_RE.search(text)
    return match.group(1) if match else "app"


def _server_run_command(module: str, text: str, port: int) -> list[str]:
    """The ``uv run`` arguments that start this entry's HTTP server.

    A module that starts the server itself (a ``__main__`` block invoking
    ``uvicorn``/``.run(``) is executed directly (``python -m <module>``), so its
    own host/port/reload config is honoured. An exported ASGI app with no runner
    (``app = FastAPI()`` in ``app.py``) is served with ``uvicorn <module>:<app>``.
    """
    low = text.lower()
    if "__main__" in text and ("uvicorn" in low or ".run(" in text or "serve(" in low):
        return ["python", "-m", module]
    return [
        "uvicorn",
        f"{module}:{_asgi_app_var(text)}",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]


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
            "Could not resolve authentication": (
                "the backend has no Anthropic API key — set ANTHROPIC_API_KEY in "
                "your shell or run `scaffold auth login`, then `agent-scaffold up --resume`"
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
        port = _default_port(ctx.manifest.language)
        run_args = _server_run_command(_module_for(entry), _safe_read_text(entry), port)

        pid_file = _pid_file_path(project_dir)
        stale = _read_pid_file(pid_file)
        if stale is not None and not _is_alive(int(stale["pid"])):
            pid_file.unlink(missing_ok=True)

        log_file = _log_file_path(project_dir)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text("", encoding="utf-8")

        spawn = self._spawn(project_dir, run_args, port, log_file, runtime_env=ctx.runtime_env)
        if isinstance(spawn, StepResult):
            return spawn
        proc, started_at = spawn
        pid = proc.pid

        outcome = self._await_ready(proc, port)
        if outcome != "ready":
            tail = "\n".join(_safe_read_text(log_file).splitlines()[-_LOG_TAIL_LINES:])
            if outcome == "timeout":
                _terminate(pid)  # still hung — kill it; an exited proc is already gone
            log_file.unlink(missing_ok=True)
            return StepResult(
                StepStatus.FAILED,
                error=self._failure_error(outcome, proc.returncode, port),
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
        if _backend_entry(ctx.project_dir) is not None:
            return None  # found an HTTP-server entry (main.py / app.py / …)
        # No server entry. A bare agent module (a main.py with no server) is the
        # common "nothing to serve" case; otherwise there's no entry at all.
        if any((d / "main.py").is_file() for d in _candidate_package_dirs(ctx.project_dir)):
            return "backend entry is an agent module — no HTTP server to start"
        return "no src/<pkg>/ or app/ {main,app,server,api}.py backend entry"

    def _spawn(
        self,
        project_dir: Path,
        run_args: list[str],
        port: int,
        log_file: Path,
        runtime_env: dict[str, str] | None = None,
    ) -> tuple[subprocess.Popen[bytes], str] | StepResult:
        """Spawn ``uv run <run_args>`` detached. Returns ``(proc, iso)`` or FAILED."""
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
                ["uv", "run", *run_args],
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
        return proc, _iso_now()

    def _await_ready(self, proc: subprocess.Popen[bytes], port: int) -> str:
        """Wait for the backend to bind ``port``, exit, or time out.

        Returns ``"ready"`` (the port is accepting connections), ``"exited"``
        (the process died before binding — a startup crash, e.g. a missing API
        key, caught immediately instead of after the full ``ready_timeout``), or
        ``"timeout"`` (still running but never bound — usually a backing service
        it needs is down).
        """
        deadline = time.monotonic() + self.ready_timeout
        while time.monotonic() < deadline:
            if _port_reachable(port, timeout=_READY_POLL_INTERVAL):
                return "ready"
            if proc.poll() is not None:
                # The process is gone. One last port check covers a fast
                # bind-then-exit race; otherwise it crashed during startup.
                if _port_reachable(port, timeout=_READY_POLL_INTERVAL):
                    return "ready"
                return "exited"
            time.sleep(_READY_POLL_INTERVAL)
        return "timeout"

    def _failure_error(self, outcome: str, returncode: int | None, port: int) -> str:
        """Human cause for a non-ready launch — distinguishes crash from timeout."""
        if outcome == "exited":
            code = "?" if returncode is None else str(returncode)
            return (
                f"backend process exited during startup (exit code {code}) before "
                f"binding port {port} — see the log tail below for the cause"
            )
        return (
            f"backend didn't start listening on port {port} within "
            f"{self.ready_timeout:.0f}s (run `agent-scaffold up` for backing services "
            "if it needs them)"
        )

    def _write_pid_file(self, path: Path, *, pid: int, port: int, started_at: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps({"pid": pid, "port": port, "started_at": started_at}, indent=2)
        path.write_text(body + "\n", encoding="utf-8")
        try:
            os.chmod(path, 0o644)
        except OSError:
            pass


__all__ = ["LaunchBackendStep"]
