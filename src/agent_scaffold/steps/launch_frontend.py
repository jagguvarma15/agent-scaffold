"""``launch_frontend`` step: spawn the frontend dev server in the background.

The frontend template ships into ``<project>/frontend/`` but until now the
orchestrator stopped at "files written"; the user had to ``cd frontend && pnpm
install && pnpm dev`` by hand. This step closes that gap: after
``install_deps`` runs, ``launch_frontend`` does the ``pnpm install`` (if needed)
and spawns ``pnpm dev`` detached, writing the PID + port to
``<project>/.scaffold/frontend.pid`` so ``cmd_down`` and ``cmd_logs`` can find
it.

Detection rules:

- No ``frontend/package.json`` → ``SKIPPED`` (recipe doesn't ship a frontend).
- PID file present + process alive → ``DONE``.
- PID file present + process dead → ``PENDING`` (the previous server died, restart).
- No PID file → ``PENDING``.

The fingerprint hashes ``frontend/package.json`` + the resolved frontend
capability id + the chosen port, so a dependency edit or capability swap
invalidates the DONE marker on the next ``--resume``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
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

_DEFAULT_PORT = 3000
_DEFAULT_BACKEND_PORT = 8000
_DEFAULT_TIMEOUT = 60.0
_READY_TIMEOUT = 10.0
_READY_POLL_INTERVAL = 0.25
_LOG_TAIL_LINES = 20
_READY_MARKERS = ("ready in", "Local:", "Local:   http")


def _pid_file_path(project_dir: Path) -> Path:
    return project_dir / SCAFFOLD_DIR / "frontend.pid"


def _log_file_path(project_dir: Path) -> Path:
    return project_dir / SCAFFOLD_DIR / "frontend.log"


def _frontend_dir(project_dir: Path) -> Path:
    return project_dir / "frontend"


def _is_alive(pid: int) -> bool:
    """Best-effort liveness check: ``os.kill(pid, 0)`` is the POSIX idiom.

    Returns ``False`` for dead PIDs, recycled PIDs we no longer own
    (PermissionError), or platforms without ``os.kill`` semantics.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # PID exists but isn't ours — treat as alive; killing it isn't our job.
        return True
    except OSError:
        return False
    return True


def _read_pid_file(path: Path) -> dict[str, Any] | None:
    """Read + minimally validate the PID file. Returns ``None`` on any error."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict) or "pid" not in data:
        return None
    try:
        int(data["pid"])
    except (TypeError, ValueError):
        return None
    return data


def _resolve_frontend_capability_id(ctx: StepContext) -> str | None:
    stack = getattr(ctx, "resolved_stack", None)
    if stack is None:
        return None
    for cap in getattr(stack, "capabilities", []) or []:
        cap_id = getattr(cap, "id", "")
        if cap_id.startswith("frontend."):
            return cap_id
    return None


def _resolve_frontend_env_vars(ctx: StepContext) -> list[str]:
    """The frontend capability's env vars — its backend-URL knobs.

    ``frontend.nextjs-chat`` → ``["NEXT_PUBLIC_AGENT_URL"]``;
    ``frontend.streamlit`` → ``["AGENT_URL"]``. Used to point the dev server at
    the (containerized or local) backend.
    """
    stack = getattr(ctx, "resolved_stack", None)
    if stack is None:
        return []
    for cap in getattr(stack, "capabilities", []) or []:
        if getattr(cap, "id", "").startswith("frontend."):
            return list(getattr(cap, "env_vars", []) or [])
    return []


def _backend_port(language: str) -> int:
    """Host port the backend listens on — the language's ``default_port`` (8000 py)."""
    try:
        hints = load_language_hints(language)
    except UnknownLanguageError:
        return _DEFAULT_BACKEND_PORT
    raw = hints.get("default_port")
    return raw if isinstance(raw, int) and raw > 0 else _DEFAULT_BACKEND_PORT


@dataclass
class LaunchFrontendStep:
    """Spawn the frontend dev server as a detached background process."""

    id: str = "launch_frontend"
    description: str = "Start frontend dev server in the background"
    depends_on: tuple[str, ...] = ("install_deps",)
    port: int = _DEFAULT_PORT
    timeout: float = _DEFAULT_TIMEOUT
    ready_timeout: float = _READY_TIMEOUT
    troubleshoot: dict[str, str] = field(
        default_factory=lambda: {
            "EADDRINUSE": (
                "port already in use — find the process with `lsof -i :<port>` "
                "and stop it, or change `port` on the step"
            ),
            "command not found": (
                "pnpm not installed — `npm install -g pnpm` or `corepack enable`"
            ),
            "ERR_PNPM_NO_LOCKFILE": (
                "frontend has no pnpm-lock.yaml — run `pnpm install` in frontend/ once "
                "to generate it, then re-run `agent-scaffold up --retry launch_frontend`"
            ),
        }
    )

    # ---- detection ----------------------------------------------------

    def detect(self, ctx: StepContext) -> DetectionResult:
        frontend = _frontend_dir(ctx.project_dir)
        if not (frontend / "package.json").is_file():
            return DetectionResult(
                StepStatus.SKIPPED, reason="no frontend/package.json — recipe ships no frontend"
            )
        pid_file = _pid_file_path(ctx.project_dir)
        data = _read_pid_file(pid_file)
        if data is None:
            return DetectionResult(StepStatus.PENDING, reason="no PID file — will spawn dev server")
        pid = int(data["pid"])
        if _is_alive(pid):
            port = data.get("port", self.port)
            return DetectionResult(
                StepStatus.DONE, reason=f"dev server live (pid={pid}, port={port})"
            )
        return DetectionResult(
            StepStatus.PENDING, reason=f"PID {pid} from stale file is dead — will respawn"
        )

    # ---- apply --------------------------------------------------------

    def apply(self, ctx: StepContext) -> StepResult:
        frontend = _frontend_dir(ctx.project_dir)
        if not (frontend / "package.json").is_file():
            return StepResult(StepStatus.SKIPPED, detail="no frontend/package.json")

        if shutil.which("pnpm") is None:
            return StepResult(
                StepStatus.SKIPPED,
                detail="pnpm not on PATH — install it then `agent-scaffold up --retry launch_frontend`",
            )

        # Clean up any stale PID file before we spawn so a crash mid-step doesn't
        # leave a confusing artifact pointing at a dead PID.
        pid_file = _pid_file_path(ctx.project_dir)
        stale = _read_pid_file(pid_file)
        if stale is not None and not _is_alive(int(stale["pid"])):
            pid_file.unlink(missing_ok=True)

        # ``pnpm install`` is idempotent and cheap on a warm cache; skip when
        # node_modules already exists to keep re-runs fast.
        node_modules = frontend / "node_modules"
        if not node_modules.is_dir():
            install_result = self._run_pnpm_install(ctx, frontend)
            if install_result is not None:
                return install_result

        log_file = _log_file_path(ctx.project_dir)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        # Truncate prior log; ``cmd_logs`` tails the file from current position.
        log_file.write_text("", encoding="utf-8")

        spawn = self._spawn_dev_server(
            frontend,
            log_file,
            runtime_env=ctx.runtime_env,
            extra_env=self._backend_url_env(ctx),
        )
        if isinstance(spawn, StepResult):
            return spawn
        pid, started_at = spawn

        ok, tail = self._wait_for_ready(log_file)
        if not ok:
            # Tear the dev server back down so we don't leave a zombie behind.
            _terminate(pid)
            log_file.unlink(missing_ok=True)
            return StepResult(
                StepStatus.FAILED,
                error=(
                    f"frontend dev server didn't emit a ready marker within "
                    f"{self.ready_timeout:.0f}s"
                ),
                stderr_tail=tail,
            )

        self._write_pid_file(pid_file, pid=pid, port=self.port, started_at=started_at)
        ctx.emit(
            StepLog(
                step_id=self.id,
                line=f"frontend ready at http://localhost:{self.port} (pid={pid})",
                stream="stdout",
            )
        )
        return StepResult(
            StepStatus.DONE, detail=f"dev server live at http://localhost:{self.port}"
        )

    # ---- fingerprint --------------------------------------------------

    def fingerprint(self, ctx: StepContext) -> str:
        pkg_path = _frontend_dir(ctx.project_dir) / "package.json"
        pkg_sha = hashlib.sha256(pkg_path.read_bytes()).hexdigest() if pkg_path.is_file() else None
        return compute_fingerprint(
            {
                "package_json_sha": pkg_sha,
                "frontend_capability": _resolve_frontend_capability_id(ctx),
                "port": self.port,
            }
        )

    # ---- internals ----------------------------------------------------

    def _run_pnpm_install(self, ctx: StepContext, frontend: Path) -> StepResult | None:
        """Run ``pnpm install --silent``; return a FAILED StepResult on error."""
        from agent_scaffold.steps._subprocess import stream_subprocess

        ctx.emit(
            StepLog(
                step_id=self.id,
                line="pnpm install (frontend/node_modules missing)",
                stream="stdout",
            )
        )
        result = stream_subprocess(
            ["pnpm", "install", "--silent"],
            cwd=frontend,
            step_id=self.id,
            callback=ctx.callback,
            timeout=self.timeout,
            env=ctx.runtime_env,
        )
        if result.exit_code != 0:
            return StepResult(
                StepStatus.FAILED,
                error=(
                    f"pnpm install timed out after {result.duration:.0f}s"
                    if result.timed_out
                    else f"pnpm install failed (exit {result.exit_code})"
                ),
                stderr_tail=result.stderr_tail,
            )
        return None

    def _backend_url_env(self, ctx: StepContext) -> dict[str, str]:
        """Default the frontend's backend-URL var(s) to the running backend.

        Only fills vars the user hasn't already set (their override wins). Empty
        when the recipe ships no frontend capability with such a var.
        """
        url_vars = _resolve_frontend_env_vars(ctx)
        if not url_vars:
            return {}
        base = ctx.runtime_env if ctx.runtime_env is not None else os.environ
        url = f"http://localhost:{_backend_port(ctx.manifest.language)}"
        return {var: url for var in url_vars if not base.get(var)}

    def _spawn_dev_server(
        self,
        frontend: Path,
        log_file: Path,
        runtime_env: dict[str, str] | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> tuple[int, str] | StepResult:
        """Spawn ``pnpm dev`` detached. Returns ``(pid, started_at_iso)`` or FAILED."""
        try:
            log_fh = log_file.open("a", encoding="utf-8")
        except OSError as exc:
            return StepResult(StepStatus.FAILED, error=f"could not open frontend.log: {exc}")
        try:
            base_env = runtime_env if runtime_env is not None else dict(os.environ)
            popen_kwargs: dict[str, Any] = {
                "cwd": str(frontend),
                "stdout": log_fh,
                "stderr": subprocess.STDOUT,
                "stdin": subprocess.DEVNULL,
                # extra_env (backend URL defaults) is collision-free with base_env —
                # _backend_url_env already drops anything the user set.
                "env": {
                    **base_env,
                    **(extra_env or {}),
                    "PORT": str(self.port),
                    "BROWSER": "none",
                },
            }
            if os.name == "nt":
                # CREATE_NEW_PROCESS_GROUP keeps the child alive after parent exit
                # and lets us signal it as a group later.
                popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            else:
                popen_kwargs["start_new_session"] = True
            proc = subprocess.Popen(  # noqa: S603 — list-form, shell=False
                ["pnpm", "dev"],
                **popen_kwargs,
            )
        except (OSError, FileNotFoundError) as exc:
            log_fh.close()
            return StepResult(StepStatus.FAILED, error=f"could not spawn pnpm dev: {exc}")
        finally:
            # Parent closes its own handle; the child inherited the underlying
            # fd and keeps it open. Discarding here is what lets the child
            # survive when the parent exits.
            try:
                log_fh.close()
            except OSError:
                pass
        return proc.pid, _iso_now()

    def _wait_for_ready(self, log_file: Path) -> tuple[bool, str]:
        """Poll the log until a ready marker appears or ``ready_timeout`` elapses."""
        deadline = time.monotonic() + self.ready_timeout
        while time.monotonic() < deadline:
            text = _safe_read_text(log_file)
            if any(marker in text for marker in _READY_MARKERS):
                return True, ""
            time.sleep(_READY_POLL_INTERVAL)
        # Failure: return the last N lines for the failure panel.
        tail = _safe_read_text(log_file).splitlines()[-_LOG_TAIL_LINES:]
        return False, "\n".join(tail)

    def _write_pid_file(self, path: Path, *, pid: int, port: int, started_at: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps({"pid": pid, "port": port, "started_at": started_at}, indent=2)
        path.write_text(body + "\n", encoding="utf-8")
        try:
            os.chmod(path, 0o644)
        except OSError:
            pass


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (OSError, FileNotFoundError):
        return ""


def _iso_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def _terminate(pid: int) -> None:
    """Best-effort tear-down of a process group we spawned."""
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, AttributeError, OSError):
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass


__all__ = ["LaunchFrontendStep"]
