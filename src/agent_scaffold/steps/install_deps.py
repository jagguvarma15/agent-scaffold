"""``install_deps`` step: ``uv lock`` (if needed) + ``uv sync``.

Python-only — TypeScript provisioning is an explicit non-goal for v2 (see
SESSION-HANDOFF.md). A non-Python project surfaces as ``SKIPPED`` rather
than ``FAILED`` so the orchestrator can still proceed with the rest of the
plan.

Detection rules:

- No ``uv.lock`` → ``PENDING`` (we'll run ``uv lock`` then ``uv sync``).
- No ``.venv`` → ``PENDING``.
- ``.venv`` older than ``uv.lock`` → ``PENDING`` (resync needed; lock file
  was regenerated since the last sync).
- Else ``DONE``.

The fingerprint hashes ``pyproject.toml`` + ``uv.lock`` content so that any
edit to either invalidates the DONE marker on the next ``--resume``.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from agent_scaffold.orchestrator import (
    DetectionResult,
    StepContext,
    StepResult,
    StepStatus,
    compute_fingerprint,
)
from agent_scaffold.steps._subprocess import stream_subprocess

_DEFAULT_TIMEOUT = 600.0


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class InstallDepsStep:
    """``uv sync`` a Python project, running ``uv lock`` first if needed."""

    id: str = "install_deps"
    description: str = "Install Python dependencies (uv lock + uv sync)"
    depends_on: tuple[str, ...] = ()
    # Per-step timeout default. Long because cold ``uv sync`` over a slow
    # network can take several minutes on dependency-heavy recipes.
    timeout: float = _DEFAULT_TIMEOUT
    troubleshoot: dict[str, str] = field(
        default_factory=lambda: {
            "No solution found": (
                "version conflict in pyproject.toml — loosen pins or run "
                "`uv lock --resolution=lowest-direct`"
            ),
            "Requested Python version": (
                "uv could not find a Python that satisfies python = ... in pyproject.toml — "
                "install a matching interpreter via pyenv / asdf / brew"
            ),
        }
    )

    # ---- detection ----------------------------------------------------

    def detect(self, ctx: StepContext) -> DetectionResult:
        if not _is_python_project(ctx):
            return DetectionResult(
                StepStatus.SKIPPED,
                reason=f"language={ctx.manifest.language!r} — install_deps only handles python",
            )
        pyproject = ctx.project_dir / "pyproject.toml"
        if not pyproject.is_file():
            return DetectionResult(
                StepStatus.SKIPPED,
                reason="no pyproject.toml — nothing to install",
            )
        lock = ctx.project_dir / "uv.lock"
        venv = ctx.project_dir / ".venv"
        if not lock.is_file():
            return DetectionResult(StepStatus.PENDING, reason="no uv.lock yet — uv lock + uv sync")
        if not venv.is_dir():
            return DetectionResult(StepStatus.PENDING, reason="no .venv — uv sync")
        try:
            if lock.stat().st_mtime > venv.stat().st_mtime:
                return DetectionResult(
                    StepStatus.PENDING, reason="uv.lock newer than .venv — re-sync needed"
                )
        except OSError:
            # Filesystem hiccup → treat as PENDING; apply() will surface real failures.
            return DetectionResult(StepStatus.PENDING, reason="could not stat lock/venv")
        return DetectionResult(StepStatus.DONE, reason=".venv present and up to date")

    # ---- apply --------------------------------------------------------

    def apply(self, ctx: StepContext) -> StepResult:
        if not _is_python_project(ctx):
            return StepResult(StepStatus.SKIPPED, detail="not a python project")
        if shutil.which("uv") is None:
            return StepResult(
                StepStatus.FAILED,
                error="`uv` not found on PATH",
                stderr_tail=(
                    "install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
                ),
            )

        lock = ctx.project_dir / "uv.lock"
        if not lock.is_file():
            lock_result = stream_subprocess(
                ["uv", "lock"],
                cwd=ctx.project_dir,
                step_id=self.id,
                callback=ctx.callback,  # type: ignore[arg-type]
                timeout=self.timeout,
            )
            if lock_result.exit_code != 0:
                return StepResult(
                    status=StepStatus.FAILED,
                    error=_failure_message("uv lock", lock_result),
                    stderr_tail=lock_result.stderr_tail,
                )

        sync_result = stream_subprocess(
            ["uv", "sync"],
            cwd=ctx.project_dir,
            step_id=self.id,
            callback=ctx.callback,  # type: ignore[arg-type]
            timeout=self.timeout,
        )
        if sync_result.exit_code != 0:
            return StepResult(
                status=StepStatus.FAILED,
                error=_failure_message("uv sync", sync_result),
                stderr_tail=sync_result.stderr_tail,
            )
        return StepResult(
            status=StepStatus.DONE,
            detail=f"uv sync ok in {sync_result.duration:.1f}s",
        )

    # ---- fingerprint --------------------------------------------------

    def fingerprint(self, ctx: StepContext) -> str:
        return compute_fingerprint(
            {
                "pyproject_sha": _sha256_file(ctx.project_dir / "pyproject.toml"),
                "lock_sha": _sha256_file(ctx.project_dir / "uv.lock"),
                "language": ctx.manifest.language,
            }
        )


def _is_python_project(ctx: StepContext) -> bool:
    return ctx.manifest.language.lower() == "python"


def _failure_message(label: str, result: object) -> str:
    """Build a one-line failure message from a ``SubprocessResult``-shaped object."""
    exit_code = getattr(result, "exit_code", "?")
    timed_out = getattr(result, "timed_out", False)
    if timed_out:
        return f"{label} timed out after {getattr(result, 'duration', 0):.0f}s"
    return f"{label} failed (exit {exit_code})"


__all__ = ["InstallDepsStep"]
