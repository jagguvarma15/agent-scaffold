"""``install_deps`` step: install the project's dependencies per language.

Python: ``uv lock`` (if needed) + ``uv sync``. TypeScript: the package
manager the lockfile implies — ``pnpm install --frozen-lockfile``,
``npm ci``, or ``yarn install --frozen-lockfile``; a project with no
lockfile yet gets a plain ``pnpm install`` (the language hints' default
manager), which also generates one. Other languages surface as ``SKIPPED``
rather than ``FAILED`` so the orchestrator can proceed with the rest of
the plan.

Detection rules (python): no ``uv.lock`` → PENDING (lock + sync); no
``.venv`` → PENDING; ``.venv`` older than ``uv.lock`` → PENDING; else
DONE. TypeScript mirrors the shape with ``package.json`` /
``node_modules`` / the lockfile.

The fingerprint hashes the manifest + lockfile content so any edit to
either invalidates the DONE marker on the next ``--resume``.
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

# TypeScript package managers by lockfile, priority order. The frozen /
# ci variants refuse to drift from the lockfile — matching uv sync's
# reproducibility contract.
_LOCKFILE_COMMANDS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("pnpm-lock.yaml", "pnpm", ("pnpm", "install", "--frozen-lockfile")),
    ("package-lock.json", "npm", ("npm", "ci")),
    ("yarn.lock", "yarn", ("yarn", "install", "--frozen-lockfile")),
)


def _detect_package_manager(project_dir: Path) -> tuple[str, list[str]]:
    """``(binary, install argv)`` for a TypeScript project, from its lockfile.

    No lockfile falls back to a plain ``pnpm install`` (the language hints'
    default manager), which generates the lockfile as a side effect.
    """
    for lockfile, binary, argv in _LOCKFILE_COMMANDS:
        if (project_dir / lockfile).is_file():
            return binary, list(argv)
    return "pnpm", ["pnpm", "install"]


def _ts_lockfile(project_dir: Path) -> Path | None:
    """The lockfile the package manager will honour, if one exists."""
    for lockfile, _binary, _argv in _LOCKFILE_COMMANDS:
        path = project_dir / lockfile
        if path.is_file():
            return path
    return None


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class InstallDepsStep:
    """``uv sync`` a Python project, running ``uv lock`` first if needed."""

    id: str = "install_deps"
    description: str = "Install project dependencies"
    depends_on: tuple[str, ...] = ()
    # The one essential step: nothing downstream can run without deps, so a
    # failure here halts the whole run (every other step is best-effort).
    optional: bool = False
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
            "SSL: CERTIFICATE_VERIFY_FAILED": (
                "corporate proxy — set `UV_NATIVE_TLS=true` or `REQUESTS_CA_BUNDLE`"
            ),
            "Permission denied": (
                "no write access to .venv — check perms; try a fresh project dir"
            ),
            "ERR_PNPM_OUTDATED_LOCKFILE": (
                "package.json drifted from pnpm-lock.yaml — run `pnpm install` once "
                "to refresh the lockfile, then retry"
            ),
            "EBADENGINE": (
                "Node version too old for a dependency — upgrade Node (nvm install --lts)"
            ),
        }
    )

    # ---- detection ----------------------------------------------------

    def detect(self, ctx: StepContext) -> DetectionResult:
        language = ctx.manifest.language.lower()
        if language == "typescript":
            return self._detect_typescript(ctx)
        if language != "python":
            return DetectionResult(
                StepStatus.SKIPPED,
                reason=(
                    f"language={ctx.manifest.language!r} — install_deps handles "
                    "python and typescript"
                ),
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

    def _detect_typescript(self, ctx: StepContext) -> DetectionResult:
        package_json = ctx.project_dir / "package.json"
        if not package_json.is_file():
            return DetectionResult(
                StepStatus.SKIPPED, reason="no package.json — nothing to install"
            )
        node_modules = ctx.project_dir / "node_modules"
        if not node_modules.is_dir():
            return DetectionResult(
                StepStatus.PENDING, reason="no node_modules — install dependencies"
            )
        lockfile = _ts_lockfile(ctx.project_dir)
        try:
            if lockfile is not None and lockfile.stat().st_mtime > node_modules.stat().st_mtime:
                return DetectionResult(
                    StepStatus.PENDING,
                    reason=f"{lockfile.name} newer than node_modules — re-install needed",
                )
        except OSError:
            return DetectionResult(StepStatus.PENDING, reason="could not stat lockfile")
        return DetectionResult(StepStatus.DONE, reason="node_modules present and up to date")

    # ---- apply --------------------------------------------------------

    def apply(self, ctx: StepContext) -> StepResult:
        language = ctx.manifest.language.lower()
        if language == "typescript":
            return self._apply_typescript(ctx)
        if language != "python":
            return StepResult(StepStatus.SKIPPED, detail="unsupported language")
        if shutil.which("uv") is None:
            return StepResult(
                StepStatus.FAILED,
                error="`uv` not found on PATH",
                stderr_tail=("install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"),
            )

        lock = ctx.project_dir / "uv.lock"
        if not lock.is_file():
            lock_result = stream_subprocess(
                ["uv", "lock"],
                cwd=ctx.project_dir,
                step_id=self.id,
                callback=ctx.callback,
                timeout=self.timeout,
                env=ctx.runtime_env,
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
            callback=ctx.callback,
            timeout=self.timeout,
            env=ctx.runtime_env,
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

    def _apply_typescript(self, ctx: StepContext) -> StepResult:
        if not (ctx.project_dir / "package.json").is_file():
            return StepResult(StepStatus.SKIPPED, detail="no package.json — nothing to install")
        binary, argv = _detect_package_manager(ctx.project_dir)
        if shutil.which(binary) is None:
            return StepResult(
                StepStatus.FAILED,
                error=f"`{binary}` not found on PATH",
                stderr_tail=(
                    f"enable it via corepack (`corepack enable`, ships with Node 16.10+) "
                    f"or install it directly (`npm install -g {binary}`)"
                ),
            )
        install_result = stream_subprocess(
            argv,
            cwd=ctx.project_dir,
            step_id=self.id,
            callback=ctx.callback,
            timeout=self.timeout,
            env=ctx.runtime_env,
        )
        if install_result.exit_code != 0:
            return StepResult(
                status=StepStatus.FAILED,
                error=_failure_message(" ".join(argv), install_result),
                stderr_tail=install_result.stderr_tail,
            )
        return StepResult(
            status=StepStatus.DONE,
            detail=f"{binary} install ok in {install_result.duration:.1f}s",
        )

    # ---- fingerprint --------------------------------------------------

    def fingerprint(self, ctx: StepContext) -> str:
        if ctx.manifest.language.lower() == "typescript":
            lockfile = _ts_lockfile(ctx.project_dir)
            return compute_fingerprint(
                {
                    "package_json_sha": _sha256_file(ctx.project_dir / "package.json"),
                    "lockfile": lockfile.name if lockfile else None,
                    "lockfile_sha": _sha256_file(lockfile) if lockfile else None,
                    "language": ctx.manifest.language,
                }
            )
        return compute_fingerprint(
            {
                "pyproject_sha": _sha256_file(ctx.project_dir / "pyproject.toml"),
                "lock_sha": _sha256_file(ctx.project_dir / "uv.lock"),
                "language": ctx.manifest.language,
            }
        )


def _failure_message(label: str, result: object) -> str:
    """Build a one-line failure message from a ``SubprocessResult``-shaped object."""
    exit_code = getattr(result, "exit_code", "?")
    timed_out = getattr(result, "timed_out", False)
    if timed_out:
        return f"{label} timed out after {getattr(result, 'duration', 0):.0f}s"
    return f"{label} failed (exit {exit_code})"


__all__ = ["InstallDepsStep"]
