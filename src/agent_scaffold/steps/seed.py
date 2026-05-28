"""``seed`` step: load development seed data.

Detection order:
- ``scripts/seed.py`` → ``uv run python scripts/seed.py``
- ``scripts/seed.sh`` → ``bash scripts/seed.sh``
- neither → ``SKIPPED`` ("no seed script")

Fingerprint includes the script's SHA so editing the seed script invalidates
the recorded DONE on ``--resume``.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from agent_scaffold.discovery import (
    DiscoveryError,
    ExternalService,
    Recipe,
    discover_recipes,
)
from agent_scaffold.orchestrator import (
    DetectionResult,
    StepContext,
    StepResult,
    StepStatus,
    compute_fingerprint,
)
from agent_scaffold.steps._subprocess import stream_subprocess

_DEFAULT_TIMEOUT = 300.0
_SEED_PY = Path("scripts") / "seed.py"
_SEED_SH = Path("scripts") / "seed.sh"


@dataclass
class SeedStep:
    """Run a project-supplied seed script. Idempotent if the script is."""

    id: str = "seed"
    description: str = "Load development seed data"
    depends_on: tuple[str, ...] = ("migrations",)
    timeout: float = _DEFAULT_TIMEOUT
    troubleshoot: dict[str, str] = field(
        default_factory=lambda: {
            "no such table": ("migrations didn't run — `agent-scaffold up --force migrations`"),
            "duplicate key value": (
                "data already seeded — pass `--force seed` to re-seed, "
                "or accept the current state"
            ),
            "ModuleNotFoundError": (
                "deps not installed — `agent-scaffold up --force install_deps`"
            ),
        }
    )

    # ---- detection ----------------------------------------------------

    def detect(self, ctx: StepContext) -> DetectionResult:
        script = self._seed_script(ctx)
        if script is None:
            return DetectionResult(
                StepStatus.SKIPPED,
                reason="no scripts/seed.py or scripts/seed.sh",
            )
        # State-file fingerprint comparison is owned by the orchestrator's
        # decide loop (via the StepState). We only have to report PENDING here
        # so the orchestrator runs apply() — re-detection after apply() returns
        # DONE keeps the step idempotent.
        return DetectionResult(StepStatus.PENDING, reason=f"will run {script.name}")

    # ---- apply --------------------------------------------------------

    def apply(self, ctx: StepContext) -> StepResult:
        script = self._seed_script(ctx)
        if script is None:
            return StepResult(StepStatus.SKIPPED, detail="no seed script")
        cmd: list[str]
        if script.suffix == ".py":
            if shutil.which("uv") is None:
                return StepResult(
                    StepStatus.FAILED,
                    error="`uv` not found on PATH — run install_deps first",
                )
            cmd = ["uv", "run", "python", str(script.relative_to(ctx.project_dir))]
        else:
            if shutil.which("bash") is None:
                return StepResult(
                    StepStatus.FAILED,
                    error="`bash` not found on PATH — install bash or supply seed.py",
                )
            cmd = ["bash", str(script.relative_to(ctx.project_dir))]
        result = stream_subprocess(
            cmd,
            cwd=ctx.project_dir,
            step_id=self.id,
            callback=ctx.callback,
            timeout=self.timeout,
        )
        if result.exit_code != 0:
            return StepResult(
                StepStatus.FAILED,
                error=(
                    f"seed script timed out after {result.duration:.0f}s"
                    if result.timed_out
                    else f"seed script failed (exit {result.exit_code})"
                ),
                stderr_tail=result.stderr_tail,
            )
        return StepResult(
            StepStatus.DONE,
            detail=f"{script.name} ok in {result.duration:.1f}s",
        )

    # ---- fingerprint --------------------------------------------------

    def fingerprint(self, ctx: StepContext) -> str:
        script = self._seed_script(ctx)
        recipe = _load_recipe(ctx)
        db_target = _first_db_target(recipe.external_services if recipe else [])
        return compute_fingerprint(
            {
                "script_sha": _sha256_file(script) if script else None,
                "db_host": db_target[0],
                "db_db": db_target[1],
            }
        )

    # ---- helpers ------------------------------------------------------

    def _seed_script(self, ctx: StepContext) -> Path | None:
        for candidate in (_SEED_PY, _SEED_SH):
            target = ctx.project_dir / candidate
            if target.is_file():
                return target
        return None


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _first_db_target(services: Sequence[ExternalService]) -> tuple[str, str]:
    """Pick the first migrating service so the fingerprint binds to the DB."""
    for svc in services:
        if not svc.migrations:
            continue
        for env_var in svc.env_vars:
            value = os.environ.get(env_var, "").strip()
            if value and "://" in value:
                parsed = urlparse(value)
                return parsed.hostname or "", (parsed.path or "").lstrip("/")
    return "", ""


def _load_recipe(ctx: StepContext) -> Recipe | None:
    from agent_scaffold.config import load_config
    from agent_scaffold.sources import SourceFetchError, resolve_deployments

    try:
        cfg = load_config()
    except Exception:  # noqa: BLE001
        return None
    try:
        dep = resolve_deployments(
            override=cfg.deployments_path,
            mode=cfg.deployments_source,
            cache_dir=cfg.cache_dir,
        )
    except SourceFetchError:
        return None
    if dep.path is None:
        return None
    try:
        recipes = discover_recipes(dep.path)
    except DiscoveryError:
        return None
    return next((r for r in recipes if r.slug == ctx.manifest.recipe), None)


__all__ = ["SeedStep"]
