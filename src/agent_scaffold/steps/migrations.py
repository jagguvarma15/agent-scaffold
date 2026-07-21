"""``migrations`` step: apply database schema migrations.

Driven by ``external_services[*].migrations`` on the recipe (Q3 schema).
v2 ships only the **alembic** engine. Detection runs ``alembic current``
plus ``alembic heads`` for each migrating service and skips the upgrade
when current is already at head.

Non-alembic engines (``prisma``, ``flyway``, ``drizzle-kit``, ...) surface
as ``SKIPPED`` with a hint explaining the v3 deferral. The dispatch shape
makes adding an engine a single registry entry.
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

_DEFAULT_TIMEOUT = 600.0
_SUPPORTED_ENGINES: frozenset[str] = frozenset({"alembic"})


@dataclass
class MigrationsStep:
    """Apply schema migrations declared by ``external_services[*].migrations``."""

    id: str = "migrations"
    description: str = "Apply database migrations"
    depends_on: tuple[str, ...] = ("docker_up", "wire_credentials")
    timeout: float = _DEFAULT_TIMEOUT
    troubleshoot: dict[str, str] = field(
        default_factory=lambda: {
            "could not connect to server": (
                "DATABASE_URL points at an unreachable host — run "
                "`agent-scaffold up --force docker_up` if local, or check your hosted URL"
            ),
            'relation "alembic_version" does not exist': (
                "first migration — this is normal; if it persists, run "
                "`alembic stamp head` then retry"
            ),
            "could not translate host name": (
                "DATABASE_URL hostname not resolvable — check DNS / typo"
            ),
            "password authentication failed": (
                "DATABASE_URL credentials wrong — re-run "
                "`agent-scaffold up --force wire_credentials`"
            ),
            "ERROR [alembic.runtime.migration]": (
                "migration script error — see stderr tail; fix the script and retry"
            ),
        }
    )

    # ---- detection ----------------------------------------------------

    def detect(self, ctx: StepContext) -> DetectionResult:
        services = self._migrating_services(ctx)
        if not services:
            return DetectionResult(
                StepStatus.SKIPPED,
                reason="no external_services declare migrations",
            )
        if ctx.manifest.language.lower() != "python":
            # alembic runs through `uv run` inside the project's Python env;
            # a TypeScript project has neither. A recipe declaring alembic on
            # a TS run is an authoring mismatch — skip instead of crashing.
            return DetectionResult(
                StepStatus.SKIPPED,
                reason=(
                    f"alembic migrations need a Python project; language={ctx.manifest.language!r}"
                ),
            )
        if shutil.which("uv") is None:
            return DetectionResult(
                StepStatus.PENDING,
                reason="uv not on PATH — install_deps will surface the install hint",
            )
        # We can only meaningfully detect alembic. Anything else is SKIPPED.
        unsupported = [s for s in services if (s.migrations or "") not in _SUPPORTED_ENGINES]
        supported = [s for s in services if (s.migrations or "") in _SUPPORTED_ENGINES]
        if not supported:
            engines = sorted({s.migrations or "?" for s in unsupported})
            return DetectionResult(
                StepStatus.SKIPPED,
                reason=(
                    f"only alembic is supported in v2; got {', '.join(engines)} — skipping for now"
                ),
            )
        pending: list[str] = []
        for svc in supported:
            current, heads = self._alembic_versions(ctx, svc)
            if current is None or heads is None:
                pending.append(svc.id)
                continue
            if current != heads:
                pending.append(svc.id)
        if pending:
            return DetectionResult(
                StepStatus.PENDING,
                reason=f"alembic out of date for: {', '.join(pending)}",
            )
        return DetectionResult(
            StepStatus.DONE,
            reason=f"alembic at head for {len(supported)} service(s)",
        )

    # ---- apply --------------------------------------------------------

    def apply(self, ctx: StepContext) -> StepResult:
        services = self._migrating_services(ctx)
        if not services:
            return StepResult(StepStatus.SKIPPED, detail="no migrating services")
        if ctx.manifest.language.lower() != "python":
            return StepResult(
                StepStatus.SKIPPED,
                detail=f"alembic needs a Python project; language={ctx.manifest.language!r}",
            )
        if shutil.which("uv") is None:
            return StepResult(
                StepStatus.FAILED,
                error="`uv` not found on PATH — run install_deps first",
            )
        ran = 0
        skipped: list[str] = []
        for svc in services:
            engine = svc.migrations or ""
            if engine not in _SUPPORTED_ENGINES:
                skipped.append(f"{svc.id} ({engine})")
                continue
            result = stream_subprocess(
                ["uv", "run", "alembic", "upgrade", "head"],
                cwd=ctx.project_dir,
                step_id=self.id,
                callback=ctx.callback,
                timeout=self.timeout,
                env=self._env_for(ctx, svc),
            )
            if result.exit_code != 0:
                return StepResult(
                    StepStatus.FAILED,
                    error=(
                        f"alembic upgrade timed out after {result.duration:.0f}s"
                        if result.timed_out
                        else f"alembic upgrade failed for {svc.id} (exit {result.exit_code})"
                    ),
                    stderr_tail=result.stderr_tail,
                )
            ran += 1
        detail_parts = [f"{ran} service(s) migrated"]
        if skipped:
            detail_parts.append(f"skipped unsupported: {', '.join(skipped)}")
        return StepResult(StepStatus.DONE, detail="; ".join(detail_parts))

    # ---- fingerprint --------------------------------------------------

    def fingerprint(self, ctx: StepContext) -> str:
        services = self._migrating_services(ctx)
        db_targets = []
        for svc in services:
            host, db = _db_host_and_name(_db_url_for_env(svc))
            db_targets.append({"id": svc.id, "host": host, "db": db})
        return compute_fingerprint(
            {
                "db_targets": db_targets,
                "migration_dir_sha": _sha256_dir(ctx.project_dir / "alembic"),
            }
        )

    # ---- helpers ------------------------------------------------------

    def _migrating_services(self, ctx: StepContext) -> list[ExternalService]:
        recipe = _load_recipe(ctx)
        if recipe is None:
            return []
        return [svc for svc in recipe.external_services if svc.migrations]

    def _alembic_versions(
        self, ctx: StepContext, svc: ExternalService
    ) -> tuple[str | None, str | None]:
        """Probe ``alembic current`` + ``alembic heads`` and return the parsed revs.

        Uses ``subprocess.run`` directly rather than the streaming helper because
        we actually need the captured stdout to parse the revision id.
        """
        env = self._env_for(ctx, svc)
        current_out, current_rc = _capture(
            ["uv", "run", "alembic", "current"], cwd=ctx.project_dir, env=env, timeout=30.0
        )
        heads_out, heads_rc = _capture(
            ["uv", "run", "alembic", "heads"], cwd=ctx.project_dir, env=env, timeout=30.0
        )
        if current_rc != 0 or heads_rc != 0:
            return None, None
        return _parse_alembic_rev(current_out), _parse_alembic_rev(heads_out)

    def _env_for(self, ctx: StepContext, svc: ExternalService) -> dict[str, str]:
        """Vault-resolved runtime env when available, else the current shell env."""
        env = dict(ctx.runtime_env) if ctx.runtime_env is not None else os.environ.copy()
        # No mutation — DATABASE_URL etc. arrive via the vault / .env.local /
        # shell. Returning a copy still matters: callers may extend it for
        # service-scoped runs.
        _ = svc
        return env


def _capture(
    cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None, timeout: float = 30.0
) -> tuple[str, int]:
    """Tiny ``subprocess.run`` wrapper used by read-only alembic queries."""
    import subprocess

    try:
        proc = subprocess.run(  # noqa: S603 — list-form, shell=False
            cmd,
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return "", -1
    return proc.stdout or "", int(proc.returncode)


def _parse_alembic_rev(text: str) -> str:
    """Pull the rev hex out of alembic's ``<rev> (head)`` style output."""
    for line in text.splitlines():
        token = line.strip().split()[:1]
        if token and all(c in "0123456789abcdef" for c in token[0].lower()) and len(token[0]) >= 8:
            return token[0]
    return ""


def _db_url_for_env(svc: ExternalService) -> str:
    for env_var in svc.env_vars:
        value = os.environ.get(env_var, "").strip()
        if value:
            return value
    return svc.default_local or ""


def _db_host_and_name(url: str) -> tuple[str, str]:
    if not url:
        return "", ""
    if "://" not in url:
        return url, ""
    parsed = urlparse(url)
    host = parsed.hostname or ""
    db = (parsed.path or "").lstrip("/")
    return host, db


def _sha256_dir(directory: Path) -> str | None:
    if not directory.is_dir():
        return None
    h = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            h.update(path.relative_to(directory).as_posix().encode("utf-8"))
            h.update(b"\0")
            h.update(path.read_bytes())
    return h.hexdigest()


def _load_recipe(ctx: StepContext) -> Recipe | None:
    from agent_scaffold.config import load_config
    from agent_scaffold.sources import SourceFetchError, resolve_deployments

    try:
        cfg = load_config()
    except Exception:  # noqa: BLE001 — config issues must not crash detect()
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


__all__: Sequence[str] = ["MigrationsStep"]
