"""``docker_up`` step: start declared services via ``docker compose``.

Driven by the recipe's ``external_services`` frontmatter (Q3 schema):
every entry whose ``docker_service`` field is set is started with
``docker compose up -d <service>``. After the start, we re-probe each
service using ``probes.PROBES[svc.probe]`` until it returns ``OK`` or the
healthcheck timeout elapses.

Edge cases this honors (lessons from earlier provisioning attempts):

- No ``docker-compose.yml`` → ``SKIPPED``. The recipe may declare services
  the user already runs natively (e.g. Homebrew redis).
- No ``docker_service`` field on any external service → ``SKIPPED``.
- ``docker`` not on PATH or daemon down → ``SKIPPED`` with a useful
  ``fix_hint``, **not** ``FAILED``. The user might intentionally use
  Colima, podman, or no container runtime at all.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from agent_scaffold.discovery import (
    DiscoveryError,
    ExternalService,
    Recipe,
    discover_recipes,
)
from agent_scaffold.orchestrator import (
    DetectionResult,
    StepContext,
    StepLog,
    StepProgress,
    StepResult,
    StepStatus,
    compute_fingerprint,
)
from agent_scaffold.steps._subprocess import stream_subprocess

_DEFAULT_PULL_TIMEOUT = 600.0  # docker pulls can be slow on cold caches
_DEFAULT_HEALTHCHECK_TIMEOUT = 60.0  # individual service healthcheck wait
_HEALTHCHECK_POLL_INTERVAL = 2.0


@dataclass
class DockerUpStep:
    """Start every external service that declares a ``docker_service`` name."""

    id: str = "docker_up"
    description: str = "Start required services via docker compose"
    # install_deps isn't a hard prerequisite but it's cheap and decides the
    # interpreter arch; running it first means a failed sync surfaces before
    # the user waits on a multi-hundred-MB image pull.
    depends_on: tuple[str, ...] = ("install_deps",)
    timeout: float = _DEFAULT_PULL_TIMEOUT
    healthcheck_timeout: float = _DEFAULT_HEALTHCHECK_TIMEOUT
    troubleshoot: dict[str, str] = field(
        default_factory=lambda: {
            "Cannot connect to the Docker daemon": (
                "start Docker Desktop or `colima start`, then re-run "
                "`agent-scaffold up --retry docker_up`"
            ),
            "address already in use": (
                "another process holds the port — find it with "
                "`lsof -i :<port>` and stop it, then `--force docker_up`"
            ),
            "no such service": (
                "the recipe's docker_service name doesn't match docker-compose.yml — "
                "fix the frontmatter or compose file"
            ),
            "manifest unknown": (
                "the requested image tag doesn't exist in the registry — "
                "check the image: pin in docker-compose.yml"
            ),
            "port is already allocated": (
                "another process holds the port — `lsof -i :<port>` to find it"
            ),
            "image not found": (
                "image name or tag wrong — check docker-compose.yml; try `docker pull <image>`"
            ),
            "pull access denied": "private image — `docker login` first",
        }
    )

    # ---- detection ----------------------------------------------------

    def detect(self, ctx: StepContext) -> DetectionResult:
        compose = ctx.project_dir / "docker-compose.yml"
        if not compose.is_file():
            return DetectionResult(StepStatus.SKIPPED, reason="no docker-compose.yml — skipping")
        services = self._declared_services(ctx)
        if not services:
            return DetectionResult(
                StepStatus.SKIPPED,
                reason="recipe declares no docker_service on any external_service",
            )
        if shutil.which("docker") is None:
            return DetectionResult(
                StepStatus.SKIPPED,
                reason="docker not on PATH — install Docker Desktop / Colima then re-run",
            )
        running = self._running_services(ctx)
        wanted = {svc.docker_service for svc in services if svc.docker_service}
        missing = sorted(wanted - running)
        if not missing:
            return DetectionResult(
                StepStatus.DONE, reason=f"{len(wanted)} service(s) already running"
            )
        return DetectionResult(StepStatus.PENDING, reason=f"need to start: {', '.join(missing)}")

    # ---- apply --------------------------------------------------------

    def apply(self, ctx: StepContext) -> StepResult:
        compose = ctx.project_dir / "docker-compose.yml"
        if not compose.is_file():
            return StepResult(StepStatus.SKIPPED, detail="no docker-compose.yml")
        services = self._declared_services(ctx)
        if not services:
            return StepResult(StepStatus.SKIPPED, detail="no docker_service declared")
        if shutil.which("docker") is None:
            return StepResult(
                StepStatus.SKIPPED, detail="docker not on PATH; user may run services natively"
            )

        # Daemon-down check before pulling: surfaces the friendly "start
        # Docker Desktop" hint instead of a 30-second timeout on `compose up`.
        ping = stream_subprocess(
            ["docker", "info"],
            cwd=ctx.project_dir,
            step_id=self.id,
            callback=None,  # noisy; we only care about exit code
            timeout=10.0,
        )
        if ping.exit_code != 0:
            return StepResult(
                StepStatus.SKIPPED,
                detail="docker daemon not reachable — `docker info` failed",
                stderr_tail=ping.stderr_tail,
            )

        service_names = [svc.docker_service for svc in services if svc.docker_service]
        # Sequential up; docker handles per-image pull concurrency itself.
        up_result = stream_subprocess(
            ["docker", "compose", "up", "-d", *service_names],
            cwd=ctx.project_dir,
            step_id=self.id,
            callback=ctx.callback,
            timeout=self.timeout,
        )
        if up_result.exit_code != 0:
            return StepResult(
                StepStatus.FAILED,
                error=(
                    f"docker compose up timed out after {up_result.duration:.0f}s"
                    if up_result.timed_out
                    else f"docker compose up failed (exit {up_result.exit_code})"
                ),
                stderr_tail=up_result.stderr_tail,
            )

        # Healthcheck wait: re-probe each service with a registered probe.
        unhealthy = self._wait_for_health(services, ctx)
        if unhealthy:
            return StepResult(
                StepStatus.FAILED,
                error=f"{len(unhealthy)} service(s) failed healthcheck: {', '.join(unhealthy)}",
                stderr_tail="probes returned non-OK; check container logs",
            )
        return StepResult(
            StepStatus.DONE,
            detail=f"{len(service_names)} service(s) up and healthy",
        )

    # ---- fingerprint --------------------------------------------------

    def fingerprint(self, ctx: StepContext) -> str:
        services = self._declared_services(ctx)
        compose = ctx.project_dir / "docker-compose.yml"
        compose_sha = None
        if compose.is_file():
            import hashlib

            compose_sha = hashlib.sha256(compose.read_bytes()).hexdigest()
        return compute_fingerprint(
            {
                "services": sorted(svc.docker_service for svc in services if svc.docker_service),
                "compose_sha": compose_sha,
            }
        )

    # ---- helpers ------------------------------------------------------

    def _declared_services(self, ctx: StepContext) -> list[ExternalService]:
        """Return external_services with a non-empty ``docker_service``."""
        recipe = _load_recipe(ctx)
        if recipe is None:
            return []
        return [svc for svc in recipe.external_services if svc.docker_service]

    def _running_services(self, ctx: StepContext) -> set[str]:
        """Names returned by ``docker compose ps --services --filter status=running``."""
        if shutil.which("docker") is None:
            return set()
        result = stream_subprocess(
            ["docker", "compose", "ps", "--services", "--filter", "status=running"],
            cwd=ctx.project_dir,
            step_id=self.id,
            callback=None,
            timeout=10.0,
        )
        if result.exit_code != 0:
            return set()
        # Parsing piggybacks on stream_subprocess's StepLog emission — but
        # we passed callback=None, so re-run a quick capture call.
        proc_out = _capture_stdout(
            ["docker", "compose", "ps", "--services", "--filter", "status=running"],
            cwd=ctx.project_dir,
            timeout=10.0,
        )
        return {line.strip() for line in proc_out.splitlines() if line.strip()}

    def _wait_for_health(self, services: list[ExternalService], ctx: StepContext) -> list[str]:
        """Probe each service repeatedly until OK or per-service timeout."""
        from agent_scaffold.doctor import CheckStatus
        from agent_scaffold.probes import PROBES, run_probe

        unhealthy: list[str] = []
        for svc in services:
            if not svc.probe or svc.probe not in PROBES:
                # No probe → can't healthcheck; trust ``docker compose up``.
                continue
            deadline = time.monotonic() + self.healthcheck_timeout
            last_detail = ""
            while True:
                ctx.emit(
                    StepProgress(
                        step_id=self.id,
                        message=f"healthcheck: {svc.id}",
                    )
                )
                check = run_probe(svc, timeout=5.0)
                if check.status == CheckStatus.OK:
                    ctx.emit(
                        StepLog(
                            step_id=self.id,
                            line=f"{svc.id}: {check.title}",
                            stream="stdout",
                        )
                    )
                    break
                last_detail = check.detail or check.title
                if time.monotonic() >= deadline:
                    ctx.emit(
                        StepLog(
                            step_id=self.id,
                            line=f"{svc.id}: healthcheck timeout — {last_detail}",
                            stream="stderr",
                        )
                    )
                    unhealthy.append(svc.id)
                    break
                time.sleep(_HEALTHCHECK_POLL_INTERVAL)
        return unhealthy


def _capture_stdout(cmd: list[str], cwd: Path, timeout: float) -> str:
    """Tiny ``subprocess.run``-style helper for read-only ``docker`` queries."""
    import subprocess

    try:
        proc = subprocess.run(  # noqa: S603 — cmd is list-form, shell=False
            cmd,
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""
    return proc.stdout or ""


def _load_recipe(ctx: StepContext) -> Recipe | None:
    """Look up the manifest's recipe slug in the configured deployments path."""
    from agent_scaffold.config import load_config

    try:
        cfg = load_config()
    except Exception:  # noqa: BLE001 — config errors must not crash detect()
        return None
    try:
        recipes = discover_recipes(cfg.deployments_path.expanduser())
    except DiscoveryError:
        return None
    return next((r for r in recipes if r.slug == ctx.manifest.recipe), None)


__all__ = ["DockerUpStep"]
