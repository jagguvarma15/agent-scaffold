"""``docker_up`` step: start the project's services via ``docker compose``.

Two modes:

1. **Declared services.** If the recipe's ``external_services`` set a
   ``docker_service`` field, start exactly those (``docker compose up -d
   <service>``) and re-probe each via ``probes.PROBES[svc.probe]`` until
   ``OK`` or the healthcheck timeout elapses.
2. **Whole stack.** If a ``docker-compose.yml`` exists but no
   ``docker_service`` is declared (e.g. bare-string ``external_services``),
   bring up the entire generated stack with ``docker compose up -d --wait``
   (native healthcheck waiting) so the DB / Redis / etc. the project needs
   actually start after generation.

Edge cases this honors (lessons from earlier provisioning attempts):

- No ``docker-compose.yml`` → ``SKIPPED``. The recipe may declare services
  the user already runs natively (e.g. Homebrew redis).
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
from agent_scaffold.steps._subprocess import SubprocessResult, stream_subprocess

_DEFAULT_PULL_TIMEOUT = 600.0  # docker pulls can be slow on cold caches
_DEFAULT_HEALTHCHECK_TIMEOUT = 60.0  # individual service healthcheck wait
_DEFAULT_STACK_WAIT_TIMEOUT = 180.0  # whole-stack `docker compose up --wait` health budget
_HEALTHCHECK_POLL_INTERVAL = 2.0


def docker_available(*, timeout: float = 10.0) -> tuple[bool, str]:
    """Is docker installed, the daemon running, and accessible? → ``(ok, reason)``.

    ``docker info`` exiting 0 is the canonical "usable" probe: it needs the CLI
    installed, the daemon reachable, and the caller to have socket access. The
    ``reason`` distinguishes the common failures so the caller can guide the user.
    """
    if shutil.which("docker") is None:
        return False, "not installed"
    result = stream_subprocess(
        ["docker", "info"],
        cwd=Path.cwd(),
        step_id="docker_available",
        callback=None,
        timeout=timeout,
    )
    if result.exit_code == 0:
        return True, "ok"
    err = result.stderr_tail.lower()
    if "permission denied" in err:
        return False, "permission denied — add your user to the docker group"
    if "cannot connect" in err or "daemon" in err or "is the docker daemon running" in err:
        return False, "daemon not running — start Docker Desktop / Colima"
    return False, "docker info failed"


@dataclass
class DockerUpStep:
    """Start every external service that declares a ``docker_service`` name."""

    id: str = "docker_up"
    description: str = "Start required services via docker compose"
    # install_deps isn't a hard prerequisite but it's cheap and decides the
    # interpreter arch; running it first means a failed sync surfaces before
    # the user waits on a multi-hundred-MB image pull.
    depends_on: tuple[str, ...] = ("install_deps",)
    # Docker is opt-in: default_steps_for sets enabled=True only when the user
    # chose docker mode (--docker / prompt). Disabled → the step skips.
    enabled: bool = True
    timeout: float = _DEFAULT_PULL_TIMEOUT
    healthcheck_timeout: float = _DEFAULT_HEALTHCHECK_TIMEOUT
    # Whole-stack `--wait` budget when the recipe declares no docker_service map.
    wait_timeout: float = _DEFAULT_STACK_WAIT_TIMEOUT
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
        if not self.enabled:
            return DetectionResult(
                StepStatus.SKIPPED, reason="docker mode off — opt in with --docker"
            )
        compose = ctx.project_dir / "docker-compose.yml"
        if not compose.is_file():
            return DetectionResult(StepStatus.SKIPPED, reason="no docker-compose.yml — skipping")
        if shutil.which("docker") is None:
            return DetectionResult(
                StepStatus.SKIPPED,
                reason="docker not on PATH — install Docker Desktop / Colima then re-run",
            )
        declared = {
            svc.docker_service for svc in self._declared_services(ctx) if svc.docker_service
        }
        # Whole-stack mode: a compose file exists but the recipe declares no
        # docker_service map (bare-string external_services). Bring up the whole
        # generated stack instead of skipping it.
        wanted = declared or set(self._compose_services(ctx))
        if not wanted:
            return DetectionResult(
                StepStatus.SKIPPED, reason="docker-compose.yml declares no services"
            )
        missing = sorted(wanted - self._running_services(ctx))
        if not missing:
            return DetectionResult(
                StepStatus.DONE, reason=f"{len(wanted)} service(s) already running"
            )
        if declared:
            return DetectionResult(
                StepStatus.PENDING, reason=f"need to start: {', '.join(missing)}"
            )
        return DetectionResult(
            StepStatus.PENDING, reason=f"will bring up the compose stack ({len(wanted)} services)"
        )

    # ---- apply --------------------------------------------------------

    def apply(self, ctx: StepContext) -> StepResult:
        if not self.enabled:
            return StepResult(StepStatus.SKIPPED, detail="docker mode off — opt in with --docker")
        compose = ctx.project_dir / "docker-compose.yml"
        if not compose.is_file():
            return StepResult(StepStatus.SKIPPED, detail="no docker-compose.yml")
        # Installed + daemon up + accessible? Friendly reason instead of a
        # 30-second timeout on `compose up`.
        ok, reason = docker_available()
        if not ok:
            return StepResult(StepStatus.SKIPPED, detail=f"docker not usable: {reason}")

        services = self._declared_services(ctx)
        if services:
            return self._apply_declared(ctx, services)
        return self._apply_whole_stack(ctx)

    def _apply_declared(self, ctx: StepContext, services: list[ExternalService]) -> StepResult:
        """Start exactly the recipe-declared ``docker_service`` names + probe them."""
        service_names = [svc.docker_service for svc in services if svc.docker_service]
        # Sequential up; docker handles per-image pull concurrency itself.
        up_result = stream_subprocess(
            ["docker", "compose", "up", "-d", *service_names],
            cwd=ctx.project_dir,
            step_id=self.id,
            callback=ctx.callback,
            # Vault-resolved env so compose ${VAR} interpolation works
            # without a plaintext .env file.
            timeout=self.timeout,
            env=ctx.runtime_env,
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

    def _apply_whole_stack(self, ctx: StepContext) -> StepResult:
        """Bring up the entire generated compose stack with native ``--wait``."""
        names = self._compose_services(ctx)
        if not names:
            return StepResult(StepStatus.SKIPPED, detail="docker-compose.yml declares no services")
        up = self._compose_up_wait(ctx)
        if up.exit_code != 0:
            return StepResult(
                StepStatus.FAILED,
                error=(
                    f"docker compose up timed out after {up.duration:.0f}s"
                    if up.timed_out
                    else f"docker compose up failed (exit {up.exit_code})"
                ),
                stderr_tail=up.stderr_tail,
            )
        return StepResult(StepStatus.DONE, detail=f"{len(names)} service(s) up and healthy")

    def _compose_up_wait(self, ctx: StepContext) -> SubprocessResult:
        """``docker compose up -d --wait``, degrading flags for older Compose.

        ``--wait`` (Compose 2.1.1+) blocks until healthchecks pass; ``--wait-timeout``
        (newer) bounds that. We try the richest form and only fall back on an
        *unknown-flag* error — a real failure (bad image, port clash) stops us
        rather than silently re-running ``up`` three times.
        """
        base = ["docker", "compose", "up", "-d"]
        attempts: tuple[list[str], ...] = (
            ["--wait", "--wait-timeout", str(int(self.wait_timeout))],
            ["--wait"],
            [],
        )
        result: SubprocessResult | None = None
        for extra in attempts:
            result = stream_subprocess(
                base + extra,
                cwd=ctx.project_dir,
                step_id=self.id,
                callback=ctx.callback,
                timeout=self.timeout,
                env=ctx.runtime_env,
            )
            if result.exit_code == 0 or not _is_unknown_flag_error(result.stderr_tail):
                return result
        assert result is not None  # attempts is non-empty
        return result

    def _compose_services(self, ctx: StepContext) -> list[str]:
        """Every service name in docker-compose.yml (``docker compose config --services``)."""
        out = _capture_stdout(
            ["docker", "compose", "config", "--services"],
            cwd=ctx.project_dir,
            timeout=15.0,
        )
        return [line.strip() for line in out.splitlines() if line.strip()]

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
            env=ctx.runtime_env,
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


def _is_unknown_flag_error(stderr: str) -> bool:
    """True if a docker CLI error is about an unsupported flag (vs a real failure)."""
    low = stderr.lower()
    return (
        "unknown flag" in low or "unknown shorthand flag" in low or "unknown docker command" in low
    )


def _load_recipe(ctx: StepContext) -> Recipe | None:
    """Look up the manifest's recipe slug in the configured deployments path."""
    from agent_scaffold.config import load_config
    from agent_scaffold.sources import SourceFetchError, resolve_deployments

    try:
        cfg = load_config()
    except Exception:  # noqa: BLE001 — config errors must not crash detect()
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


__all__ = ["DockerUpStep"]
