"""``bootstrap_observability`` step: provision Grafana datasources + dashboards.

Runs after ``docker_up`` for the ``obs.grafana-stack`` capability:

1. Wait for Grafana's ``GET /api/health`` to return ``database: ok``.
2. POST Prometheus + Tempo datasources via ``/api/datasources`` (idempotent
   on duplicate-name 409).
3. POST each dashboard JSON shipped under ``ops/grafana/dashboards/*.json``
   via ``/api/dashboards/db`` (idempotent on ``uid``).

Uses stdlib ``urllib.request`` only — no new dependency. Admin auth via
HTTP Basic with ``GRAFANA_ADMIN_PASSWORD`` (default ``admin``).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from agent_scaffold.orchestrator import (
    DetectionResult,
    StepContext,
    StepLog,
    StepProgress,
    StepResult,
    StepStatus,
    compute_fingerprint,
)

log = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 60.0
_HEALTH_POLL_INTERVAL = 2.0
_DEFAULT_GRAFANA_URL = "http://localhost:3002"

_DATASOURCES = [
    {
        "env_url": "PROMETHEUS_URL",
        "default_url": "http://prometheus:9090",
        "payload": {
            "name": "Prometheus",
            "type": "prometheus",
            "access": "proxy",
            "isDefault": True,
        },
    },
    {
        "env_url": "TEMPO_URL",
        "default_url": "http://tempo:3200",
        "payload": {
            "name": "Tempo",
            "type": "tempo",
            "access": "proxy",
            "isDefault": False,
        },
    },
]


@dataclass
class BootstrapObservabilityStep:
    """Provision Grafana datasources + dashboards declared by the recipe."""

    id: str = "bootstrap_observability"
    description: str = "Provision Grafana datasources + dashboards"
    depends_on: tuple[str, ...] = ("docker_up",)
    timeout: float = _DEFAULT_TIMEOUT
    troubleshoot: dict[str, str] = field(
        default_factory=lambda: {
            "401": "rotate GRAFANA_ADMIN_PASSWORD (default 'admin' rejected in prod images)",
            "connection refused": (
                "grafana container not reachable — `agent-scaffold up --retry docker_up`"
            ),
        }
    )

    # ---- detection ----------------------------------------------------

    def detect(self, ctx: StepContext) -> DetectionResult:
        if not self._has_capability(ctx):
            return DetectionResult(
                StepStatus.SKIPPED,
                reason="recipe declares no obs.grafana-stack capability",
            )
        dashboards = _list_dashboards(ctx.project_dir)
        return DetectionResult(
            StepStatus.PENDING,
            reason=f"datasources + {len(dashboards)} dashboard(s)",
        )

    # ---- apply --------------------------------------------------------

    def apply(self, ctx: StepContext) -> StepResult:
        if not self._has_capability(ctx):
            return StepResult(StepStatus.SKIPPED, detail="no obs.grafana-stack capability")
        base = os.environ.get("GRAFANA_URL", _DEFAULT_GRAFANA_URL).rstrip("/")
        admin_password = os.environ.get("GRAFANA_ADMIN_PASSWORD", "admin")
        auth_header = _basic_auth("admin", admin_password)
        if not _wait_for_health(base, timeout=self.timeout, ctx=ctx, step_id=self.id):
            return StepResult(
                StepStatus.FAILED,
                error=f"grafana: /api/health never returned ok at {base}",
            )

        ds_added, ds_skipped = _ensure_datasources(base, auth_header, ctx, self.id)
        dash_added, dash_skipped = _ensure_dashboards(
            base, auth_header, ctx.project_dir, ctx, self.id
        )
        return StepResult(
            StepStatus.DONE,
            detail=(
                f"datasources: {ds_added} added / {ds_skipped} existed; "
                f"dashboards: {dash_added} added / {dash_skipped} existed"
            ),
        )

    # ---- fingerprint --------------------------------------------------

    def fingerprint(self, ctx: StepContext) -> str:
        dashboards = _list_dashboards(ctx.project_dir)
        return compute_fingerprint(
            {
                "has_capability": self._has_capability(ctx),
                "grafana_url": os.environ.get("GRAFANA_URL", _DEFAULT_GRAFANA_URL),
                "dashboards": [str(p) for p in sorted(dashboards)],
            }
        )

    # ---- helpers ------------------------------------------------------

    def _has_capability(self, ctx: StepContext) -> bool:
        stack = ctx.resolved_stack
        if stack is None:
            return False
        return any(c.id == "obs.grafana-stack" for c in stack.capabilities)


def _basic_auth(user: str, password: str) -> str:
    token = base64.b64encode(f"{user}:{password}".encode()).decode("ascii")
    return f"Basic {token}"


def _http_request(
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes | None = None,
    timeout: float = 10.0,
) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, method=method)
    for key, value in headers.items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — list-form URL, no shell
            return resp.getcode(), resp.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read() or b""


def _wait_for_health(
    base: str, *, timeout: float, ctx: StepContext, step_id: str
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ctx.emit(StepProgress(step_id=step_id, message=f"healthcheck: {base}/api/health"))
        try:
            code, body = _http_request("GET", f"{base}/api/health", headers={}, timeout=5.0)
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(_HEALTH_POLL_INTERVAL)
            continue
        if code == 200:
            try:
                data = json.loads(body)
            except ValueError:
                data = {}
            if isinstance(data, dict) and data.get("database") == "ok":
                ctx.emit(StepLog(step_id=step_id, line="grafana: /api/health OK"))
                return True
        time.sleep(_HEALTH_POLL_INTERVAL)
    return False


def _ensure_datasources(
    base: str, auth_header: str, ctx: StepContext, step_id: str
) -> tuple[int, int]:
    headers = {"Content-Type": "application/json", "Authorization": auth_header}
    added = 0
    skipped = 0
    for spec in _DATASOURCES:
        url_env = str(spec["env_url"])
        default_url = str(spec["default_url"])
        ds_url = os.environ.get(url_env, default_url)
        payload = {**spec["payload"], "url": ds_url}  # type: ignore[dict-item]
        body = json.dumps(payload).encode("utf-8")
        code, raw = _http_request(
            "POST", f"{base}/api/datasources", headers=headers, body=body, timeout=10.0
        )
        if code == 200 or code == 201:
            added += 1
            ctx.emit(
                StepLog(step_id=step_id, line=f"grafana: datasource {payload['name']} added")
            )
        elif code == 409 or b"data source with the same name" in raw.lower():
            skipped += 1
        else:
            log.warning(
                "bootstrap_observability: datasource %s POST returned %d: %s",
                payload["name"],
                code,
                raw[:200],
            )
            skipped += 1
    return added, skipped


def _list_dashboards(project_dir: Path) -> list[Path]:
    target = project_dir / "ops" / "grafana" / "dashboards"
    if not target.is_dir():
        return []
    return sorted(target.glob("*.json"))


def _ensure_dashboards(
    base: str,
    auth_header: str,
    project_dir: Path,
    ctx: StepContext,
    step_id: str,
) -> tuple[int, int]:
    headers = {"Content-Type": "application/json", "Authorization": auth_header}
    added = 0
    skipped = 0
    for dashboard_path in _list_dashboards(project_dir):
        try:
            dashboard_json = json.loads(dashboard_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("bootstrap_observability: skipping %s: %s", dashboard_path, exc)
            skipped += 1
            continue
        payload = {
            "dashboard": dashboard_json,
            "overwrite": True,  # idempotent: update by uid if exists
        }
        body = json.dumps(payload).encode("utf-8")
        code, raw = _http_request(
            "POST", f"{base}/api/dashboards/db", headers=headers, body=body, timeout=10.0
        )
        if code in (200, 412):  # 412 = version conflict, treated as already-present
            added += 1
            ctx.emit(
                StepLog(step_id=step_id, line=f"grafana: dashboard {dashboard_path.name} added")
            )
        else:
            log.warning(
                "bootstrap_observability: dashboard %s POST returned %d: %s",
                dashboard_path.name,
                code,
                raw[:200],
            )
            skipped += 1
    return added, skipped


__all__ = ["BootstrapObservabilityStep"]
