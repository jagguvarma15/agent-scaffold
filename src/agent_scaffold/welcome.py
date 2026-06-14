"""Render the post-``up`` welcome panel: every live local URL in one place.

After ``agent-scaffold up`` finishes the user wants to know two things:
"what URL do I open?" and "how do I stop everything?". The welcome panel
answers both by collecting URLs from three sources:

1. ``frontend.pid`` — written by ``launch_frontend`` after the dev server
   reports ready. Its ``port`` is authoritative for the frontend row.
2. The resolved capability stack — each capability's ``docker.ports`` gives
   us the host port to surface (Grafana, Langfuse, Qdrant, Tempo…).
3. Language hints — ``default_port`` tells us where the backend HTTP server
   lives (8000 for Python, 3000 for TypeScript).

The panel is fixed-order so the experience is stable across runs and is
easy to assert in tests. Rows whose source isn't present (no PID file, no
capability) are silently dropped. Each service URL is then liveness-probed
(a quick TCP connect) so we mark what's actually reachable — a row whose
service isn't running (``docker_up`` skipped, backend not started yet) shows
a dim ``○`` and a "not running" note instead of masquerading as a live link.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.panel import Panel
from rich.table import Table

from agent_scaffold._scaffold_dir import SCAFFOLD_DIR
from agent_scaffold.language_hints import UnknownLanguageError, load_language_hints
from agent_scaffold.manifest import Manifest

_GRAFANA_ADMIN_NOTE = "admin / $GRAFANA_ADMIN_PASSWORD (default: admin)"
_DEFAULT_BACKEND_PORT_BY_LANGUAGE = {"python": 8000, "typescript": 3000}


@dataclass(frozen=True)
class WelcomeRow:
    """One row in the welcome panel."""

    label: str
    url: str
    note: str = ""


def render_welcome_panel(
    project_dir: Path,
    manifest: Manifest,
    resolved_stack: Any | None,
    *,
    run_log_dir: str = "",
    probe: bool = True,
) -> Panel:
    """Build a Rich panel listing the local URLs, each marked live or not running.

    ``resolved_stack`` is ``agent_scaffold.capabilities.ResolvedStack | None``;
    typed as ``Any`` to avoid pulling the discovery dependency chain into
    every importer of this leaf module. ``run_log_dir`` adds a pointer row
    when the caller has a persistent run log for this invocation. ``probe``
    (default on) liveness-checks each service URL; pass ``False`` to skip the
    network round-trip (tests, ``--no-probe``).
    """
    rows = list(_collect_rows(project_dir, manifest, resolved_stack, run_log_dir=run_log_dir))
    live: dict[str, bool] = {}
    if probe:
        live = _probe_urls_live([row.url for row in rows if _is_probeable(row.url)])

    table = Table(show_header=False, box=None, expand=False, pad_edge=False)
    table.add_column("Service", style="bold cyan", no_wrap=True)
    table.add_column("URL")
    table.add_column("Note", style="dim", overflow="fold")
    for row in rows:
        url_cell = row.url
        note = row.note
        if probe and _is_probeable(row.url):
            if live.get(row.url, False):
                url_cell = f"[green]●[/] {row.url}"
            else:
                url_cell = f"[dim]○[/] {row.url}"
                note = "not running" + (f" · {row.note}" if row.note else "")
        table.add_row(row.label, url_cell, note)
    return Panel(
        table,
        title="[bold green]Ready[/] — local URLs",
        title_align="left",
        border_style="green",
        expand=False,
    )


def _is_probeable(url: str) -> bool:
    """Only http(s) service URLs get a liveness check — not file paths or commands."""
    return url.startswith(("http://", "https://"))


def _probe_urls_live(urls: list[str], *, timeout: float = 0.3) -> dict[str, bool]:
    """TCP-connect each URL's host:port concurrently; return ``{url: reachable}``.

    Best-effort liveness only: a refused or timed-out connection means "not
    running", never an error. Concurrent so a panel full of down services
    costs one timeout, not one per row.
    """
    import socket
    from concurrent.futures import ThreadPoolExecutor
    from urllib.parse import urlparse

    def _reachable(url: str) -> bool:
        try:
            parsed = urlparse(url)
            host = parsed.hostname or "localhost"
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    if not urls:
        return {}
    with ThreadPoolExecutor(max_workers=min(8, len(urls))) as pool:
        return dict(zip(urls, pool.map(_reachable, urls), strict=True))


def _collect_rows(
    project_dir: Path,
    manifest: Manifest,
    resolved_stack: Any | None,
    *,
    run_log_dir: str = "",
) -> Iterable[WelcomeRow]:
    """Yield rows in the fixed order documented in the module docstring."""
    capabilities_by_id = _capabilities_by_id(resolved_stack)

    frontend = _frontend_row(project_dir)
    if frontend is not None:
        yield frontend

    yield _backend_row(manifest)

    if "obs.grafana-stack" in capabilities_by_id:
        port = _first_host_port(capabilities_by_id["obs.grafana-stack"])
        if port is not None:
            yield WelcomeRow(
                label="Grafana",
                url=f"http://localhost:{port}",
                note=_GRAFANA_ADMIN_NOTE,
            )
        # Tempo is derived from the grafana-stack capability. Its container
        # listens on 3200 by default; the capability's emit_files plumbs the
        # tempo.yaml so the service is up. Surface the URL even when the
        # service doesn't appear in this capability's docker.ports (Tempo is
        # a sibling service emitted via emit_files).
        yield WelcomeRow(label="Tempo", url="http://localhost:3200")

    if "obs.langfuse" in capabilities_by_id:
        port = _first_host_port(capabilities_by_id["obs.langfuse"])
        if port is not None:
            yield WelcomeRow(label="Langfuse", url=f"http://localhost:{port}")

    if "vector_db.qdrant" in capabilities_by_id:
        port = _first_host_port(capabilities_by_id["vector_db.qdrant"])
        if port is not None:
            yield WelcomeRow(label="Qdrant", url=f"http://localhost:{port}/dashboard")

    if _manifest_has_eval_capability(manifest, capabilities_by_id):
        baseline = _read_eval_baseline_text(manifest)
        note = (
            f"baseline {baseline} — exits 1 on regression"
            if baseline is not None
            else "run the eval suite against this project"
        )
        yield WelcomeRow(label="Eval", url="agent-scaffold eval", note=note)

    summary_path = project_dir / SCAFFOLD_DIR / "run-summary.md"
    if summary_path.is_file():
        yield WelcomeRow(
            label="Run summary",
            url=str(summary_path),
            note="what was generated + how to start it",
        )
    if run_log_dir:
        yield WelcomeRow(label="Run log", url=run_log_dir)

    yield WelcomeRow(
        label="Stop everything",
        url="agent-scaffold down",
        note="-v also wipes named volumes",
    )


def _frontend_row(project_dir: Path) -> WelcomeRow | None:
    pid_file = project_dir / SCAFFOLD_DIR / "frontend.pid"
    if not pid_file.is_file():
        return None
    try:
        data = json.loads(pid_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    port = data.get("port")
    if not isinstance(port, int) or port <= 0:
        return None
    return WelcomeRow(label="Frontend", url=f"http://localhost:{port}")


def _backend_row(manifest: Manifest) -> WelcomeRow:
    port = _default_backend_port(manifest.language)
    return WelcomeRow(label="Backend", url=f"http://localhost:{port}")


def _default_backend_port(language: str) -> int:
    """Read ``default_port`` from the language hints; fall back to language map."""
    try:
        hints = load_language_hints(language)
    except UnknownLanguageError:
        return _DEFAULT_BACKEND_PORT_BY_LANGUAGE.get(language, 8000)
    raw = hints.get("default_port")
    if isinstance(raw, int) and raw > 0:
        return raw
    return _DEFAULT_BACKEND_PORT_BY_LANGUAGE.get(language, 8000)


def _capabilities_by_id(resolved_stack: Any | None) -> dict[str, Any]:
    if resolved_stack is None:
        return {}
    caps = getattr(resolved_stack, "capabilities", None) or []
    return {getattr(cap, "id", ""): cap for cap in caps if getattr(cap, "id", "")}


def _first_host_port(capability: Any) -> int | None:
    """Return the host port from the first ``"host:container"`` mapping.

    Returns ``None`` if the capability has no docker block, no ports
    declared, or the first entry is malformed.
    """
    docker = getattr(capability, "docker", None)
    if docker is None:
        return None
    ports = getattr(docker, "ports", None) or []
    for entry in ports:
        port = _parse_host_port(entry)
        if port is not None:
            return port
    return None


def _parse_host_port(entry: str) -> int | None:
    """Parse a ``"host:container"`` (or ``"host:container/proto"``) mapping."""
    if not isinstance(entry, str) or ":" not in entry:
        return None
    head = entry.split(":", 1)[0]
    try:
        return int(head)
    except ValueError:
        return None


def _read_eval_baseline_text(manifest: Manifest) -> str | None:
    """Return ``manifest.answers["eval_baseline"]`` formatted for display, or ``None``."""
    raw = (manifest.answers or {}).get("eval_baseline")
    if not raw:
        return None
    try:
        return f"{float(raw):.2f}"
    except (TypeError, ValueError):
        return None


def _manifest_has_eval_capability(manifest: Manifest, capabilities_by_id: dict[str, Any]) -> bool:
    """True iff the recipe declared any ``eval.*`` capability."""
    for cap_id in manifest.capabilities or capabilities_by_id.keys():
        if cap_id.startswith("eval."):
            return True
    return False


def _open_browser_safe(url: str) -> bool:
    """Open ``url`` in the user's default browser. Swallows headless/CI failures.

    Lives here rather than in ``cli.py`` so the autorun brief (next PR) can
    call it from a flow that doesn't depend on Typer.
    """
    import webbrowser

    if os.environ.get("BROWSER") == "none":
        return False
    try:
        return webbrowser.open(url, new=2)
    except Exception:  # noqa: BLE001 — webbrowser raises a grab-bag of OS errors
        return False


__all__ = [
    "WelcomeRow",
    "render_welcome_panel",
    "_open_browser_safe",
]
