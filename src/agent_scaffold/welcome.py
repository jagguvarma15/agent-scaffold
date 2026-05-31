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
capability) are silently dropped — only what the user can actually click is
shown.
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
) -> Panel:
    """Build a Rich panel listing every live local URL the user can hit.

    ``resolved_stack`` is ``agent_scaffold.capabilities.ResolvedStack | None``;
    typed as ``Any`` to avoid pulling the discovery dependency chain into
    every importer of this leaf module.
    """
    rows = list(_collect_rows(project_dir, manifest, resolved_stack))
    table = Table(show_header=False, box=None, expand=False, pad_edge=False)
    table.add_column("Service", style="bold cyan", no_wrap=True)
    table.add_column("URL")
    table.add_column("Note", style="dim", overflow="fold")
    for row in rows:
        table.add_row(row.label, row.url, row.note)
    return Panel(
        table,
        title="[bold green]Ready[/] — local URLs",
        title_align="left",
        border_style="green",
        expand=False,
    )


def _collect_rows(
    project_dir: Path,
    manifest: Manifest,
    resolved_stack: Any | None,
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
        yield WelcomeRow(
            label="Eval",
            url="agent-scaffold eval",
            note="run the eval suite against this project",
        )

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
