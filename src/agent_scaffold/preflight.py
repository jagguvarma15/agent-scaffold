"""Pre-flight environment + service checks, run before the LLM call.

``agent-scaffold new`` spends real money on generation; until now the first
moment a missing secret or unreachable service surfaced was during ``up`` —
*after* the spend. This module gates the golden path:

1. **Environment**: union of every env var the run will eventually need —
   the recipe's ``external_services[].env_vars``, the catalog's auto-derived
   ``env_contract`` (entries with a ``default`` count as satisfied), and the
   resolved capability stack's ``env_vars``. Presence is checked through the
   same :mod:`agent_scaffold.envfile` helpers the ``wire_credentials`` step
   uses, so the two stages can never disagree about what "set" means.
2. **Fill now (optional)**: interactively prompt (``getpass`` — values never
   echo) for anything missing. ``ANTHROPIC_API_KEY`` persists to the auth
   backend immediately; project secrets are exported to ``os.environ`` for
   this run and persisted to the project's ``.env.local`` *after* generation
   writes the destination directory (it doesn't exist yet at gate time).
3. **Services**: the recipe's external services are probed concurrently.
   Failures are warn-only — a service backed by ``docker_service`` is
   *expected* to be down before ``up`` starts compose — so the gate never
   blocks generation; it just makes the state visible before the spend.

Non-interactive runs never prompt: missing names (never values) are printed
to stderr and the report is returned without probing.
"""

from __future__ import annotations

import getpass as _getpass
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import SecretStr
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agent_scaffold.auth import ENV_API_KEY, AuthError, store_key
from agent_scaffold.catalog import RecipeEntry
from agent_scaffold.discovery import ExternalService, Recipe
from agent_scaffold.doctor import CheckResult, CheckStatus
from agent_scaffold.envfile import append_env_local, is_present, read_env_local

_STATUS_SYMBOL: dict[CheckStatus, tuple[str, str]] = {
    CheckStatus.OK: ("✓", "green"),
    CheckStatus.WARN: ("⚠", "yellow"),
    CheckStatus.FAIL: ("✗", "red"),
    CheckStatus.SKIP: ("⏭", "dim"),
}


@dataclass(frozen=True)
class EnvRequirement:
    """One env var the generated project will need, with where it came from."""

    name: str
    source: str
    required: bool
    satisfied: bool
    has_default: bool = False


@dataclass
class PreflightReport:
    requirements: list[EnvRequirement] = field(default_factory=list)
    probe_results: list[CheckResult] = field(default_factory=list)
    # Values collected interactively at the gate (name → secret). Exported to
    # os.environ immediately; persisted to .env.local post-write by
    # :func:`persist_filled`. ANTHROPIC_API_KEY never lands here — it goes
    # straight to the auth backend.
    filled: dict[str, SecretStr] = field(default_factory=dict)

    @property
    def missing(self) -> list[EnvRequirement]:
        return [r for r in self.requirements if not r.satisfied]

    @property
    def missing_required(self) -> list[EnvRequirement]:
        return [r for r in self.missing if r.required]


def collect_env_requirements(
    recipe: Recipe,
    catalog_entry: RecipeEntry | None,
    resolved_stack: Any | None,
    project_dir: Path,
) -> list[EnvRequirement]:
    """Union the three env-var sources, first-seen order, presence-checked.

    Merge rules when a var appears in several sources: the first source
    label wins (recipe-declared services are most specific), ``required``
    is OR'd, ``has_default`` is OR'd.
    """
    merged: dict[str, dict[str, Any]] = {}

    def _add(name: str, source: str, *, required: bool, has_default: bool = False) -> None:
        name = name.strip()
        if not name:
            return
        entry = merged.get(name)
        if entry is None:
            merged[name] = {"source": source, "required": required, "has_default": has_default}
        else:
            entry["required"] = entry["required"] or required
            entry["has_default"] = entry["has_default"] or has_default

    for svc in recipe.external_services:
        for var in svc.env_vars:
            _add(var, svc.id, required=bool(svc.required))

    if catalog_entry is not None:
        for contract in catalog_entry.env_contract:
            _add(
                contract.name,
                contract.source_capability or "recipe",
                required=True,
                has_default=contract.default is not None,
            )

    if resolved_stack is not None:
        for cap in getattr(resolved_stack, "capabilities", []):
            for var in getattr(cap, "env_vars", []):
                _add(var, cap.id, required=True)

    env_local = read_env_local(project_dir) if project_dir.is_dir() else {}
    return [
        EnvRequirement(
            name=name,
            source=str(entry["source"]),
            required=bool(entry["required"]),
            satisfied=bool(entry["has_default"]) or is_present(name, env_local),
            has_default=bool(entry["has_default"]),
        )
        for name, entry in merged.items()
    ]


def render_env_panel(requirements: list[EnvRequirement]) -> Panel:
    table = Table.grid(padding=(0, 2))
    table.add_column(width=1)
    table.add_column()
    table.add_column(style="dim")
    for req in requirements:
        if req.satisfied:
            sym, style = ("✓", "green") if not req.has_default else ("✓", "cyan")
        elif req.required:
            sym, style = "✗", "red"
        else:
            sym, style = "○", "yellow"
        note = req.source
        if req.has_default:
            note += "  (recipe default)"
        elif not req.required and not req.satisfied:
            note += "  (optional)"
        table.add_row(Text(sym, style=style), req.name, note)
    missing = [r for r in requirements if not r.satisfied]
    footer = (
        Text(f"\n{len(missing)} value(s) missing — fill now or during `up`.", style="yellow")
        if missing
        else Text("\nAll declared env vars resolvable.", style="green")
    )
    from rich.console import Group

    return Panel(
        Group(table, footer),
        title="Pre-flight: environment",
        expand=False,
    )


def render_service_panel(results: list[CheckResult], services: list[ExternalService]) -> Panel:
    """Service probe outcomes, softened for docker-managed services.

    A FAIL against a service that ``up`` will start via docker compose is
    expected before first provisioning — render it as informational instead
    of alarming.
    """
    docker_backed = {svc.id for svc in services if svc.docker_service}
    table = Table.grid(padding=(0, 2))
    table.add_column(width=1)
    table.add_column()
    table.add_column(style="dim")
    for res in results:
        sym, style = _STATUS_SYMBOL.get(res.status, ("•", "white"))
        note = res.detail or ""
        if res.status is CheckStatus.FAIL and res.id in docker_backed:
            sym, style = "⏸", "dim"
            note = "not running — `up` starts it via docker compose"
        table.add_row(Text(sym, style=style), res.title, note)
    return Panel(table, title="Pre-flight: services", expand=False)


def fill_missing(
    report: PreflightReport,
    console: Console,
    *,
    ask: Callable[[str], str] | None = None,
) -> None:
    """Prompt for each missing var; empty input skips. Mutates ``report``.

    Values are read via getpass (no echo). ``ANTHROPIC_API_KEY`` persists to
    the auth backend (keyring → 0600 file fallback) right away; everything
    else is exported to ``os.environ`` for this process and queued on
    ``report.filled`` for post-write persistence to ``.env.local``.
    """
    import os

    def _getpass_ask(prompt: str) -> str:
        return _getpass.getpass(prompt)

    asker: Callable[[str], str] = ask if ask is not None else _getpass_ask
    updated: dict[str, EnvRequirement] = {}
    for req in report.missing:
        label = "required" if req.required else "optional — Enter to skip"
        try:
            raw = asker(f"  {req.name} ({req.source}, {label}): ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("[yellow]Fill aborted — remaining values stay unset.[/]")
            break
        if not raw:
            continue
        secret = SecretStr(raw)
        if req.name == ENV_API_KEY:
            backend = _store_anthropic_key(secret, console)
            if backend is None:
                continue
            console.print(f"  [green]{req.name}[/] → stored ({backend})")
        else:
            os.environ[req.name] = raw
            report.filled[req.name] = secret
            console.print(f"  [green]{req.name}[/] → set for this run (persists after write)")
        updated[req.name] = EnvRequirement(
            name=req.name,
            source=req.source,
            required=req.required,
            satisfied=True,
            has_default=req.has_default,
        )
    if updated:
        report.requirements = [updated.get(r.name, r) for r in report.requirements]


def _store_anthropic_key(secret: SecretStr, console: Console) -> str | None:
    try:
        store_key("anthropic", secret, backend="keyring")
        return "keyring"
    except AuthError:
        try:
            store_key("anthropic", secret, backend="file")
            return "credentials file (mode 0600)"
        except AuthError as exc:
            console.print(f"[red]Could not store key:[/] {exc}")
            return None


def persist_filled(project_dir: Path, filled: dict[str, SecretStr]) -> list[str]:
    """Write gate-collected secrets to the project's ``.env.local`` (0600).

    Called after generation has written ``project_dir`` (the directory does
    not exist at gate time). The pipeline's gitignore pass already covers
    ``.env.local``. Returns the names persisted; failures drop to os.environ
    -only (the value still works for this run's ``up``).
    """
    persisted: list[str] = []
    if not project_dir.is_dir():
        return persisted
    for name, secret in filled.items():
        try:
            append_env_local(project_dir, name, secret)
        except OSError:
            continue
        persisted.append(name)
    return persisted


def run_preflight(
    *,
    recipe: Recipe,
    catalog_entry: RecipeEntry | None,
    resolved_stack: Any | None,
    project_dir: Path,
    console: Console,
    interactive: bool,
    probe: Callable[[list[ExternalService]], list[CheckResult]],
    confirm: Callable[[str], bool] | None = None,
    ask: Callable[[str], str] | None = None,
) -> PreflightReport:
    """The gate. Never blocks generation — warn-only by design."""
    requirements = collect_env_requirements(recipe, catalog_entry, resolved_stack, project_dir)
    report = PreflightReport(requirements=requirements)

    if not interactive:
        missing = report.missing_required
        if missing:
            names = ", ".join(r.name for r in missing)
            print(
                f"agent-scaffold: pre-flight: missing env var(s): {names} — "
                "`up` will need them; set them in the environment beforehand.",
                file=sys.stderr,
            )
        return report

    if requirements:
        console.print(render_env_panel(report.requirements))
    if report.missing:
        asker_confirm = confirm if confirm is not None else _default_confirm
        if asker_confirm("Fill missing values now?"):
            fill_missing(report, console, ask=ask)

    report.probe_results = probe(recipe.external_services)
    if report.probe_results:
        console.print(render_service_panel(report.probe_results, recipe.external_services))
    return report


def _default_confirm(prompt: str) -> bool:
    import typer

    try:
        return bool(typer.confirm(prompt, default=True))
    except (typer.Abort, EOFError):
        return False


__all__ = [
    "EnvRequirement",
    "PreflightReport",
    "collect_env_requirements",
    "fill_missing",
    "persist_filled",
    "render_env_panel",
    "render_service_panel",
    "run_preflight",
]
