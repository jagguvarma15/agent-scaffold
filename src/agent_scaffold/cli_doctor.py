"""``agent-scaffold doctor`` sub-app: read-only environment + recipe audit.

Lifted out of ``cli.py`` so all doctor concerns — the report renderer,
the JSON encoder, the ``--explain`` doc resolver, the auth + service
:class:`Check` adapters, and the Typer callback — live in one module.
Previously these were scattered across ~1100 LOC of ``cli.py``.

The Typer sub-app is exported as ``doctor_app``; ``cli.py`` wires it
onto the parent ``app`` via ``app.add_typer(doctor_app, name="doctor")``.

``doctor`` is intentionally read-only: it never mutates user state. Its
job is to surface what's broken and how to fix it.
"""

from __future__ import annotations

import importlib.resources as resources
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer

from agent_scaffold.auth import (
    AuthError,
    describe_backend,
    detect_backend,
    resolve_active,
)
from agent_scaffold.capabilities import (
    ResolvedStack,
    load_capabilities,
)
from agent_scaffold.capabilities import resolve as resolve_capabilities
from agent_scaffold.cli_shared import console
from agent_scaffold.config import ConfigError, load_config
from agent_scaffold.discovery import (
    DiscoveryError,
    ExternalService,
    Recipe,
    discover_recipes,
)
from agent_scaffold.doctor import (
    Check,
    CheckResult,
    CheckStatus,
    DoctorReport,
    baseline_checks,
    run_checks,
)
from agent_scaffold.sources import SourceFetchError, resolve_deployments

doctor_app = typer.Typer(
    name="doctor",
    help="Read-only environment + recipe audit. Never mutates.",
    invoke_without_command=True,
)


# ---------------------------------------------------------------------------
# Rendering — Rich and JSON outputs
# ---------------------------------------------------------------------------

_DOCTOR_ICONS: dict[CheckStatus, str] = {
    CheckStatus.OK: "✓",
    CheckStatus.WARN: "⚠",
    CheckStatus.FAIL: "✗",
    CheckStatus.SKIP: "⏭",
}

_DOCTOR_COLORS: dict[CheckStatus, str] = {
    CheckStatus.OK: "green",
    CheckStatus.WARN: "yellow",
    CheckStatus.FAIL: "red",
    CheckStatus.SKIP: "dim cyan",
}


def _doctor_render(report: DoctorReport) -> None:
    """Render the report as grouped category sections — one-shot, no Live."""
    if not report.results:
        console.print("[dim]No checks ran.[/]")
        return
    # Preserve the order categories first appeared in.
    seen: dict[str, list[CheckResult]] = {}
    for r in report.results:
        seen.setdefault(r.category, []).append(r)
    for idx, (category, rows) in enumerate(seen.items()):
        if idx > 0:
            console.print()
        console.print(f"[bold]{category}[/]")
        for r in rows:
            color = _DOCTOR_COLORS[r.status]
            icon = _DOCTOR_ICONS[r.status]
            line = f"  [{color}]{icon}[/] {r.title}"
            if r.detail:
                line += f"   [dim]{r.detail}[/]"
            console.print(line)
            if r.fix_hint:
                console.print(f"      [dim]→[/] {r.fix_hint}")
            if r.status == CheckStatus.FAIL and r.explain_topic:
                console.print(f"      [dim]→[/] agent-scaffold doctor --explain {r.explain_topic}")
    s = report.summary
    console.print()
    console.print(
        f"Summary: {s[CheckStatus.OK.value]} ok, {s[CheckStatus.WARN.value]} warn, "
        f"{s[CheckStatus.FAIL.value]} fail, {s[CheckStatus.SKIP.value]} skip"
    )


def _doctor_json(report: DoctorReport) -> str:
    payload = {
        "schema_version": 1,
        "results": [
            {
                "id": r.id,
                "category": r.category,
                "status": r.status.value,
                "title": r.title,
                "detail": r.detail,
                "fix_hint": r.fix_hint,
                "explain_topic": r.explain_topic,
            }
            for r in report.results
        ],
        "summary": report.summary,
        "exit_code": report.exit_code,
    }
    return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# --explain <topic> — pager-friendly getting-started doc lookup
# ---------------------------------------------------------------------------


def _resolve_explain_doc(topic: str) -> Path | None:
    """Resolve ``--explain <topic>`` to a markdown path.

    Bundled docs win over the live deployments checkout to keep the offline
    story honest. Q4 will write these getting-started docs; Q1 may return
    ``None`` if neither location has the slug yet — the caller fails soft.
    """
    try:
        ref = resources.files("agent_scaffold._bundled_deployments").joinpath(
            f"docs/getting-started/{topic}.md"
        )
        candidate = Path(str(ref))
        if candidate.is_file():
            return candidate
    except (FileNotFoundError, ModuleNotFoundError):
        pass

    try:
        cfg = load_config()
    except ConfigError:
        return None
    # Best-effort: only consult an explicit local path (env / TOML / flag).
    # We don't auto-fetch here — this helper runs on basically every CLI
    # invocation through the doctor and we don't want a network hop.
    if cfg.deployments_path is None:
        return None
    live_candidate = cfg.deployments_path.expanduser() / "docs" / "getting-started" / f"{topic}.md"
    if live_candidate.is_file():
        return live_candidate
    return None


def _explain_topic(topic: str) -> int:
    """Show the getting-started doc for ``topic``. Returns process exit code."""
    chosen = _resolve_explain_doc(topic)
    if chosen is None:
        console.print(f"[yellow]No docs yet for {topic!r}[/] — see Q4")
        return 0

    text = chosen.read_text(encoding="utf-8")
    pager = os.environ.get("PAGER")
    if not pager or not sys.stdout.isatty():
        console.print(text)
        return 0

    try:
        proc = subprocess.run(
            [*pager.split(), str(chosen)],
            check=False,
            shell=False,
        )
        return int(proc.returncode)
    except (FileNotFoundError, OSError):
        console.print(text)
        return 0


# ---------------------------------------------------------------------------
# Auth + service checks — Check Protocol adapters owned by doctor
# ---------------------------------------------------------------------------


@dataclass
class _AuthBackendCheck:
    id: str = "auth.backend"
    category: str = "Authentication"

    def run(self) -> CheckResult:
        try:
            detect_backend()
        except AuthError as exc:
            return CheckResult(
                id=self.id,
                category=self.category,
                status=CheckStatus.WARN,
                title=f"keyring backend: {describe_backend()}",
                detail=str(exc),
                fix_hint="agent-scaffold auth login --use-file (falls back to mode-0600 file)",
                explain_topic="keyring",
            )
        return CheckResult(
            id=self.id,
            category=self.category,
            status=CheckStatus.OK,
            title=f"keyring backend: {describe_backend()}",
            explain_topic="keyring",
        )


@dataclass
class _AuthKeyCheck:
    id: str = "auth.anthropic_key"
    category: str = "Authentication"

    def run(self) -> CheckResult:
        active = resolve_active()
        if active is None:
            return CheckResult(
                id=self.id,
                category=self.category,
                status=CheckStatus.FAIL,
                title="anthropic key: not resolved",
                detail="checked ANTHROPIC_API_KEY, keyring, credentials file",
                fix_hint="agent-scaffold auth login",
                explain_topic="anthropic",
            )
        _, source = active
        return CheckResult(
            id=self.id,
            category=self.category,
            status=CheckStatus.OK,
            title=f"anthropic key: resolved from {source}",
            explain_topic="anthropic",
        )


def _auth_checks() -> list[Check]:
    return [_AuthBackendCheck(), _AuthKeyCheck()]


@dataclass
class _ServiceCheck:
    """``Check`` wrapper around ``probes.run_probe``.

    The runner builds these in ``cmd_doctor`` / ``cmd_new`` and hands them to
    ``run_checks``. ``timeout`` and ``skip`` are baked in at construction time
    so the ``run()`` signature stays Protocol-compatible.
    """

    service: ExternalService
    timeout: float = 5.0
    skip: bool = False
    id: str = ""  # populated in __post_init__; declared so the Protocol matches
    category: str = "Recipe services"

    def __post_init__(self) -> None:
        self.id = f"service.{self.service.id}"

    def run(self) -> CheckResult:
        from agent_scaffold.probes import run_probe

        return run_probe(self.service, timeout=self.timeout, skip=self.skip)


def _service_checks(services: list[ExternalService], *, timeout: float, skip: bool) -> list[Check]:
    checks: list[Check] = [
        _ServiceCheck(service=svc, timeout=timeout, skip=skip) for svc in services
    ]
    return checks


@dataclass
class _CapabilityCheck:
    """``Check`` adapter that reports a single resolved capability.

    Capabilities themselves don't have a network probe at the resolver layer
    (probes live with the underlying service in :data:`probes.PROBES`); this
    check exists so the resolved set surfaces in ``doctor --recipe`` as one
    OK row per capability with kind + probe + bootstrap metadata. Unresolved
    capability ids land as WARN rows.
    """

    id: str
    category: str = "Capabilities"
    title: str = ""
    detail: str = ""
    status: CheckStatus = CheckStatus.OK
    fix_hint: str = ""

    def run(self) -> CheckResult:
        return CheckResult(
            id=self.id,
            category=self.category,
            status=self.status,
            title=self.title,
            detail=self.detail,
            fix_hint=self.fix_hint,
        )


def _capability_meta(cap: Any) -> str:
    bits: list[str] = []
    if cap.probe:
        bits.append(f"probe: {cap.probe}")
    if cap.bootstrap_step:
        bits.append(f"bootstrap: {cap.bootstrap_step}")
    if cap.docker is not None:
        bits.append(f"docker: {cap.docker.service}")
    return ", ".join(bits) if bits else "(no probe / bootstrap declared)"


def _capability_checks(stack: ResolvedStack) -> list[Check]:
    """Build one ``_CapabilityCheck`` per resolved capability + per unresolved id.

    Static metadata rows only — ``doctor --recipe`` runs pre-generation with
    no project env, so nothing is probed here. A generated project's live view
    is :func:`probed_capability_results` (used by ``agent-scaffold status``).
    """
    checks: list[Check] = []
    for cap in stack.capabilities:
        checks.append(
            _CapabilityCheck(
                id=f"capability.{cap.id}",
                status=CheckStatus.OK,
                title=f"{cap.id} ({cap.kind})",
                detail=_capability_meta(cap),
            )
        )
    checks.extend(_unresolved_capability_checks(stack))
    return checks


def _unresolved_capability_checks(stack: ResolvedStack) -> list[Check]:
    return [
        _CapabilityCheck(
            id=f"capability.{cap_id}",
            status=CheckStatus.WARN,
            title=f"{cap_id} (unresolved)",
            detail="not found in deployments docs/capabilities/",
            fix_hint="upgrade your deployments source or remove from the recipe",
        )
        for cap_id in stack.unresolved
    ]


def _service_from_capability(cap: Any) -> ExternalService:
    """Bridge a resolved capability into the probeable ``ExternalService`` shape."""
    from agent_scaffold.stack_options import default_local_endpoint

    return ExternalService(
        id=cap.id,
        required=False,
        env_vars=list(cap.env_vars),
        default_local=default_local_endpoint(_capability_stem(cap.id)),
        docker_service=cap.docker.service if cap.docker is not None else None,
        probe=cap.probe,
    )


def _capability_stem(cap_id: str) -> str:
    return cap_id.split(".", 1)[1] if "." in cap_id else cap_id


def probed_capability_results(
    stack: ResolvedStack, *, env: Mapping[str, str] | None, timeout: float
) -> list[CheckResult]:
    """Live capability health: run each declared probe with the project env.

    Capabilities with a ``probe`` run it (in parallel, through the shared
    thread pool) against ``env`` — the project's runtime env, so vault-stored
    managed credentials count. Probe-less capabilities keep their static
    metadata row. Failing rows carry a remediation hint: the probe's own hint
    when it has one, otherwise ``agent-scaffold connect <option>`` for cloud
    capabilities and ``agent-scaffold up`` for docker-backed ones.
    """
    import dataclasses

    from agent_scaffold.probes import probe_external_services

    probed = [cap for cap in stack.capabilities if cap.probe]
    results_by_id: dict[str, CheckResult] = {}
    if probed:
        raw = probe_external_services(
            [_service_from_capability(cap) for cap in probed], timeout=timeout, env=env
        )
        for cap, result in zip(probed, raw, strict=True):
            if result.status in (CheckStatus.OK, CheckStatus.WARN):
                hint = result.fix_hint
            elif result.fix_hint:
                hint = result.fix_hint
            elif cap.docker is not None:
                hint = "agent-scaffold up"
            else:
                hint = f"agent-scaffold connect {_capability_stem(cap.id)}"
            results_by_id[cap.id] = dataclasses.replace(
                result,
                id=f"capability.{cap.id}",
                category="Capabilities",
                detail=(result.detail or result.title) + f" ({_capability_meta(cap)})",
                fix_hint=hint,
            )
    results: list[CheckResult] = []
    for cap in stack.capabilities:
        if cap.id in results_by_id:
            results.append(results_by_id[cap.id])
        else:
            results.append(
                CheckResult(
                    id=f"capability.{cap.id}",
                    category="Capabilities",
                    status=CheckStatus.OK,
                    title=f"{cap.id} ({cap.kind})",
                    detail=_capability_meta(cap),
                )
            )
    results.extend(check.run() for check in _unresolved_capability_checks(stack))
    return results


def _resolve_recipe_for_doctor(slug: str) -> Recipe:
    """Find ``slug`` among configured deployments. Raises ``typer.Exit`` on miss."""
    try:
        cfg = load_config()
    except ConfigError as exc:
        console.print(f"[red]Configuration error:[/] {exc}")
        raise typer.Exit(code=1) from exc
    try:
        dep_source = resolve_deployments(
            override=cfg.deployments_path,
            mode=cfg.deployments_source,
            cache_dir=cfg.cache_dir,
        )
    except SourceFetchError as exc:
        console.print(f"[red]Source resolution error:[/] {exc}")
        raise typer.Exit(code=1) from exc
    if dep_source.path is None:
        console.print("[red]Could not resolve deployments source.[/]")
        raise typer.Exit(code=1)
    try:
        recipes = discover_recipes(dep_source.path)
    except DiscoveryError as exc:
        console.print(f"[red]Error:[/] {exc}")
        raise typer.Exit(code=1) from exc
    match = next((r for r in recipes if r.slug == slug), None)
    if match is None:
        available = ", ".join(r.slug for r in recipes) or "(none)"
        console.print(f"[red]Unknown recipe:[/] {slug}. Available: {available}")
        raise typer.Exit(code=1)
    return match


# ---------------------------------------------------------------------------
# Typer callback
# ---------------------------------------------------------------------------


@doctor_app.callback(invoke_without_command=True)
def cmd_doctor(
    recipe: str | None = typer.Option(
        None,
        "--recipe",
        "-r",
        help=(
            "Recipe slug. Adds Authentication + per-`external_services` rows. "
            "Without this flag, doctor only checks local tools."
        ),
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit machine-readable JSON; suppresses Rich output.",
    ),
    no_probes: bool = typer.Option(
        False,
        "--no-probes",
        help="Skip network/daemon probes; service rows report SKIP.",
    ),
    explain: str | None = typer.Option(
        None,
        "--explain",
        help="Open the getting-started doc for <topic> in $PAGER and exit.",
    ),
    timeout: float = typer.Option(
        5.0,
        "--timeout",
        min=1.0,
        max=30.0,
        help="Per-probe timeout in seconds.",
    ),
) -> None:
    """Audit local tools, (with --recipe) auth + recipe-declared services. Never mutates."""
    if explain is not None:
        rc = _explain_topic(explain)
        raise typer.Exit(code=rc)

    checks: list[Check] = baseline_checks()
    if recipe is not None:
        chosen = _resolve_recipe_for_doctor(recipe)
        checks.extend(_auth_checks())
        checks.extend(_service_checks(chosen.external_services, timeout=timeout, skip=no_probes))
        if chosen.capabilities:
            cfg = load_config()
            dep_source = resolve_deployments(
                override=cfg.deployments_path,
                mode=cfg.deployments_source,
                cache_dir=cfg.cache_dir,
            )
            if dep_source.path is not None:
                catalog = load_capabilities(dep_source.path)
                stack = resolve_capabilities(chosen, catalog)
                checks.extend(_capability_checks(stack))
    report = run_checks(checks)

    if json_output:
        typer.echo(_doctor_json(report))
    else:
        _doctor_render(report)

    raise typer.Exit(code=report.exit_code)
