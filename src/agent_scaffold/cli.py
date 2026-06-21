"""Typer CLI entry point for agent-scaffold.

Commands:
- ``agent-scaffold new``      : interactive (or ``--non-interactive``) project generator.
- ``agent-scaffold config``   : print resolved configuration.
- ``agent-scaffold validate`` : re-run validation tiers on an existing generated project.
- ``agent-scaffold --version``
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer
from rich.logging import RichHandler
from rich.panel import Panel

from agent_scaffold import __version__
from agent_scaffold._scaffold_dir import SCAFFOLD_DIR
from agent_scaffold.auth import project_namespace
from agent_scaffold.branding import print_banner
from agent_scaffold.capabilities import load_capabilities
from agent_scaffold.capabilities import resolve as resolve_capabilities
from agent_scaffold.cli_auth import auth_app
from agent_scaffold.cli_doctor import doctor_app
from agent_scaffold.cli_secrets import secrets_app
from agent_scaffold.cli_shared import console, prompt_to_raise_context_cap
from agent_scaffold.config import Config, ConfigError, load_config
from agent_scaffold.context import AssembledContext, ContextBudgetError, assemble
from agent_scaffold.costs import estimate_preflight as estimate_preflight_cost
from agent_scaffold.discovery import (
    DiscoveryError,
    ExternalService,
    Recipe,
    discover_recipes,
)
from agent_scaffold.doctor import CheckResult
from agent_scaffold.effort import EFFORT_PRESETS
from agent_scaffold.envfile import build_runtime_env
from agent_scaffold.generator import (
    extract_fenced_content,
    generate_single_file,
)
from agent_scaffold.imports import discover_neighbours
from agent_scaffold.language_hints import (
    UnknownLanguageError,
    load_language_hints,
)
from agent_scaffold.manifest import (
    Manifest,
    ManifestNotFoundError,
    read_manifest,
    update_file_entry,
    write_manifest,
)
from agent_scaffold.orchestrator import (
    Orchestrator,
    OrchestratorError,
    StepEvent,
    StepFinished,
    StepResult,
    StepStatus,
    render_plan_table,
)
from agent_scaffold.pipeline import (
    PipelineError,
    PipelineInputs,
    print_next_steps,
    print_phase_summary,
    run_generation,
    run_post_gen_formatter,
)
from agent_scaffold.plan import GenerationPlan
from agent_scaffold.plan import confirm as confirm_plan
from agent_scaffold.preflight import persist_filled, run_preflight
from agent_scaffold.progress import (
    GenerationDisplay,
    NullProgressDisplay,
    PlainProgressDisplay,
    ProgressEvent,
    RichProgressDisplay,
    make_step_display,
    render_failure_panel,
)
from agent_scaffold.run_log import RunLogger, TeeProgressSink
from agent_scaffold.sources import (
    BlueprintsMode,
    DeploymentsMode,
    ResolvedSource,
    SourceConfigError,
    SourceFetchError,
    resolve_blueprints,
    resolve_deployments,
)
from agent_scaffold.steps import default_steps_for
from agent_scaffold.topology import resolve as resolve_topology
from agent_scaffold.validator import ValidationTier
from agent_scaffold.validator import validate as run_validate
from agent_scaffold.writer import (
    WriteMode,
)

app = typer.Typer(
    name="agent-scaffold",
    help="Generate runnable AI agent projects from markdown specs.",
    add_completion=False,
    invoke_without_command=True,
)

# Sub-apps that own their own Typer instance live in dedicated modules to
# keep this file focused on the project-generation pipeline.
app.add_typer(doctor_app, name="doctor", rich_help_panel="Setup")
app.add_typer(auth_app, name="auth", rich_help_panel="Setup")
app.add_typer(secrets_app, name="secrets", rich_help_panel="Setup")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"agent-scaffold {__version__}")
        raise typer.Exit()


_LOGO_BODY = [
    "[bold]Agent Scaffold[/]  [dim]v{version}[/]",
    "[dim]Generate runnable AI agent projects from markdown specs.[/]",
    "",
    "[dim]Pipeline:[/]  [#FFB347]blueprints[/] → [#FF6347]deployments[/] → [bold #DC143C]scaffold[/]",
    "",
    "[bold]Start here:[/]",
    "  [bold #FFA500]scaffold[/]   interactive shell — configure, create, and run, all in one place",
    "",
    "[dim]Inside the shell:[/] [dim]/config → /status → /new → /generate.[/]",
    "[dim]The other `agent-scaffold <command>` verbs are for scripting/CI; "
    "run `agent-scaffold --help` to see them.[/]",
]


def _print_banner() -> None:
    body_lines = [line.format(version=__version__) for line in _LOGO_BODY]
    print_banner(console, body_lines)


@app.callback()
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the agent-scaffold version and exit.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable debug logging.",
    ),
) -> None:
    """agent-scaffold: generate runnable AI agent projects from markdown specs."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
    # Stash on the Typer context so subcommands (cmd_new) can read it without
    # redeclaring the flag on every subcommand signature.
    ctx.ensure_object(dict)["verbose"] = verbose
    if ctx.invoked_subcommand is None:
        _print_banner()
        raise typer.Exit()


def _load_language_hints(language: str) -> dict[str, Any]:
    """Thin wrapper around :func:`language_hints.load_language_hints`.

    Translates :class:`UnknownLanguageError` into ``typer.BadParameter`` so
    the CLI's error surface stays consistent with how it reports other bad
    flag values.
    """
    try:
        return load_language_hints(language)
    except UnknownLanguageError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _coerce_deployments_mode(raw: str) -> DeploymentsMode:
    # vX+1: 'bundled' is no longer accepted — the bundled snapshot has been
    # removed in favor of the catalog flow. Existing scripts that pass
    # --deployments-source=bundled get a clear error.
    if raw != "auto":
        raise typer.BadParameter(
            f"--deployments-source must be 'auto' (got {raw!r}). The 'bundled' mode "
            "was removed; the catalog + on-disk fetch cache replaces it."
        )
    return raw  # type: ignore[return-value]


def _coerce_blueprints_mode(raw: str) -> BlueprintsMode:
    if raw not in ("auto", "skip"):
        raise typer.BadParameter(f"--blueprints-source must be 'auto' or 'skip', got {raw!r}")
    return raw  # type: ignore[return-value]


def _print_source_status(label: str, source: ResolvedSource) -> None:
    """Print the resolved source status; highlight fallback paths in yellow.

    The default-dim line is kept for the "fresh fetch / cache" happy path.
    When ``used_fallback`` is True, swap to a yellow warning so users notice
    they're on the bundled snapshot rather than the live deployments tree —
    silently using stale docs leads to confusing scaffold output.
    """
    if source.used_fallback:
        reason = source.fallback_reason or "GitHub unreachable"
        console.print(
            f"[yellow]⚠ {label}:[/] {source.label}  " f"[dim](offline fallback — fix: {reason})[/]"
        )
    else:
        console.print(f"[dim]{label}:[/] {source.label}")


def _exit_on_source_config_error(exc: SourceConfigError) -> None:
    """Render a SourceConfigError and exit. Centralizes the message format."""
    console.print(f"[red]✗ Source config error:[/] {exc}")
    raise typer.Exit(code=2) from exc


# Interactive prompts + name validators live in cli_interactive. Imported
# here rather than at the top of the file so the import sits next to the
# `# Pipeline helpers` boundary comment that documents the extraction
# layering — this is a deliberate placement, not an oversight.
from agent_scaffold.cli_interactive import (  # noqa: E402
    _interactive_path,
    _interactive_select,
    _interactive_text,
    _python_module_name,
    _select_framework,
    _select_language,
    _select_model,
    _select_recipe,
    _select_write_mode,
    _validate_project_name,
)

# Pipeline helpers + run_generation moved to agent_scaffold.pipeline so the
# upcoming REPL can reuse them. cmd_regenerate still imports
# ``run_post_gen_formatter`` from there.


@app.command("scaffold", rich_help_panel="Start here")
def cmd_scaffold(
    deployments_path: Path | None = typer.Option(
        None,
        "--deployments-path",
        help="Local agent-deployments checkout (defaults to GitHub auto-fetch).",
    ),
    blueprints_path: Path | None = typer.Option(
        None,
        "--blueprints-path",
        help="Local agent-blueprints checkout (defaults to GitHub auto-fetch).",
    ),
    deployments_source: str = typer.Option(
        "auto",
        "--deployments-source",
        help="auto | bundled — where to fetch deployments docs from.",
    ),
    blueprints_source: str = typer.Option(
        "auto",
        "--blueprints-source",
        help="auto | skip — fetch blueprints from GitHub or skip entirely.",
    ),
) -> None:
    """Open the interactive scaffold shell.

    Persistent REPL: pick a recipe, language, framework with slash commands
    (``/recipe``, ``/language``, …), refine the plan with free text
    (``swap to sonnet, add postgres``), and generate with ``/go``. Stays
    open until ``/exit`` or Ctrl-D, so you can scaffold multiple projects
    in one session.
    """
    # Lazy import keeps the cli import-fast for non-REPL commands and
    # avoids pulling prompt_toolkit into doctor / auth / secrets paths.
    from agent_scaffold.repl.shell import run_shell

    try:
        cfg = load_config()
    except ConfigError as exc:
        console.print(f"[red]Configuration error:[/] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        dep_source = resolve_deployments(
            override=deployments_path,
            mode=_coerce_deployments_mode(deployments_source),
            cache_dir=cfg.cache_dir,
        )
        bp_source = resolve_blueprints(
            override=blueprints_path,
            mode=_coerce_blueprints_mode(blueprints_source),
            cache_dir=cfg.cache_dir,
            deployments_path=dep_source.path,
        )
    except SourceConfigError as exc:
        _exit_on_source_config_error(exc)
    except SourceFetchError as exc:
        # SourceNetworkError shouldn't normally land here — the auto-resolver
        # eats network failures and falls back. If it does (e.g. blueprints
        # with no fallback + network down), the message is still informative.
        console.print(f"[red]Source resolution error:[/] {exc}")
        raise typer.Exit(code=1) from exc

    exit_code = run_shell(cfg, dep_source, bp_source, console=console)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


@app.command("config", rich_help_panel="Setup")
def cmd_config() -> None:
    """Show the resolved configuration."""
    try:
        cfg = load_config()
    except ConfigError as exc:
        console.print(f"[red]Configuration error:[/] {exc}")
        raise typer.Exit(code=1) from exc
    payload = cfg.model_dump()
    payload["anthropic_api_key"] = "***" if payload.get("anthropic_api_key") else ""
    payload = {k: (str(v) if isinstance(v, Path) else v) for k, v in payload.items()}
    console.print(Panel(json.dumps(payload, indent=2), title="agent-scaffold config"))


@app.command("new", rich_help_panel="Generate")
def cmd_new(
    typer_ctx: typer.Context,
    non_interactive: bool = typer.Option(
        False,
        "--non-interactive",
        help="Skip prompts; use the --recipe/--language/... flags instead.",
    ),
    recipe_slug: str | None = typer.Option(None, "--recipe", help="Recipe slug to use."),
    language: str | None = typer.Option(None, "--language", help="Target language."),
    framework: str | None = typer.Option(
        None, "--framework", help="Framework key (matches language hints)."
    ),
    project_name: str | None = typer.Option(None, "--project-name"),
    dest: Path | None = typer.Option(None, "--dest", help="Destination directory."),
    write_mode: WriteMode = typer.Option(
        WriteMode.abort,
        "--write-mode",
        help="What to do if the destination already has files.",
    ),
    deployments_path: Path | None = typer.Option(
        None,
        "--deployments-path",
        help="Override path to your agent-deployments repo.",
    ),
    blueprints_path: Path | None = typer.Option(
        None,
        "--blueprints-path",
        help="Override path to your agent-blueprints repo.",
    ),
    deployments_source: str = typer.Option(
        "auto",
        "--deployments-source",
        help="auto | bundled — where to fetch deployments docs from.",
    ),
    blueprints_source: str = typer.Option(
        "auto",
        "--blueprints-source",
        help="auto | skip — fetch blueprints from GitHub or skip entirely.",
    ),
    skip_validation: bool = typer.Option(
        False,
        "--skip-validation",
        help="Do not run the post-generation static validation tier.",
    ),
    format_output: bool = typer.Option(
        True,
        "--format/--no-format",
        help=(
            "Run a post-write formatter pass (ruff for Python, prettier/biome "
            "for TypeScript) before static validation. Override with "
            "AGENT_SCAFFOLD_FORMAT={0,1}."
        ),
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Skip response cache and always call the LLM.",
    ),
    effort: str | None = typer.Option(
        None,
        "--effort",
        help=(
            "Preset bundle: low | medium | high. Sets model, max_tokens, "
            "thinking_budget, and prompt strictness. Explicit --model / "
            "--max-tokens / --thinking / --strict flags override the preset."
        ),
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Anthropic model ID. Overrides --effort and config.",
    ),
    max_tokens: int | None = typer.Option(
        None,
        "--max-tokens",
        help="Override the API max_tokens for this run.",
    ),
    thinking: int | None = typer.Option(
        None,
        "--thinking",
        help="Extended-thinking budget in tokens. Omit to disable.",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Use the strict system prompt (demands Docker, CI, structlog, three-tier tests).",
    ),
    max_context_tokens: int | None = typer.Option(
        None,
        "--max-context-tokens",
        help=(
            "Hard cap on assembled-context tokens. Lowest-priority docs are "
            "dropped to fit. Recipe + Composes that exceed the cap raise a hard error."
        ),
    ),
    max_link_depth: int | None = typer.Option(
        None,
        "--max-link-depth",
        help="Transitive markdown-link walk depth (0 = recipe only).",
    ),
    max_tokens_per_doc: int | None = typer.Option(
        None,
        "--max-tokens-per-doc",
        help="Per-doc token cap; longer docs are truncated with a marker.",
    ),
    plan: bool | None = typer.Option(
        None,
        "--plan/--no-plan",
        help=(
            "Show a generation plan (recipe / topology / model / context / files / "
            "warnings) and prompt Y/n before calling the LLM. Default on for "
            "interactive runs; --no-plan skips the gate."
        ),
    ),
    probe_services: bool = typer.Option(
        True,
        "--probe-services/--no-probe-services",
        help=(
            "Probe recipe-declared external services (Anthropic / Redis / Postgres / ...) "
            "before showing the plan panel. Probes run concurrently with a 5s per-probe cap. "
            "Disable with --no-probe-services in CI or when offline."
        ),
    ),
    autorun: bool = typer.Option(
        True,
        "--autorun/--no-autorun",
        help=(
            "After generation succeeds, run `up` (install deps, docker compose, "
            "frontend dev server, …) and print the welcome panel. "
            "Implicitly disabled by --non-interactive so CI scripts stay one-shot."
        ),
    ),
    open_browser: bool = typer.Option(
        True,
        "--open-browser/--no-open-browser",
        help=(
            "After autorun, open the frontend URL in the default browser. "
            "No effect when --no-autorun. Best-effort: headless / no-browser "
            "environments fail silently."
        ),
    ),
    autorun_yes: bool = typer.Option(
        False,
        "--autorun-yes",
        help=(
            "Skip the autorun confirm prompt — proceed straight into "
            "install_deps / docker compose / frontend dev server / etc. "
            "Use for CI scripts and unattended runs."
        ),
    ),
    use_docker: bool | None = typer.Option(
        None,
        "--docker/--no-docker",
        help=(
            "Run the stack in Docker (containers) vs local processes during "
            "autorun. Default: ask interactively, else local."
        ),
    ),
) -> None:
    """Generate a new agent project."""
    try:
        cfg = load_config()
    except ConfigError as exc:
        console.print(f"[red]Configuration error:[/] {exc}")
        raise typer.Exit(code=1) from exc

    if effort is not None and effort not in EFFORT_PRESETS:
        raise typer.BadParameter(
            f"Unknown effort: {effort!r}. Choose from {', '.join(EFFORT_PRESETS)}."
        )
    preset = EFFORT_PRESETS[effort] if effort else None
    if preset is not None:
        cfg = cfg.model_copy(
            update={
                "model": preset.model,
                "max_tokens": preset.max_tokens,
                "thinking_budget": preset.thinking,
                "max_context_tokens": preset.max_context_tokens,
                "max_link_depth": preset.max_link_depth,
                "max_tokens_per_doc": preset.max_tokens_per_doc,
            }
        )
        if preset.strict:
            strict = True

    # Explicit flags override the preset.
    cfg_updates: dict[str, Any] = {}
    if model is not None:
        cfg_updates["model"] = model
    if max_tokens is not None:
        cfg_updates["max_tokens"] = max_tokens
    if thinking is not None:
        cfg_updates["thinking_budget"] = thinking
    if max_context_tokens is not None:
        cfg_updates["max_context_tokens"] = max_context_tokens
    if max_link_depth is not None:
        cfg_updates["max_link_depth"] = max_link_depth
    if max_tokens_per_doc is not None:
        cfg_updates["max_tokens_per_doc"] = max_tokens_per_doc
    if cfg_updates:
        cfg = cfg.model_copy(update=cfg_updates)

    # Resolve deployments + blueprints sources. The deployments resolver
    # auto-fetches the latest main commit from GitHub (cached by SHA) and
    # falls back to the bundled copy when offline; the blueprints resolver
    # returns None when offline or explicitly skipped, in which case
    # blueprint URLs in deployments docs are dropped from context.
    try:
        dep_source = resolve_deployments(
            override=deployments_path,
            mode=_coerce_deployments_mode(deployments_source),
            cache_dir=cfg.cache_dir,
        )
        bp_source = resolve_blueprints(
            override=blueprints_path,
            mode=_coerce_blueprints_mode(blueprints_source),
            cache_dir=cfg.cache_dir,
            deployments_path=dep_source.path,
        )
    except SourceConfigError as exc:
        _exit_on_source_config_error(exc)
    except SourceFetchError as exc:
        # SourceNetworkError shouldn't normally land here — the auto-resolver
        # eats network failures and falls back. If it does (e.g. blueprints
        # with no fallback + network down), the message is still informative.
        console.print(f"[red]Source resolution error:[/] {exc}")
        raise typer.Exit(code=1) from exc
    _print_source_status("Deployments", dep_source)
    _print_source_status("Blueprints ", bp_source)
    if dep_source.path is None:
        # Shouldn't happen — deployments always has a bundled fallback.
        console.print("[red]Could not resolve deployments source.[/]")
        raise typer.Exit(code=1)
    deployments = dep_source.path
    blueprints = bp_source.path

    try:
        recipes = discover_recipes(deployments)
    except DiscoveryError as exc:
        console.print(f"[red]Error:[/] {exc}")
        raise typer.Exit(code=1) from exc

    recipe = _select_recipe(recipes, recipe_slug, non_interactive)
    chosen_language = _select_language(recipe, language, non_interactive)
    hints = _load_language_hints(chosen_language)
    recipe_lang_deps = recipe.recipe_dependencies.get(chosen_language, {})
    if recipe_lang_deps:
        pinned = dict(hints.get("pinned_dependencies") or {})
        pinned.update(recipe_lang_deps)
        hints = {**hints, "pinned_dependencies": pinned}
    chosen_framework = _select_framework(deployments, chosen_language, framework, non_interactive)

    chosen_model = _select_model(cfg, model, non_interactive)
    cfg = cfg.model_copy(update={"model": chosen_model})

    if non_interactive and project_name is None:
        raise typer.BadParameter("--project-name is required in --non-interactive mode")
    raw_name = project_name or _interactive_text("Project name:", default=recipe.slug)
    raw_name = _validate_project_name(raw_name)
    final_name = _python_module_name(raw_name, chosen_language)

    if dest is None:
        if non_interactive:
            dest = Path.cwd() / raw_name
        else:
            chosen_dest = _interactive_path(
                "Destination:",
                default=str(Path.cwd() / raw_name),
            )
            dest = Path(chosen_dest).expanduser()
    dest = dest.resolve()

    if not non_interactive and dest.exists() and any(dest.iterdir()):
        write_mode = _select_write_mode()

    catalog = load_capabilities(deployments)
    resolved_stack = resolve_capabilities(recipe, catalog, default_frontend=True)
    if resolved_stack.unresolved:
        console.print(
            f"[yellow]Capabilities not in catalog:[/] {', '.join(resolved_stack.unresolved)} "
            "(upgrade your deployments source or remove from the recipe)"
        )

    # Load the top-level deployments Catalog. Required — assemble() consults
    # catalog data for aliases / cross-cutting / framework gating / blueprint
    # URL rewriting. CatalogError propagates so the user sees a clear failure
    # rather than silently-degraded context.
    from agent_scaffold.catalog import load_catalog_for_config

    top_catalog = load_catalog_for_config(cfg)

    # Pre-flight gate: surface missing env vars + unreachable services BEFORE
    # the LLM spend. Warn-only by design — generation never blocks on it; a
    # value filled here is exported for this run and persisted to the
    # project's .env.local after the write phase.
    catalog_recipe_entry = next((r for r in top_catalog.recipes if r.slug == recipe.slug), None)
    preflight_report = run_preflight(
        recipe=recipe,
        catalog_entry=catalog_recipe_entry,
        resolved_stack=resolved_stack if resolved_stack.capabilities else None,
        project_dir=dest,
        console=console,
        interactive=not non_interactive,
        probe=lambda svcs: _probe_services_for_plan(svcs, probe_services=probe_services),
    )

    def _assemble_with_cfg(active_cfg: Config) -> AssembledContext:
        return assemble(
            recipe,
            chosen_language,
            chosen_framework,
            deployments,
            blueprints_path=blueprints,
            max_context_tokens=active_cfg.max_context_tokens,
            max_link_depth=active_cfg.max_link_depth,
            max_tokens_per_doc=active_cfg.max_tokens_per_doc,
            resolved_stack=resolved_stack if resolved_stack.capabilities else None,
            catalog=top_catalog,
        )

    with console.status("Assembling context..."):
        try:
            ctx = _assemble_with_cfg(cfg)
        except ContextBudgetError as exc:
            bumped = prompt_to_raise_context_cap(console, exc, non_interactive=non_interactive)
            if bumped is None:
                raise typer.Exit(code=1) from exc
            new_cap, new_per_doc = bumped
            cfg = cfg.model_copy(
                update={"max_context_tokens": new_cap, "max_tokens_per_doc": new_per_doc}
            )
            ctx = _assemble_with_cfg(cfg)
    if ctx.summary is not None:
        console.print(Panel(ctx.summary.render(), title="Assembled context", expand=False))
    else:
        console.print(
            f"[green]Context ready:[/] {len(ctx.referenced_paths)} reference(s), "
            f"~{ctx.token_estimate} tokens."
        )

    topology, roles = resolve_topology(recipe, ctx.body)

    # One confirm gate before money is spent — the plan panel (context, cost
    # estimate, service readiness) defaults ON for every interactive run.
    # --no-plan opts out; non-interactive runs never prompt.
    plan_enabled = plan if plan is not None else True
    if plan_enabled and not non_interactive:
        warnings: list[str] = []
        if ctx.summary is not None and ctx.summary.total_tokens > int(0.95 * ctx.summary.cap):
            warnings.append(
                f"Context is {int(100 * ctx.summary.total_tokens / max(1, ctx.summary.cap))}% of cap"
            )
        if not recipe.required_files:
            warnings.append("Recipe declares no required_files — hard to validate output")
        # Probes already ran inside the pre-flight gate; reuse the results so
        # the plan panel never triggers a second probe sweep.
        readiness = preflight_report.probe_results
        # Show the user what this call is likely to cost before they confirm.
        # output_range adapts to the configured max_tokens: low bound is the
        # assumed minimum useful response (8k), high bound is the configured
        # max so users see the worst case.
        preflight = estimate_preflight_cost(
            cfg.model,
            input_tokens=ctx.token_estimate,
            output_range=(min(8_000, cfg.max_tokens), cfg.max_tokens),
        )
        gen_plan = GenerationPlan(
            recipe_slug=recipe.slug,
            recipe_status=recipe.status,
            language=chosen_language,
            framework=chosen_framework,
            project_name=final_name,
            dest=dest,
            topology=topology,
            roles=roles,
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            thinking_budget=cfg.thinking_budget,
            required_files=recipe.required_files,
            context_summary=ctx.summary,
            write_mode=write_mode,
            warnings=warnings,
            strict=strict,
            service_readiness=readiness,
            preflight_cost=preflight,
        )
        if not confirm_plan(gen_plan, console):
            console.print("[yellow]Aborted before LLM call.[/]")
            raise typer.Exit(code=0)

    env_format = os.environ.get("AGENT_SCAFFOLD_FORMAT")
    if env_format is not None and env_format.strip() != "":
        format_output = env_format.strip() not in {"0", "false", "False", "no"}

    expected_files = len(recipe.required_files) or None
    verbose_flag = bool((typer_ctx.obj or {}).get("verbose", False))
    # Persistent run artifacts (run.log + events.jsonl). Logging must never
    # block generation — on any filesystem error we degrade to console-only.
    run_logger: RunLogger | None
    try:
        run_logger = RunLogger(cfg.cache_dir, command="new")
    except OSError as exc:
        run_logger = None
        console.print(f"[yellow]Run logging disabled:[/] {exc}")
    base_display: GenerationDisplay
    if non_interactive:
        base_display = NullProgressDisplay()
    elif not console.is_terminal:
        # CI / piped output: flat grep-able lines instead of a Live panel.
        base_display = PlainProgressDisplay()
    else:
        base_display = RichProgressDisplay(
            console,
            cfg.model,
            verbose=verbose_flag,
            expected_files=expected_files,
        )
    display: GenerationDisplay = (
        base_display if run_logger is None else TeeProgressSink(base_display, run_logger)
    )

    pipeline_inputs = PipelineInputs(
        cfg=cfg,
        recipe=recipe,
        language=chosen_language,
        framework=chosen_framework,
        project_name=final_name,
        raw_project_name=project_name,
        dest=dest,
        deployments=deployments,
        ctx=ctx,
        hints=hints,
        topology=topology,
        roles=roles,
        write_mode=write_mode,
        strict=strict,
        format_output=format_output,
        skip_validation=skip_validation,
        no_cache=no_cache,
        resolved_stack=resolved_stack if resolved_stack.capabilities else None,
    )
    run_status = "completed"
    try:
        try:
            run_report = run_generation(pipeline_inputs, display=display)
        except PipelineError as exc:
            # Pipeline already printed phase-specific progress events; surface the
            # error message + hint and exit with non-zero so callers in shell
            # scripts can detect the failure.
            run_status = "failed"
            console.print(f"[red]{exc.phase or 'pipeline'} failed:[/] {exc.message}")
            if exc.hint:
                console.print(exc.hint)
            if run_logger is not None:
                console.print(f"[dim]Full log: {run_logger.log_path}[/]")
            raise typer.Exit(code=1) from exc

        result = run_report.result
        report = run_report.report
        validation_results = run_report.validation_results

        # Secrets collected at the pre-flight gate can land in .env.local now
        # that the destination directory exists.
        if result is not None and preflight_report.filled:
            persisted = persist_filled(dest, preflight_report.filled)
            if persisted:
                console.print(
                    f"[green]Persisted[/] {len(persisted)} pre-flight secret(s) "
                    "to .env.local (mode 0600)."
                )

        if result is not None:
            console.print(f"[green]Generated[/] {len(result.files)} files.")
        if report is not None:
            console.print(
                f"[green]Wrote[/] {len(report.written)} new, "
                f"{len(report.overwritten)} overwritten, {len(report.skipped)} skipped."
            )
        for vr in validation_results:
            mark = "[green][OK][/]" if vr.passed else "[red][FAIL][/]"
            console.print(f"{mark} {vr.tier.value}")
            if not vr.passed:
                console.print(vr.output)

        print_phase_summary(
            getattr(display, "phase_durations", {}),
            getattr(display, "warnings", []),
            getattr(display, "errors", []),
        )

        # Autorun is on by default for interactive runs and is suppressed by
        # --non-interactive so CI scripts that generate sample projects in tests
        # don't suddenly start spinning up docker. The user can also opt out via
        # --no-autorun for a staged-by-hand flow.
        should_autorun = autorun and not non_interactive and result is not None

        if result is not None and not should_autorun:
            print_next_steps(dest, chosen_language, result.smoke_check, result.post_install)

        if should_autorun:
            rc = _autorun_after_new(
                project_dir=dest,
                recipe=recipe,
                resolved_stack=(
                    resolved_stack if resolved_stack and resolved_stack.capabilities else None
                ),
                open_browser=open_browser,
                autorun_yes=autorun_yes,
                use_docker=use_docker,
                run_logger=run_logger,
            )
            if rc != 0:
                run_status = "failed"
                raise typer.Exit(code=rc)
    finally:
        if run_logger is not None:
            run_logger.close(status=run_status)


def _autorun_after_new(
    project_dir: Path,
    recipe: Any | None,
    resolved_stack: Any | None,
    open_browser: bool,
    autorun_yes: bool = False,
    use_docker: bool | None = None,
    run_logger: RunLogger | None = None,
) -> int:
    """Gate autorun behind a confirmation prompt + return the exit code.

    Pre-Phase-4 this ran the orchestrator silently with ``yes=True``, so
    docker compose, the frontend dev server, and ``$EDITOR`` all launched
    without a chance to opt out. Now the same prompt that ``cmd_up``
    already had fires here too: the orchestrator's "Provisioning plan"
    table is printed, then the user picks yes / edit / dry-run / no.

    ``autorun_yes=True`` (``--autorun-yes`` on the CLI) restores the
    silent pre-Phase-4 behavior for CI.
    """
    try:
        manifest = read_manifest(project_dir)
    except ManifestNotFoundError as exc:
        console.print(f"[yellow]Autorun skipped:[/] {exc}")
        return 0  # Generation succeeded; autorun is a courtesy.

    flags = StepFlags(
        only=[],
        skip=[],
        force=[],
        retry=[],
        resume=False,
        plan_only=False,
        yes=autorun_yes,
        debug=False,
        use_docker=use_docker,
    )
    rc = _run_up_inline(
        project_dir=project_dir,
        manifest=manifest,
        recipe=recipe,
        resolved_stack=resolved_stack,
        flags=flags,
        interactive=not autorun_yes,
        step_logger=run_logger,
    )
    if rc != 0:
        return rc

    if open_browser and not autorun_yes:
        _maybe_open_browser_with_confirm(project_dir, default_yes=True)
    elif open_browser and autorun_yes:
        # CI: honor --open-browser literally — no prompt, just try.
        _maybe_open_browser_with_confirm(project_dir, default_yes=True, prompt=False)
    return 0


def _maybe_open_browser_with_confirm(
    project_dir: Path, *, default_yes: bool, prompt: bool = True
) -> None:
    """Ask before opening the frontend URL in the user's browser.

    Browser-open is a separate confirm from the autorun gate because by
    the time we get here the project is provisioned and the user might
    want to inspect the frontend manually first. Skips silently when no
    frontend was launched (no pid file).
    """
    url = _resolve_frontend_url(project_dir)
    if url is None:
        return
    if prompt:
        suffix = "[Y/n]" if default_yes else "[y/N]"
        answer = console.input(f"[bold]Open {url} in your browser?[/] {suffix} ").strip().lower()
        if default_yes:
            consent = answer in ("", "y", "yes")
        else:
            consent = answer in ("y", "yes")
        if not consent:
            console.print(f"[dim]Skipping browser open. Visit {url} when ready.[/]")
            return
    from agent_scaffold.welcome import _open_browser_safe

    if _open_browser_safe(url):
        console.print(f"[dim]Opening {url} in your browser…[/]")


def _resolve_frontend_url(project_dir: Path) -> str | None:
    """Read ``.scaffold/frontend.pid`` and return the dev server URL, or ``None``.

    ``None`` for any kind of missing / malformed PID file — autorun never
    fails on it because not every recipe ships a frontend.
    """
    pid_file = project_dir / SCAFFOLD_DIR / "frontend.pid"
    if not pid_file.is_file():
        return None
    try:
        data = json.loads(pid_file.read_text(encoding="utf-8"))
        port = int(data["port"])
    except (json.JSONDecodeError, KeyError, ValueError, TypeError, OSError):
        return None
    if port <= 0:
        return None
    return f"http://localhost:{port}"


# _select_recipe / _select_language / _select_model / _select_framework /
# _select_write_mode live in cli_interactive and are re-imported at the
# top of this module.


def _render_unified_diff(rel_path: str, old: str, new: str) -> str:
    import difflib

    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{rel_path}",
        tofile=f"b/{rel_path}",
    )
    return "".join(diff)


@app.command("regenerate", rich_help_panel="Generate")
def cmd_regenerate(
    typer_ctx: typer.Context,
    project_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Existing project directory generated by `agent-scaffold new`.",
    ),
    file_path: str = typer.Argument(
        ...,
        help="Path to the file inside the project to regenerate (relative).",
    ),
    reason: str = typer.Option(
        "",
        "--reason",
        help="Free-text instruction describing why this file is being regenerated.",
    ),
    diff_only: bool = typer.Option(
        False,
        "--diff",
        help="Print a unified diff against the existing file and exit without writing.",
    ),
    deployments_path: Path | None = typer.Option(
        None,
        "--deployments-path",
        help="Override path to your agent-deployments repo (defaults to env/config).",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Anthropic model ID for this regen call. Defaults to the manifest's model.",
    ),
    max_tokens: int | None = typer.Option(
        None,
        "--max-tokens",
        help="Override the API max_tokens for this regen call.",
    ),
    thinking: int | None = typer.Option(
        None,
        "--thinking",
        help="Extended-thinking budget in tokens.",
    ),
    no_format: bool = typer.Option(
        False,
        "--no-format",
        help="Skip the post-regen formatter pass.",
    ),
) -> None:
    """Re-prompt the model for a single file in an existing project.

    Reads ``<project>/.scaffold/manifest.json`` to recover the original
    recipe + language + framework, prompts the model with the target file +
    its import neighbours + the ``--reason``, then writes the replacement
    (or just shows the diff under ``--diff``). On validation failure the
    user is asked whether to keep the change; declining restores the backup.
    """
    project_dir = project_dir.resolve()
    target_abs = project_dir / file_path
    if not target_abs.is_file():
        console.print(f"[red]Error:[/] no such file: {target_abs}")
        raise typer.Exit(code=1)

    try:
        manifest = read_manifest(project_dir)
    except ManifestNotFoundError as exc:
        console.print(f"[red]Error:[/] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        cfg = load_config()
    except ConfigError as exc:
        console.print(f"[red]Configuration error:[/] {exc}")
        raise typer.Exit(code=1) from exc

    cfg_updates: dict[str, Any] = {"model": model or manifest.model}
    if max_tokens is not None:
        cfg_updates["max_tokens"] = max_tokens
    if thinking is not None:
        cfg_updates["thinking_budget"] = thinking
    cfg = cfg.model_copy(update=cfg_updates)

    try:
        dep_source = resolve_deployments(
            override=deployments_path,
            mode=cfg.deployments_source,
            cache_dir=cfg.cache_dir,
        )
    except SourceConfigError as exc:
        _exit_on_source_config_error(exc)
    except SourceFetchError as exc:
        # SourceNetworkError shouldn't normally land here — the auto-resolver
        # eats network failures and falls back. If it does (e.g. blueprints
        # with no fallback + network down), the message is still informative.
        console.print(f"[red]Source resolution error:[/] {exc}")
        raise typer.Exit(code=1) from exc
    if dep_source.path is None:
        console.print("[red]Could not resolve deployments source.[/]")
        raise typer.Exit(code=1)
    deployments = dep_source.path
    try:
        recipes = discover_recipes(deployments)
    except DiscoveryError as exc:
        console.print(f"[red]Error:[/] {exc}")
        raise typer.Exit(code=1) from exc
    recipe = next((r for r in recipes if r.slug == manifest.recipe), None)
    if recipe is None:
        console.print(
            f"[red]Error:[/] manifest references recipe {manifest.recipe!r} "
            f"which was not found under {deployments}."
        )
        raise typer.Exit(code=1)
    recipe_body = recipe.path.read_text(encoding="utf-8")

    neighbour_paths = discover_neighbours(project_dir, file_path)
    neighbours = {
        str(p.relative_to(project_dir)): p.read_text(encoding="utf-8", errors="replace")
        for p in neighbour_paths
    }
    current_content = target_abs.read_text(encoding="utf-8")

    console.print(
        f"[bold]Regenerating[/] {file_path} with {cfg.model} " f"(neighbours: {len(neighbours)})"
    )

    verbose_flag = bool((typer_ctx.obj or {}).get("verbose", False))
    display: RichProgressDisplay | NullProgressDisplay
    display = (
        NullProgressDisplay()
        if verbose_flag is False and diff_only
        else RichProgressDisplay(console, cfg.model, verbose=verbose_flag)
    )
    with display as progress:
        progress.on_event(
            ProgressEvent(
                kind="operation_started",
                payload={"name": "regenerate", "hint": file_path},
            )
        )
        raw = generate_single_file(
            config=cfg,
            recipe_body=recipe_body,
            target_path=file_path,
            current_content=current_content,
            neighbours=neighbours,
            reason=reason,
            language=manifest.language,
            progress=progress.on_event,
        )
        progress.on_event(
            ProgressEvent(
                kind="operation_done",
                payload={"name": "regenerate", "status": "ok"},
            )
        )

    try:
        new_content = extract_fenced_content(raw)
    except ValueError as exc:
        console.print(f"[red]Error:[/] {exc}")
        console.print("[dim]--- raw response head ---[/]")
        console.print(raw[:500])
        raise typer.Exit(code=1) from exc

    if new_content == current_content:
        console.print("[yellow]No change:[/] model returned the same content.")
        raise typer.Exit(code=0)

    diff_text = _render_unified_diff(file_path, current_content, new_content)
    if diff_only:
        console.print(diff_text or "(no diff)")
        raise typer.Exit(code=0)

    backup_text = current_content
    target_abs.write_text(new_content, encoding="utf-8")
    console.print(f"[green]Wrote[/] {file_path}")

    hints = _load_language_hints(manifest.language)
    if not no_format:
        with console.status("Formatting..."):
            run_post_gen_formatter(project_dir, manifest.language)

    results = run_validate(project_dir, hints, smoke_check="", tiers=[ValidationTier.static])
    failed = [r for r in results if not r.passed]
    if failed:
        console.print("[yellow]Static validation failed after regen:[/]")
        for r in failed:
            console.print(r.output)
        keep = _confirm_keep_after_failure()
        if not keep:
            target_abs.write_text(backup_text, encoding="utf-8")
            console.print(f"[yellow]Reverted[/] {file_path} to its previous contents.")
            raise typer.Exit(code=1)

    try:
        new_manifest = update_file_entry(manifest, project_dir, file_path)
        write_manifest(project_dir, new_manifest)
    except OSError as exc:
        console.print(f"[yellow]Warning:[/] could not update manifest: {exc}")


def _confirm_keep_after_failure() -> bool:
    """Y/n confirm whether to keep a regen that failed static validation."""
    try:
        import questionary
    except ImportError:  # pragma: no cover - questionary is a hard dep
        return True
    answer = questionary.confirm("Keep the regenerated file?", default=True).ask()
    return bool(answer)


@app.command("validate", rich_help_panel="Generate")
def cmd_validate(
    path: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    tier: str = typer.Option("static", "--tier", help="static|build|smoke"),
    language: str = typer.Option("python", "--language", help="Generated project language."),
    smoke_check: str | None = typer.Option(
        None,
        "--smoke-check",
        help="Override smoke-check command (defaults to language hints).",
    ),
) -> None:
    """Re-run a validation tier on an already-generated project."""
    hints = _load_language_hints(language)
    sc = smoke_check or str(hints.get("smoke_check", "")).replace("{project_name}", path.name)
    try:
        chosen = ValidationTier(tier)
    except ValueError as exc:
        raise typer.BadParameter(f"Unknown tier: {tier}") from exc
    results = run_validate(path, hints, sc, [chosen])
    for vr in results:
        mark = "[green][OK][/]" if vr.passed else "[red][FAIL][/]"
        console.print(f"{mark} {vr.tier.value}")
        console.print(vr.output)
    if any(not r.passed for r in results):
        raise typer.Exit(code=1)


@dataclass(frozen=True)
class StepFlags:
    """Shared flag set for orchestrator-driven commands (``up``, ``update``, ...).

    Q5 owns this dataclass; Q6 wires it into ``cmd_up`` and Q8 into
    ``cmd_update``. Defining it here prevents flag drift between siblings
    (e.g. so ``--retry`` always means the same thing).
    """

    only: list[str]
    skip: list[str]
    force: list[str]
    retry: list[str]
    resume: bool
    plan_only: bool
    yes: bool
    debug: bool
    # Q7: paired with --yes to opt out of the commit_push always-prompt rule.
    # Set on its own (without --yes) it's a no-op — the per-step prompts still fire.
    confirm_commit_push: bool = False
    # Opt-in: re-include the slow eval baseline (bootstrap_evals) in the chain.
    with_evals: bool = False
    # Opt-in docker mode: None = ask (interactive), True/False = explicit.
    use_docker: bool | None = None


def step_flags_callback(
    only: list[str] = typer.Option(
        [],
        "--only",
        help="Run only these steps + their transitive dependencies.",
    ),
    skip: list[str] = typer.Option(
        [],
        "--skip",
        help="Mark steps as skipped without running them.",
    ),
    force: list[str] = typer.Option(
        [],
        "--force",
        help="Re-run steps regardless of stored state.",
    ),
    retry: list[str] = typer.Option(
        [],
        "--retry",
        help="Re-run steps that previously failed.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Skip steps the state file marks as DONE without re-detecting.",
    ),
    plan_only: bool = typer.Option(
        False,
        "--plan",
        help="Print the orchestrator plan table and exit.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip the interactive Y/n confirmation.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Emit step-level debug logs.",
    ),
) -> StepFlags:
    return StepFlags(
        only=list(only),
        skip=list(skip),
        force=list(force),
        retry=list(retry),
        resume=resume,
        plan_only=plan_only,
        yes=yes,
        debug=debug,
    )


def _resolve_recipe_silently(slug: str) -> Recipe | None:
    """Look up ``slug`` without prompting; returns ``None`` if not found.

    ``cmd_up`` runs against a generated project where the recipe slug is
    already on the manifest, so we don't want the doctor-style hard exit if
    the deployments repo is missing — let the steps surface that themselves.
    """
    try:
        cfg = load_config()
    except ConfigError:
        return None
    try:
        dep_source = resolve_deployments(
            override=cfg.deployments_path,
            mode=cfg.deployments_source,
            cache_dir=cfg.cache_dir,
        )
    except SourceFetchError:
        return None
    if dep_source.path is None:
        return None
    try:
        recipes = discover_recipes(dep_source.path)
    except DiscoveryError:
        return None
    return next((r for r in recipes if r.slug == slug), None)


def _resolve_capability_stack_silently(
    recipe: Recipe | None, *, capabilities: list[str] | None = None
) -> Any | None:
    """Resolve a ``ResolvedStack`` without prompting.

    Mirrors :func:`_resolve_recipe_silently` — failures (no deployments,
    no catalog, recipe without capabilities) return ``None`` so the
    bootstrap steps SKIP cleanly.

    When ``capabilities`` (the manifest's *chosen* resolved ids) is given,
    resolve THAT set instead of the recipe's declared defaults — so a
    post-generation panel/run reflects the user's actual choices (e.g.
    ``obs.langsmith`` swapped in over the recipe's default ``obs.langfuse``)
    rather than a phantom service the user never picked.
    """
    if recipe is None:
        return None
    declared = set(recipe.capabilities)
    chosen = set(capabilities) if capabilities else declared
    if not chosen:
        return None
    try:
        cfg = load_config()
    except ConfigError:
        return None
    try:
        dep_source = resolve_deployments(
            override=cfg.deployments_path,
            mode=cfg.deployments_source,
            cache_dir=cfg.cache_dir,
        )
    except SourceFetchError:
        return None
    if dep_source.path is None:
        return None
    catalog = load_capabilities(dep_source.path)
    stack = resolve_capabilities(
        recipe,
        catalog,
        add_capabilities=sorted(chosen - declared),
        remove_capabilities=declared - chosen,
    )
    return stack if stack.capabilities else None


def _select_active_steps(all_step_specs: list[tuple[str, str]], current_ids: set[str]) -> list[str]:
    """Interactive checkbox over step ids — used by the ``edit`` confirm path."""
    import questionary

    choices = [
        questionary.Choice(title=f"{sid}  {desc}", value=sid, checked=(sid in current_ids))
        for sid, desc in all_step_specs
    ]
    chosen = questionary.checkbox(
        "Toggle which steps to run (space to toggle, enter to confirm):",
        choices=choices,
    ).ask()
    if chosen is None:
        raise typer.Abort()
    return [str(c) for c in chosen]


@app.command("up", rich_help_panel="Run & deploy")
def cmd_up(
    project_dir: Path = typer.Argument(
        Path("."),
        help="Path to a generated project (where .scaffold/manifest.json lives).",
        exists=False,
    ),
    only: list[str] = typer.Option(
        [], "--only", help="Run only these steps + their transitive dependencies."
    ),
    skip: list[str] = typer.Option(
        [], "--skip", help="Mark steps as skipped without running them."
    ),
    force: list[str] = typer.Option([], "--force", help="Re-run steps regardless of stored state."),
    retry: list[str] = typer.Option([], "--retry", help="Re-run steps that previously failed."),
    resume: bool = typer.Option(
        False, "--resume", help="Skip steps the state file marks as DONE without re-detecting."
    ),
    plan_only: bool = typer.Option(
        False, "--plan", help="Print the orchestrator plan table and exit."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the interactive Y/n confirmation."),
    debug: bool = typer.Option(False, "--debug", help="Emit step-level debug logs."),
    confirm_commit_push: bool = typer.Option(
        False,
        "--confirm-commit-push",
        help=(
            "Pair with --yes to skip the commit_push step's per-action prompts. "
            "Without this flag, --yes still prompts before commit and before push."
        ),
    ),
    with_evals: bool = typer.Option(
        False,
        "--with-evals",
        help=(
            "Also run the eval baseline (slow, makes real LLM calls). Off by "
            "default — prefer `agent-scaffold eval --update-baseline`."
        ),
    ),
    use_docker: bool | None = typer.Option(
        None,
        "--docker/--no-docker",
        help=(
            "Run the stack in Docker (backend + services as containers) vs local "
            "processes. Default: ask interactively, else local."
        ),
    ),
) -> None:
    """Interactively provision a local environment for a generated project.

    Reads ``.scaffold/manifest.json`` to learn what recipe + language to wire
    up, then runs the Q5 orchestrator with all configured steps. Idempotent:
    re-runs skip steps already DONE per ``.scaffold/state.json``.
    """
    flags = StepFlags(
        only=list(only),
        skip=list(skip),
        force=list(force),
        retry=list(retry),
        resume=resume,
        plan_only=plan_only,
        yes=yes,
        debug=debug,
        confirm_commit_push=confirm_commit_push,
        with_evals=with_evals,
        use_docker=use_docker,
    )
    project_dir = project_dir.expanduser().resolve()
    try:
        manifest = read_manifest(project_dir)
    except ManifestNotFoundError as exc:
        console.print(f"[red]Error:[/] {exc}")
        raise typer.Exit(code=1) from exc

    recipe = _resolve_recipe_silently(manifest.recipe)
    if recipe is None:
        console.print(
            f"[yellow]Note:[/] recipe {manifest.recipe!r} not found in deployments "
            "path; docker/credentials steps will skip if they need it."
        )

    resolved_stack = _resolve_capability_stack_silently(
        recipe, capabilities=manifest.capabilities
    )

    exit_code = _run_up_inline(
        project_dir=project_dir,
        manifest=manifest,
        recipe=recipe,
        resolved_stack=resolved_stack,
        flags=flags,
        interactive=True,
    )
    raise typer.Exit(code=exit_code)


def _resolve_use_docker(flags: StepFlags, interactive: bool, project_dir: Path) -> bool:
    """Resolve docker vs local mode: explicit flag → interactive prompt → local.

    When docker is chosen, verify it's actually usable (installed, daemon up,
    socket access) and fall back to local with a clear note if not — so the
    servers still come up either way.
    """
    intent = flags.use_docker
    if intent is None:
        if interactive and (project_dir / "docker-compose.yml").is_file():
            choice = _interactive_select(
                "How should the stack run?",
                choices=[
                    ("local", "Local — backend/frontend as local processes (default)"),
                    ("docker", "Docker — backend + services as containers"),
                ],
                default="local",
            )
            intent = choice == "docker"
        else:
            intent = False
    if not intent:
        return False
    from agent_scaffold.steps.docker_up import docker_available

    ok, reason = docker_available()
    if not ok:
        console.print(f"[yellow]Docker not available:[/] {reason} — running locally instead.")
        return False
    return True


def _run_up_inline(
    project_dir: Path,
    manifest: Manifest,
    recipe: Any | None,  # Recipe | None — Any avoids the optional import dep here
    resolved_stack: Any | None,  # ResolvedStack | None
    flags: StepFlags,
    *,
    interactive: bool = True,
    step_logger: RunLogger | None = None,
) -> int:
    """Run the orchestrator + welcome panel for a generated project.

    ``interactive=False`` skips the plan-confirm prompt and the
    "edit which steps" picker — used by autorun after ``new`` where the user
    has already implicitly approved by typing ``new`` with autorun on.
    Returns the shell exit code instead of raising ``typer.Exit`` so callers
    that chain (``cmd_new`` autorun) can decide what to do next.
    """
    use_docker = _resolve_use_docker(flags, interactive, project_dir)
    steps = default_steps_for(
        manifest,
        recipe,
        yes=flags.yes,
        confirm_commit_push=flags.confirm_commit_push,
        with_evals=flags.with_evals,
        use_docker=use_docker,
    )
    # Resolve the subprocess environment once per run: shell env > project
    # secrets vault (OS keyring, batched read) > .env.local. Steps thread it
    # into every spawned process so docker compose ${VAR} interpolation works
    # without a plaintext file.
    namespace = manifest.secrets_namespace or project_namespace(project_dir.name, project_dir)
    runtime_env = build_runtime_env(project_dir, namespace)
    try:
        orch = Orchestrator(
            steps,
            project_dir,
            manifest,
            resolved_stack=resolved_stack,
            runtime_env=runtime_env,
        )
    except OrchestratorError as exc:
        console.print(f"[red]Orchestrator error:[/] {exc}")
        return 1

    rows = orch.plan()
    console.print(render_plan_table(rows))

    if flags.plan_only:
        return 0

    step_specs: list[tuple[str, str]] = [(s.id, s.description) for s in steps]

    if interactive and not flags.yes:
        action = _interactive_select(
            "Proceed?",
            choices=[
                ("yes", "yes — run the plan above"),
                ("edit", "edit — toggle which steps to run"),
                ("dry_run", "dry-run — print the plan and exit (no changes)"),
                ("no", "no — abort without changes"),
            ],
            default="yes",
        )
        if action == "no":
            console.print("[yellow]Aborted.[/]")
            return 0
        if action == "dry_run":
            console.print(
                "[dim]Dry-run only — no commands were executed. "
                "Re-run without --no-autorun to apply.[/]"
            )
            return 0
        if action == "edit":
            current = set(flags.only) if flags.only else {sid for sid, _ in step_specs}
            chosen_ids = _select_active_steps(step_specs, current)
            if not chosen_ids:
                console.print("[yellow]No steps selected; aborted.[/]")
                return 0
            flags = StepFlags(
                only=chosen_ids,
                skip=list(flags.skip),
                force=list(flags.force),
                retry=list(flags.retry),
                resume=flags.resume,
                plan_only=flags.plan_only,
                yes=flags.yes,
                debug=flags.debug,
            )

    troubleshoot_by_step = {s.id: dict(getattr(s, "troubleshoot", {}) or {}) for s in steps}

    force_plain = flags.yes or not console.is_terminal
    display = make_step_display(console, step_specs, force_plain=force_plain)

    # Collect failed-step results so we can render their panels after the
    # Live display releases stdout. PlainStepProgressDisplay doesn't track
    # results itself, so we tee them through a shim callback.
    failed_results: dict[str, StepResult] = {}

    def _on_event(event: StepEvent) -> None:
        display.on_event(event)
        if step_logger is not None:
            step_logger.log_step_event(event)
        if (
            isinstance(event, StepFinished)
            and event.result is not None
            and event.result.status == StepStatus.FAILED
        ):
            failed_results[event.step_id] = event.result

    with display:
        orch.callback = _on_event
        result = orch.run(
            only=flags.only,
            skip=flags.skip,
            force=flags.force,
            retry=flags.retry,
            resume=flags.resume,
        )

    for sid, step_result in failed_results.items():
        console.print(render_failure_panel(sid, step_result, troubleshoot_by_step.get(sid)))

    # One bounded smoke-repair round: the post-write repair loop can't cover
    # the smoke tier (services weren't up yet). Interactive offer only —
    # repair is an LLM call with real cost.
    if (
        result.exit_code != 0
        and "smoke_test" in failed_results
        and recipe is not None
        and interactive
        and not flags.yes
    ):
        result = _offer_smoke_repair(
            orch=orch,
            project_dir=project_dir,
            manifest=manifest,
            recipe=recipe,
            failure=failed_results["smoke_test"],
            step_specs=step_specs,
            step_logger=step_logger,
            previous=result,
        )

    _print_step_summary(result.summary)

    # Keep the project's durable record current: each `up` refreshes the
    # Provisioning section in .scaffold/run-summary.md (best-effort).
    from agent_scaffold.run_summary import append_provisioning_section

    append_provisioning_section(project_dir, result.summary)

    # Surface every live URL the user can open after a successful run. The
    # panel quietly handles missing capabilities / missing frontend PID file,
    # so it stays useful on partial runs too.
    if result.exit_code == 0:
        from agent_scaffold.welcome import render_welcome_panel

        console.print(
            render_welcome_panel(
                project_dir,
                manifest,
                resolved_stack,
                run_log_dir=str(step_logger.run_dir) if step_logger is not None else "",
            )
        )

    return result.exit_code


def _offer_smoke_repair(
    *,
    orch: Orchestrator,
    project_dir: Path,
    manifest: Manifest,
    recipe: Any,
    failure: StepResult,
    step_specs: list[tuple[str, str]],
    step_logger: RunLogger | None,
    previous: Any,
) -> Any:
    """Prompt for one model-driven smoke repair; re-run the smoke step on success.

    Returns the retry's RunResult when the repair landed, else ``previous``.
    The original failure stays authoritative on any repair-side error.
    """
    from agent_scaffold._redact import redact

    try:
        proceed = typer.confirm(
            "smoke_test failed — attempt one model-driven repair round (LLM call)?",
            default=False,
        )
    except (typer.Abort, EOFError):
        return previous
    if not proceed:
        return previous

    from agent_scaffold.pipeline import repair_smoke_failure

    try:
        cfg = load_config()
        patched = repair_smoke_failure(
            project_dir=project_dir,
            manifest=manifest,
            recipe=recipe,
            cfg=cfg,
            failure_output=(failure.stderr_tail or failure.error or ""),
        )
    except Exception as exc:  # noqa: BLE001 — repair must never crash `up`
        console.print(f"[yellow]Smoke repair did not land:[/] {redact(str(exc))}")
        return previous

    console.print(f"[green]Repair patched {patched} file(s)[/] — re-running smoke test.")
    if step_logger is not None:
        step_logger.note(f"smoke repair patched {patched} file(s); retrying smoke_test")

    display = make_step_display(console, step_specs, force_plain=not console.is_terminal)

    def _on_retry_event(event: StepEvent) -> None:
        display.on_event(event)
        if step_logger is not None:
            step_logger.log_step_event(event)

    with display:
        orch.callback = _on_retry_event
        retry_result = orch.run(only=["smoke_test"], retry=["smoke_test"])
    return retry_result


def _print_step_summary(summary: dict[str, int]) -> None:
    parts = [
        f"{summary.get('done', 0)} done",
        f"{summary.get('skipped', 0)} skipped",
        f"{summary.get('failed', 0)} failed",
    ]
    if summary.get("partial", 0):
        parts.append(f"{summary['partial']} partial")
    if summary.get("pending", 0):
        parts.append(f"{summary['pending']} pending")
    console.print("[bold]Run summary:[/] " + ", ".join(parts))


@app.command("update", rich_help_panel="Run & deploy")
def cmd_update(
    project_dir: Path = typer.Argument(
        Path("."),
        help="Path to a generated project (where .scaffold/manifest.json lives).",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Compute the merge and show the plan; don't write."
    ),
    continue_: bool = typer.Option(
        False,
        "--continue",
        help="Continue a previous update after manual conflict resolution.",
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip the Apply? confirmation; never asks about removals."
    ),
    debug: bool = typer.Option(False, "--debug", help="Emit verbose update logs."),
) -> None:
    """Re-run the recipe against this project and 3-way-merge template changes.

    Copier-style template evolution: pulls in new files added to the recipe
    since last generation, merges template-side edits with your local edits,
    and leaves conflict markers for hunks that touch the same lines. Resolve
    in your editor, then run ``agent-scaffold update --continue``.
    """
    # Lazy import keeps the update flow's dependencies (manifest, merge,
    # template_snapshot) out of every CLI entry point that doesn't run
    # the update command.
    from agent_scaffold import cli_update

    cli_update.run(
        project_dir,
        dry_run=dry_run,
        continue_=continue_,
        yes=yes,
        debug=debug,
    )


def _probe_services_for_plan(
    services: list[ExternalService],
    *,
    probe_services: bool,
    timeout: float = 5.0,
    max_workers: int = 4,
) -> list[CheckResult]:
    """Run service probes concurrently for the plan panel.

    Thin wrapper over :func:`agent_scaffold.probes.probe_external_services`:
    adds the CLI's `--no-probes` skip semantics and the ``console.status``
    spinner the plan-panel flow expects. The REPL's ``cmd_recipe`` path
    calls the underlying helper directly without the spinner.
    """
    if not services:
        return []
    from agent_scaffold.probes import probe_external_services, run_probe

    if not probe_services:
        return [run_probe(svc, timeout=timeout, skip=True) for svc in services]
    with console.status(f"Probing {len(services)} service(s)..."):
        return probe_external_services(services, timeout=timeout, max_workers=max_workers)


# ---------------------------------------------------------------------------
# Lifecycle verbs: deploy / down / status / logs
# ---------------------------------------------------------------------------


@app.command("deploy", rich_help_panel="Run & deploy")
def cmd_deploy(
    target: str = typer.Option(
        ...,
        "--target",
        "-t",
        help="Cloud provider: vercel | railway | fly",
    ),
    cwd: Path = typer.Option(
        Path("."),
        "--cwd",
        help="Project directory (must contain .scaffold/manifest.json).",
    ),
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--no-dry-run",
        help="Print the deploy command instead of running it (default: on).",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip the confirmation prompt and actually deploy.",
    ),
) -> None:
    """Push the project to a cloud provider declared by a host.* capability.

    Local-first by design: ``--dry-run`` is on by default. The plugin
    inspects the project (config file present, CLI installed, project
    linked), prints the command it WOULD run plus the dashboard URL, and
    exits without touching the cloud. Use ``--no-dry-run --yes`` to
    actually deploy.
    """
    from agent_scaffold.deploy import get_plugin

    project_dir = cwd.expanduser().resolve()
    try:
        manifest = read_manifest(project_dir)
    except ManifestNotFoundError as exc:
        console.print(f"[red]Error:[/] {exc}")
        raise typer.Exit(code=1) from exc

    # Resolve the host.* capability declared on the manifest. If the user
    # passed a target that doesn't match any declared capability, fail
    # loudly rather than silently picking the wrong one.
    capability_targets = _resolve_deploy_targets(manifest)
    if capability_targets and target not in capability_targets:
        console.print(
            f"[red]Target {target!r} not declared by the recipe.[/] "
            f"Recipe declares: {', '.join(capability_targets) or '(none)'}"
        )
        raise typer.Exit(code=1)

    try:
        plugin = get_plugin(target)
    except KeyError as exc:
        console.print(
            f"[red]Unknown deploy target {target!r}.[/] " "Supported: vercel, railway, fly"
        )
        raise typer.Exit(code=1) from exc

    result = plugin.deploy(project_dir, dry_run=dry_run, yes=yes)
    _render_deploy_result(result)
    # Exit non-zero only on a real failed provider run.
    if result.exit_code is not None and result.exit_code != 0:
        raise typer.Exit(code=1)


@app.command("down", rich_help_panel="Run & deploy")
def cmd_down(
    cwd: Path = typer.Option(
        Path("."),
        "--cwd",
        help="Project directory containing docker-compose.yml.",
    ),
    volumes: bool = typer.Option(
        False,
        "-v",
        "--volumes",
        help="Also remove named volumes — DESTROYS local data.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompts."),
) -> None:
    """Tear down the local docker-compose stack. Never touches cloud.

    Without ``-v`` this is safe and reversible: just stops + removes the
    containers. With ``-v`` it also deletes named volumes, which wipes
    Postgres / Qdrant / Redis state on disk — requires typing ``yes``
    (or ``--yes``) to confirm.
    """
    project_dir = cwd.expanduser().resolve()
    rc = _down_inline(project_dir, volumes=volumes, yes=yes)
    raise typer.Exit(code=rc)


def _down_inline(project_dir: Path, *, volumes: bool = False, yes: bool = False) -> int:
    """Tear down the local stack; returns an exit code instead of raising.

    Shared by ``cmd_down`` (terminal) and the REPL ``/down`` so both stop the
    dev servers, ``docker compose down`` the stack, and reset the docker_up step
    state. ``volumes=True`` also deletes named volumes (confirmed unless ``yes``).
    """
    # Stop the dev servers before tearing down compose so the user doesn't see a
    # "Backend gone" error in the browser tab they still have open.
    _stop_frontend(project_dir)
    _stop_backend(project_dir)

    compose_path = _find_docker_compose(project_dir)
    if compose_path is None:
        console.print(f"[red]Error:[/] no docker-compose.yml found under {project_dir}")
        return 1
    if shutil.which("docker") is None:
        console.print("[red]Error:[/] docker not on PATH — install Docker Desktop / Colima first")
        return 1

    if volumes and not yes:
        from agent_scaffold.deploy._common import confirm

        ok = confirm(
            "This will DELETE local data in named volumes "
            "(postgres, qdrant, redis, etc.). Type 'yes' to continue."
        )
        if not ok:
            console.print("[yellow]Aborted.[/]")
            return 0

    cmd = ["docker", "compose", "down"]
    if volumes:
        cmd.append("-v")
    console.print(f"[cyan]Running:[/] {' '.join(cmd)} (cwd: {compose_path.parent})")
    rc = subprocess.run(  # noqa: S603 — list-form, shell=False
        cmd, cwd=str(compose_path.parent), check=False
    ).returncode
    if rc != 0:
        console.print(f"[red]docker compose down exited {rc}[/]")
        return 1
    console.print("[green]Local stack stopped.[/]")

    # Reset docker_up step state so the next `agent-scaffold up` re-detects
    # fresh containers rather than skipping based on stale orchestrator state.
    _reset_step_state(project_dir, "docker_up")
    return 0


@app.command("status", rich_help_panel="Setup")
def cmd_status(
    cwd: Path = typer.Option(
        Path("."),
        "--cwd",
        help="Project directory.",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON; suppresses Rich output."
    ),
    timeout: float = typer.Option(
        5.0, "--timeout", min=1.0, max=30.0, help="Per-probe timeout in seconds."
    ),
) -> None:
    """Probe every capability the recipe declared and print a health table.

    Loads the project manifest, resolves the recipe's capabilities against
    the deployments catalog, runs each capability's declared probe, and
    renders a table with OK / WARN / FAIL / SKIP. Exit 1 if any FAIL.
    """
    from agent_scaffold.cli_doctor import _capability_checks
    from agent_scaffold.doctor import CheckStatus

    project_dir = cwd.expanduser().resolve()
    try:
        manifest = read_manifest(project_dir)
    except ManifestNotFoundError as exc:
        console.print(f"[red]Error:[/] {exc}")
        raise typer.Exit(code=1) from exc

    recipe = _resolve_recipe_silently(manifest.recipe)
    resolved_stack = _resolve_capability_stack_silently(
        recipe, capabilities=manifest.capabilities
    )
    service_results: list[CheckResult] = []
    if recipe is not None and recipe.external_services:
        service_results = _probe_services_for_plan(
            recipe.external_services, probe_services=True, timeout=timeout
        )
    capability_results: list[CheckResult] = []
    if resolved_stack is not None:
        capability_results = [c.run() for c in _capability_checks(resolved_stack)]

    if json_output:
        import dataclasses
        import json as _json

        body = {
            "services": [dataclasses.asdict(r) for r in service_results],
            "capabilities": [dataclasses.asdict(r) for r in capability_results],
        }
        typer.echo(_json.dumps(body, indent=2, default=str))
    else:
        _render_status_table(service_results, capability_results)

    any_fail = any(r.status == CheckStatus.FAIL for r in (*service_results, *capability_results))
    raise typer.Exit(code=1 if any_fail else 0)


@app.command("eval", rich_help_panel="Run & deploy")
def cmd_eval(
    cwd: Path = typer.Option(
        Path("."),
        "--cwd",
        help="Project directory containing the manifest + evals/ tree.",
    ),
    target: str = typer.Option(
        "promptfoo",
        "--target",
        "-t",
        help="Eval framework. Currently supported: promptfoo.",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON; suppresses Rich output."
    ),
    update_baseline: bool = typer.Option(
        False,
        "--update-baseline",
        help=(
            "Persist this run's total as the new baseline in the manifest. "
            "Use after intentional improvements; exits 0 even on regression."
        ),
    ),
) -> None:
    """Run the project's eval suite. Exits 1 on regression vs the stored baseline.

    Reads the baseline from ``manifest.answers["eval_baseline"]`` (set by the
    ``bootstrap_evals`` step during ``up``). If no eval capability is declared
    on the recipe, exits 0 with a friendly note rather than an error — recipes
    without evals shouldn't be punished.
    """
    from agent_scaffold.eval import get_plugin

    project_dir = cwd.expanduser().resolve()
    try:
        manifest = read_manifest(project_dir)
    except ManifestNotFoundError as exc:
        console.print(f"[red]Error:[/] {exc}")
        raise typer.Exit(code=1) from exc

    eval_caps = [c for c in (manifest.capabilities or []) if c.startswith("eval.")]
    if not eval_caps:
        console.print("[yellow]No eval capability declared by this recipe — nothing to run.[/]")
        raise typer.Exit(code=0)

    try:
        plugin = get_plugin(target)
    except KeyError as exc:
        from agent_scaffold.eval import EVAL_PLUGINS as _plugins  # populated by get_plugin

        registered = ", ".join(sorted((_plugins or {}).keys()))
        console.print(f"[red]Unknown eval target {target!r}.[/] Supported: {registered}")
        raise typer.Exit(code=1) from exc

    baseline = _read_eval_baseline(manifest)
    result = plugin.run(project_dir, baseline)

    if json_output:
        _emit_eval_json(result)
    else:
        _render_eval_result(result)

    if update_baseline and not result.skipped and result.error is None:
        from agent_scaffold.manifest import update_manifest_answer

        update_manifest_answer(project_dir, "eval_baseline", f"{result.total:.4f}")
        console.print(f"[green]Baseline updated:[/] eval_baseline = {result.total:.4f}")
        raise typer.Exit(code=0)

    if result.error is not None or (result.skipped and result.skip_reason):
        raise typer.Exit(code=1 if result.error is not None else 0)
    if result.is_regression:
        raise typer.Exit(code=1)
    raise typer.Exit(code=0)


def _read_eval_baseline(manifest: Manifest) -> float | None:
    """Parse ``manifest.answers["eval_baseline"]`` as a float, ``None`` if absent."""
    raw = (manifest.answers or {}).get("eval_baseline")
    if not raw:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _render_eval_result(result: Any) -> None:
    """Render a Rich panel + per-case table for an EvalResult."""
    from rich.panel import Panel as RichPanel
    from rich.table import Table as RichTable

    if result.skipped:
        console.print(
            RichPanel(
                f"[yellow]Skipped:[/] {result.skip_reason}",
                title=f"eval/{result.target}",
                border_style="yellow",
                expand=False,
            )
        )
        return
    if result.error is not None:
        console.print(
            RichPanel(
                f"[red]Error:[/] {result.error}",
                title=f"eval/{result.target}",
                border_style="red",
                expand=False,
            )
        )
        return

    table = RichTable(show_header=True, header_style="bold", expand=False)
    table.add_column("Case", overflow="fold")
    table.add_column("Score", justify="right")
    table.add_column("Pass", justify="center")
    for case in result.cases:
        mark = "[green]✓[/]" if case.passed else "[red]✗[/]"
        table.add_row(case.name, f"{case.score:.2f}", mark)
    table.add_section()
    summary_cells = [f"Total ({len(result.cases)} cases)", f"{result.total:.2f}", ""]
    table.add_row(*summary_cells, style="bold")
    if result.baseline_total is not None:
        table.add_row("Baseline", f"{result.baseline_total:.2f}", "")
        delta = result.delta or 0.0
        delta_color = "red" if result.is_regression else ("green" if delta > 0 else "dim")
        table.add_row("Δ", f"[{delta_color}]{delta:+.2f}[/]", "")

    border = "red" if result.is_regression else "green"
    title = f"eval/{result.target} — {result.passed_count}/{len(result.cases)} passed"
    console.print(RichPanel(table, title=title, border_style=border, expand=False))
    if result.is_regression:
        console.print(
            f"[red]Regression detected:[/] total {result.delta:+.2f} vs baseline. "
            f"Re-run with --update-baseline if this was intentional."
        )


def _emit_eval_json(result: Any) -> None:
    """Emit a stable JSON shape for the eval result."""
    import json as _json

    body = {
        "target": result.target,
        "skipped": result.skipped,
        "skip_reason": result.skip_reason,
        "error": result.error,
        "total": result.total,
        "baseline_total": result.baseline_total,
        "delta": result.delta,
        "is_regression": result.is_regression,
        "cmd_run": list(result.cmd_run),
        "cases": [{"name": c.name, "score": c.score, "passed": c.passed} for c in result.cases],
    }
    typer.echo(_json.dumps(body, indent=2))


@app.command("logs", rich_help_panel="Run & deploy")
def cmd_logs(
    service: str = typer.Argument(..., help="docker-compose service name."),
    follow: bool = typer.Option(True, "-f/--no-follow", help="Stream new log lines."),
    tail: int = typer.Option(100, "--tail", min=0, help="Number of past lines to show."),
    cwd: Path = typer.Option(Path("."), "--cwd", help="Project directory."),
) -> None:
    """Tail container logs. Thin wrapper around ``docker compose logs``.

    The reserved service names ``frontend`` and ``backend`` tail the dev
    servers' log files under ``.scaffold/`` rather than going through docker.
    """
    project_dir = cwd.expanduser().resolve()

    reserved = {"frontend": "frontend.log", "backend": "backend.log"}
    if service in reserved:
        _tail_scaffold_log(
            project_dir, log_name=reserved[service], label=service, follow=follow, tail=tail
        )
        return

    compose_path = _find_docker_compose(project_dir)
    if compose_path is None:
        console.print(f"[red]Error:[/] no docker-compose.yml found under {project_dir}")
        raise typer.Exit(code=1)
    if shutil.which("docker") is None:
        console.print("[red]Error:[/] docker not on PATH")
        raise typer.Exit(code=1)

    cmd = ["docker", "compose", "logs", "--tail", str(tail)]
    if follow:
        cmd.append("-f")
    cmd.append(service)
    rc = subprocess.run(  # noqa: S603 — list-form, shell=False
        cmd, cwd=str(compose_path.parent), check=False
    ).returncode
    if rc != 0:
        raise typer.Exit(code=rc)


def _tail_scaffold_log(
    project_dir: Path, *, log_name: str, label: str, follow: bool, tail: int
) -> None:
    """Tail a detached server's ``.scaffold`` log file. Friendly error if missing."""
    log_file = project_dir / SCAFFOLD_DIR / log_name
    if not log_file.is_file():
        console.print(f"[yellow]No {label} log — has `up` been run?[/]")
        raise typer.Exit(code=1)
    tail_bin = shutil.which("tail")
    if tail_bin is not None:
        cmd = [tail_bin, "-n", str(tail)]
        if follow:
            cmd.append("-f")
        cmd.append(str(log_file))
        # exec replaces the current process so SIGINT goes straight to tail —
        # no extra process to manage, and the user's ^C is instant.
        os.execvp(cmd[0], cmd)  # noqa: S606 — list-form, no shell
        return
    # Pure-Python fallback for Windows / minimal containers.
    _python_tail(log_file, follow=follow, tail=tail)


def _python_tail(path: Path, *, follow: bool, tail: int) -> None:
    """Minimal ``tail -f``-style reader for when ``tail`` isn't on PATH."""
    import time

    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    for line in lines[-tail:] if tail > 0 else []:
        console.print(line)
    if not follow:
        return
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        fh.seek(0, os.SEEK_END)
        try:
            while True:
                chunk = fh.readline()
                if not chunk:
                    time.sleep(0.25)
                    continue
                console.print(chunk.rstrip("\n"))
        except KeyboardInterrupt:
            return


# ---------------------------------------------------------------------------
# Lifecycle helpers
# ---------------------------------------------------------------------------


def _resolve_deploy_targets(manifest: Manifest) -> list[str]:
    """Return target names declared by the manifest's host.* capabilities.

    Reads the resolved capability stack lazily — if the deployments source
    isn't available, returns an empty list and the deploy command falls
    through to whatever target the user passed.
    """
    if not manifest.capabilities:
        return []
    recipe = _resolve_recipe_silently(manifest.recipe)
    stack = _resolve_capability_stack_silently(recipe, capabilities=manifest.capabilities)
    if stack is None:
        return []
    targets: list[str] = []
    for cap in stack.capabilities:
        for cfg in cap.deploy_configs:
            if cfg.target not in targets:
                targets.append(cfg.target)
    return targets


def _find_docker_compose(project_dir: Path) -> Path | None:
    for candidate in (
        project_dir / "docker-compose.yml",
        project_dir / "infra" / "docker-compose.yml",
        project_dir / "compose.yaml",
    ):
        if candidate.is_file():
            return candidate
    return None


def _stop_pid_service(project_dir: Path, *, pid_name: str, step_id: str, label: str) -> None:
    """Best-effort teardown of a detached server we spawned (frontend / backend).

    Reads ``<project>/.scaffold/<pid_name>``, kills the process group with
    SIGTERM, removes the PID file, and resets the step state so the next
    ``up`` re-launches. Missing/malformed PID files are silently OK.
    """
    import signal

    pid_file = project_dir / SCAFFOLD_DIR / pid_name
    if not pid_file.is_file():
        return
    try:
        data = json.loads(pid_file.read_text(encoding="utf-8"))
        pid = int(data["pid"])
    except (json.JSONDecodeError, KeyError, ValueError, OSError):
        pid_file.unlink(missing_ok=True)
        _reset_step_state(project_dir, step_id)
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, AttributeError, OSError):
        # Windows / already-dead / not our process — fall back to direct kill.
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    pid_file.unlink(missing_ok=True)
    _reset_step_state(project_dir, step_id)
    console.print(f"[green]{label} stopped.[/]")


def _stop_frontend(project_dir: Path) -> None:
    """Stop the dev server spawned by ``launch_frontend``."""
    _stop_pid_service(
        project_dir,
        pid_name="frontend.pid",
        step_id="launch_frontend",
        label="Frontend dev server",
    )


def _stop_backend(project_dir: Path) -> None:
    """Stop the HTTP server spawned by ``launch_backend``."""
    _stop_pid_service(
        project_dir,
        pid_name="backend.pid",
        step_id="launch_backend",
        label="Backend server",
    )


def _open_browser_safe(url: str) -> bool:
    """Open ``url`` in the user's default browser. Swallows headless/CI failures.

    Helper for the autorun brief (next PR). Shipped here so that change is a
    smaller delta.
    """
    import webbrowser

    if os.environ.get("BROWSER") == "none":
        return False
    try:
        return webbrowser.open(url, new=2)
    except Exception:  # noqa: BLE001 — webbrowser raises a grab-bag of OS errors
        return False


def _reset_step_state(project_dir: Path, step_id: str) -> None:
    """Best-effort: mark ``step_id`` PENDING in .scaffold/state.json.

    Lets the next ``agent-scaffold up`` re-detect from a clean slate after
    ``down`` has removed the containers. Failures here are silently OK —
    the orchestrator handles a missing state file gracefully.
    """
    try:
        from agent_scaffold.orchestrator import (
            StepState,
            StepStatus,
            read_state,
            write_state,
        )
    except ImportError:
        return
    try:
        state = read_state(project_dir)
    except Exception:  # noqa: BLE001 — defensive; state file may be malformed
        return
    if step_id in state.steps:
        state.steps[step_id] = StepState(status=StepStatus.PENDING)
        try:
            write_state(project_dir, state)
        except OSError:
            pass


def _render_deploy_result(result: Any) -> None:
    from rich.panel import Panel

    color = "yellow" if result.skipped else ("green" if result.exit_code == 0 else "red")
    lines = [f"[bold]{result.target}[/]: {result.summary}"]
    if result.cmd_run:
        lines.append(f"command: [cyan]{' '.join(result.cmd_run)}[/]")
    if result.dashboard_url:
        lines.append(f"dashboard: {result.dashboard_url}")
    console.print(Panel("\n".join(lines), title="Deploy", border_style=color, expand=False))


def _render_status_table(services: list[Any], capabilities: list[Any]) -> None:
    from rich.table import Table

    if not services and not capabilities:
        console.print("[yellow]No services or capabilities declared by this project.[/]")
        return
    table = Table(title="Status", header_style="bold cyan")
    table.add_column("Kind")
    table.add_column("ID")
    table.add_column("Status")
    table.add_column("Detail", overflow="fold")
    for row in services:
        table.add_row("service", row.id, _status_glyph(row.status), row.detail or row.title)
    for row in capabilities:
        table.add_row("capability", row.id, _status_glyph(row.status), row.detail or row.title)
    console.print(table)


def _status_glyph(status: Any) -> str:
    text = str(status.value if hasattr(status, "value") else status)
    glyphs = {
        "ok": "[green]✓ ok[/]",
        "warn": "[yellow]⚠ warn[/]",
        "fail": "[red]✗ fail[/]",
        "skip": "[dim]⏭ skip[/]",
    }
    return glyphs.get(text, text)


def scaffold_main() -> None:
    """Entry point for the ``scaffold`` binary alias.

    Bare ``scaffold`` opens the REPL (like ``claude``). With a subcommand or
    ``--help``/``--version``, routes through the main Typer app exactly like
    ``agent-scaffold`` does.
    """
    args = sys.argv[1:]
    wants_help_or_version = any(a in {"--help", "-h", "--version"} for a in args)
    has_subcommand = any(not a.startswith("-") for a in args)
    if not has_subcommand and not wants_help_or_version:
        sys.argv.insert(1, "scaffold")
    app()


# Re-export for ``python -m agent_scaffold``.
__all__ = ["app", "scaffold_main"]
