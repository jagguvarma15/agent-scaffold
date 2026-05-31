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
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
from rich.logging import RichHandler
from rich.panel import Panel

from agent_scaffold import __version__
from agent_scaffold._scaffold_dir import SCAFFOLD_DIR
from agent_scaffold.branding import print_banner
from agent_scaffold.capabilities import load_capabilities
from agent_scaffold.capabilities import resolve as resolve_capabilities
from agent_scaffold.cli_auth import auth_app
from agent_scaffold.cli_doctor import doctor_app
from agent_scaffold.cli_secrets import secrets_app
from agent_scaffold.cli_shared import console
from agent_scaffold.config import Config, ConfigError, load_config
from agent_scaffold.context import ContextBudgetError, assemble
from agent_scaffold.contract import (
    ContractParseError,
    parse,
)
from agent_scaffold.costs import estimate_preflight as estimate_preflight_cost
from agent_scaffold.discovery import (
    DiscoveryError,
    ExternalService,
    Recipe,
    discover_recipes,
)
from agent_scaffold.doctor import CheckResult
from agent_scaffold.effort import EFFORT_PRESETS
from agent_scaffold.generator import (
    GenerationRequest,
    extract_fenced_content,
    generate,
    generate_single_file,
)
from agent_scaffold.imports import discover_neighbours
from agent_scaffold.language_hints import (
    UnknownLanguageError,
    available_languages,
    load_language_hints,
)
from agent_scaffold.manifest import (
    Manifest,
    ManifestNotFoundError,
    UpdateEntry,
    build_file_entries,
    read_manifest,
    update_file_entry,
    write_manifest,
)
from agent_scaffold.merge import (
    has_unresolved_markers,
    three_way_merge,
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
from agent_scaffold.progress import (
    NullProgressDisplay,
    ProgressEvent,
    RichProgressDisplay,
    make_step_display,
    render_failure_panel,
)
from agent_scaffold.sources import (
    BlueprintsMode,
    DeploymentsMode,
    SourceFetchError,
    resolve_blueprints,
    resolve_deployments,
)
from agent_scaffold.steps import default_steps_for
from agent_scaffold.template_snapshot import (
    cleanup_tempdir,
    compute_template_sha,
    has_snapshot,
    load_generation_snapshot,
    prune_snapshots,
    save_generation_snapshot,
)
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
app.add_typer(doctor_app, name="doctor")
app.add_typer(auth_app, name="auth")
app.add_typer(secrets_app, name="secrets")

PROJECT_NAME_RE = re.compile(r"^[a-z0-9_-]+$")

KNOWN_MODELS: list[tuple[str, str]] = [
    ("claude-opus-4-7", "Opus 4.7 — highest quality (slowest, most expensive)"),
    ("claude-sonnet-4-6", "Sonnet 4.6 — balanced (recommended for most runs)"),
    ("claude-haiku-4-5-20251001", "Haiku 4.5 — fast iteration (lowest quality)"),
]


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
    "[bold]Quick start:[/]",
    "  [#FFA500]agent-scaffold scaffold[/]  interactive shell (recommended)",
    "  [#FF8C00]agent-scaffold doctor[/]    verify environment + service probes",
    "  [#FF6347]agent-scaffold new[/]       one-shot project generator",
    "  [#FF4500]agent-scaffold up[/]        install, wire creds, migrate, smoke",
    "  [#DC143C]agent-scaffold update[/]    3-way merge against template evolution",
    "",
    "[dim]Run `agent-scaffold --help` for the full command reference.[/]",
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
    if raw not in ("auto", "bundled"):
        raise typer.BadParameter(f"--deployments-source must be 'auto' or 'bundled', got {raw!r}")
    return raw  # type: ignore[return-value]


def _coerce_blueprints_mode(raw: str) -> BlueprintsMode:
    if raw not in ("auto", "skip"):
        raise typer.BadParameter(f"--blueprints-source must be 'auto' or 'skip', got {raw!r}")
    return raw  # type: ignore[return-value]


def _validate_project_name(name: str) -> str:
    if not PROJECT_NAME_RE.match(name):
        raise typer.BadParameter(
            "Project name must contain only lowercase letters, digits, hyphens, " "and underscores."
        )
    return name


def _python_module_name(project_name: str, language: str) -> str:
    if language == "python" and "-" in project_name:
        replaced = project_name.replace("-", "_")
        console.print(
            f"[yellow]Note:[/] Python module name will be '{replaced}' "
            "(hyphens replaced with underscores)."
        )
        return replaced
    return project_name


def _interactive_select(
    prompt: str, choices: list[tuple[str, str]], default: str | None = None
) -> str:
    """Wrap ``questionary.select`` so we only import it when needed."""
    import questionary

    options = [questionary.Choice(title=label, value=value) for value, label in choices]
    answer = questionary.select(prompt, choices=options, default=default).ask()
    if answer is None:
        raise typer.Abort()
    return str(answer)


def _interactive_text(prompt: str, default: str | None = None) -> str:
    import questionary

    answer = questionary.text(prompt, default=default or "").ask()
    if answer is None:
        raise typer.Abort()
    return str(answer)


def _interactive_path(prompt: str, default: str | None = None) -> str:
    import questionary

    answer = questionary.path(prompt, default=default or "").ask()
    if answer is None:
        raise typer.Abort()
    return str(answer)


# Pipeline helpers + run_generation moved to agent_scaffold.pipeline so the
# upcoming REPL can reuse them. cmd_regenerate still imports
# ``run_post_gen_formatter`` from there.


@app.command("scaffold")
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
        )
    except SourceFetchError as exc:
        console.print(f"[red]Source resolution error:[/] {exc}")
        raise typer.Exit(code=1) from exc

    exit_code = run_shell(cfg, dep_source, bp_source, console=console)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


@app.command("config")
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


@app.command("new")
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
            "warnings) and prompt Y/n before calling the LLM. Default on for --effort high."
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
        )
    except SourceFetchError as exc:
        console.print(f"[red]Source resolution error:[/] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[dim]Deployments:[/] {dep_source.label}")
    console.print(f"[dim]Blueprints: [/] {bp_source.label}")
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
    chosen_framework = _select_framework(hints, framework, non_interactive)

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
    resolved_stack = resolve_capabilities(recipe, catalog)
    if resolved_stack.unresolved:
        console.print(
            f"[yellow]Capabilities not in catalog:[/] {', '.join(resolved_stack.unresolved)} "
            "(upgrade your deployments source or remove from the recipe)"
        )

    with console.status("Assembling context..."):
        try:
            ctx = assemble(
                recipe,
                chosen_language,
                chosen_framework,
                deployments,
                blueprints_path=blueprints,
                max_context_tokens=cfg.max_context_tokens,
                max_link_depth=cfg.max_link_depth,
                max_tokens_per_doc=cfg.max_tokens_per_doc,
                resolved_stack=resolved_stack if resolved_stack.capabilities else None,
            )
        except ContextBudgetError as exc:
            console.print(f"[red]Context budget error:[/] {exc}")
            raise typer.Exit(code=1) from exc
    if ctx.summary is not None:
        console.print(Panel(ctx.summary.render(), title="Assembled context", expand=False))
    else:
        console.print(
            f"[green]Context ready:[/] {len(ctx.referenced_paths)} reference(s), "
            f"~{ctx.token_estimate} tokens."
        )

    topology, roles = resolve_topology(recipe, ctx.body)

    plan_default_on = effort == "high"
    plan_enabled = plan if plan is not None else plan_default_on
    if plan_enabled and not non_interactive:
        warnings: list[str] = []
        if ctx.summary is not None and ctx.summary.total_tokens > int(0.95 * ctx.summary.cap):
            warnings.append(
                f"Context is {int(100 * ctx.summary.total_tokens / max(1, ctx.summary.cap))}% of cap"
            )
        if not recipe.required_files:
            warnings.append("Recipe declares no required_files — hard to validate output")
        readiness = _probe_services_for_plan(
            recipe.external_services, probe_services=probe_services
        )
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
    display: RichProgressDisplay | NullProgressDisplay
    if non_interactive:
        display = NullProgressDisplay()
    else:
        display = RichProgressDisplay(
            console,
            cfg.model,
            verbose=verbose_flag,
            expected_files=expected_files,
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
    try:
        run_report = run_generation(pipeline_inputs, display=display)
    except PipelineError as exc:
        # Pipeline already printed phase-specific progress events; surface the
        # error message + hint and exit with non-zero so callers in shell
        # scripts can detect the failure.
        console.print(f"[red]{exc.phase or 'pipeline'} failed:[/] {exc.message}")
        if exc.hint:
            console.print(exc.hint)
        raise typer.Exit(code=1) from exc

    result = run_report.result
    report = run_report.report
    validation_results = run_report.validation_results

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

    if result is not None:
        print_next_steps(dest, chosen_language, result.smoke_check, result.post_install)


def _select_recipe(recipes: list[Recipe], slug: str | None, non_interactive: bool) -> Recipe:
    if slug is not None:
        match = next((r for r in recipes if r.slug == slug), None)
        if match is None:
            available = ", ".join(r.slug for r in recipes)
            raise typer.BadParameter(f"Unknown recipe slug: {slug}. Available: {available}")
        return match
    if non_interactive:
        raise typer.BadParameter("--recipe is required in --non-interactive mode")
    choices = [(r.slug, f"{r.title}  [{r.status}]") for r in recipes]
    chosen_slug = _interactive_select("Pick a recipe:", choices)
    return next(r for r in recipes if r.slug == chosen_slug)


def _select_language(recipe: Recipe, language: str | None, non_interactive: bool) -> str:
    candidates = [lang for lang in recipe.languages if lang in available_languages()]
    if not candidates:
        candidates = available_languages()
    if language is not None:
        if language not in candidates:
            raise typer.BadParameter(
                f"Language {language} not supported by recipe {recipe.slug}. "
                f"Allowed: {', '.join(candidates)}"
            )
        return language
    if non_interactive:
        raise typer.BadParameter("--language is required in --non-interactive mode")
    if len(candidates) == 1:
        return candidates[0]
    return _interactive_select("Pick a target language:", [(c, c) for c in candidates])


def _select_model(cfg: Config, override: str | None, non_interactive: bool) -> str:
    if override:
        return override
    if non_interactive:
        return cfg.model
    default = cfg.model if any(mid == cfg.model for mid, _ in KNOWN_MODELS) else None
    return _interactive_select(
        "Pick a model:",
        [(mid, label) for mid, label in KNOWN_MODELS],
        default=default,
    )


def _select_framework(hints: dict[str, Any], framework: str | None, non_interactive: bool) -> str:
    available = list((hints.get("framework_dependencies") or {}).keys())
    available.append("none")
    if framework is not None:
        if framework not in available:
            raise typer.BadParameter(
                f"Framework {framework} not in language hints. " f"Allowed: {', '.join(available)}"
            )
        return framework
    if non_interactive:
        return "none"
    return _interactive_select("Pick a framework:", [(f, f.replace("_", " ")) for f in available])


def _select_write_mode() -> WriteMode:
    chosen = _interactive_select(
        "Destination is not empty. What should I do?",
        [
            (WriteMode.skip.value, "skip existing files"),
            (WriteMode.diff.value, "show diffs and prompt"),
            (WriteMode.overwrite.value, "overwrite all"),
            (WriteMode.abort.value, "abort"),
        ],
        default=WriteMode.skip.value,
    )
    return WriteMode(chosen)


def _render_unified_diff(rel_path: str, old: str, new: str) -> str:
    import difflib

    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{rel_path}",
        tofile=f"b/{rel_path}",
    )
    return "".join(diff)


@app.command("regenerate")
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
    except SourceFetchError as exc:
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


@app.command("validate")
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


def _resolve_capability_stack_silently(recipe: Recipe | None) -> Any | None:
    """Resolve the recipe's capabilities to a ``ResolvedStack`` without prompting.

    Mirrors :func:`_resolve_recipe_silently` — failures (no deployments,
    no catalog, recipe without capabilities) return ``None`` so the
    bootstrap steps SKIP cleanly. Returning ``None`` is the back-compat
    signal: a stack with an empty ``capabilities`` list would also surface
    as "nothing to do" to every step, but ``None`` is unambiguous.
    """
    if recipe is None or not recipe.capabilities:
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
    stack = resolve_capabilities(recipe, catalog)
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


@app.command("up")
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

    resolved_stack = _resolve_capability_stack_silently(recipe)

    steps = default_steps_for(
        manifest,
        recipe,
        yes=flags.yes,
        confirm_commit_push=flags.confirm_commit_push,
    )
    try:
        orch = Orchestrator(steps, project_dir, manifest, resolved_stack=resolved_stack)
    except OrchestratorError as exc:
        console.print(f"[red]Orchestrator error:[/] {exc}")
        raise typer.Exit(code=1) from exc

    rows = orch.plan()
    console.print(render_plan_table(rows))

    if flags.plan_only:
        raise typer.Exit(code=0)

    step_specs: list[tuple[str, str]] = [(s.id, s.description) for s in steps]

    if not flags.yes:
        action = _interactive_select(
            "Proceed?",
            choices=[
                ("yes", "yes — run the plan above"),
                ("edit", "edit — toggle which steps to run"),
                ("no", "no — abort without changes"),
            ],
            default="yes",
        )
        if action == "no":
            console.print("[yellow]Aborted.[/]")
            raise typer.Exit(code=0)
        if action == "edit":
            current = set(flags.only) if flags.only else {sid for sid, _ in step_specs}
            chosen_ids = _select_active_steps(step_specs, current)
            if not chosen_ids:
                console.print("[yellow]No steps selected; aborted.[/]")
                raise typer.Exit(code=0)
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

    _print_step_summary(result.summary)

    # Surface every live URL the user can open after a successful run. The
    # panel quietly handles missing capabilities / missing frontend PID file,
    # so it stays useful on partial runs too.
    if result.exit_code == 0:
        from agent_scaffold.welcome import render_welcome_panel

        console.print(render_welcome_panel(project_dir, manifest, resolved_stack))

    raise typer.Exit(code=result.exit_code)


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


UPDATE_IN_PROGRESS_FILENAME = "update.in-progress.json"


@dataclass(frozen=True)
class _UpdateClassification:
    added: list[str]
    modified: list[str]
    conflicted: list[str]
    removed: list[str]
    binary_skipped: list[str]
    merge_results: dict[str, object]  # path -> MergeResult; object to avoid forward-ref


def _in_progress_path(project_dir: Path) -> Path:
    return project_dir / SCAFFOLD_DIR / UPDATE_IN_PROGRESS_FILENAME


def _read_in_progress(project_dir: Path) -> dict[str, Any] | None:
    path = _in_progress_path(project_dir)
    if not path.is_file():
        return None
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _write_in_progress(project_dir: Path, payload: dict[str, Any]) -> None:
    path = _in_progress_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _clear_in_progress(project_dir: Path) -> None:
    _in_progress_path(project_dir).unlink(missing_ok=True)


def _classify_update(
    base_dir: Path,
    project_dir: Path,
    fresh_files: dict[str, str],
    *,
    previous_files: list[str],
) -> _UpdateClassification:
    """Compute the 3-way classification per file.

    ``previous_files`` is the manifest's list of paths from the prior gen;
    used to identify removals (files that *were* templated but aren't now).
    """
    added: list[str] = []
    modified: list[str] = []
    conflicted: list[str] = []
    binary_skipped: list[str] = []
    merge_results: dict[str, Any] = {}

    previous_set = set(previous_files)
    base_lookup = {rel: (base_dir / rel) for rel in _iter_base_files(base_dir)}

    for rel, fresh_text in fresh_files.items():
        on_disk = project_dir / rel
        base_path = base_lookup.get(rel)
        if not on_disk.exists():
            added.append(rel)
            continue
        # Both base and ours present → 3-way; missing base → 2-way (theirs vs ours).
        ours = on_disk.read_bytes()
        theirs = fresh_text.encode("utf-8")
        base_bytes = base_path.read_bytes() if base_path and base_path.is_file() else theirs
        merge = three_way_merge(base_bytes, ours, theirs)
        merge_results[rel] = merge
        if merge.binary:
            binary_skipped.append(rel)
            continue
        if ours.decode("utf-8", errors="replace") == merge.text:
            continue  # nothing actually changes — skip silently
        if merge.conflicted:
            conflicted.append(rel)
        else:
            modified.append(rel)

    removed = sorted(
        rel for rel in previous_set if rel not in fresh_files and (project_dir / rel).is_file()
    )
    return _UpdateClassification(
        added=sorted(added),
        modified=sorted(modified),
        conflicted=sorted(conflicted),
        removed=removed,
        binary_skipped=sorted(binary_skipped),
        merge_results=merge_results,
    )


def _iter_base_files(base_dir: Path) -> list[str]:
    """Return file paths from a previously-extracted snapshot, relative to base_dir."""
    if not base_dir.is_dir():
        return []
    out: list[str] = []
    for path in base_dir.rglob("*"):
        if path.is_file():
            out.append(path.relative_to(base_dir).as_posix())
    return out


def _render_update_plan(classification: _UpdateClassification) -> Panel:
    rows: list[str] = []
    rows.append(
        f"[green]Files added   ({len(classification.added)}):[/]  "
        + (", ".join(classification.added) or "[dim](none)[/]")
    )
    rows.append(
        f"[cyan]Files updated ({len(classification.modified)}):[/]  "
        + (", ".join(classification.modified) or "[dim](none)[/]")
    )
    if classification.conflicted:
        rows.append(
            f"[red]Conflicts     ({len(classification.conflicted)}):[/]  "
            + ", ".join(classification.conflicted)
            + "\n                     → conflict markers will be written"
        )
    else:
        rows.append("[red]Conflicts     (0):[/]")
    rows.append(
        f"[yellow]Files removed ({len(classification.removed)}):[/]  "
        + (", ".join(classification.removed) or "[dim](none)[/]")
    )
    if classification.binary_skipped:
        rows.append(
            f"[dim]Binary kept   ({len(classification.binary_skipped)}):[/] "
            + ", ".join(classification.binary_skipped)
        )
    return Panel("\n".join(rows), title="Update plan", expand=False)


def _apply_update(
    project_dir: Path,
    fresh_files: dict[str, str],
    classification: _UpdateClassification,
    *,
    remove_decisions: dict[str, bool],
) -> None:
    """Write the merged content + handle removals. No prompting here."""
    for rel in classification.added:
        target = project_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(fresh_files[rel], encoding="utf-8")
    for rel in classification.modified + classification.conflicted:
        merge = classification.merge_results[rel]
        target = project_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(merge.text, encoding="utf-8")  # type: ignore[attr-defined]
    for rel, remove in remove_decisions.items():
        if remove:
            (project_dir / rel).unlink(missing_ok=True)


@app.command("update")
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
    del debug  # currently unused; reserved for future verbose flag
    project_dir = project_dir.expanduser().resolve()
    try:
        manifest = read_manifest(project_dir)
    except ManifestNotFoundError as exc:
        console.print(f"[red]Error:[/] {exc}")
        raise typer.Exit(code=1) from exc

    # --continue path: skip re-generation; just close the merge out.
    if continue_:
        _continue_update(project_dir, manifest)
        return

    in_progress = _read_in_progress(project_dir)
    if in_progress is not None:
        console.print(
            "[yellow]Previous update in progress.[/] "
            "Pass --continue to resume after resolving markers, "
            f"or `rm {_in_progress_path(project_dir)}` to abandon."
        )
        raise typer.Exit(code=1)

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
    deployments = dep_source.path
    console.print(f"[dim]Deployments:[/] {dep_source.label}")

    current_sha = compute_template_sha(deployments)
    if manifest.template_snapshot_sha == current_sha:
        console.print("[green]Template unchanged since last generation.[/] Nothing to update.")
        raise typer.Exit(code=0)

    # Locate the previous generation snapshot (the merge base). If the manifest
    # is a v1 upgrade and there's no snapshot to compare against, bootstrap by
    # snapshotting the current on-disk state under the *current* template sha
    # — next update will have a real merge base.
    bootstrap = not (
        manifest.template_snapshot_sha and has_snapshot(project_dir, manifest.template_snapshot_sha)
    )
    if bootstrap:
        console.print(
            "[yellow]No prior generation snapshot;[/] bootstrapping by snapshotting "
            "the current project files. Next update will produce real diffs."
        )
        on_disk_files = {
            f.path: (project_dir / f.path).read_text(encoding="utf-8", errors="replace")
            for f in manifest.files
            if (project_dir / f.path).is_file()
        }
        snap = save_generation_snapshot(project_dir, current_sha, on_disk_files)
        prune_snapshots(project_dir)
        manifest = manifest.model_copy(update={"template_snapshot_sha": snap.sha})
        write_manifest(project_dir, manifest)
        raise typer.Exit(code=0)

    base_tmp = load_generation_snapshot(project_dir, manifest.template_snapshot_sha or "")
    try:
        base_files_dir = base_tmp
        fresh_files = _regenerate_for_update(manifest, deployments, cfg)
        if fresh_files is None:
            console.print(
                "[red]Regeneration failed.[/] Original files untouched. "
                "Run with --debug to see the raw model response (if cached)."
            )
            raise typer.Exit(code=1)
        previous_paths = [f.path for f in manifest.files]
        classification = _classify_update(
            base_files_dir, project_dir, fresh_files, previous_files=previous_paths
        )
        console.print(_render_update_plan(classification))
        if dry_run:
            raise typer.Exit(code=0)
        if not yes:
            action = _interactive_select(
                "Apply?",
                choices=[
                    ("yes", "yes — apply the merge above"),
                    ("dry-run", "dry-run — print the plan only (no writes)"),
                    ("no", "no — abort without changes"),
                ],
                default="yes",
            )
            if action == "no":
                console.print("[yellow]Aborted.[/]")
                raise typer.Exit(code=0)
            if action == "dry-run":
                raise typer.Exit(code=0)

        remove_decisions = _decide_removals(classification.removed, yes=yes)
        _apply_update(project_dir, fresh_files, classification, remove_decisions=remove_decisions)

        if classification.conflicted:
            _write_in_progress(
                project_dir,
                {
                    "from_template_sha": manifest.template_snapshot_sha,
                    "to_template_sha": current_sha,
                    "from_schema": manifest.schema_version,
                    "to_schema": manifest.schema_version,
                    "conflicts": classification.conflicted,
                    "files_added": classification.added,
                    "files_modified": classification.modified,
                    "files_removed": [r for r, drop in remove_decisions.items() if drop],
                    "model": manifest.model,
                },
            )
            console.print(
                f"[yellow]{len(classification.conflicted)} conflict(s) require manual "
                "resolution.[/]\n  "
                + "\n  ".join(classification.conflicted)
                + "\n\nResolve markers, then run `agent-scaffold update --continue`."
            )
            raise typer.Exit(code=2)

        _finalise_update(
            project_dir,
            manifest,
            current_sha,
            classification,
            removed=[r for r, drop in remove_decisions.items() if drop],
            generated_files=fresh_files,
        )
        console.print(
            f"[green]Update complete.[/] "
            f"+{len(classification.added)} / ~{len(classification.modified)} / "
            f"-{sum(remove_decisions.values())} files."
        )
    finally:
        cleanup_tempdir(base_tmp)


def _decide_removals(removed: list[str], *, yes: bool) -> dict[str, bool]:
    """Per file, ask the user whether to remove it. ``--yes`` keeps everything.

    Default is **no**: removals are sticky — we'd rather over-keep than delete
    something the user values. Output dict maps path → ``True if remove``.
    """
    if not removed:
        return {}
    if yes:
        # In --yes mode we never silently delete files. WARN and keep.
        for rel in removed:
            console.print(
                f"[yellow]Warning:[/] {rel} is gone from the template; "
                "keeping (pass without --yes to confirm removal)."
            )
        return {rel: False for rel in removed}
    decisions: dict[str, bool] = {}
    import questionary

    for rel in removed:
        answer = questionary.confirm(
            f"File {rel} was in the template but is gone now. Remove from your project?",
            default=False,
        ).ask()
        decisions[rel] = bool(answer)
    return decisions


def _continue_update(project_dir: Path, manifest: Manifest) -> None:
    """``--continue`` path: verify markers cleared then finalise."""
    in_progress = _read_in_progress(project_dir)
    if in_progress is None:
        console.print("[red]No update in progress.[/] Run `agent-scaffold update` first.")
        raise typer.Exit(code=1)
    conflicted: list[str] = list(in_progress.get("conflicts", []))
    still_marked: list[tuple[str, int]] = []
    for rel in conflicted:
        target = project_dir / rel
        if not target.is_file():
            still_marked.append((rel, -1))
            continue
        text = target.read_text(encoding="utf-8", errors="replace")
        if has_unresolved_markers(text):
            from agent_scaffold.merge import count_unresolved_markers

            still_marked.append((rel, count_unresolved_markers(text)))
    if still_marked:
        console.print(
            "[red]Conflict markers still present in:[/]\n  "
            + "\n  ".join(f"{rel} ({n} marker line(s))" for rel, n in still_marked)
            + "\n\nResolve all `<<<<<<< / ======= / >>>>>>>` regions, then "
            "re-run `agent-scaffold update --continue`.\n"
            "If you want to abandon: "
            f"`rm {_in_progress_path(project_dir)} && git checkout .`"
        )
        raise typer.Exit(code=1)

    classification = _UpdateClassification(
        added=list(in_progress.get("files_added", [])),
        modified=list(in_progress.get("files_modified", [])),
        conflicted=conflicted,
        removed=list(in_progress.get("files_removed", [])),
        binary_skipped=[],
        merge_results={},
    )
    # On --continue we don't have the freshly-generated text in memory anymore,
    # so use the now-resolved on-disk files as the snapshot contents.
    on_disk = {
        rel: (project_dir / rel).read_text(encoding="utf-8", errors="replace")
        for rel in {*classification.added, *classification.modified, *classification.conflicted}
        if (project_dir / rel).is_file()
    }
    _finalise_update(
        project_dir,
        manifest,
        str(in_progress["to_template_sha"]),
        classification,
        removed=classification.removed,
        generated_files=on_disk,
    )
    _clear_in_progress(project_dir)
    console.print("[green]Conflicts resolved. Update finalised.[/]")


def _finalise_update(
    project_dir: Path,
    manifest: Manifest,
    new_sha: str,
    classification: _UpdateClassification,
    *,
    removed: list[str],
    generated_files: dict[str, str],
) -> None:
    """Append the update history entry, save the new snapshot, prune LRU."""
    save_generation_snapshot(project_dir, new_sha, generated_files)
    prune_snapshots(project_dir)
    new_entry = UpdateEntry(
        timestamp=datetime.now(UTC).isoformat(),
        from_schema=manifest.schema_version,
        to_schema=manifest.schema_version,
        from_template_sha=manifest.template_snapshot_sha,
        to_template_sha=new_sha,
        model=manifest.model,
        files_added=classification.added,
        files_modified=classification.modified,
        files_removed=removed,
        files_conflicted=classification.conflicted,
    )
    updated_files: list[str] = list({f.path for f in manifest.files} | set(classification.added))
    for rel in removed:
        if rel in updated_files:
            updated_files.remove(rel)
    new_manifest = manifest.model_copy(
        update={
            "template_snapshot_sha": new_sha,
            "update_history": [*manifest.update_history, new_entry],
            "files": build_file_entries(project_dir, sorted(updated_files)),
        }
    )
    write_manifest(project_dir, new_manifest)


def _regenerate_for_update(
    manifest: Manifest, deployments: Path, cfg: Config
) -> dict[str, str] | None:
    """Re-run generation with the manifest's captured answers; return ``{rel: text}``.

    Returns ``None`` on a hard failure so the caller can keep the project
    untouched.
    """
    try:
        recipes = discover_recipes(deployments)
    except DiscoveryError as exc:
        console.print(f"[red]Discovery failed:[/] {exc}")
        return None
    recipe = next((r for r in recipes if r.slug == manifest.recipe), None)
    if recipe is None:
        console.print(f"[red]Recipe {manifest.recipe!r} not found in deployments.[/]")
        return None

    language = manifest.language
    framework = manifest.framework
    hints = _load_language_hints(language)
    recipe_lang_deps = recipe.recipe_dependencies.get(language, {})
    if recipe_lang_deps:
        pinned = dict(hints.get("pinned_dependencies") or {})
        pinned.update(recipe_lang_deps)
        hints = {**hints, "pinned_dependencies": pinned}

    project_name = manifest.answers.get("project_name") or manifest.recipe
    catalog = load_capabilities(deployments)
    resolved_stack = resolve_capabilities(recipe, catalog)
    try:
        assembled = assemble(
            recipe,
            language,
            framework,
            deployments,
            max_context_tokens=cfg.max_context_tokens,
            max_link_depth=cfg.max_link_depth,
            max_tokens_per_doc=cfg.max_tokens_per_doc,
            resolved_stack=resolved_stack if resolved_stack.capabilities else None,
        )
    except ContextBudgetError as exc:
        console.print(f"[red]Context assembly failed:[/] {exc}")
        return None
    req = GenerationRequest(
        project_name=project_name,
        target_language=language,
        framework=framework,
        assembled_context=assembled,
        language_hints=hints,
        extra_required=list(recipe.required_files),
    )
    update_cfg = cfg.model_copy(update={"model": manifest.model})
    try:
        raw = generate(req, update_cfg)
    except Exception as exc:  # noqa: BLE001 — surface any model/network failure
        console.print(f"[red]Generation failed:[/] {exc}")
        return None
    try:
        result = parse(raw)
    except ContractParseError as exc:
        console.print(f"[red]Contract parse failed:[/] {exc.reason}")
        return None
    return {f.path.replace("\\", "/"): f.content for f in result.files}


def _probe_services_for_plan(
    services: list[ExternalService],
    *,
    probe_services: bool,
    timeout: float = 5.0,
    max_workers: int = 4,
) -> list[CheckResult]:
    """Run service probes concurrently for the plan panel.

    Returns an empty list when there are no services to probe so the plan
    panel skips the section entirely. Probes run in a thread pool so total
    wall time is bounded by ~max(timeout) rather than sum(timeout).
    """
    if not services:
        return []
    from concurrent.futures import ThreadPoolExecutor

    from agent_scaffold.probes import run_probe

    if not probe_services:
        return [run_probe(svc, timeout=timeout, skip=True) for svc in services]
    with console.status(f"Probing {len(services)} service(s)..."):
        with ThreadPoolExecutor(max_workers=min(max_workers, len(services))) as pool:
            futures = [pool.submit(run_probe, svc, timeout=timeout, skip=False) for svc in services]
            return [f.result() for f in futures]


# ---------------------------------------------------------------------------
# Lifecycle verbs: deploy / down / status / logs
# ---------------------------------------------------------------------------


@app.command("deploy")
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


@app.command("down")
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

    # Stop the frontend dev server before tearing down compose so the user
    # doesn't see a "Backend gone" error in the browser tab they still have open.
    _stop_frontend(project_dir)

    compose_path = _find_docker_compose(project_dir)
    if compose_path is None:
        console.print(f"[red]Error:[/] no docker-compose.yml found under {project_dir}")
        raise typer.Exit(code=1)
    if shutil.which("docker") is None:
        console.print("[red]Error:[/] docker not on PATH — install Docker Desktop / Colima first")
        raise typer.Exit(code=1)

    if volumes and not yes:
        from agent_scaffold.deploy._common import confirm

        ok = confirm(
            "This will DELETE local data in named volumes "
            "(postgres, qdrant, redis, etc.). Type 'yes' to continue."
        )
        if not ok:
            console.print("[yellow]Aborted.[/]")
            raise typer.Exit(code=0)

    cmd = ["docker", "compose", "down"]
    if volumes:
        cmd.append("-v")
    console.print(f"[cyan]Running:[/] {' '.join(cmd)} (cwd: {compose_path.parent})")
    rc = subprocess.run(  # noqa: S603 — list-form, shell=False
        cmd, cwd=str(compose_path.parent), check=False
    ).returncode
    if rc != 0:
        console.print(f"[red]docker compose down exited {rc}[/]")
        raise typer.Exit(code=1)
    console.print("[green]Local stack stopped.[/]")

    # Reset docker_up step state so the next `agent-scaffold up` re-detects
    # fresh containers rather than skipping based on stale orchestrator state.
    _reset_step_state(project_dir, "docker_up")


@app.command("status")
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
    resolved_stack = _resolve_capability_stack_silently(recipe)
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


@app.command("logs")
def cmd_logs(
    service: str = typer.Argument(..., help="docker-compose service name."),
    follow: bool = typer.Option(True, "-f/--no-follow", help="Stream new log lines."),
    tail: int = typer.Option(100, "--tail", min=0, help="Number of past lines to show."),
    cwd: Path = typer.Option(Path("."), "--cwd", help="Project directory."),
) -> None:
    """Tail container logs. Thin wrapper around ``docker compose logs``.

    The reserved service name ``frontend`` tails ``launch_frontend``'s log
    file at ``.scaffold/frontend.log`` rather than going through docker.
    """
    project_dir = cwd.expanduser().resolve()

    if service == "frontend":
        _tail_frontend_log(project_dir, follow=follow, tail=tail)
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


def _tail_frontend_log(project_dir: Path, *, follow: bool, tail: int) -> None:
    """Tail the frontend dev server's log file. Friendly error if missing."""
    log_file = project_dir / SCAFFOLD_DIR / "frontend.log"
    if not log_file.is_file():
        console.print("[yellow]No frontend log — has `up` been run?[/]")
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
    stack = _resolve_capability_stack_silently(recipe)
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


def _stop_frontend(project_dir: Path) -> None:
    """Best-effort teardown of the dev server spawned by ``launch_frontend``.

    Reads ``<project>/.scaffold/frontend.pid``, kills the process group with
    SIGTERM, removes the PID file, and resets the step state so the next
    ``up`` re-launches. Missing/malformed PID files are silently OK.
    """
    import signal

    pid_file = project_dir / SCAFFOLD_DIR / "frontend.pid"
    if not pid_file.is_file():
        return
    try:
        data = json.loads(pid_file.read_text(encoding="utf-8"))
        pid = int(data["pid"])
    except (json.JSONDecodeError, KeyError, ValueError, OSError):
        pid_file.unlink(missing_ok=True)
        _reset_step_state(project_dir, "launch_frontend")
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
    _reset_step_state(project_dir, "launch_frontend")
    console.print("[green]Frontend dev server stopped.[/]")


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


# Re-export for ``python -m agent_scaffold``.
__all__ = ["app"]
