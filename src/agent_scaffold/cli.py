"""Typer CLI entry point for agent-scaffold.

Commands:
- ``agent-scaffold new``      : interactive (or ``--non-interactive``) project generator.
- ``agent-scaffold config``   : print resolved configuration.
- ``agent-scaffold validate`` : re-run validation tiers on an existing generated project.
- ``agent-scaffold --version``
"""

from __future__ import annotations

import importlib.resources as resources
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

from agent_scaffold import __version__
from agent_scaffold.auth import (
    DEFAULT_KEY_NAME,
    AuthError,
    BackendKind,
    StoredCredential,
    delete_key,
    describe_backend,
    detect_backend,
    list_credentials,
    resolve_active,
    store_key,
    validate_anthropic_key,
)
from agent_scaffold.cache import get_cached, save_cache
from agent_scaffold.config import Config, ConfigError, load_config
from agent_scaffold.context import ContextBudgetError, assemble
from agent_scaffold.contract import (
    ContractParseError,
    GenerationResult,
    parse,
    validate_paths,
    validate_required_files,
)
from agent_scaffold.costs import estimate as estimate_cost
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
from agent_scaffold.generator import (
    GenerationRequest,
    extract_fenced_content,
    generate,
    generate_single_file,
    get_last_usage,
    prompts_signature,
    repair,
)
from agent_scaffold.imports import discover_neighbours
from agent_scaffold.manifest import (
    Manifest,
    ManifestNotFoundError,
    build_file_entries,
    read_manifest,
    update_file_entry,
    write_manifest,
)
from agent_scaffold.plan import GenerationPlan
from agent_scaffold.plan import confirm as confirm_plan
from agent_scaffold.progress import (
    NullProgressDisplay,
    ProgressEvent,
    RichProgressDisplay,
)
from agent_scaffold.topology import Topology, coerce_roles, coerce_topology, infer_topology
from agent_scaffold.validator import ValidationTier, verify_required_files_on_disk
from agent_scaffold.validator import validate as run_validate
from agent_scaffold.writer import (
    DestinationExistsError,
    WriteMode,
    write_project,
)

app = typer.Typer(
    name="agent-scaffold",
    help="Generate runnable AI agent projects from markdown specs.",
    add_completion=False,
    invoke_without_command=True,
)

console = Console()

LANGUAGES_PACKAGE = "agent_scaffold.languages"
PROJECT_NAME_RE = re.compile(r"^[a-z0-9_-]+$")

KNOWN_MODELS: list[tuple[str, str]] = [
    ("claude-opus-4-7", "Opus 4.7 — highest quality (slowest, most expensive)"),
    ("claude-sonnet-4-6", "Sonnet 4.6 — balanced (recommended for most runs)"),
    ("claude-haiku-4-5-20251001", "Haiku 4.5 — fast iteration (lowest quality)"),
]

# Each preset bundles model + max_tokens + thinking + strict prompt into one
# knob. Order of overrides applied in cmd_new: preset -> explicit flags -> env.
EFFORT_PRESETS: dict[str, dict[str, Any]] = {
    "low": {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 16000,
        "thinking": None,
        "strict": False,
        "max_context_tokens": 30_000,
        "max_link_depth": 1,
        "max_tokens_per_doc": 4_000,
    },
    "medium": {
        "model": "claude-sonnet-4-6",
        "max_tokens": 32000,
        "thinking": 8000,
        "strict": False,
        "max_context_tokens": 60_000,
        "max_link_depth": 2,
        "max_tokens_per_doc": 8_000,
    },
    "high": {
        "model": "claude-opus-4-7",
        "max_tokens": 64000,
        "thinking": 16000,
        "strict": True,
        "max_context_tokens": 100_000,
        "max_link_depth": 3,
        "max_tokens_per_doc": 12_000,
    },
}


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"agent-scaffold {__version__}")
        raise typer.Exit()


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
        typer.echo(ctx.get_help())
        raise typer.Exit()


def _load_language_hints(language: str) -> dict[str, Any]:
    filename = f"{language}.yaml"
    try:
        text = resources.files(LANGUAGES_PACKAGE).joinpath(filename).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise typer.BadParameter(f"Unknown language: {language}") from exc
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise typer.BadParameter(f"Malformed language hints in {filename}")
    return data


def _available_languages() -> list[str]:
    langs: list[str] = []
    for entry in resources.files(LANGUAGES_PACKAGE).iterdir():
        name = entry.name
        if name.endswith(".yaml"):
            langs.append(name[: -len(".yaml")])
    return sorted(langs)


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


def _save_failure(raw: str, failures_dir: Path) -> Path:
    failures_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    path = failures_dir / f"{ts}.json"
    path.write_text(raw, encoding="utf-8")
    return path


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


def _format_hint(language: str) -> str:
    if language == "python":
        return "ruff check --fix + ruff format"
    if language == "typescript":
        if shutil.which("prettier"):
            return "prettier --write"
        if shutil.which("biome"):
            return "biome format --write"
    return f"no formatter for {language}"


def _run_subprocess_with_events(
    cmd: list[str],
    on_event: Callable[[ProgressEvent], None] | None,
) -> int:
    if on_event is not None:
        on_event(ProgressEvent(kind="bash_started", payload={"cmd": cmd}))
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if on_event is not None:
        on_event(
            ProgressEvent(
                kind="bash_done",
                payload={
                    "cmd": cmd,
                    "exit_code": proc.returncode,
                    "stdout_tail": (proc.stdout or "")[-200:],
                    "stderr_tail": (proc.stderr or "")[-200:],
                },
            )
        )
    return proc.returncode


def _run_post_gen_formatter(
    dest: Path,
    language: str,
    on_event: Callable[[ProgressEvent], None] | None = None,
) -> None:
    """Auto-fix trivial lint + reformat freshly-written files.

    Idempotent and best-effort: a missing formatter or non-zero exit must not
    fail the run, since the static-validation tier will surface anything that
    still matters. Runs ``ruff check --fix --unsafe-fixes`` followed by
    ``ruff format`` for Python; ``prettier`` (or ``biome``) for TypeScript.

    When ``on_event`` is supplied, each subprocess invocation surfaces as a
    ``bash_started`` / ``bash_done`` pair so the progress display can log it.
    """
    if language == "python":
        if shutil.which("ruff") is None:
            return
        _run_subprocess_with_events(
            ["ruff", "check", "--fix", "--unsafe-fixes", "--quiet", str(dest)], on_event
        )
        _run_subprocess_with_events(["ruff", "format", "--quiet", str(dest)], on_event)
    elif language == "typescript":
        if shutil.which("prettier"):
            _run_subprocess_with_events(
                ["prettier", "--write", "--log-level", "silent", str(dest)], on_event
            )
        elif shutil.which("biome"):
            _run_subprocess_with_events(["biome", "format", "--write", str(dest)], on_event)
    # Other languages: no formatter wired up — silently no-op.


def _print_usage_summary(model: str, wall_seconds: float, *, cached: bool) -> None:
    """Print a token + cost + wall-time summary. Always called, even on failure."""
    usage = get_last_usage()
    if usage.input_tokens == 0 and usage.output_tokens == 0:
        return
    mins, secs = divmod(int(wall_seconds), 60)
    wall_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
    cache_total = usage.cache_read_input_tokens + usage.cache_creation_input_tokens
    cache_ratio = ""
    if cache_total:
        denom = max(1, usage.input_tokens + cache_total)
        pct = int(100 * usage.cache_read_input_tokens / denom)
        cache_ratio = f" (cache hit {pct}%)"
    suffix = " [cached]" if cached else ""
    lines = [
        f"Tokens: {usage.input_tokens:,} in{cache_ratio} / {usage.output_tokens:,} out",
        f"Wall time: {wall_str}{suffix}",
    ]
    cost = estimate_cost(
        model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_input_tokens,
        cache_write_tokens=usage.cache_creation_input_tokens,
    )
    if cost is not None:
        lines.insert(
            1,
            f"Estimated cost: ${cost.total:.2f} "
            f"(in ${cost.input_uncached:.2f}, out ${cost.output:.2f}, "
            f"cache r ${cost.cache_read:.2f} / w ${cost.cache_write:.2f})",
        )
    console.print(Panel("\n".join(lines), title="Run summary", expand=False))


def _print_phase_summary(
    phase_durations: dict[str, float], warnings: list[str], errors: list[str]
) -> None:
    """Render per-phase wall times plus any warnings/errors collected during the run."""
    if not phase_durations and not warnings and not errors:
        return
    lines: list[str] = []
    if phase_durations:
        lines.append("[bold]Phase timings:[/]")
        for name, secs in phase_durations.items():
            mins, s = divmod(int(secs), 60)
            label = f"{mins}m {s:02d}s" if mins else f"{secs:.1f}s"
            lines.append(f"  {name}: {label}")
    if warnings:
        if lines:
            lines.append("")
        lines.append("[bold yellow]Warnings:[/]")
        for w in warnings:
            lines.append(f"  ⚠ {w}")
    if errors:
        if lines:
            lines.append("")
        lines.append("[bold red]Errors:[/]")
        for e in errors:
            lines.append(f"  ✗ {e}")
    console.print(Panel("\n".join(lines), title="Phase summary", expand=False))


def _print_next_steps(dest: Path, language: str, smoke_check: str, post_install: list[str]) -> None:
    lines = [f"Project written to: [bold]{dest}[/]\n"]
    lines.append("Next steps:")
    lines.append(f"  cd {dest}")
    if post_install:
        for cmd in post_install:
            lines.append(f"  {cmd}")
    elif language == "python":
        lines.append("  uv sync")
    elif language == "typescript":
        lines.append("  pnpm install")
    lines.append(f"  {smoke_check}")
    console.print(Panel("\n".join(lines), title="Next steps", expand=False))


def _attempt_parse(
    raw: str,
    dest: Path,
    hints: dict[str, Any],
    project_name: str,
    extra_required: list[str],
) -> GenerationResult:
    result = parse(raw)
    validate_paths(result, dest)
    validate_required_files(result, hints, extra_required)
    if result.project_name != project_name:
        # Allow the LLM to canonicalize hyphens -> underscores for python.
        result = result.model_copy(update={"project_name": project_name})
    return result


def _generate_with_repair(
    req: GenerationRequest,
    config: Config,
    dest: Path,
    hints: dict[str, Any],
    project_name: str,
    extra_required: list[str],
    progress: Callable[[ProgressEvent], None] | None = None,
) -> tuple[GenerationResult, str]:
    """Return ``(parsed_result, raw_response_text_that_succeeded)``."""
    raw = generate(req, config, progress=progress)
    try:
        return _attempt_parse(raw, dest, hints, project_name, extra_required), raw
    except ContractParseError as exc:
        failure_path = _save_failure(raw, config.failures_dir)
        console.print(
            f"[yellow]Warning:[/] contract parse failed: {exc.reason}.\n"
            f"Raw response saved to: {failure_path}\n"
            "Attempting repair..."
        )
        repaired = repair(raw, exc.reason, config, strict=req.strict, progress=progress)
        try:
            return (
                _attempt_parse(repaired, dest, hints, project_name, extra_required),
                repaired,
            )
        except ContractParseError as exc2:
            second_failure = _save_failure(repaired, config.failures_dir)
            console.print(
                f"[red]Error:[/] repair also failed: {exc2.reason}\n"
                f"Original raw response: {failure_path}\n"
                f"Repaired raw response: {second_failure}"
            )
            raise typer.Exit(code=1) from exc2


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
                "model": preset["model"],
                "max_tokens": preset["max_tokens"],
                "thinking_budget": preset["thinking"],
                "max_context_tokens": preset["max_context_tokens"],
                "max_link_depth": preset["max_link_depth"],
                "max_tokens_per_doc": preset["max_tokens_per_doc"],
            }
        )
        if preset["strict"]:
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

    deployments = (deployments_path or cfg.deployments_path).expanduser()
    if not non_interactive and deployments_path is None:
        chosen = _interactive_path(
            "Path to agent-deployments repo:",
            default=str(deployments),
        )
        deployments = Path(chosen).expanduser()

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

    with console.status("Assembling context..."):
        try:
            ctx = assemble(
                recipe,
                chosen_language,
                chosen_framework,
                deployments,
                max_context_tokens=cfg.max_context_tokens,
                max_link_depth=cfg.max_link_depth,
                max_tokens_per_doc=cfg.max_tokens_per_doc,
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

    topology = (
        coerce_topology(recipe.topology) if recipe.topology else infer_topology(recipe, ctx.body)
    )
    if topology is None:
        topology = Topology.SINGLE
    roles = coerce_roles(recipe.roles)

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
        )
        if not confirm_plan(gen_plan, console):
            console.print("[yellow]Aborted before LLM call.[/]")
            raise typer.Exit(code=0)

    req = GenerationRequest(
        project_name=final_name,
        target_language=chosen_language,
        framework=chosen_framework,
        assembled_context=ctx,
        language_hints=hints,
        extra_required=recipe.required_files,
        strict=strict,
    )

    cache_inputs = {
        "project_name": final_name,
        "language": chosen_language,
        "framework": chosen_framework,
        "context": ctx.body,
        "model": cfg.model,
        "hints": hints,
        "prompts": prompts_signature(),
        "required_files": recipe.required_files,
        "strict": strict,
        "thinking_budget": cfg.thinking_budget,
    }
    cached_raw = None if no_cache else get_cached(cfg.cache_dir, cache_inputs)
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

    wall_start = time.time()
    result: GenerationResult | None = None
    report: Any = None
    validation_results: list[Any] = []
    try:
        with display as progress:
            # --- Generate (or load from cache) ---------------------------------
            if cached_raw is not None:
                progress.on_event(
                    ProgressEvent(
                        kind="operation_started",
                        payload={"name": "cached lookup", "hint": "skipping LLM call"},
                    )
                )
                result = _attempt_parse(cached_raw, dest, hints, final_name, recipe.required_files)
                progress.on_event(
                    ProgressEvent(
                        kind="operation_done",
                        payload={
                            "name": "cached lookup",
                            "status": "ok",
                            "summary": f"{len(result.files)} files",
                        },
                    )
                )
            else:
                progress.on_event(
                    ProgressEvent(
                        kind="operation_started",
                        payload={"name": "generate", "hint": f"model={cfg.model}"},
                    )
                )
                result, raw_response = _generate_with_repair(
                    req,
                    cfg,
                    dest,
                    hints,
                    final_name,
                    recipe.required_files,
                    progress=progress.on_event,
                )
                progress.on_event(
                    ProgressEvent(
                        kind="operation_done",
                        payload={
                            "name": "generate",
                            "status": "ok",
                            "summary": f"{len(result.files)} files",
                        },
                    )
                )
                save_cache(cfg.cache_dir, cache_inputs, raw_response)

            # --- Write to disk -------------------------------------------------
            progress.on_event(
                ProgressEvent(
                    kind="operation_started",
                    payload={"name": "write", "hint": f"{len(result.files)} files"},
                )
            )
            try:
                report = write_project(result, dest, write_mode, on_event=progress.on_event)
            except DestinationExistsError as exc:
                progress.on_event(
                    ProgressEvent(
                        kind="operation_done",
                        payload={"name": "write", "status": "fail", "summary": str(exc)},
                    )
                )
                console.print(f"[red]Error:[/] {exc}")
                raise typer.Exit(code=1) from exc
            except ContractParseError as exc:
                progress.on_event(
                    ProgressEvent(
                        kind="operation_done",
                        payload={"name": "write", "status": "fail", "summary": exc.reason},
                    )
                )
                console.print(f"[red]Path validation error:[/] {exc.reason}")
                raise typer.Exit(code=1) from exc
            progress.on_event(
                ProgressEvent(
                    kind="operation_done",
                    payload={
                        "name": "write",
                        "status": "ok",
                        "summary": (
                            f"{len(report.written)} new, "
                            f"{len(report.overwritten)} overwritten, "
                            f"{len(report.skipped)} skipped"
                        ),
                    },
                )
            )

            # --- Verify required files actually landed on disk -----------------
            if recipe.required_files:
                progress.on_event(
                    ProgressEvent(
                        kind="operation_started",
                        payload={
                            "name": "verify",
                            "hint": f"{len(recipe.required_files)} required files",
                        },
                    )
                )
                on_disk_missing = verify_required_files_on_disk(dest, recipe.required_files)
                if on_disk_missing:
                    summary = f"missing: {', '.join(on_disk_missing)}"
                    progress.on_event(
                        ProgressEvent(
                            kind="operation_done",
                            payload={"name": "verify", "status": "fail", "summary": summary},
                        )
                    )
                    console.print(
                        "[red]Required files missing after write:[/]\n  "
                        + "\n  ".join(on_disk_missing)
                        + "\n\nLikely causes:\n"
                        + "  - --write-mode skip with a non-empty destination "
                        + "containing colliding paths\n"
                        + "  - write permissions / disk full / path-traversal sanitisation\n\n"
                        + "Try: --write-mode overwrite (BE CAREFUL — irreversible)"
                    )
                    raise typer.Exit(code=1)
                progress.on_event(
                    ProgressEvent(
                        kind="operation_done",
                        payload={
                            "name": "verify",
                            "status": "ok",
                            "summary": f"{len(recipe.required_files)} present",
                        },
                    )
                )

            # --- Format --------------------------------------------------------
            if format_output:
                progress.on_event(
                    ProgressEvent(
                        kind="operation_started",
                        payload={"name": "format", "hint": _format_hint(chosen_language)},
                    )
                )
                _run_post_gen_formatter(dest, chosen_language, on_event=progress.on_event)
                progress.on_event(
                    ProgressEvent(
                        kind="operation_done",
                        payload={"name": "format", "status": "ok"},
                    )
                )

            # --- Static validation ---------------------------------------------
            if not skip_validation:
                progress.on_event(
                    ProgressEvent(
                        kind="operation_started",
                        payload={"name": "validate", "hint": "static tier"},
                    )
                )
                validation_results = run_validate(
                    dest,
                    hints,
                    result.smoke_check,
                    [ValidationTier.static],
                    on_event=progress.on_event,
                )
                status = "ok" if all(r.passed for r in validation_results) else "fail"
                summary = "; ".join(
                    f"{r.tier.value}={'ok' if r.passed else 'fail'}" for r in validation_results
                )
                progress.on_event(
                    ProgressEvent(
                        kind="operation_done",
                        payload={"name": "validate", "status": status, "summary": summary},
                    )
                )

            # --- Write .scaffold/manifest.json so `regenerate` knows the
            # recipe/language/framework/model without re-prompting the user.
            if result is not None and report is not None:
                progress.on_event(
                    ProgressEvent(
                        kind="operation_started",
                        payload={"name": "manifest", "hint": ".scaffold/manifest.json"},
                    )
                )
                try:
                    manifest = Manifest(
                        recipe=recipe.slug,
                        language=chosen_language,
                        framework=chosen_framework,
                        topology=topology.value if topology else None,
                        roles=[
                            {
                                "name": r.name,
                                "description": r.description,
                                "model_hint": r.model_hint,
                                "tools": list(r.tools),
                            }
                            for r in roles
                        ],
                        model=cfg.model,
                        generated_at=datetime.now(UTC).isoformat(),
                        files=build_file_entries(dest, [f.path for f in result.files]),
                    )
                    write_manifest(dest, manifest)
                    progress.on_event(
                        ProgressEvent(
                            kind="operation_done",
                            payload={
                                "name": "manifest",
                                "status": "ok",
                                "summary": f"{len(manifest.files)} files indexed",
                            },
                        )
                    )
                except OSError as exc:
                    progress.on_event(
                        ProgressEvent(
                            kind="operation_done",
                            payload={
                                "name": "manifest",
                                "status": "warn",
                                "summary": f"could not write manifest: {exc}",
                            },
                        )
                    )
    finally:
        _print_usage_summary(cfg.model, time.time() - wall_start, cached=cached_raw is not None)

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

    _print_phase_summary(
        getattr(display, "phase_durations", {}),
        getattr(display, "warnings", []),
        getattr(display, "errors", []),
    )

    if result is not None:
        _print_next_steps(dest, chosen_language, result.smoke_check, result.post_install)


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
    candidates = [lang for lang in recipe.languages if lang in _available_languages()]
    if not candidates:
        candidates = _available_languages()
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

    deployments = (deployments_path or cfg.deployments_path).expanduser()
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
            _run_post_gen_formatter(project_dir, manifest.language)

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


doctor_app = typer.Typer(
    name="doctor",
    help="Read-only environment + recipe audit. Never mutates.",
    invoke_without_command=True,
)
app.add_typer(doctor_app, name="doctor")


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


def _doctor_render(console: Console, report: DoctorReport) -> None:
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

    @property
    def id(self) -> str:
        return f"service.{self.service.id}"

    @property
    def category(self) -> str:
        return "Recipe services"

    def run(self) -> CheckResult:
        from agent_scaffold.probes import run_probe

        return run_probe(self.service, timeout=self.timeout, skip=self.skip)


def _service_checks(
    services: list[ExternalService], *, timeout: float, skip: bool
) -> list[Check]:
    return [_ServiceCheck(svc, timeout=timeout, skip=skip) for svc in services]


def _resolve_recipe_for_doctor(slug: str) -> Recipe:
    """Find ``slug`` among configured deployments. Raises ``typer.Exit`` on miss."""
    try:
        cfg = load_config()
    except ConfigError as exc:
        console.print(f"[red]Configuration error:[/] {exc}")
        raise typer.Exit(code=1) from exc
    try:
        recipes = discover_recipes(cfg.deployments_path.expanduser())
    except DiscoveryError as exc:
        console.print(f"[red]Error:[/] {exc}")
        raise typer.Exit(code=1) from exc
    match = next((r for r in recipes if r.slug == slug), None)
    if match is None:
        available = ", ".join(r.slug for r in recipes) or "(none)"
        console.print(f"[red]Unknown recipe:[/] {slug}. Available: {available}")
        raise typer.Exit(code=1)
    return match


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
        checks.extend(
            _service_checks(chosen.external_services, timeout=timeout, skip=no_probes)
        )
    report = run_checks(checks)

    if json_output:
        typer.echo(_doctor_json(report))
    else:
        _doctor_render(console, report)

    raise typer.Exit(code=report.exit_code)


auth_app = typer.Typer(
    name="auth",
    help="Manage Anthropic credentials (keyring-first; mode-0600 file fallback).",
)
app.add_typer(auth_app, name="auth")


def _select_backend(use_keyring: bool, use_file: bool, use_env: bool) -> BackendKind:
    chosen = [
        name
        for name, flag in (
            ("keyring", use_keyring),
            ("file", use_file),
            ("env", use_env),
        )
        if flag
    ]
    if len(chosen) > 1:
        raise typer.BadParameter("--use-keyring / --use-file / --use-env are mutually exclusive.")
    if chosen:
        return chosen[0]  # type: ignore[return-value]
    try:
        return detect_backend()
    except AuthError:
        # No native keyring available — degrade to the file backend rather
        # than failing the user mid-flow. This is the explicit v2 fallback.
        console.print(
            "[yellow]Warning:[/] no native keyring backend detected; "
            "falling back to mode-0600 credentials file."
        )
        return "file"


def _prompt_paste(prompt: str = "Paste your Anthropic key (input hidden):") -> str:
    import getpass

    try:
        return getpass.getpass(prompt).strip()
    except (EOFError, KeyboardInterrupt) as exc:
        raise typer.Abort() from exc


@auth_app.command("login")
def auth_login(
    name: str = typer.Option(
        DEFAULT_KEY_NAME, "--name", "-n", help="Credential name (for multi-key setups)."
    ),
    use_keyring: bool = typer.Option(False, "--use-keyring", help="Force keyring backend."),
    use_file: bool = typer.Option(False, "--use-file", help="Force mode-0600 file backend."),
    use_env: bool = typer.Option(
        False, "--use-env", help="Don't store; just print the export line."
    ),
    no_browser: bool = typer.Option(
        False, "--no-browser", help="Skip the browser flow; prompt for paste instead."
    ),
    no_validate: bool = typer.Option(
        False, "--no-validate", help="Skip the models.list() probe before storing."
    ),
) -> None:
    """Capture an Anthropic key (browser or paste), validate, store."""
    from pydantic import SecretStr

    backend = _select_backend(use_keyring, use_file, use_env)

    key_text: str | None = None
    if not no_browser:
        from agent_scaffold.auth_browser import browser_paste_flow

        console.print("Opening your browser to paste your Anthropic key...")
        key_text = browser_paste_flow()
        if not key_text:
            console.print("[yellow]No key captured from browser flow.[/] Falling back to paste.")
    if not key_text:
        key_text = _prompt_paste()
    if not key_text:
        console.print("[red]No key supplied.[/]")
        raise typer.Exit(code=1)

    secret = SecretStr(key_text)
    if not no_validate:
        ok, msg = validate_anthropic_key(secret)
        if not ok:
            console.print(f"[red]Validation failed:[/] {msg}")
            raise typer.Exit(code=1)
        console.print(f"[green]Key {msg}.[/]")

    try:
        stored = store_key(name, secret, backend=backend)
    except AuthError as exc:
        console.print(f"[red]Store failed:[/] {exc}")
        raise typer.Exit(code=1) from exc

    if stored.backend == "env":
        console.print(
            f"[bold]Add to your shell:[/]  export ANTHROPIC_API_KEY='{secret.get_secret_value()}'"
        )
    else:
        console.print(
            f"[green]Stored[/] '{stored.name}' in {stored.backend} " f"({stored.masked_value})."
        )


@auth_app.command("status")
def auth_status(
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Show backend health, stored credentials, and the active resolution."""
    backend_label: str
    backend_ok: bool
    try:
        detect_backend()
        backend_label = describe_backend()
        backend_ok = True
    except AuthError as exc:
        backend_label = f"{describe_backend()} (refused: {exc})"
        backend_ok = False

    creds = list_credentials()
    active = resolve_active()

    if json_output:
        payload = {
            "schema_version": 1,
            "backend": backend_label,
            "backend_ok": backend_ok,
            "credentials": [
                {
                    "name": c.name,
                    "backend": c.backend,
                    "masked": c.masked_value,
                    "created": c.created,
                }
                for c in creds
            ],
            "active": (
                {"name": DEFAULT_KEY_NAME, "backend": active[1]} if active is not None else None
            ),
            "resolution_order": ["env (ANTHROPIC_API_KEY)", "keyring", "file"],
        }
        typer.echo(json.dumps(payload, indent=2))
        return

    health = "[green]good[/]" if backend_ok else "[red]refused[/]"
    console.print(f"[bold]Backend:[/] {backend_label} {health}")
    if creds:
        console.print("[bold]Stored credentials:[/]")
        for c in creds:
            created = f"   created {c.created}" if c.created else ""
            console.print(f"  {c.name:<14}  {c.masked_value:<18}  ({c.backend}){created}")
    else:
        console.print("[dim]No stored credentials.[/]")
    console.print("[bold]Resolution order:[/] ANTHROPIC_API_KEY (env) > keyring > file")
    if active is not None:
        _, src = active
        console.print(f"[bold]Currently resolved:[/] name={DEFAULT_KEY_NAME} from {src}")
    else:
        console.print("[yellow]No key resolved.[/] Run `agent-scaffold auth login`.")


@auth_app.command("logout")
def auth_logout(
    name: str = typer.Option(DEFAULT_KEY_NAME, "--name", "-n", help="Credential name to remove."),
    all_: bool = typer.Option(
        False, "--all", help="Remove every stored credential, not just --name."
    ),
) -> None:
    """Remove a stored credential from every backend it lives in."""
    if all_:
        creds: list[StoredCredential] = list_credentials()
        names = {c.name for c in creds}
        removed_any = False
        for n in names:
            if delete_key(n):
                removed_any = True
                console.print(f"[green]Removed[/] {n}")
        if not removed_any:
            console.print("[dim]No credentials to remove.[/]")
        return
    if delete_key(name):
        console.print(f"[green]Removed[/] {name}")
    else:
        console.print(f"[yellow]No credential named[/] {name}")
        raise typer.Exit(code=1)


@auth_app.command("setup-token")
def auth_setup_token(
    name: str = typer.Argument(..., help="Token name (e.g. ci-prod)."),
    from_stdin: bool = typer.Option(False, "--stdin", help="Read token from stdin (for CI)."),
) -> None:
    """Store a long-lived token in the mode-0600 file backend (for CI)."""
    from pydantic import SecretStr

    if from_stdin:
        text = sys.stdin.read().strip()
    else:
        text = _prompt_paste("Paste the token:")
    if not text:
        console.print("[red]No token supplied.[/]")
        raise typer.Exit(code=1)
    stored = store_key(name, SecretStr(text), backend="file")
    console.print(f"[green]Stored[/] '{stored.name}' in {stored.backend} ({stored.masked_value}).")


# Re-export for ``python -m agent_scaffold``.
__all__ = ["app"]
