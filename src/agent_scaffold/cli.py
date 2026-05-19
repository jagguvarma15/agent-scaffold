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
import re
import time
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

from agent_scaffold import __version__
from agent_scaffold.cache import get_cached, save_cache
from agent_scaffold.config import Config, ConfigError, load_config
from agent_scaffold.context import assemble
from agent_scaffold.contract import (
    ContractParseError,
    GenerationResult,
    parse,
    validate_paths,
    validate_required_files,
)
from agent_scaffold.discovery import DiscoveryError, Recipe, discover_recipes
from agent_scaffold.generator import (
    GenerationRequest,
    generate,
    get_last_usage,
    prompts_signature,
    repair,
)
from agent_scaffold.validator import ValidationTier
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

# Each preset bundles model + max_tokens + thinking + strict prompt into one
# knob. Order of overrides applied in cmd_new: preset -> explicit flags -> env.
EFFORT_PRESETS: dict[str, dict[str, Any]] = {
    "low": {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 16000,
        "thinking": None,
        "strict": False,
    },
    "medium": {
        "model": "claude-sonnet-4-6",
        "max_tokens": 32000,
        "thinking": 8000,
        "strict": False,
    },
    "high": {
        "model": "claude-opus-4-7",
        "max_tokens": 64000,
        "thinking": 16000,
        "strict": True,
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
    raw: str, dest: Path, hints: dict[str, Any], project_name: str
) -> GenerationResult:
    result = parse(raw)
    validate_paths(result, dest)
    validate_required_files(result, hints)
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
) -> tuple[GenerationResult, str]:
    """Return ``(parsed_result, raw_response_text_that_succeeded)``."""
    raw = generate(req, config)
    try:
        return _attempt_parse(raw, dest, hints, project_name), raw
    except ContractParseError as exc:
        failure_path = _save_failure(raw, config.failures_dir)
        console.print(
            f"[yellow]Warning:[/] contract parse failed: {exc.reason}.\n"
            f"Raw response saved to: {failure_path}\n"
            "Attempting repair..."
        )
        repaired = repair(raw, exc.reason, config, strict=req.strict)
        try:
            return _attempt_parse(repaired, dest, hints, project_name), repaired
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
    chosen_framework = _select_framework(hints, framework, non_interactive)

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
        ctx = assemble(recipe, chosen_language, chosen_framework, deployments)
    console.print(
        f"[green]Context ready:[/] {len(ctx.referenced_paths)} reference(s), "
        f"~{ctx.token_estimate} tokens."
    )

    req = GenerationRequest(
        project_name=final_name,
        target_language=chosen_language,
        framework=chosen_framework,
        assembled_context=ctx,
        language_hints=hints,
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
        "strict": strict,
        "thinking_budget": cfg.thinking_budget,
    }
    cached_raw = None if no_cache else get_cached(cfg.cache_dir, cache_inputs)
    if cached_raw is not None:
        console.print("[dim]Using cached response.[/]")
        result = _attempt_parse(cached_raw, dest, hints, final_name)
    else:
        with console.status(f"Generating with {cfg.model}..."):
            result, raw_response = _generate_with_repair(req, cfg, dest, hints, final_name)
        save_cache(cfg.cache_dir, cache_inputs, raw_response)

    usage = get_last_usage()
    if usage.input_tokens > 0:
        console.print(
            f"[green]Generated[/] {len(result.files)} files. "
            f"Tokens: {usage.input_tokens} in / {usage.output_tokens} out"
        )
    else:
        console.print(f"[green]Generated[/] {len(result.files)} files.")

    try:
        report = write_project(result, dest, write_mode)
    except DestinationExistsError as exc:
        console.print(f"[red]Error:[/] {exc}")
        raise typer.Exit(code=1) from exc
    except ContractParseError as exc:
        console.print(f"[red]Path validation error:[/] {exc.reason}")
        raise typer.Exit(code=1) from exc

    console.print(
        f"[green]Wrote[/] {len(report.written)} new, "
        f"{len(report.overwritten)} overwritten, {len(report.skipped)} skipped."
    )

    if not skip_validation:
        with console.status("Running static validation..."):
            results = run_validate(dest, hints, result.smoke_check, [ValidationTier.static])
        for vr in results:
            mark = "[green][OK][/]" if vr.passed else "[red][FAIL][/]"
            console.print(f"{mark} {vr.tier.value}")
            if not vr.passed:
                console.print(vr.output)

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


# Re-export for ``python -m agent_scaffold``.
__all__ = ["app"]
