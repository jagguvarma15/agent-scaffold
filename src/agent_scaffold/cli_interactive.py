"""Interactive prompts + name validators for the CLI.

Extracted from ``cli.py`` so the ``cmd_new`` flow's questionary helpers
have their own module. These are lazy on ``questionary`` import so a
non-interactive command path (``doctor``, ``auth``, ``config``, …)
doesn't pay the prompt-toolkit import cost.

Every helper raises ``typer.Abort`` on user cancellation (Ctrl-C /
Ctrl-D) rather than returning ``None`` — the call sites all want to bail
to the shell on cancel, not branch on it.

This module imports the shared ``console`` from ``cli_shared`` so test
fixtures that monkeypatch ``cli_shared.console`` see the same instance
the live code uses.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from agent_scaffold.cli_shared import console
from agent_scaffold.discovery import Recipe
from agent_scaffold.language_hints import available_languages
from agent_scaffold.writer import WriteMode

if TYPE_CHECKING:
    from agent_scaffold.config import Config


PROJECT_NAME_RE = re.compile(r"^[a-z0-9_-]+$")

KNOWN_MODELS: list[tuple[str, str]] = [
    ("claude-opus-4-7", "Opus 4.7 — highest quality (slowest, most expensive)"),
    ("claude-sonnet-4-6", "Sonnet 4.6 — balanced (recommended for most runs)"),
    ("claude-haiku-4-5-20251001", "Haiku 4.5 — fast iteration (lowest quality)"),
]


# ── name validation ──────────────────────────────────────────────────────────


def _validate_project_name(name: str) -> str:
    if not PROJECT_NAME_RE.match(name):
        raise typer.BadParameter(
            "Project name must contain only lowercase letters, digits, hyphens, and underscores."
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


# ── low-level questionary wrappers ───────────────────────────────────────────


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


# ── per-field selectors ──────────────────────────────────────────────────────


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
        list(KNOWN_MODELS),
        default=default,
    )


def _select_framework(
    deployments_root: Path | None,
    language: str,
    framework: str | None,
    non_interactive: bool,
) -> str:
    """Pick a framework from the deployments-doc frontmatter (SR1b).

    Reads ``docs/frameworks/<name>.md`` YAML frontmatter from the resolved
    deployments tree and surfaces the ids whose ``language`` matches.
    Falls back to ``["none"]`` only when the deployments tree predates
    SR1a frontmatter (offline / stale snapshot); the caller still gets a
    working picker.
    """
    from agent_scaffold.framework_versions import available_frameworks_for_language

    available: list[str] = []
    if deployments_root is not None:
        available = available_frameworks_for_language(deployments_root, language)
    available.append("none")
    if framework is not None:
        if framework not in available:
            raise typer.BadParameter(
                f"Framework {framework} not available for {language}. "
                f"Allowed: {', '.join(available)}"
            )
        return framework
    if non_interactive:
        return "none"
    return _interactive_select("Pick a framework:", [(f, f.replace("_", " ")) for f in available])


def _select_write_mode() -> WriteMode:
    chosen = _interactive_select(
        "Destination is not empty. What should I do?",
        [
            (WriteMode.overwrite.value, "overwrite — replace with freshly generated files"),
            (WriteMode.skip.value, "skip — only add files that don't exist yet"),
            (WriteMode.abort.value, "abort"),
        ],
        default=WriteMode.skip.value,
    )
    return WriteMode(chosen)
