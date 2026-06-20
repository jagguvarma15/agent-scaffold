"""Language-target hints (YAML) loader — a leaf module both CLI and REPL share.

Each language target has a YAML hints file under
``agent_scaffold/languages/`` (e.g. ``python.yaml``, ``typescript.yaml``)
that describes the manifest filename, entry-point path, framework
dependencies, and any pinned package version hints. ``load_language_hints``
parses one file; ``available_languages`` enumerates the package.

Lives outside ``cli.py`` so the REPL can read it without importing the
Typer machinery — previously ``repl/shell.py`` shipped its own near-copy
to avoid that cycle.
"""

from __future__ import annotations

import importlib.resources as resources
from typing import Any

import yaml

LANGUAGES_PACKAGE = "agent_scaffold.languages"


class UnknownLanguageError(ValueError):
    """Raised when ``load_language_hints`` can't find a YAML for the language.

    Callers wrap it in their own surface error: ``typer.BadParameter`` from
    the CLI, ``CommandError`` from the REPL. Keeping it framework-neutral
    here lets the leaf module stay dependency-free.
    """


def load_language_hints(language: str) -> dict[str, Any]:
    """Read ``<language>.yaml`` from the languages package and return its dict.

    Raises :class:`UnknownLanguageError` if the file is missing or
    malformed (non-dict at the top level). Callers translate that into
    their preferred surface error.
    """
    filename = f"{language}.yaml"
    try:
        text = resources.files(LANGUAGES_PACKAGE).joinpath(filename).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise UnknownLanguageError(f"Unknown language: {language}") from exc
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise UnknownLanguageError(f"Malformed language hints in {filename}")
    return data


def _entry_module(entry_path: str) -> str:
    """Dotted import module for an entry path.

    ``app/main.py`` → ``app.main``; ``src/{project_name}/main.py`` →
    ``{project_name}.main`` (the ``src`` prefix is dropped, any ``{...}``
    placeholder is preserved so later substitution still works).
    """
    parts = [p for p in entry_path.replace("\\", "/").split("/") if p]
    if parts and parts[0] == "src":
        parts = parts[1:]
    if parts and parts[-1].endswith(".py"):
        parts[-1] = parts[-1][: -len(".py")]
    return ".".join(parts)


def reconcile_entry_point(hints: dict[str, Any], required_files: list[str]) -> dict[str, Any]:
    """Align ``entry_point`` / ``project_layout`` / ``smoke_check`` to the
    layout the recipe actually declares.

    The per-language hints carry a *default* source layout (Python's
    ``src/{project_name}/``). Recipes are authoritative about their own tree via
    ``required_files`` — most Python recipes ship a flat ``app/`` package, which
    contradicts that default. Feeding the model both layouts at once makes it emit
    one and fail the required-files contract on the other (the failure mode behind
    ``missing recipe-required file: app/...``).

    When a required file names the same entry-point basename as the language
    default (e.g. ``app/main.py`` vs the default ``src/{project_name}/main.py``),
    rewrite the hints to the recipe's path so the model sees a single coherent
    layout. No-op when the recipe declares no matching entry — the language
    default stands (so ``src/``-layout recipes are untouched).

    Returns a new dict; the input is never mutated.
    """
    entry_default = str(hints.get("entry_point", "")).replace("\\", "/")
    basename = entry_default.rsplit("/", 1)[-1] if entry_default else ""
    if not basename:
        return hints
    declared = next(
        (
            r.replace("\\", "/")
            for r in required_files
            if r.replace("\\", "/").rsplit("/", 1)[-1] == basename
        ),
        None,
    )
    if not declared or declared == entry_default:
        return hints
    new_hints = dict(hints)
    new_hints["entry_point"] = declared
    top = declared.split("/", 1)[0] if "/" in declared else ""
    if top and hints.get("project_layout"):
        new_hints["project_layout"] = top
    smoke = hints.get("smoke_check")
    if isinstance(smoke, str) and smoke:
        old_mod = _entry_module(entry_default)
        new_mod = _entry_module(declared)
        if old_mod and new_mod and old_mod in smoke:
            new_hints["smoke_check"] = smoke.replace(old_mod, new_mod)
    return new_hints


def available_languages() -> list[str]:
    """Return the sorted slugs of every ``<lang>.yaml`` in the languages package.

    Walking the package resources rather than hardcoding a list means
    adding a new language target — drop in ``rust.yaml`` — is automatically
    picked up by both ``/language`` validation and the wizard's choice list.
    """
    langs: list[str] = []
    for entry in resources.files(LANGUAGES_PACKAGE).iterdir():
        name = entry.name
        if name.endswith(".yaml"):
            langs.append(name[: -len(".yaml")])
    return sorted(langs)
