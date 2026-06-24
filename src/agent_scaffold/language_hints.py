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
from dataclasses import dataclass
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


@dataclass(frozen=True)
class EntryPointSpec:
    """The single source of truth for a project's entry point + smoke contract.

    ``entry_point`` is the project-relative source file (e.g. ``app/main.py`` or
    ``src/{project_name}/main.py``); ``module`` is its importable dotted form;
    ``smoke_check`` is the recipe/language smoke command. All three may still
    carry a ``{project_name}`` placeholder — callers persisting them to a
    per-project manifest substitute the real name.
    """

    entry_point: str
    module: str
    project_layout: str | None
    smoke_check: str


def resolve_entry_point(hints: dict[str, Any], required_files: list[str]) -> EntryPointSpec:
    """Resolve the canonical entry point + smoke contract — the SoT both
    generation (:func:`reconcile_entry_point`) and run (manifest →
    ``launch_backend``) agree on.

    The per-language hints carry a *default* source layout (Python's
    ``src/{project_name}/``). Recipes are authoritative about their own tree via
    ``required_files`` — most Python recipes ship a flat ``app/`` package. When a
    required file names the same entry-point basename as the language default
    (e.g. ``app/main.py`` vs ``src/{project_name}/main.py``), the recipe's path
    wins; otherwise the language default stands.

    Unlike the old reconcile, this is **never a silent no-op** — it always
    returns a concrete spec (the language default when the recipe declares no
    matching entry), so every project gets a recorded, runnable entry point.
    """
    entry_default = str(hints.get("entry_point", "")).replace("\\", "/")
    basename = entry_default.rsplit("/", 1)[-1] if entry_default else ""
    declared: str | None = None
    if basename:
        declared = next(
            (
                r.replace("\\", "/")
                for r in required_files
                if r.replace("\\", "/").rsplit("/", 1)[-1] == basename
            ),
            None,
        )
    resolved = declared if (declared and declared != entry_default) else entry_default

    layout = hints.get("project_layout")
    project_layout = str(layout) if isinstance(layout, str) and layout else None
    top = resolved.split("/", 1)[0] if "/" in resolved else ""
    if top and project_layout:
        project_layout = top

    smoke_raw = hints.get("smoke_check")
    smoke_check = smoke_raw if isinstance(smoke_raw, str) else ""
    if smoke_check and resolved != entry_default:
        old_mod = _entry_module(entry_default)
        new_mod = _entry_module(resolved)
        if old_mod and new_mod and old_mod in smoke_check:
            smoke_check = smoke_check.replace(old_mod, new_mod)

    return EntryPointSpec(
        entry_point=resolved,
        module=_entry_module(resolved),
        project_layout=project_layout,
        smoke_check=smoke_check,
    )


def reconcile_entry_point(hints: dict[str, Any], required_files: list[str]) -> dict[str, Any]:
    """Align ``entry_point`` / ``project_layout`` / ``smoke_check`` in ``hints`` to
    the layout the recipe declares — a thin wrapper over :func:`resolve_entry_point`.

    Feeding the model both the language default and the recipe's layout at once
    makes it emit one and fail the required-files contract on the other (the
    ``missing recipe-required file: app/...`` failure mode). Rewriting the hints
    to the resolved path gives it a single coherent layout.

    No-op (returns the input object unchanged) when the resolved entry equals the
    language default — ``src/``-layout recipes are untouched. Otherwise returns a
    new dict; the input is never mutated.
    """
    entry_default = str(hints.get("entry_point", "")).replace("\\", "/")
    spec = resolve_entry_point(hints, required_files)
    if spec.entry_point == entry_default:
        return hints
    new_hints = dict(hints)
    new_hints["entry_point"] = spec.entry_point
    if hints.get("project_layout") and spec.project_layout:
        new_hints["project_layout"] = spec.project_layout
    if isinstance(hints.get("smoke_check"), str) and hints["smoke_check"]:
        new_hints["smoke_check"] = spec.smoke_check
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
