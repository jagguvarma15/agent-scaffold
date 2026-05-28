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
