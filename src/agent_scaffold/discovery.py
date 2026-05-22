"""Discover agent recipes inside an agent-deployments repo.

A recipe is any markdown file under ``{deployments_path}/docs/recipes/`` with
an H1 title. Optional YAML frontmatter at the top may provide ``status`` and
``languages``; otherwise sane defaults are used.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

DEFAULT_LANGUAGES = ("python", "typescript")
DEFAULT_STATUS = "unknown"

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


class DiscoveryError(Exception):
    """Raised when recipes cannot be discovered."""


class Recipe(BaseModel):
    slug: str
    title: str
    status: str = DEFAULT_STATUS
    path: Path
    languages: list[str] = Field(default_factory=lambda: list(DEFAULT_LANGUAGES))
    required_files: list[str] = Field(default_factory=list)
    recipe_dependencies: dict[str, dict[str, str]] = Field(default_factory=dict)


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    raw = match.group(1)
    try:
        loaded = yaml.safe_load(raw) or {}
    except yaml.YAMLError:
        return {}, text[match.end() :]
    if not isinstance(loaded, dict):
        return {}, text[match.end() :]
    return loaded, text[match.end() :]


def _first_h1(text: str) -> str | None:
    match = _H1_RE.search(text)
    if not match:
        return None
    return match.group(1).strip()


def _coerce_languages(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v).lower() for v in value]
    if isinstance(value, str):
        return [value.lower()]
    return list(DEFAULT_LANGUAGES)


def _coerce_str_list(value: Any, *, context: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value]
    _warn(f"{context}: expected list of strings, got {type(value).__name__}; ignoring")
    return []


def _sanitize_required_paths(entries: list[str], *, recipe_name: str) -> list[str]:
    """Apply the same path-safety rules used by validate_paths."""
    cleaned: list[str] = []
    for raw in entries:
        if not raw or raw != raw.strip():
            _warn(f"{recipe_name}: empty/whitespace required_files entry {raw!r}; dropping")
            continue
        if raw.startswith(("/", "\\")):
            _warn(f"{recipe_name}: absolute required_files path {raw!r}; dropping")
            continue
        normalized = raw.replace("\\", "/")
        if any(part == ".." for part in normalized.split("/")):
            _warn(f"{recipe_name}: required_files path contains '..': {raw!r}; dropping")
            continue
        cleaned.append(raw)
    return cleaned


def _coerce_recipe_dependencies(
    value: Any, recipe_name: str
) -> dict[str, dict[str, str]]:
    """Coerce frontmatter ``recipe_dependencies`` into ``{lang: {pkg: version}}``.

    Per-language entries must be ``dict[str, str]`` mappings; anything else is
    dropped with a warning. Package version values are coerced via ``str``.
    Language keys are normalized to lowercase.
    """
    if not isinstance(value, dict):
        _warn(
            f"{recipe_name}: recipe_dependencies must be a mapping of language "
            f"to {{package: version}}; got {type(value).__name__}; ignoring"
        )
        return {}
    result: dict[str, dict[str, str]] = {}
    for lang_key, deps in value.items():
        if not isinstance(lang_key, str):
            _warn(
                f"{recipe_name}: skipping malformed recipe_dependencies entry: "
                f"language key {lang_key!r} is not a string"
            )
            continue
        lang = lang_key.lower()
        if not isinstance(deps, dict):
            _warn(
                f"{recipe_name}: skipping malformed recipe_dependencies for "
                f"language {lang!r}: expected mapping of package to version, "
                f"got {type(deps).__name__}"
            )
            continue
        lang_entries: dict[str, str] = {}
        for pkg, version in deps.items():
            if not isinstance(pkg, str):
                _warn(
                    f"{recipe_name}: skipping malformed recipe_dependencies "
                    f"package name {pkg!r} for language {lang!r}; not a string"
                )
                continue
            lang_entries[pkg] = str(version)
        if lang_entries:
            result[lang] = lang_entries
    return result


def _warn(msg: str) -> None:
    print(f"agent-scaffold: warning: {msg}", file=sys.stderr)


def discover_recipes(deployments_path: Path) -> list[Recipe]:
    """Scan ``{deployments_path}/docs/recipes/*.md`` and return all valid recipes."""
    recipes_dir = deployments_path / "docs" / "recipes"
    if not recipes_dir.is_dir():
        raise DiscoveryError(f"No recipes found at {deployments_path}/docs/recipes")

    recipes: list[Recipe] = []
    for entry in sorted(recipes_dir.iterdir()):
        if entry.name.startswith("."):
            continue
        if not entry.is_file() or entry.suffix.lower() != ".md":
            continue

        try:
            text = entry.read_text(encoding="utf-8")
        except OSError as exc:
            _warn(f"could not read {entry}: {exc}")
            continue

        frontmatter, body = _parse_frontmatter(text)
        title = _first_h1(body) or _first_h1(text)
        if title is None:
            _warn(f"skipping {entry.name}: no H1 title")
            continue

        status = str(frontmatter.get("status", DEFAULT_STATUS))
        languages = _coerce_languages(frontmatter.get("languages", DEFAULT_LANGUAGES))
        slug = entry.stem
        required_files = _sanitize_required_paths(
            _coerce_str_list(
                frontmatter.get("required_files"),
                context=f"{entry.name}: required_files",
            ),
            recipe_name=entry.name,
        )
        recipe_dependencies = _coerce_recipe_dependencies(
            frontmatter.get("recipe_dependencies") or {},
            entry.name,
        )

        recipes.append(
            Recipe(
                slug=slug,
                title=title,
                status=status,
                path=entry.resolve(),
                languages=languages,
                required_files=required_files,
                recipe_dependencies=recipe_dependencies,
            )
        )

    recipes.sort(key=lambda r: r.slug)
    return recipes
