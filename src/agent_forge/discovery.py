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


def _warn(msg: str) -> None:
    print(f"agent-forge: warning: {msg}", file=sys.stderr)


def discover_recipes(deployments_path: Path) -> list[Recipe]:
    """Scan ``{deployments_path}/docs/recipes/*.md`` and return all valid recipes."""
    recipes_dir = deployments_path / "docs" / "recipes"
    if not recipes_dir.is_dir():
        raise DiscoveryError(
            f"No recipes found at {deployments_path}/docs/recipes"
        )

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

        recipes.append(
            Recipe(
                slug=slug,
                title=title,
                status=status,
                path=entry.resolve(),
                languages=languages,
            )
        )

    recipes.sort(key=lambda r: r.slug)
    return recipes
