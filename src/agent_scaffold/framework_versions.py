"""Loader for canonical framework version pins declared in agent-deployments.

`agent-deployments/docs/frameworks/<name>.md` files carry YAML frontmatter
that names the canonical PyPI / npm package + version pin for each
framework. This loader walks the resolved deployments tree and returns
the parsed specs so the scaffold's framework picker can enumerate
available frameworks without duplicating the pins in language YAML.

Before this module existed, the same data lived in two places:

- `agent-deployments/docs/frameworks/<name>.md` body prose
- `src/agent_scaffold/languages/{python,typescript}.yaml` under
  `framework_dependencies`

The duplication drifted. With this module + the SR1a frontmatter, the
deployments doc is the single source of truth; the language YAML
carries only language-level concerns (entry point, manifest, smoke check).

Resilience: docs that lack frontmatter are skipped with a warning rather
than raising — keeps offline / older-snapshot users functional.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

# Set by every doc Frontmatter we accept. Loosely typed because PyPI uses
# "name>=ver" or just "ver"; npm uses "^ver" or "~ver". The scaffold
# treats the string opaquely — it goes into the generated manifest as-is.
Language = Literal["python", "typescript"]


class ExtraPackage(BaseModel):
    """Companion dependency a framework requires beyond the language baseline."""

    model_config = {"frozen": True}

    name: str
    minimum: str


class FrameworkSpec(BaseModel):
    """One framework's pin spec, parsed from a deployments doc frontmatter."""

    model_config = {"frozen": True}

    id: str = Field(description="Snake-case slug used by the scaffold picker.")
    language: Language
    package: str = Field(description="PyPI or npm distribution name.")
    minimum: str = Field(
        description="Version constraint (with operator, e.g. '>=0.1.0' or '^4.0.0')."
    )
    last_known_good: str | None = Field(default=None)
    notes: str | None = Field(default=None)
    extra_packages: list[ExtraPackage] = Field(default_factory=list)


def load_framework_versions(deployments_root: Path) -> dict[str, FrameworkSpec]:
    """Walk ``deployments_root/docs/frameworks/*.md`` and return {id: FrameworkSpec}.

    Docs without frontmatter are skipped with a warning (the loader treats
    them as transitional / docs-only files like README.md). Docs whose
    frontmatter is present but malformed (missing required fields, bad
    types) raise a clear error — those are bugs to fix, not data drift.

    Returns an empty dict if the frameworks directory is absent entirely
    (i.e. the deployments tree predates the frameworks docs); the caller
    decides how to degrade.
    """
    frameworks_dir = deployments_root / "docs" / "frameworks"
    if not frameworks_dir.is_dir():
        return {}

    specs: dict[str, FrameworkSpec] = {}
    for md in sorted(frameworks_dir.glob("*.md")):
        if md.name.lower() in {"readme.md", "schema.md", "comparison.md"}:
            continue
        spec = _parse_one(md)
        if spec is None:
            continue
        if spec.id in specs:
            warnings.warn(
                f"duplicate framework id {spec.id!r} in {md} (already loaded); "
                "second occurrence ignored.",
                stacklevel=2,
            )
            continue
        specs[spec.id] = spec
    return specs


def available_frameworks_for_language(deployments_root: Path, language: str) -> list[str]:
    """Return sorted framework ids that target ``language``.

    Convenience used by the CLI / REPL framework pickers. The empty list
    is a valid return — the picker should fall back to "none" in that case.
    """
    return sorted(
        spec.id
        for spec in load_framework_versions(deployments_root).values()
        if spec.language == language
    )


def _parse_one(md: Path) -> FrameworkSpec | None:
    """Parse one framework doc's frontmatter into a FrameworkSpec.

    Returns ``None`` (with a warning) when the file has no frontmatter at
    all — those are transitional docs the loader skips. Raises ValueError
    when frontmatter exists but is malformed.
    """
    text = md.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        warnings.warn(
            f"framework doc {md.name} lacks YAML frontmatter; skipping. "
            "Add `id: …` etc. per docs/frameworks/README.md if this is intentional.",
            stacklevel=3,
        )
        return None

    parts = text.split("---", 2)
    if len(parts) < 3:
        warnings.warn(
            f"framework doc {md.name} starts with '---' but never closes the "
            "frontmatter block; skipping.",
            stacklevel=3,
        )
        return None

    try:
        raw = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"{md}: invalid YAML frontmatter — {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{md}: frontmatter must be a mapping, got {type(raw).__name__}.")

    # Normalize the nested ``versions`` block — early SR1a docs use
    # ``versions: {minimum, last_known_good?, notes?}``; flatten so
    # FrameworkSpec sees a flat shape.
    versions = raw.pop("versions", None)
    if isinstance(versions, dict):
        if "minimum" in versions:
            raw["minimum"] = versions["minimum"]
        if "last_known_good" in versions:
            raw["last_known_good"] = versions["last_known_good"]
        if "notes" in versions:
            raw["notes"] = versions["notes"]

    try:
        return FrameworkSpec(**raw)
    except Exception as exc:
        raise ValueError(f"{md}: invalid framework frontmatter — {exc}") from exc
