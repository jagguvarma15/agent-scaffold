"""Tests for ``agent_scaffold.framework_versions`` — the SR1b loader.

The loader reads YAML frontmatter from ``docs/frameworks/<name>.md`` in
the resolved deployments tree (SR1a established the frontmatter shape).
This module replaces what used to live in
``src/agent_scaffold/languages/{python,typescript}.yaml`` under
``framework_dependencies``.

These tests build a tiny fake deployments tree in ``tmp_path`` so they
don't depend on the bundled snapshot or the live agent-deployments repo.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_scaffold.framework_versions import (
    FrameworkSpec,
    available_frameworks_for_language,
    load_framework_versions,
)


def _write_framework_doc(
    root: Path, filename: str, frontmatter: str, body: str = "Body.\n"
) -> None:
    """Helper: write a framework markdown doc with frontmatter + body."""
    frameworks_dir = root / "docs" / "frameworks"
    frameworks_dir.mkdir(parents=True, exist_ok=True)
    (frameworks_dir / filename).write_text(f"---\n{frontmatter}---\n\n{body}", encoding="utf-8")


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_loader_parses_minimal_valid_frontmatter(tmp_path: Path) -> None:
    _write_framework_doc(
        tmp_path,
        "langgraph.md",
        'id: langgraph\nlanguage: python\npackage: langgraph\nversions:\n  minimum: "0.3.21"\n',
    )
    specs = load_framework_versions(tmp_path)
    assert set(specs.keys()) == {"langgraph"}
    spec = specs["langgraph"]
    assert isinstance(spec, FrameworkSpec)
    assert spec.language == "python"
    assert spec.package == "langgraph"
    assert spec.minimum == "0.3.21"
    assert spec.extra_packages == []


def test_loader_carries_extra_packages(tmp_path: Path) -> None:
    _write_framework_doc(
        tmp_path,
        "vercel-ai-sdk.md",
        (
            "id: vercel_ai_sdk\nlanguage: typescript\npackage: ai\n"
            'versions:\n  minimum: "^4.0.0"\n'
            'extra_packages:\n  - {name: "@ai-sdk/anthropic", minimum: "^1.0.0"}\n'
        ),
    )
    spec = load_framework_versions(tmp_path)["vercel_ai_sdk"]
    assert len(spec.extra_packages) == 1
    assert spec.extra_packages[0].name == "@ai-sdk/anthropic"
    assert spec.extra_packages[0].minimum == "^1.0.0"


def test_loader_carries_notes_and_last_known_good(tmp_path: Path) -> None:
    _write_framework_doc(
        tmp_path,
        "pydantic-ai.md",
        (
            "id: pydantic_ai\nlanguage: python\npackage: pydantic-ai\n"
            'versions:\n  minimum: ">=0.1.0"\n'
            '  last_known_good: "0.1.7"\n'
            '  notes: "@tool decorator signature stable since 0.1.0."\n'
        ),
    )
    spec = load_framework_versions(tmp_path)["pydantic_ai"]
    assert spec.last_known_good == "0.1.7"
    assert spec.notes is not None
    assert "decorator" in spec.notes


# ---------------------------------------------------------------------------
# Filter by language
# ---------------------------------------------------------------------------


def test_available_frameworks_filters_by_language(tmp_path: Path) -> None:
    _write_framework_doc(
        tmp_path,
        "langgraph.md",
        'id: langgraph\nlanguage: python\npackage: langgraph\nversions:\n  minimum: "0.3.21"\n',
    )
    _write_framework_doc(
        tmp_path,
        "vercel-ai-sdk.md",
        (
            "id: vercel_ai_sdk\nlanguage: typescript\npackage: ai\n"
            'versions:\n  minimum: "^4.0.0"\n'
        ),
    )
    assert available_frameworks_for_language(tmp_path, "python") == ["langgraph"]
    assert available_frameworks_for_language(tmp_path, "typescript") == ["vercel_ai_sdk"]
    assert available_frameworks_for_language(tmp_path, "rust") == []


# ---------------------------------------------------------------------------
# Resilience
# ---------------------------------------------------------------------------


def test_loader_returns_empty_when_frameworks_dir_missing(tmp_path: Path) -> None:
    # tmp_path has no docs/frameworks/ — the loader should return {} rather
    # than raise, so offline / pre-SR1a users don't see crashes.
    assert load_framework_versions(tmp_path) == {}


def test_loader_skips_docs_without_frontmatter(tmp_path: Path) -> None:
    (tmp_path / "docs" / "frameworks").mkdir(parents=True)
    # A README-style doc with no frontmatter — must be skipped with a warning,
    # not crash the loader.
    (tmp_path / "docs" / "frameworks" / "transitional.md").write_text(
        "# Framework: Transitional\n\nNo frontmatter yet.\n", encoding="utf-8"
    )
    # Also add a valid doc so we know the loader continues past the skip.
    _write_framework_doc(
        tmp_path,
        "langgraph.md",
        'id: langgraph\nlanguage: python\npackage: langgraph\nversions:\n  minimum: "0.3.21"\n',
    )
    with pytest.warns(UserWarning, match="lacks YAML frontmatter"):
        specs = load_framework_versions(tmp_path)
    assert set(specs.keys()) == {"langgraph"}


def test_loader_skips_readme_and_schema_files(tmp_path: Path) -> None:
    frameworks_dir = tmp_path / "docs" / "frameworks"
    frameworks_dir.mkdir(parents=True)
    # README.md and SCHEMA.md should be ignored without warnings.
    (frameworks_dir / "README.md").write_text("# index\n", encoding="utf-8")
    (frameworks_dir / "comparison.md").write_text("# comparison matrix\n", encoding="utf-8")
    _write_framework_doc(
        tmp_path,
        "langgraph.md",
        'id: langgraph\nlanguage: python\npackage: langgraph\nversions:\n  minimum: "0.3.21"\n',
    )
    specs = load_framework_versions(tmp_path)
    assert set(specs.keys()) == {"langgraph"}


def test_loader_raises_on_malformed_frontmatter(tmp_path: Path) -> None:
    # Missing the required ``package`` field → FrameworkSpec validation fails.
    _write_framework_doc(
        tmp_path,
        "broken.md",
        'id: broken\nlanguage: python\nversions:\n  minimum: "1.0"\n',
    )
    with pytest.raises(ValueError, match="invalid framework frontmatter"):
        load_framework_versions(tmp_path)


def test_loader_raises_on_bad_language_value(tmp_path: Path) -> None:
    # ``language`` is a Literal — values outside python|typescript reject.
    _write_framework_doc(
        tmp_path,
        "exotic.md",
        'id: exotic\nlanguage: ruby\npackage: exotic\nversions:\n  minimum: "1.0"\n',
    )
    with pytest.raises(ValueError):
        load_framework_versions(tmp_path)


def test_loader_warns_and_skips_on_duplicate_id(tmp_path: Path) -> None:
    # Filenames chosen so sort order is deterministic: "01-…" loads before "02-…".
    _write_framework_doc(
        tmp_path,
        "01-langgraph.md",
        'id: langgraph\nlanguage: python\npackage: langgraph\nversions:\n  minimum: "0.3.21"\n',
    )
    _write_framework_doc(
        tmp_path,
        "02-langgraph-alt.md",
        'id: langgraph\nlanguage: python\npackage: langgraph\nversions:\n  minimum: "0.4.0"\n',
    )
    with pytest.warns(UserWarning, match="duplicate framework id"):
        specs = load_framework_versions(tmp_path)
    # First filename in sort order wins; the second is dropped with a warning.
    assert specs["langgraph"].minimum == "0.3.21"
