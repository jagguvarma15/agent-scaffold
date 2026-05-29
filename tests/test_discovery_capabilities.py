"""Tests for the Phase 1b ``Recipe.capabilities`` frontmatter parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_scaffold.discovery import discover_recipes


def test_recipe_capabilities_parsed_and_validated(
    mock_deployments_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    recipes = {r.slug: r for r in discover_recipes(mock_deployments_path)}
    recipe = recipes["with-capabilities"]
    # BAD_FORMAT dropped; duplicate cache.redis deduped; vector_db.nonexistent
    # passes shape validation (it's the resolver's job to flag unknown ids).
    assert recipe.capabilities == [
        "cache.redis",
        "vector_db.qdrant",
        "host.vercel",
        "vector_db.nonexistent",
    ]
    err = capsys.readouterr().err
    assert "BAD_FORMAT" in err
    assert "must match" in err
    assert "declared twice" in err


def test_recipe_without_capabilities_field(mock_deployments_path: Path) -> None:
    recipes = {r.slug: r for r in discover_recipes(mock_deployments_path)}
    # Pre-existing fixture recipe has no `capabilities:` field — must default empty.
    assert recipes["customer-support-triage"].capabilities == []


def test_recipe_capabilities_round_trip_in_model() -> None:
    # Direct model construction must accept an empty list (pydantic default).
    from agent_scaffold.discovery import Recipe

    r = Recipe(slug="x", title="X", path=Path("/tmp/x.md"))
    assert r.capabilities == []
