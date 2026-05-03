"""Tests for agent_forge.discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_forge.discovery import DiscoveryError, discover_recipes


def test_recipes_alphabetical(mock_deployments_path: Path) -> None:
    recipes = discover_recipes(mock_deployments_path)
    slugs = [r.slug for r in recipes]
    assert slugs == sorted(slugs)
    assert "customer-support-triage" in slugs
    assert "docs-rag-qa" in slugs


def test_frontmatter_parsed(mock_deployments_path: Path) -> None:
    recipes = discover_recipes(mock_deployments_path)
    triage = next(r for r in recipes if r.slug == "customer-support-triage")
    assert triage.title == "Customer Support Triage"
    assert triage.status == "validated"
    assert triage.languages == ["python", "typescript"]


def test_no_frontmatter_defaults(mock_deployments_path: Path) -> None:
    recipes = discover_recipes(mock_deployments_path)
    rag = next(r for r in recipes if r.slug == "docs-rag-qa")
    assert rag.title == "Docs RAG QA"
    assert rag.status == "unknown"
    assert rag.languages == ["python", "typescript"]


def test_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(DiscoveryError, match="No recipes found"):
        discover_recipes(tmp_path)


def test_skips_no_h1(
    mock_deployments_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    recipes = discover_recipes(mock_deployments_path)
    slugs = {r.slug for r in recipes}
    assert "no-h1-recipe" not in slugs
    err = capsys.readouterr().err
    assert "no-h1-recipe.md" in err
    assert "no H1" in err


def test_ignores_hidden_files(mock_deployments_path: Path) -> None:
    recipes = discover_recipes(mock_deployments_path)
    slugs = {r.slug for r in recipes}
    assert ".DS_Store" not in slugs
