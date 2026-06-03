"""Tests for agent_scaffold.discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_scaffold.discovery import DiscoveryError, discover_recipes


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


def test_skips_no_h1(mock_deployments_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
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


def test_skips_doc_files_in_recipes_dir(mock_deployments_path: Path) -> None:
    """README.md / SCHEMA.md live next to recipes but are docs about the dir.

    They have valid H1s so the existing no-H1 filter doesn't catch them — the
    filename-based denylist must.
    """
    recipes = discover_recipes(mock_deployments_path)
    slugs = {r.slug for r in recipes}
    assert "README" not in slugs
    assert "readme" not in slugs
    assert "SCHEMA" not in slugs
    assert "schema" not in slugs


def test_required_files_parsed(mock_deployments_path: Path) -> None:
    recipes = discover_recipes(mock_deployments_path)
    by_slug = {r.slug: r for r in recipes}
    assert by_slug["with-required-files"].required_files == [
        "Dockerfile",
        "docker-compose.yml",
    ]
    # Recipes without the field default to empty.
    assert by_slug["customer-support-triage"].required_files == []


def test_required_files_unsafe_entries_dropped(
    mock_deployments_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    recipes = discover_recipes(mock_deployments_path)
    by_slug = {r.slug: r for r in recipes}
    assert by_slug["bad-required-files"].required_files == ["Dockerfile"]
    err = capsys.readouterr().err
    assert "/etc/passwd" in err
    assert "../escape.txt" in err


def test_recipe_dependencies_parsed(mock_deployments_path: Path) -> None:
    recipes = discover_recipes(mock_deployments_path)
    by_slug = {r.slug: r for r in recipes}
    assert by_slug["with-recipe-deps"].recipe_dependencies == {
        "python": {"redis": ">=5.0.0", "structlog": ">=24.1.0"},
        "typescript": {"ioredis": "^5.4.0"},
    }
    # Recipes without the field default to empty.
    assert by_slug["customer-support-triage"].recipe_dependencies == {}


def test_recipe_dependencies_malformed_warns(
    mock_deployments_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    recipes = discover_recipes(mock_deployments_path)
    by_slug = {r.slug: r for r in recipes}
    assert by_slug["bad-recipe-deps"].recipe_dependencies == {}
    err = capsys.readouterr().err
    assert "bad-recipe-deps.md" in err
    assert "recipe_dependencies" in err


def test_external_services_parsed(mock_deployments_path: Path) -> None:
    recipes = discover_recipes(mock_deployments_path)
    by_slug = {r.slug: r for r in recipes}
    svc = by_slug["with-external-services"]
    by_id = {s.id: s for s in svc.external_services}

    # All declared services that survive validation should be present.
    assert {"anthropic", "redis", "langfuse", "unknown-service", "no-probe-svc"} <= set(by_id)

    redis = by_id["redis"]
    assert redis.required is True
    assert redis.env_vars == ["REDIS_URL"]
    assert redis.default_local == "redis://localhost:6379"
    assert redis.docker_service == "redis"
    assert redis.probe == "redis_ping"
    assert redis.explain == "redis"

    langfuse = by_id["langfuse"]
    assert langfuse.required is False


def test_external_services_default_empty(mock_deployments_path: Path) -> None:
    recipes = discover_recipes(mock_deployments_path)
    by_slug = {r.slug: r for r in recipes}
    # A recipe without external_services field gets an empty list.
    assert by_slug["customer-support-triage"].external_services == []


def test_external_services_malformed_entries_warn(
    mock_deployments_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    discover_recipes(mock_deployments_path)
    err = capsys.readouterr().err
    # The fixture intentionally includes `not a mapping` and `{}` — both must warn.
    assert "external_services" in err
