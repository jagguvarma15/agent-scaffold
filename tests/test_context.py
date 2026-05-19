"""Tests for agent_scaffold.context."""

from __future__ import annotations

from pathlib import Path

from agent_scaffold.context import assemble
from agent_scaffold.discovery import discover_recipes


def _recipe(deployments: Path, slug: str):  # type: ignore[no-untyped-def]
    recipes = discover_recipes(deployments)
    return next(r for r in recipes if r.slug == slug)


def test_assemble_no_references(mock_deployments_path: Path) -> None:
    recipe = _recipe(mock_deployments_path, "lonely-recipe")
    out = assemble(
        recipe, language="python", framework="none", deployments_path=mock_deployments_path
    )
    recipe_text = recipe.path.read_text(encoding="utf-8").rstrip()
    assert recipe_text in out.body
    assert out.referenced_paths == []
    assert "<!-- ===== referenced:" not in out.body


def test_assemble_relative_links(mock_deployments_path: Path) -> None:
    recipe = _recipe(mock_deployments_path, "customer-support-triage")
    out = assemble(
        recipe, language="python", framework="langgraph", deployments_path=mock_deployments_path
    )
    rel_paths = [p.name for p in out.referenced_paths]
    assert "react.md" in rel_paths
    assert "langgraph.md" in rel_paths
    assert "qdrant.md" in rel_paths
    # Section markers are present.
    assert "<!-- ===== referenced: patterns/react.md ===== -->" in out.body


def test_assemble_alias_resolution(mock_deployments_path: Path) -> None:
    recipe = _recipe(mock_deployments_path, "docs-rag-qa")
    out = assemble(
        recipe, language="python", framework="pydantic_ai", deployments_path=mock_deployments_path
    )
    rel_paths = {p.name for p in out.referenced_paths}
    # "pattern: RAG" alias maps to patterns/rag.md.
    assert "rag.md" in rel_paths
    # "Qdrant" alias maps to stack/qdrant.md.
    assert "qdrant.md" in rel_paths
    # "Pydantic AI" alias for python; "Vercel AI SDK" should NOT be included for python.
    assert "pydantic-ai.md" in rel_paths
    assert "vercel-ai-sdk.md" not in rel_paths


def test_assemble_filters_wrong_language_framework(mock_deployments_path: Path) -> None:
    recipe = _recipe(mock_deployments_path, "docs-rag-qa")
    out = assemble(
        recipe,
        language="typescript",
        framework="vercel_ai_sdk",
        deployments_path=mock_deployments_path,
    )
    rel_paths = {p.name for p in out.referenced_paths}
    # For typescript: vercel-ai-sdk.md included, langgraph/pydantic-ai dropped.
    assert "vercel-ai-sdk.md" in rel_paths
    assert "langgraph.md" not in rel_paths
    assert "pydantic-ai.md" not in rel_paths


def test_assemble_cross_cutting(mock_deployments_path: Path) -> None:
    recipe = _recipe(mock_deployments_path, "customer-support-triage")
    out = assemble(
        recipe, language="python", framework="langgraph", deployments_path=mock_deployments_path
    )
    rel_paths = {p.name for p in out.referenced_paths}
    assert "auth.md" in rel_paths
    assert "logging.md" in rel_paths


def test_assemble_skips_missing_reference(mock_deployments_path: Path, capsys) -> None:
    recipe = _recipe(mock_deployments_path, "missing-ref-recipe")
    out = assemble(
        recipe, language="python", framework="none", deployments_path=mock_deployments_path
    )
    err = capsys.readouterr().err
    assert "does-not-exist.md" in err
    # Body should still include the recipe content.
    assert "Missing Ref Recipe" in out.body


def test_assemble_handles_circular_references(mock_deployments_path: Path) -> None:
    recipe = _recipe(mock_deployments_path, "cycle-recipe")
    out = assemble(
        recipe, language="python", framework="none", deployments_path=mock_deployments_path
    )
    # No file should appear twice.
    paths = [p.resolve() for p in out.referenced_paths]
    assert len(paths) == len(set(paths))
    # Both loop docs should appear exactly once.
    names = [p.name for p in paths]
    assert names.count("loop-a.md") == 1
    assert names.count("loop-b.md") == 1


def test_token_estimate_monotonic(mock_deployments_path: Path) -> None:
    short = _recipe(mock_deployments_path, "lonely-recipe")
    long = _recipe(mock_deployments_path, "customer-support-triage")
    short_ctx = assemble(
        short, language="python", framework="none", deployments_path=mock_deployments_path
    )
    long_ctx = assemble(
        long, language="python", framework="langgraph", deployments_path=mock_deployments_path
    )
    assert long_ctx.token_estimate > short_ctx.token_estimate
    # Estimate should grow with body length.
    assert long_ctx.token_estimate >= len(long_ctx.body) // 4 - 1
