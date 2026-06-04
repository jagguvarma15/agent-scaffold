"""Tests for agent_scaffold.context."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent_scaffold.context import (
    ALIAS_TABLE,
    CROSS_CUTTING,
    ContextBudgetError,
    _alias_matches,
    _docs_root,
    _truncate,
    assemble,
    evaluate_load_list_predicate,
)
from agent_scaffold.discovery import Recipe, discover_recipes


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
    assert "vector-qdrant.md" in rel_paths
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
    # "Qdrant" alias maps to stack/vector-qdrant.md.
    assert "vector-qdrant.md" in rel_paths
    # "Pydantic AI" alias for python; "Vercel AI SDK" should NOT be included for python.
    assert "pydantic-ai.md" in rel_paths
    assert "vercel-ai-sdk.md" not in rel_paths


def test_alias_matches_event_driven_variants() -> None:
    # Hyphen, lowercase.
    assert "event-driven" in _alias_matches("the event-driven approach")
    # Space, lowercase.
    assert "event driven" in _alias_matches("we use event driven semantics")
    # Mixed case (matcher is case-insensitive).
    assert "event-driven" in _alias_matches("This is an Event-Driven pattern")
    assert "event driven" in _alias_matches("a fully Event Driven design")
    # Bare "events" must not trigger either variant.
    hits = _alias_matches("the system emits many events on each request")
    assert "event-driven" not in hits
    assert "event driven" not in hits


def test_assemble_resolves_event_driven_alias_from_prose(mock_deployments_path: Path) -> None:
    recipe = _recipe(mock_deployments_path, "event-driven-recipe")
    out = assemble(
        recipe, language="python", framework="none", deployments_path=mock_deployments_path
    )
    rel_paths = {p.name for p in out.referenced_paths}
    assert "event-driven.md" in rel_paths


# ---------------------------------------------------------------------------
# D6-follow: structured load_list integration
# ---------------------------------------------------------------------------


def test_evaluate_load_list_predicate_empty_is_true() -> None:
    assert evaluate_load_list_predicate(
        None, language="python", framework="none", capabilities=[], topology=None
    )
    assert evaluate_load_list_predicate(
        "  ", language="python", framework="none", capabilities=[], topology=None
    )


def test_evaluate_load_list_predicate_scalar_equality() -> None:
    pred = "language == 'python'"
    assert evaluate_load_list_predicate(
        pred, language="python", framework="none", capabilities=[], topology=None
    )
    assert not evaluate_load_list_predicate(
        pred, language="typescript", framework="none", capabilities=[], topology=None
    )
    # framework + topology supported too
    assert evaluate_load_list_predicate(
        "framework == 'langgraph'",
        language="python",
        framework="langgraph",
        capabilities=[],
        topology=None,
    )
    assert evaluate_load_list_predicate(
        "topology == 'multi-agent-flat'",
        language="python",
        framework="none",
        capabilities=[],
        topology="multi-agent-flat",
    )


def test_evaluate_load_list_predicate_capabilities_contains() -> None:
    pred = "capabilities contains 'obs.langfuse'"
    assert evaluate_load_list_predicate(
        pred,
        language="python",
        framework="none",
        capabilities=["obs.langfuse", "cache.redis"],
        topology=None,
    )
    assert not evaluate_load_list_predicate(
        pred,
        language="python",
        framework="none",
        capabilities=["cache.redis"],
        topology=None,
    )


def test_evaluate_load_list_predicate_unknown_syntax_warns_and_returns_true(capsys) -> None:
    # Fail-open: a malformed predicate must never silently drop a required doc.
    result = evaluate_load_list_predicate(
        "language is awesome",
        language="python",
        framework="none",
        capabilities=[],
        topology=None,
    )
    assert result is True
    err = capsys.readouterr().err
    assert "unknown load_list predicate" in err


def test_assemble_load_list_required_loads_regardless_of_prose(
    mock_deployments_path: Path,
) -> None:
    """A recipe whose body mentions nothing must still load every required
    load_list entry whose `when` passes."""
    recipe = _recipe(mock_deployments_path, "with-load-list")
    out = assemble(
        recipe,
        language="python",
        framework="pydantic_ai",
        deployments_path=mock_deployments_path,
    )
    rel_paths = {p.name for p in out.referenced_paths}
    # Required, no when -> loaded.
    assert "react.md" in rel_paths
    # Required + when matches Python -> loaded.
    assert "pydantic-ai.md" in rel_paths
    # Required + when matches TypeScript -> NOT loaded.
    assert "vercel-ai-sdk.md" not in rel_paths


def test_assemble_load_list_optional_capability_predicate(
    mock_deployments_path: Path,
) -> None:
    """`required: false` entries still load when the predicate passes —
    they're just demoted to a lower tier so they drop first on budget pressure."""
    recipe = _recipe(mock_deployments_path, "with-load-list")
    # The fixture declares `capabilities: [obs.langfuse]` in frontmatter, so
    # the `capabilities contains 'obs.langfuse'` predicate must pass.
    out = assemble(
        recipe,
        language="python",
        framework="pydantic_ai",
        deployments_path=mock_deployments_path,
    )
    rel_paths = {p.name for p in out.referenced_paths}
    # Optional + no when -> always loads.
    assert "logging-structured.md" in rel_paths
    # Optional + when matches a declared capability -> loads.
    assert "observability.md" in rel_paths
    # Optional + when fails (multi-tenancy capability not declared) -> dropped.
    assert "multi-tenancy.md" not in rel_paths


def test_assemble_framework_filter_drops_other_framework_alias(
    mock_deployments_path: Path,
) -> None:
    """SR2: when a framework is selected, alias mentions of OTHER frameworks
    in the SAME language are dropped — they used to leak in via the alias tier."""
    recipe = _recipe(mock_deployments_path, "framework-mixed-aliases")
    out = assemble(
        recipe,
        language="python",
        framework="langgraph",
        deployments_path=mock_deployments_path,
    )
    rel_paths = {p.name for p in out.referenced_paths}
    # Selected framework loads via alias.
    assert "langgraph.md" in rel_paths
    # Other Python framework (mentioned in prose but not selected) is filtered.
    assert "pydantic-ai.md" not in rel_paths


def test_assemble_framework_none_loads_every_aliased_framework(
    mock_deployments_path: Path,
) -> None:
    """SR2: framework="none" disables the framework filter — every Python
    framework alias still resolves (existing behavior preserved)."""
    recipe = _recipe(mock_deployments_path, "framework-mixed-aliases")
    out = assemble(
        recipe,
        language="python",
        framework="none",
        deployments_path=mock_deployments_path,
    )
    rel_paths = {p.name for p in out.referenced_paths}
    assert "langgraph.md" in rel_paths
    assert "pydantic-ai.md" in rel_paths


def test_assemble_explicit_composes_overrides_framework_filter(
    mock_deployments_path: Path,
) -> None:
    """SR2: when a recipe explicitly composes a framework doc, the framework
    filter must NOT drop it — recipe-author intent wins over the picker."""
    recipe = _recipe(mock_deployments_path, "framework-composes-override")
    out = assemble(
        recipe,
        language="python",
        framework="langgraph",
        deployments_path=mock_deployments_path,
    )
    rel_paths = {p.name for p in out.referenced_paths}
    # Selected framework — naturally loaded.
    assert "langgraph.md" in rel_paths
    # Other framework — explicitly composed, must still load.
    assert "pydantic-ai.md" in rel_paths


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
    assert "authorization-rbac.md" in rel_paths
    assert "logging-structured.md" in rel_paths


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


_DEPLOYMENTS_ENV = "AGENT_SCAFFOLD_DEPLOYMENTS_PATH"


@pytest.mark.integration
def test_alias_table_targets_exist_in_deployments() -> None:
    """Every ALIAS_TABLE and CROSS_CUTTING entry must point at an existing file.

    Run with AGENT_SCAFFOLD_DEPLOYMENTS_PATH pointed at a real agent-deployments
    checkout. Skipped when unset so unit-test runs don't depend on a sibling repo.
    """
    raw = os.environ.get(_DEPLOYMENTS_ENV)
    if not raw:
        pytest.skip(f"{_DEPLOYMENTS_ENV} not set; skipping alias-target integration check")
    deployments = Path(raw).expanduser()
    if not deployments.is_dir():
        pytest.skip(f"{_DEPLOYMENTS_ENV}={deployments} is not a directory")
    docs = _docs_root(deployments)

    missing: list[tuple[str, str]] = []
    for alias, rel_path in {**ALIAS_TABLE, **CROSS_CUTTING}.items():
        target = docs / rel_path
        if not target.is_file():
            missing.append((alias, rel_path))

    assert not missing, "Alias table points at files that don't exist:\n" + "\n".join(
        f"  {alias!r} -> {path}" for alias, path in missing
    )


def _budget_fixture(tmp_path: Path, *, doc_size_chars: int = 4_000, n_extra: int = 0) -> Path:
    """Build a synthetic deployments tree with one recipe linking many big docs."""
    docs = tmp_path / "docs"
    (docs / "recipes").mkdir(parents=True)
    (docs / "patterns").mkdir()
    body = "x " * (doc_size_chars // 2)
    composes_links = [
        "[A](../patterns/big-a.md)",
        "[B](../patterns/big-b.md)",
    ]
    extra_links = [f"[E{i}](../patterns/extra-{i}.md)" for i in range(n_extra)]
    recipe_text = (
        "---\nstatus: blueprint\nlanguages: [python]\n---\n\n# Big Recipe\n\n"
        "## Composes\n\n" + "\n\n".join(composes_links) + "\n\n"
        "## Extras\n\n" + "\n\n".join(extra_links) + "\n"
    )
    (docs / "recipes" / "big.md").write_text(recipe_text, encoding="utf-8")
    (docs / "patterns" / "big-a.md").write_text(f"# A\n\n{body}", encoding="utf-8")
    (docs / "patterns" / "big-b.md").write_text(f"# B\n\n{body}", encoding="utf-8")
    for i in range(n_extra):
        (docs / "patterns" / f"extra-{i}.md").write_text(f"# Extra {i}\n\n{body}", encoding="utf-8")
    return tmp_path


def _load_recipe(tmp_path: Path) -> Recipe:
    return next(r for r in discover_recipes(tmp_path) if r.slug == "big")


def test_truncate_appends_marker_when_over_cap() -> None:
    text = "a" * 10_000  # ~2500 tokens
    truncated, was = _truncate(text, max_tokens=500)
    assert was is True
    assert "[truncated for context budget]" in truncated
    assert len(truncated) <= 500 * 4 + 1


def test_truncate_passes_through_when_under_cap() -> None:
    text = "small"
    out, was = _truncate(text, max_tokens=500)
    assert out == text
    assert was is False


def test_assemble_drops_lowest_tier_first_when_over_budget(tmp_path: Path) -> None:
    deployments = _budget_fixture(tmp_path, doc_size_chars=4_000, n_extra=5)
    recipe = _load_recipe(deployments)
    # Cap is tight enough to keep the recipe + the 2 Composes but drop the extras.
    out = assemble(
        recipe,
        language="python",
        framework="none",
        deployments_path=deployments,
        max_context_tokens=3_500,
        max_link_depth=0,
        max_tokens_per_doc=2_000,
    )
    assert out.summary is not None
    rel_kept = [p.relative_to(deployments / "docs").as_posix() for p in out.referenced_paths]
    assert "patterns/big-a.md" in rel_kept
    assert "patterns/big-b.md" in rel_kept
    # Lower-priority extras get dropped.
    assert any("extra-" in p for p in out.summary.dropped)


def test_assemble_truncates_oversized_doc(tmp_path: Path) -> None:
    deployments = _budget_fixture(tmp_path, doc_size_chars=20_000, n_extra=0)
    recipe = _load_recipe(deployments)
    out = assemble(
        recipe,
        language="python",
        framework="none",
        deployments_path=deployments,
        max_context_tokens=50_000,
        max_link_depth=0,
        max_tokens_per_doc=500,
    )
    assert out.summary is not None
    assert "[truncated for context budget]" in out.body
    assert len(out.summary.truncated) >= 1


def test_assemble_hard_fails_when_essentials_over_cap(tmp_path: Path) -> None:
    deployments = _budget_fixture(tmp_path, doc_size_chars=20_000, n_extra=0)
    recipe = _load_recipe(deployments)
    with pytest.raises(ContextBudgetError, match="Composes"):
        assemble(
            recipe,
            language="python",
            framework="none",
            deployments_path=deployments,
            max_context_tokens=1_000,
            max_link_depth=0,
            max_tokens_per_doc=50_000,
        )


def test_context_budget_error_carries_structured_fields(tmp_path: Path) -> None:
    """The wizard/REPL need essentials_tokens + current_cap to decide whether
    bumping to a higher preset would fit, without parsing the message."""
    deployments = _budget_fixture(tmp_path, doc_size_chars=20_000, n_extra=0)
    recipe = _load_recipe(deployments)
    with pytest.raises(ContextBudgetError) as exc_info:
        assemble(
            recipe,
            language="python",
            framework="none",
            deployments_path=deployments,
            max_context_tokens=1_000,
            max_link_depth=0,
            max_tokens_per_doc=50_000,
        )
    assert exc_info.value.current_cap == 1_000
    assert exc_info.value.essentials_tokens > 1_000


def test_assemble_link_depth_zero_skips_transitive(tmp_path: Path) -> None:
    # big-a links to a transitive doc; depth 0 must not follow it.
    deployments = _budget_fixture(tmp_path, doc_size_chars=200, n_extra=0)
    (deployments / "docs" / "patterns" / "transitive.md").write_text(
        "# T\n\nthrowaway", encoding="utf-8"
    )
    big_a = deployments / "docs" / "patterns" / "big-a.md"
    big_a.write_text(big_a.read_text() + "\n\n[t](transitive.md)\n", encoding="utf-8")

    recipe = _load_recipe(deployments)
    out = assemble(
        recipe,
        language="python",
        framework="none",
        deployments_path=deployments,
        max_context_tokens=10_000,
        max_link_depth=0,
        max_tokens_per_doc=1_000,
    )
    names = {p.name for p in out.referenced_paths}
    assert "transitive.md" not in names

    out2 = assemble(
        recipe,
        language="python",
        framework="none",
        deployments_path=deployments,
        max_context_tokens=10_000,
        max_link_depth=2,
        max_tokens_per_doc=1_000,
    )
    names2 = {p.name for p in out2.referenced_paths}
    assert "transitive.md" in names2


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
