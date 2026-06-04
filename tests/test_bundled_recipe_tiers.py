"""Pin tier inference for the 10 bundled recipes.

The Phase-2 wizard's tier-grouped picker depends on every bundled recipe
landing in the right tier; an unannotated recipe defaults to ``basic``
and gets buried in the wrong header. This test sweeps the bundled path
and asserts the expected tier per recipe slug.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_scaffold.discovery import discover_recipes, infer_complexity

_EXPECTED_TIERS: dict[str, str] = {
    "research-assistant": "basic",
    "docs-rag-qa": "basic",
    "customer-support-triage": "mid",
    "code-review-agent": "mid",
    "content-pipeline": "mid",
    "hierarchical-agent": "mid",
    "memory-assistant": "mid",
    "parallel-enricher": "mid",
    "ops-crew": "complex",
    "restaurant-rebooking": "complex",
}


@pytest.fixture
def bundled_path() -> Path:
    """Locate the in-repo ``_bundled_deployments`` source."""
    here = Path(__file__).resolve()
    root = here.parent.parent  # tests/ → repo root
    return root / "src" / "agent_scaffold" / "_bundled_deployments"


def test_every_bundled_recipe_has_explicit_tier(bundled_path: Path) -> None:
    recipes = {r.slug: r for r in discover_recipes(bundled_path)}
    # Each expected slug should still be present in the bundled set.
    missing = sorted(set(_EXPECTED_TIERS) - set(recipes))
    assert not missing, f"bundled recipes missing: {missing}"
    # And each must self-declare its complexity in frontmatter.
    for slug, recipe in recipes.items():
        assert recipe.complexity in {
            "basic",
            "mid",
            "complex",
        }, f"{slug}: missing or invalid `complexity:` frontmatter (got {recipe.complexity!r})"


def test_inferred_tier_matches_expected_per_recipe(bundled_path: Path) -> None:
    recipes = {r.slug: r for r in discover_recipes(bundled_path)}
    mismatches: list[str] = []
    for slug, expected in _EXPECTED_TIERS.items():
        actual = infer_complexity(recipes[slug])
        if actual != expected:
            mismatches.append(f"{slug}: expected {expected}, got {actual}")
    assert not mismatches, "\n".join(mismatches)


def test_every_bundled_recipe_has_agent_pattern(bundled_path: Path) -> None:
    recipes = discover_recipes(bundled_path)
    missing = [r.slug for r in recipes if not r.agent_pattern]
    assert not missing, f"recipes without agent_pattern frontmatter: {missing}"


def test_at_least_one_recipe_in_each_tier(bundled_path: Path) -> None:
    # The wizard's grouped picker needs at least one recipe per tier so users
    # see the spread. If a future deletion empties a tier, this test surfaces it.
    recipes = discover_recipes(bundled_path)
    tiers = {infer_complexity(r) for r in recipes}
    assert tiers == {"basic", "mid", "complex"}
