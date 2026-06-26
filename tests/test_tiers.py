"""Tests for tier presets — the resolution layer that turns a named tier
(T0 chat … T4 enterprise) into a seeded capability set.

These lock the core contract of the first redesign unit: a tier is sugar over
``capabilities.resolve(add_capabilities=...)`` (no parallel code path), the
expanded id sets form a superset chain, overlays never auto-seed, and a tier
the catalog doesn't carry resolves inertly rather than raising.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_scaffold import tiers
from agent_scaffold.capabilities import Capability, resolve
from agent_scaffold.catalog import BlueprintsPointer, Catalog, TierEntry
from agent_scaffold.discovery import Recipe, _reset_warn_dedupe, discover_recipes
from agent_scaffold.tiers import (
    active_tier,
    default_presets,
    expand_tier,
    load_tier_presets,
    tier_seed_ids,
)


def _catalog_with_tiers(tier_entries: list[TierEntry]) -> Catalog:
    return Catalog(
        schema_version=1,
        blueprints=BlueprintsPointer(repo="x/y", branch="main"),
        tiers=tier_entries,
    )


# ---------------------------------------------------------------------------
# Preset table + chain
# ---------------------------------------------------------------------------


def test_default_presets_cover_known_tiers() -> None:
    presets = default_presets()
    assert set(presets) == set(tiers.KNOWN_TIERS)
    assert presets["T0"].extends is None
    assert presets["T1"].extends == "T0"
    assert presets["T4"].extends == "T3"


def test_expand_tier_is_a_superset_chain() -> None:
    presets = default_presets()
    t0 = set(tier_seed_ids(expand_tier("T0", presets)))
    t1 = set(tier_seed_ids(expand_tier("T1", presets)))
    t2 = set(tier_seed_ids(expand_tier("T2", presets)))
    t3 = set(tier_seed_ids(expand_tier("T3", presets)))
    assert t0 < t1 < t2 < t3  # strict supersets: T3 ⊇ T2 ⊇ T1 ⊇ T0
    assert {"core.spec", "core.prompts", "core.io_schema"} <= t0
    assert "core.tool_registry" in t1 and "core.tool_registry" not in t0


def test_expand_tier_dedupes_and_orders_base_first() -> None:
    presets = {
        "T0": tiers.TierPreset(name="T0", title="base", capabilities=["a", "b"]),
        "T1": tiers.TierPreset(name="T1", title="mid", extends="T0", capabilities=["b", "c"]),
    }
    assert tier_seed_ids(expand_tier("T1", presets)) == ["a", "b", "c"]


def test_expand_tier_breaks_extends_cycles() -> None:
    # A malformed catalog with a cycle must not hang or recurse forever.
    presets = {
        "A": tiers.TierPreset(name="A", title="a", extends="B", capabilities=["x"]),
        "B": tiers.TierPreset(name="B", title="b", extends="A", capabilities=["y"]),
    }
    seeds = tier_seed_ids(expand_tier("A", presets))
    assert set(seeds) == {"x", "y"}


def test_overlays_are_not_seeded_by_default() -> None:
    t4 = expand_tier("T4", default_presets())
    assert "durable.temporal" not in tier_seed_ids(t4)
    assert "durable.temporal" in tier_seed_ids(t4, include_overlays=True)


def test_unknown_tier_warns_and_falls_back_to_floor(
    capsys: pytest.CaptureFixture[str],
) -> None:
    presets = default_presets()
    expanded = expand_tier("T9", presets)
    assert "unknown tier" in capsys.readouterr().err.lower()
    assert set(tier_seed_ids(expanded)) == set(tier_seed_ids(expand_tier("T0", presets)))


# ---------------------------------------------------------------------------
# CLI/recipe precedence + catalog loading
# ---------------------------------------------------------------------------


def test_active_tier_cli_overrides_recipe() -> None:
    assert active_tier("T3", "T1") == "T3"
    assert active_tier(None, "T1") == "T1"
    assert active_tier(None, None) is None


def test_load_tier_presets_uses_catalog_when_published() -> None:
    catalog = _catalog_with_tiers(
        [
            TierEntry(name="T0", title="Chat", capabilities=["core.spec"]),
            TierEntry(name="T1", title="Tools", extends="T0", capabilities=["core.tool_registry"]),
        ]
    )
    presets = load_tier_presets(catalog)
    assert set(presets) == {"T0", "T1"}
    assert tier_seed_ids(expand_tier("T1", presets)) == ["core.spec", "core.tool_registry"]


def test_load_tier_presets_falls_back_to_embedded() -> None:
    assert set(load_tier_presets(_catalog_with_tiers([]))) == set(tiers.KNOWN_TIERS)
    assert load_tier_presets(None) == default_presets()


# ---------------------------------------------------------------------------
# The lever: tier seeds flow through capabilities.resolve()
# ---------------------------------------------------------------------------


def test_tier_seeds_reach_resolve_as_capabilities() -> None:
    recipe = Recipe(slug="demo", title="Demo", path=Path("demo.md"))
    seeds = tier_seed_ids(expand_tier("T0", default_presets()))
    # core.* uses an arbitrary known kind here purely to exercise the wiring.
    caps_catalog = {cid: Capability(id=cid, kind="eval", path=Path(f"{cid}.md")) for cid in seeds}
    stack = resolve(recipe, caps_catalog, add_capabilities=seeds)
    assert set(seeds) <= set(stack.ids())


def test_tier_seeds_unknown_to_catalog_are_inert() -> None:
    # Forward-referenced core.* aren't in an empty catalog → unresolved, never raise.
    recipe = Recipe(slug="demo", title="Demo", path=Path("demo.md"))
    seeds = tier_seed_ids(expand_tier("T1", default_presets()))
    stack = resolve(recipe, {}, add_capabilities=seeds)
    assert set(seeds) <= set(stack.unresolved)
    assert stack.capabilities == []


# ---------------------------------------------------------------------------
# Recipe frontmatter parsing
# ---------------------------------------------------------------------------


def test_recipe_tier_parsed_and_normalized(tmp_path: Path) -> None:
    _reset_warn_dedupe()
    recipes_dir = tmp_path / "docs" / "recipes"
    recipes_dir.mkdir(parents=True)
    (recipes_dir / "demo.md").write_text("---\ntier: t2\n---\n# Demo Recipe\n", encoding="utf-8")
    (recipes_dir / "plain.md").write_text("# Plain Recipe\n", encoding="utf-8")
    recipes = {r.slug: r for r in discover_recipes(tmp_path)}
    assert recipes["demo"].tier == "T2"  # normalized uppercase
    assert recipes["plain"].tier is None  # absent → no tier
