"""Tests for agent_scaffold.topology."""

from __future__ import annotations

from pathlib import Path

from agent_scaffold.discovery import Recipe
from agent_scaffold.topology import (
    Role,
    Topology,
    coerce_roles,
    coerce_topology,
    infer_topology,
    resolve,
)


def _recipe(roles: list[dict[str, str]] | None = None, topology: str | None = None) -> Recipe:
    return Recipe(
        slug="test",
        title="Test",
        path=Path("/tmp/test.md"),
        languages=["python"],
        roles=roles or [],
        topology=topology,
    )


# The canonical topology value list, mirrored from agent-deployments
# docs/recipes/SCHEMA.md (`#### topology` → Allowed values). The mirror test
# below fails CI if the scaffold enum and this list ever diverge — keep both in
# lockstep with SCHEMA.md, which is the single source of truth.
CANONICAL_TOPOLOGIES = frozenset(
    {
        "single",
        "chain",
        "parallel",
        "event-driven",
        "multi-agent-flat",
        "multi-agent-hierarchical",
    }
)


def test_topology_enum_matches_canonical_schema_list() -> None:
    """Mirror guard: the enum must equal SCHEMA.md's allowed `topology` values.

    Adding a value to the enum without updating SCHEMA.md (or vice versa) — or
    reintroducing the long-gone `swarm`/`fleet` — fails here."""
    assert {t.value for t in Topology} == CANONICAL_TOPOLOGIES


def test_coerce_topology_known_values() -> None:
    assert coerce_topology("single") == Topology.SINGLE
    assert coerce_topology("chain") == Topology.CHAIN
    assert coerce_topology("parallel") == Topology.PARALLEL
    assert coerce_topology("event-driven") == Topology.EVENT_DRIVEN
    assert coerce_topology("multi-agent-flat") == Topology.MULTI
    assert coerce_topology("multi-agent-hierarchical") == Topology.MULTI_HIERARCHICAL


def test_coerce_topology_dropped_values_are_unknown() -> None:
    # swarm/fleet were never in SCHEMA.md and have no consumers — they must not
    # resolve to anything now.
    assert coerce_topology("swarm") is None
    assert coerce_topology("fleet") is None


def test_coerce_topology_aliases() -> None:
    assert coerce_topology("multi") == Topology.MULTI
    assert coerce_topology("multi-agent") == Topology.MULTI
    assert coerce_topology("hierarchical") == Topology.MULTI_HIERARCHICAL
    # Friendly tolerance for casing / underscores, incl. the new values.
    assert coerce_topology("MULTI_AGENT_FLAT") == Topology.MULTI
    assert coerce_topology("event_driven") == Topology.EVENT_DRIVEN
    assert coerce_topology("EVENT-DRIVEN") == Topology.EVENT_DRIVEN


def test_coerce_topology_unknown_returns_none() -> None:
    assert coerce_topology("nope") is None
    assert coerce_topology(None) is None


def test_coerce_roles_skips_bad_entries() -> None:
    roles = coerce_roles(
        [
            {"name": "intake", "description": "parse"},
            {"description": "no name — skipped"},
            "not a dict — skipped",
            {"name": "  ", "description": "blank name — skipped"},
            {
                "name": "search",
                "model_hint": "opus",
                "tools": ["resy", "opentable", 123],
            },
        ]
    )
    assert [r.name for r in roles] == ["intake", "search"]
    search = roles[1]
    assert search.model_hint == "opus"
    # Non-string tool entries dropped.
    assert search.tools == ["resy", "opentable"]


def test_infer_topology_from_explicit_pattern_link() -> None:
    body = "see [multi](../patterns/multi-agent-flat.md) for details"
    assert infer_topology(_recipe(), body) == Topology.MULTI

    body_h = "see [h](../patterns/multi-agent-hierarchical.md)"
    assert infer_topology(_recipe(), body_h) == Topology.MULTI_HIERARCHICAL


def test_infer_topology_from_role_count() -> None:
    body = "plain prose with no pattern link"
    three = [{"name": f"r{i}"} for i in range(3)]
    assert infer_topology(_recipe(roles=three), body) == Topology.MULTI

    two = [{"name": f"r{i}"} for i in range(2)]
    assert infer_topology(_recipe(roles=two), body) == Topology.SINGLE


def test_role_dataclass_round_trip() -> None:
    role = Role(name="notifier", model_hint="haiku", tools=["email_send"])
    assert role.model_hint == "haiku"
    assert role.tools == ["email_send"]


def test_resolve_prefers_explicit_frontmatter() -> None:
    """Explicit recipe.topology beats body inference."""
    recipe = _recipe(topology="single", roles=[{"name": "a"}, {"name": "b"}, {"name": "c"}])
    body = "this body links to ../patterns/multi-agent-flat.md so inference would say MULTI"
    topology, roles = resolve(recipe, body)
    assert topology == Topology.SINGLE
    assert [r.name for r in roles] == ["a", "b", "c"]


def test_resolve_honors_explicit_pipeline_topologies() -> None:
    """A `chain`/`parallel`/`event-driven` recipe is modeled as itself, not
    silently downgraded to SINGLE (the bug this brief fixes)."""
    for raw, expected in (
        ("chain", Topology.CHAIN),
        ("parallel", Topology.PARALLEL),
        ("event-driven", Topology.EVENT_DRIVEN),
    ):
        topology, _ = resolve(_recipe(topology=raw), "plain body, no pattern link")
        assert topology == expected


def test_resolve_falls_back_to_inference_then_single() -> None:
    """No frontmatter → infer from body; no signal → SINGLE."""
    recipe = _recipe(roles=[{"name": "a"}])
    topology, roles = resolve(recipe, "plain body, no pattern link")
    assert topology == Topology.SINGLE
    assert [r.name for r in roles] == ["a"]

    body = "see [m](../patterns/multi-agent-flat.md)"
    topology, _ = resolve(_recipe(), body)
    assert topology == Topology.MULTI
