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


def test_coerce_topology_known_values() -> None:
    assert coerce_topology("single") == Topology.SINGLE
    assert coerce_topology("multi-agent-flat") == Topology.MULTI
    assert coerce_topology("multi-agent-hierarchical") == Topology.MULTI_HIERARCHICAL
    assert coerce_topology("fleet") == Topology.FLEET
    assert coerce_topology("swarm") == Topology.SWARM


def test_coerce_topology_aliases() -> None:
    assert coerce_topology("multi") == Topology.MULTI
    assert coerce_topology("multi-agent") == Topology.MULTI
    assert coerce_topology("hierarchical") == Topology.MULTI_HIERARCHICAL
    # Friendly tolerance for casing / underscores.
    assert coerce_topology("MULTI_AGENT_FLAT") == Topology.MULTI


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
