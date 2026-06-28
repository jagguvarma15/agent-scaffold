"""Tests for the port → adapter feature-model analyzer (resolver.py)."""

from __future__ import annotations

from pathlib import Path

from agent_scaffold.capabilities import Capability, ResolvedStack
from agent_scaffold.catalog import (
    CapabilityEntry,
    Catalog,
    CompatibilityEdge,
    PortEntry,
    VerificationEntry,
)
from agent_scaffold.resolver import analyze_configuration


def _cap(cid: str, kind: str) -> Capability:
    return Capability(id=cid, kind=kind, path=Path(f"/{cid}.md"))


def _entry(cid: str, kind: str, tier: str | None = "T1") -> CapabilityEntry:
    return CapabilityEntry(
        id=cid,
        kind=kind,
        path=f"docs/{cid}.md",
        verification=VerificationEntry(tier=tier) if tier else None,
    )


def _catalog(caps: list[CapabilityEntry], ports: list[PortEntry], compat=()) -> Catalog:
    # model_construct: skip whole-catalog validation (no blueprints needed) — the
    # analyzer only reads ports / compatibility / capabilities.
    return Catalog.model_construct(
        capabilities=list(caps), ports=list(ports), compatibility=list(compat)
    )


def test_valid_one_adapter_per_exactly_one_port() -> None:
    stack = ResolvedStack(
        capabilities=[_cap("vector_db.qdrant", "vector_db"), _cap("cache.redis", "cache")]
    )
    cat = _catalog(
        [_entry("vector_db.qdrant", "vector_db"), _entry("cache.redis", "cache")],
        [PortEntry(id="vector_db", cardinality="one"), PortEntry(id="cache", cardinality="one")],
    )
    rep = analyze_configuration(stack, cat)
    assert rep.ok and not rep.issues
    assert rep.min_tier == "T1"
    assert rep.bindings == {"vector_db": ["vector_db.qdrant"], "cache": ["cache.redis"]}


def test_cardinality_violation_is_issue() -> None:
    stack = ResolvedStack(
        capabilities=[_cap("vector_db.qdrant", "vector_db"), _cap("vector_db.chroma", "vector_db")]
    )
    cat = _catalog(
        [_entry("vector_db.qdrant", "vector_db"), _entry("vector_db.chroma", "vector_db")],
        [PortEntry(id="vector_db", cardinality="one")],
    )
    rep = analyze_configuration(stack, cat)
    assert not rep.ok
    assert any("exactly-one" in m for m in rep.issues)


def test_many_port_allows_multiple() -> None:
    stack = ResolvedStack(capabilities=[_cap("obs.langfuse", "obs"), _cap("obs.grafana", "obs")])
    cat = _catalog(
        [_entry("obs.langfuse", "obs"), _entry("obs.grafana", "obs")],
        [PortEntry(id="obs", cardinality="many")],
    )
    assert analyze_configuration(stack, cat).ok


def test_requires_edge_unsatisfied_is_issue() -> None:
    stack = ResolvedStack(capabilities=[_cap("durable.temporal", "durable")])
    cat = _catalog(
        [_entry("durable.temporal", "durable")],
        [PortEntry(id="durable", cardinality="one")],
        [CompatibilityEdge(a="durable.temporal", b="relational.postgres", relation="requires")],
    )
    rep = analyze_configuration(stack, cat)
    assert not rep.ok
    assert any("requires" in m for m in rep.issues)


def test_conflicts_edge_is_warning_not_issue() -> None:
    stack = ResolvedStack(capabilities=[_cap("cache.redis", "cache"), _cap("queue.kafka", "queue")])
    cat = _catalog(
        [_entry("cache.redis", "cache"), _entry("queue.kafka", "queue")],
        [PortEntry(id="cache", cardinality="one"), PortEntry(id="queue", cardinality="many")],
        [CompatibilityEdge(a="cache.redis", b="queue.kafka", relation="conflicts")],
    )
    rep = analyze_configuration(stack, cat)
    assert rep.ok  # conflicts are soft — a warning, not an issue
    assert any("conflicts" in m for m in rep.warnings)


def test_min_tier_is_the_weakest() -> None:
    stack = ResolvedStack(capabilities=[_cap("cache.redis", "cache"), _cap("queue.kafka", "queue")])
    cat = _catalog(
        [_entry("cache.redis", "cache", "T2"), _entry("queue.kafka", "queue", "T1")],
        [PortEntry(id="cache", cardinality="one"), PortEntry(id="queue", cardinality="many")],
    )
    assert analyze_configuration(stack, cat).min_tier == "T1"


def test_untiered_adapter_warns() -> None:
    stack = ResolvedStack(capabilities=[_cap("cache.redis", "cache")])
    cat = _catalog(
        [_entry("cache.redis", "cache", tier=None)],
        [PortEntry(id="cache", cardinality="one")],
    )
    rep = analyze_configuration(stack, cat)
    assert rep.min_tier is None
    assert any("verification tier" in m for m in rep.warnings)
