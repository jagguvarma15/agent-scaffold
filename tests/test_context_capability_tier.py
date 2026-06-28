"""Tests for the capability context tier."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import yaml

from agent_scaffold.capabilities import (
    Capability,
    ResolvedStack,
    load_capabilities,
    resolve,
)
from agent_scaffold.catalog import Catalog
from agent_scaffold.context import (
    _TIER_CAPABILITY,
    _TIER_LABELS,
    assemble_capability_tier,
)
from agent_scaffold.context import (
    assemble as _real_assemble,
)
from agent_scaffold.discovery import discover_recipes

_TEST_CATALOG_PATH = Path(__file__).parent / "fixtures" / "catalog_minimal.yaml"
_TEST_CATALOG: Catalog = Catalog.model_validate(
    yaml.safe_load(_TEST_CATALOG_PATH.read_text(encoding="utf-8"))
)
assemble = partial(_real_assemble, catalog=_TEST_CATALOG)


def test_capability_tier_constant_position() -> None:
    # The tier label must exist and read as "Capabilities".
    assert _TIER_LABELS[_TIER_CAPABILITY] == "Capabilities"


def test_assemble_capability_tier_renders_blocks(mock_deployments_path: Path) -> None:
    catalog = load_capabilities(mock_deployments_path)
    cap_redis = catalog["cache.redis"]
    cap_qdrant = catalog["vector_db.qdrant"]
    stack = ResolvedStack(capabilities=[cap_redis, cap_qdrant])

    body, paths, tokens = assemble_capability_tier(stack, budget=100_000)
    assert "## Capability: cache.redis" in body
    assert "## Capability: vector_db.qdrant" in body
    # Order preserved.
    assert body.index("cache.redis") < body.index("vector_db.qdrant")
    assert [p.name for p in paths] == ["redis.md", "qdrant.md"]
    assert tokens > 0


def test_assemble_capability_tier_drops_on_budget_exhausted(
    mock_deployments_path: Path,
) -> None:
    catalog = load_capabilities(mock_deployments_path)
    stack = ResolvedStack(capabilities=[catalog["cache.redis"], catalog["vector_db.qdrant"]])
    # Budget of 1 token forces the first capability to be the only one kept
    # iff it fits — but our cache.redis body is bigger than that, so all drop.
    body, paths, _ = assemble_capability_tier(stack, budget=1)
    assert body == ""
    assert paths == []


def test_assemble_capability_tier_header_carries_metadata(
    mock_deployments_path: Path,
) -> None:
    catalog = load_capabilities(mock_deployments_path)
    body, _, _ = assemble_capability_tier(
        ResolvedStack(capabilities=[catalog["vector_db.qdrant"]]),
        budget=100_000,
    )
    assert "kind: `vector_db`" in body
    assert "`QDRANT_URL`" in body
    assert "docker service: `qdrant`" in body
    assert "bootstrap step: `bootstrap_vector_db`" in body


def test_assemble_capability_tier_handles_caps_without_docker(
    mock_deployments_path: Path,
) -> None:
    catalog = load_capabilities(mock_deployments_path)
    body, _, _ = assemble_capability_tier(
        ResolvedStack(capabilities=[catalog["host.vercel"]]),
        budget=100_000,
    )
    assert "## Capability: host.vercel" in body
    assert "docker service" not in body
    assert "deploy targets: `vercel`" in body


def test_format_capability_body_prefers_summary(tmp_path: Path) -> None:
    """With a catalog context_summary, the capability block ships the compact
    summary instead of the full body + duplicated metadata header."""
    from agent_scaffold.context import _format_capability_body

    cap = Capability(
        id="vector_db.qdrant",
        kind="vector_db",
        path=tmp_path / "q.md",
        env_vars=["QDRANT_URL"],
        docs="",
        body="FULL BODY PROSE HERE",
    )
    full = _format_capability_body(cap)
    assert "FULL BODY PROSE HERE" in full
    assert "kind: `vector_db`" in full

    summarized = _format_capability_body(
        cap, summary="Qdrant (vector_db) — vector store. Env vars: QDRANT_URL."
    )
    assert "## Capability: vector_db.qdrant" in summarized
    assert "Qdrant (vector_db)" in summarized
    assert "FULL BODY PROSE HERE" not in summarized  # full body dropped
    assert "kind: `vector_db`" not in summarized  # metadata header dropped (summary carries it)


def test_assemble_includes_capability_tier_in_summary(
    mock_deployments_path: Path,
) -> None:
    recipes = {r.slug: r for r in discover_recipes(mock_deployments_path)}
    recipe = recipes["with-capabilities"]
    catalog = load_capabilities(mock_deployments_path)
    stack = resolve(recipe, catalog)

    ctx = assemble(
        recipe=recipe,
        language="python",
        framework="langgraph",
        deployments_path=mock_deployments_path,
        resolved_stack=stack,
        max_context_tokens=200_000,
    )

    assert ctx.summary is not None
    cap_tier_stats = next((t for t in ctx.summary.tiers if t.tier == _TIER_CAPABILITY), None)
    assert cap_tier_stats is not None
    # Three capabilities resolve (redis, qdrant, vercel); summary should count them all.
    assert cap_tier_stats.docs == 3
    assert "## Capability: cache.redis" in ctx.body
    assert "## Capability: vector_db.qdrant" in ctx.body
    assert "## Capability: host.vercel" in ctx.body


def test_assemble_without_resolved_stack_is_unchanged(
    mock_deployments_path: Path,
) -> None:
    recipes = {r.slug: r for r in discover_recipes(mock_deployments_path)}
    recipe = recipes["customer-support-triage"]
    ctx = assemble(
        recipe=recipe,
        language="python",
        framework="langgraph",
        deployments_path=mock_deployments_path,
        # resolved_stack=None — back-compat path
        max_context_tokens=200_000,
    )
    assert "## Capability:" not in ctx.body
    assert ctx.summary is not None
    assert all(t.tier != _TIER_CAPABILITY for t in ctx.summary.tiers)


def test_capability_body_falls_back_to_docs_when_body_empty(tmp_path: Path) -> None:
    # Bare capability with only frontmatter `docs:` — _format_capability_body
    # should fall through to docs so the LLM still sees something.
    from agent_scaffold.context import _format_capability_body

    cap = Capability(
        id="obs.minimal",
        kind="obs",
        path=tmp_path / "minimal.md",
        env_vars=["MIN_URL"],
        docs="Minimal doc string.",
        body="",
    )
    rendered = _format_capability_body(cap)
    assert "Minimal doc string." in rendered
