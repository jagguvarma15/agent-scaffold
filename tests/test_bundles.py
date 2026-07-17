"""Tests for bundle presets — flat named capability sets (RAG, guardrails).

A bundle is sugar over ``capabilities.resolve(add_capabilities=...)`` exactly
like tiers, minus the extends chain: catalog-published entries win, the
embedded defaults cover catalogs that predate ``bundles:``, and an unknown
name expands inertly instead of raising.
"""

from __future__ import annotations

import pytest

from agent_scaffold.bundles import (
    RAG_PRESET_BUNDLES,
    default_presets,
    expand_bundle,
    load_bundles,
)
from agent_scaffold.catalog import BlueprintsPointer, BundleEntry, Catalog


def _catalog_with_bundles(entries: list[BundleEntry]) -> Catalog:
    return Catalog(
        schema_version=1,
        blueprints=BlueprintsPointer(repo="x/y", branch="main"),
        bundles=entries,
    )


def test_default_presets_cover_the_rag_choices() -> None:
    presets = default_presets()
    for bundle_name in RAG_PRESET_BUNDLES.values():
        assert bundle_name in presets
    assert expand_bundle("rag-simple", presets) == ["vector_db.pgvector", "embedding.openai"]
    assert expand_bundle("rag-complex", presets) == [
        "vector_db.qdrant",
        "embedding.openai",
        "rerank.cohere",
    ]
    assert expand_bundle("guardrails-basic", presets) == ["guardrail.llama-guard"]


def test_load_bundles_prefers_catalog_entries() -> None:
    catalog = _catalog_with_bundles(
        [BundleEntry(name="rag-simple", title="Custom", capabilities=["vector_db.chroma"])]
    )
    presets = load_bundles(catalog)
    assert set(presets) == {"rag-simple"}
    assert presets["rag-simple"].title == "Custom"
    assert expand_bundle("rag-simple", presets) == ["vector_db.chroma"]


def test_load_bundles_falls_back_without_a_bundles_block() -> None:
    empty = _catalog_with_bundles([])
    assert set(load_bundles(empty)) == set(default_presets())
    assert set(load_bundles(None)) == set(default_presets())


def test_expand_unknown_bundle_warns_and_expands_to_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert expand_bundle("nope", default_presets()) == []
    assert "unknown bundle 'nope'" in capsys.readouterr().err


def test_expand_bundle_dedupes_ids() -> None:
    presets = load_bundles(
        _catalog_with_bundles(
            [BundleEntry(name="b", capabilities=["cache.redis", "cache.redis", "obs.langfuse"])]
        )
    )
    assert expand_bundle("b", presets) == ["cache.redis", "obs.langfuse"]
