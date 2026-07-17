"""Bundle presets — flat named capability sets (the RAG and guardrails presets).

A *bundle* expands to a fixed list of capability ids in one pick. Like tiers
(:mod:`agent_scaffold.tiers`) it is sugar over the existing capability
composition machinery — the expanded ids are seeded into
:func:`agent_scaffold.capabilities.resolve` via ``add_capabilities`` — but
unlike tiers there is no ``extends`` chain: each bundle stands alone.

Presets are published by the deployments catalog (``catalog.bundles``); this
module also carries an embedded default table so the tool works against a
catalog that predates the ``bundles:`` block — the same offline-fallback
contract tiers use.

This is a leaf module — no Typer / Anthropic imports — so both the CLI and
the REPL can share it.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_scaffold.catalog import Catalog


@dataclass(frozen=True)
class BundlePreset:
    """One bundle: a flat curated capability set."""

    name: str
    title: str
    description: str = ""
    capabilities: list[str] = field(default_factory=list)


RAG_PRESET_BUNDLES: dict[str, str] = {
    "simple": "rag-simple",
    "complex": "rag-complex",
}
"""Wizard RAG choices → bundle names. The wizard asks "simple or complex";
the catalog publishes the expansions under these bundle names."""


# Embedded fallbacks matching the catalog-published bundles, so the RAG and
# guardrails pickers work against a catalog that predates ``bundles:``.
_DEFAULT_PRESETS: tuple[BundlePreset, ...] = (
    BundlePreset(
        name="rag-simple",
        title="Simple RAG",
        description=(
            "Single-stage retrieval: pgvector rides the existing postgres, "
            "OpenAI embeddings, top-k cosine into the prompt."
        ),
        capabilities=["vector_db.pgvector", "embedding.openai"],
    ),
    BundlePreset(
        name="rag-complex",
        title="Advanced RAG",
        description=(
            "Hybrid dense plus keyword retrieval on Qdrant with Cohere "
            "reranking between retrieval and the LLM."
        ),
        capabilities=["vector_db.qdrant", "embedding.openai", "rerank.cohere"],
    ),
    BundlePreset(
        name="guardrails-basic",
        title="Guardrails",
        description=(
            "Input and output classification with Llama Guard before and " "after the agent loop."
        ),
        capabilities=["guardrail.llama-guard"],
    ),
)


def default_presets() -> dict[str, BundlePreset]:
    """The embedded fallback bundle table, keyed by bundle name."""
    return {p.name: p for p in _DEFAULT_PRESETS}


def load_bundles(catalog: Catalog | None) -> dict[str, BundlePreset]:
    """Return the bundle presets for ``catalog``.

    Uses the catalog's published ``bundles`` when present; otherwise falls
    back to the embedded default table. ``None`` is treated as "no catalog".
    """
    entries = catalog.bundles if catalog is not None else None
    if not entries:
        return default_presets()
    presets: dict[str, BundlePreset] = {}
    for entry in entries:
        presets[entry.name] = BundlePreset(
            name=entry.name,
            title=entry.title or entry.name,
            description=entry.description or "",
            capabilities=list(entry.capabilities),
        )
    return presets


def expand_bundle(name: str, presets: dict[str, BundlePreset]) -> list[str]:
    """The capability ids ``name`` seeds into resolution.

    An unknown bundle warns once and expands to nothing — the resolver
    treats missing seeds inertly, matching the tier contract.
    """
    preset = presets.get(name)
    if preset is None:
        _warn(f"unknown bundle {name!r}; expected one of {sorted(presets)}")
        return []
    return _dedupe(preset.capabilities)


def _dedupe(ids: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for cid in ids:
        if cid and cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


def _warn(msg: str) -> None:
    print(f"agent-scaffold: warning: {msg}", file=sys.stderr)


__all__ = [
    "RAG_PRESET_BUNDLES",
    "BundlePreset",
    "default_presets",
    "expand_bundle",
    "load_bundles",
]
