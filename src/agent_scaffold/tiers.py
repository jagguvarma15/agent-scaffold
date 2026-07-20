"""Tier presets — curated capability bundles spanning a simple chat agent (T0)
to an enterprise stack (T4).

A *tier* is a named preset that expands to a list of capability ids. At
generation time the active tier's ids are seeded into
:func:`agent_scaffold.capabilities.resolve` via its ``add_capabilities`` hook —
so a tier is sugar over the existing capability-composition machinery, never a
parallel code path. Tiers stack: each ``extends`` its predecessor, so the
expanded id set satisfies ``T4 ⊇ T3 ⊇ T2 ⊇ T1 ⊇ T0``.

Presets are published by the deployments catalog (``catalog.tiers``); this
module also carries an embedded default table so the tool works before a
catalog ships a ``tiers:`` block — the same offline-fallback contract
:mod:`agent_scaffold.catalog` uses for the catalog itself.

This is a leaf module — no Typer / Anthropic imports — so both the CLI and the
REPL can share it, mirroring :mod:`agent_scaffold.language_hints`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_scaffold.catalog import Catalog


KNOWN_TIERS: tuple[str, ...] = ("T0", "T1", "T2", "T3", "T4")
"""Conventional tier names, lowest → highest. A *tuple*, deliberately not a
``Literal`` gate: the catalog may publish additional tiers, and a stricter type
would brick on that additive change (the deployments forward-compat contract)."""


@dataclass(frozen=True)
class TierPreset:
    """One tier: a curated capability bundle.

    ``capabilities`` are seeded into resolution for this tier; ``overlays`` are
    opt-in extras surfaced at this tier (multi-agent, HITL, durable, guardrail,
    obs backends) — never auto-seeded. ``extends`` names the tier this one
    builds on, so presets stack into a superset chain.
    """

    name: str
    title: str
    description: str = ""
    extends: str | None = None
    capabilities: list[str] = field(default_factory=list)
    overlays: list[str] = field(default_factory=list)


# The embedded default presets — used when the catalog publishes no ``tiers:``
# block. Capability ids reference the ``core.*`` primitives (spec / prompts / io
# / tool registry / step-log / tracing) emitted by the tiered generation
# contract, plus the existing ``eval.promptfoo``. Ids not yet in the catalog
# resolve inertly into ``ResolvedStack.unresolved`` until their capability docs
# land — so this table is safe to ship ahead of that content.
_DEFAULT_PRESETS: tuple[TierPreset, ...] = (
    TierPreset(
        name="T0",
        title="Chat",
        description="Conversational agent: owned editable prompts + schema-validated I/O.",
        extends=None,
        capabilities=["core.spec", "core.prompts", "core.io_schema"],
    ),
    TierPreset(
        name="T1",
        title="Tool agent",
        description="Adds a typed tool registry, permission tiers, and compact-error retry.",
        extends="T0",
        capabilities=["core.tool_registry"],
    ),
    TierPreset(
        name="T2",
        title="Workflow",
        description="Adds a serializable step-log as state (pause / resume / retry / trace).",
        extends="T1",
        capabilities=["core.step_log"],
    ),
    TierPreset(
        name="T3",
        title="Production",
        description="Adds an eval seam seeded from the spec + structured tracing.",
        extends="T2",
        capabilities=["core.tracing", "eval.promptfoo"],
    ),
    TierPreset(
        name="T4",
        title="Enterprise",
        description="Production plus opt-in overlays: multi-agent, HITL, durable, guardrails, obs.",
        extends="T3",
        capabilities=[],
        overlays=[
            "multi_agent",
            "human_in_the_loop",
            "durable.temporal",
            "guardrail.llama-guard",
            "obs.langfuse",
        ],
    ),
)


def default_presets() -> dict[str, TierPreset]:
    """The embedded fallback preset table, keyed by tier name."""
    return {p.name: p for p in _DEFAULT_PRESETS}


def load_tier_presets(catalog: Catalog | None) -> dict[str, TierPreset]:
    """Return the tier presets for ``catalog``.

    Uses the catalog's published ``tiers`` when present; otherwise falls back to
    the embedded default table so the tool works before a catalog ships a
    ``tiers:`` block. ``None`` is treated as "no catalog" → embedded defaults.
    """
    entries = catalog.tiers if catalog is not None else None
    if not entries:
        return default_presets()
    presets: dict[str, TierPreset] = {}
    for entry in entries:
        presets[entry.name] = TierPreset(
            name=entry.name,
            title=entry.title or entry.name,
            description=entry.description or "",
            extends=entry.extends,
            capabilities=list(entry.capabilities),
            overlays=list(entry.overlays),
        )
    return presets


def expand_tier(name: str, presets: dict[str, TierPreset]) -> TierPreset:
    """Flatten ``name``'s ``extends`` chain into a single cumulative preset.

    Capabilities and overlays are unioned base-first (T0 before T1 …), deduped
    first-seen, so the result is a strict superset of every tier it extends —
    encoding the ``T4 ⊇ T3 ⊇ T2 ⊇ T1 ⊇ T0`` invariant. An unknown ``name`` warns
    once and falls back to the T0 floor; if even T0 is absent, returns an empty
    preset. Cycles in ``extends`` are broken defensively.
    """
    target = presets.get(name)
    if target is None:
        _warn(
            f"unknown tier {name!r}; expected one of "
            f"{sorted(presets) or list(KNOWN_TIERS)}; falling back to T0"
        )
        target = presets.get("T0")
        if target is None:
            return TierPreset(name=name, title=name)

    chain: list[TierPreset] = []
    seen: set[str] = set()
    cursor: TierPreset | None = target
    while cursor is not None and cursor.name not in seen:
        seen.add(cursor.name)
        chain.append(cursor)
        cursor = presets.get(cursor.extends) if cursor.extends else None
    chain.reverse()  # root-first: base capabilities sort ahead of derived ones.

    capabilities: list[str] = []
    overlays: list[str] = []
    for preset in chain:
        capabilities.extend(preset.capabilities)
        overlays.extend(preset.overlays)
    return replace(
        target,
        extends=None,
        capabilities=_dedupe(capabilities),
        overlays=_dedupe(overlays),
    )


def tier_seed_ids(preset: TierPreset, *, include_overlays: bool = False) -> list[str]:
    """The capability ids a (flattened) tier seeds into resolution.

    ``include_overlays`` appends the opt-in overlay ids — off by default so
    selecting a tier never silently pulls in its heavy overlays.
    """
    ids = list(preset.capabilities)
    if include_overlays:
        ids.extend(preset.overlays)
    return _dedupe(ids)


def active_tier(cli_tier: str | None, recipe_tier: str | None) -> str | None:
    """Resolve the effective tier: an explicit CLI ``--tier`` wins over the
    recipe's declared ``tier``; ``None`` when neither is set (no seeding)."""
    return cli_tier or recipe_tier


def resolve_tier_seeds(
    explicit: str | None,
    recipe_tier: str | None,
    catalog: Catalog | None,
) -> tuple[str | None, list[str]]:
    """Resolve the effective tier and its expanded capability seeds.

    The one shared implementation behind the CLI's ``--tier`` and the REPL's
    ``/tier`` / wizard step, so the two seeding paths cannot drift. Returns
    ``(chosen_tier, seed_ids)``; ``(None, [])`` when no tier applies.
    """
    chosen = active_tier(explicit, recipe_tier)
    if not chosen:
        return None, []
    return chosen, tier_seed_ids(expand_tier(chosen, load_tier_presets(catalog)))


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
    "KNOWN_TIERS",
    "TierPreset",
    "active_tier",
    "default_presets",
    "expand_tier",
    "load_tier_presets",
    "resolve_tier_seeds",
    "tier_seed_ids",
]
