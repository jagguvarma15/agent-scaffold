"""Port → adapter feature-model resolver (analysis).

Validates a resolved capability stack against the catalog's port-typed registry:
each adapter binds a port (``port == kind``), and the catalog's ``ports[]``
(cardinality / required / default) + ``compatibility[]`` (requires / excludes /
conflicts / substitutes) define what a *valid, verified* configuration is.
:func:`analyze_configuration` reports cardinality and compatibility violations
plus the selection's weakest verification tier, so the plan step can show whether
the stack is a vetted configuration.

Analysis-only: it does not mutate the stack (no default-fill or swap rewriting),
so generation output is unchanged. Default-fill / swap resolution can build on
this later.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from agent_scaffold.capabilities import ResolvedStack
    from agent_scaffold.catalog import Catalog

# Higher rank = stronger guarantee. ``T3+`` (and any ``T3*``) → 3; unknown → 0.
_TIER_RANK = {"T1": 1, "T2": 2, "T3": 3}


def _tier_rank(tier: str | None) -> int:
    if not tier:
        return 0
    return _TIER_RANK.get(tier, 3 if tier.startswith("T3") else 0)


class ConfigReport(BaseModel):
    """Feature-model verdict for a resolved stack."""

    bindings: dict[str, list[str]]
    issues: list[str]
    """Hard violations — the configuration is invalid (cardinality over an
    exactly-one port, or a ``requires`` / ``excludes`` edge)."""
    warnings: list[str]
    """Soft signals — ``conflicts`` edges and adapters lacking a verification tier."""
    min_tier: str | None
    ok: bool

    def render(self) -> str:
        n = len(self.bindings)
        head = "verified configuration" if self.ok else "INVALID configuration"
        tier = f" · min verification {self.min_tier}" if self.min_tier else ""
        lines = [f"{head}: {n} port(s) bound{tier}"]
        lines.extend(f"  ✗ {m}" for m in self.issues)
        lines.extend(f"  ! {m}" for m in self.warnings)
        return "\n".join(lines)


def analyze_configuration(stack: ResolvedStack, catalog: Catalog) -> ConfigReport:
    """Validate ``stack`` against ``catalog``'s ports + compatibility model.

    Cardinality (an exactly-one port bound once) and ``requires`` / ``excludes``
    edges are hard issues; ``conflicts`` and adapters without a verification tier
    are warnings. The port of an adapter is its ``kind`` (the deployments
    invariant ``implements.port == kind``).
    """
    ports_by_id = {p.id: p for p in catalog.ports}
    entry_by_id = {c.id: c for c in catalog.capabilities}
    selected = [c.id for c in stack.capabilities]
    selected_set = set(selected)

    bindings: dict[str, list[str]] = {}
    for cap in stack.capabilities:
        bindings.setdefault(str(cap.kind), []).append(cap.id)

    issues: list[str] = []
    warnings: list[str] = []

    # Cardinality — an exactly-one port may bind at most one adapter.
    for port, ids in sorted(bindings.items()):
        p = ports_by_id.get(port)
        if p is not None and p.cardinality == "one" and len(ids) > 1:
            issues.append(f"port '{port}' is exactly-one but binds {sorted(ids)}")

    # NB: required-port *coverage* is intentionally not enforced here. The
    # resolved stack only covers ``adapter_home: capabilities`` ports; the
    # required model / framework / api_layer ports are bound by scaffold's
    # framework + model selection (outside the capability stack), so checking
    # them against the stack alone would false-positive on every recipe. Coverage
    # can be added once those bindings are threaded in.

    # Cross-tree edges over the selected set.
    for e in catalog.compatibility:
        if e.a not in selected_set:
            continue
        if e.relation == "requires" and e.b not in selected_set:
            issues.append(f"{e.a} requires {e.b}, which is not selected")
        elif e.relation == "excludes" and e.b in selected_set:
            issues.append(f"{e.a} excludes {e.b}, but both are selected")
        elif e.relation == "conflicts" and e.b in selected_set:
            warnings.append(f"{e.a} conflicts with {e.b} (both selected)")

    # Verification — weakest tier across selected adapters.
    tiers: list[str] = []
    untiered = 0
    for cid in selected:
        entry = entry_by_id.get(cid)
        if entry is None:
            continue
        tier = entry.verification.tier if entry.verification else None
        if tier:
            tiers.append(tier)
        else:
            untiered += 1
    min_tier = min(tiers, key=_tier_rank) if tiers else None
    if untiered:
        warnings.append(f"{untiered} selected adapter(s) without a verification tier")

    return ConfigReport(
        bindings=bindings,
        issues=issues,
        warnings=warnings,
        min_tier=min_tier,
        ok=not issues,
    )


def runtime_mode_swaps(runtime_modes: dict[str, object], mode: str) -> dict[str, str]:
    """The ``swaps`` map declared for ``mode`` in a recipe's ``runtime_modes``.

    Returns ``{}`` for ``default`` (or any mode with no swaps). Raises ``KeyError``
    for a non-``default`` mode that isn't declared — callers surface that to the user.
    """
    if mode == "default" and mode not in runtime_modes:
        return {}
    spec = runtime_modes[mode]  # KeyError on unknown mode → caller handles
    swaps = spec.get("swaps") if isinstance(spec, dict) else None
    return {str(k): str(v) for k, v in (swaps or {}).items()}


def partition_swaps(swaps: dict[str, str]) -> tuple[set[str], list[str], dict[str, str]]:
    """Split a runtime-mode swaps map into ``(capability_removes, capability_adds,
    doc_swaps)``.

    A ref containing ``/`` (e.g. ``stack/llm-claude``) is a doc/stack swap applied
    to the recipe's ``load_list``; otherwise it's a ``<kind>.<name>`` capability id
    re-routed through capability resolution (remove the ``from``, add the ``to``).
    """
    removes: set[str] = set()
    adds: list[str] = []
    doc_swaps: dict[str, str] = {}
    for frm, to in swaps.items():
        if "/" in frm or "/" in to:
            doc_swaps[frm] = to
        else:
            removes.add(frm)
            adds.append(to)
    return removes, adds, doc_swaps
