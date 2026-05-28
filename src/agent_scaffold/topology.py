"""Topology and Role types for shaping generated project structure.

A topology describes the runtime shape of the generated agent project:
single-process, multiple cooperating agents, a swarm, or a supervised fleet.
Recipes declare topology + roles in frontmatter; the CLI infers when absent.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from agent_scaffold.discovery import Recipe


class Topology(str, Enum):
    SINGLE = "single"
    MULTI = "multi-agent-flat"
    MULTI_HIERARCHICAL = "multi-agent-hierarchical"
    SWARM = "swarm"
    FLEET = "fleet"


class Role(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    name: str
    description: str = ""
    model_hint: str | None = None
    tools: list[str] = Field(default_factory=list)


def coerce_topology(value: Any) -> Topology | None:
    """Coerce a frontmatter ``topology`` value to a :class:`Topology`."""
    if value is None:
        return None
    if isinstance(value, Topology):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower().replace("_", "-")
        # Allow some friendly aliases.
        aliases = {
            "multi": Topology.MULTI,
            "multi-agent": Topology.MULTI,
            "flat": Topology.MULTI,
            "hierarchical": Topology.MULTI_HIERARCHICAL,
        }
        if normalized in aliases:
            return aliases[normalized]
        for t in Topology:
            if t.value == normalized:
                return t
    return None


def coerce_roles(value: Any) -> list[Role]:
    """Coerce a frontmatter ``roles`` list into :class:`Role` objects."""
    if not isinstance(value, list):
        return []
    roles: list[Role] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        roles.append(
            Role(
                name=name.strip(),
                description=str(item.get("description", "") or ""),
                model_hint=(
                    str(item["model_hint"]) if item.get("model_hint") is not None else None
                ),
                tools=[str(t) for t in (item.get("tools") or []) if isinstance(t, str)],
            )
        )
    return roles


def infer_topology(recipe: Recipe, body: str) -> Topology:
    """Infer the topology from recipe metadata + body links.

    Precedence:
    1. Explicit frontmatter (handled by the caller before calling this).
    2. Recipe links to a multi-agent pattern doc → MULTI / MULTI_HIERARCHICAL.
    3. Recipe declares 3+ roles → MULTI.
    4. Default → SINGLE.
    """
    lowered = body.lower()
    if "patterns/multi-agent-hierarchical.md" in lowered:
        return Topology.MULTI_HIERARCHICAL
    if "patterns/multi-agent-flat.md" in lowered or "patterns/multi-agent.md" in lowered:
        return Topology.MULTI
    if len(recipe.roles) >= 3:
        return Topology.MULTI
    return Topology.SINGLE


def resolve(recipe: Recipe, ctx_body: str) -> tuple[Topology, list[Role]]:
    """Resolve a recipe's ``(topology, roles)`` for the assembled context.

    Combines the explicit-frontmatter check with the inference fallback and
    the SINGLE default, then coerces ``recipe.roles`` into typed
    :class:`Role` objects. Used by every site that needs to render a plan
    or build :class:`agent_scaffold.pipeline.PipelineInputs` so the
    derivation doesn't drift between the CLI ``new`` flow, the REPL's
    ``/plan``, and the REPL's pre-generate handoff.
    """
    topology = (
        coerce_topology(recipe.topology) if recipe.topology else infer_topology(recipe, ctx_body)
    ) or Topology.SINGLE
    return topology, coerce_roles(recipe.roles)
