"""The single source for layer groupings shared by the wizard and slash commands.

``repl/shell.py`` (the wizard's layer walk) and ``repl/commands.py``
(``/layer``, ``/stack``) previously each carried their own copy of the
layer → capability-kind map, with a comment promising they mirror each
other. This module is that mirror made real: both import from here, so a
new capability kind lands in the wizard, ``/layer``, and ``/stack`` in one
edit.

``WIZARD_TOOLS_KINDS`` is the wizard-only narrowing of the tools layer:
mcp and guardrail capabilities have dedicated wizard steps, so the walk's
Tools step must not re-ask them. The ``/layer tools`` command keeps the
full kind set — a direct command addresses the whole layer.
"""

from __future__ import annotations

from agent_scaffold.capabilities import CapabilityKind

# Layer groupings surfaced in reading order. Memory merges the storage
# kinds so the user sees "memory layer" as one decision; infrastructure
# covers the stateful backbones; tools covers agent-tier API integrations.
# Hosting and auth are deliberately not wizard steps (late/rare decisions)
# but stay pickable via /layer and visible in /stack.
LAYER_GROUPS: tuple[tuple[str, str, tuple[CapabilityKind, ...]], ...] = (
    ("memory", "Memory", ("relational", "cache", "vector_db", "memory_store")),
    ("infrastructure", "Infrastructure", ("queue", "durable")),
    ("tools", "Tools", ("live_data", "mcp", "embedding", "rerank", "sandbox", "guardrail")),
    ("observability", "Observability", ("obs",)),
    ("eval", "Eval", ("eval",)),
    ("interface", "Interface", ("frontend",)),
)

# Key → kinds lookup for the slash commands, including aliases and the
# non-wizard layers (hosting, auth).
LAYER_GROUPS_BY_KEY: dict[str, tuple[CapabilityKind, ...]] = {
    **{key: kinds for key, _label, kinds in LAYER_GROUPS},
    "obs": ("obs",),
    "frontend": ("frontend",),
    "hosting": ("host",),
    "auth": ("auth",),
}

# The layer keys /layer (no args) and /stack iterate, in reading order.
# Aliases (obs, frontend) are skipped to avoid duplicate rows.
LAYER_DISPLAY_ORDER: tuple[str, ...] = (
    "memory",
    "infrastructure",
    "tools",
    "observability",
    "eval",
    "interface",
    "hosting",
    "auth",
)

# Wizard Tools step kinds: the full tools layer minus the kinds owned by
# dedicated wizard steps (mcp, guardrail), so no capability is asked twice
# in one walk.
WIZARD_TOOLS_KINDS: tuple[CapabilityKind, ...] = ("live_data", "embedding", "rerank", "sandbox")
