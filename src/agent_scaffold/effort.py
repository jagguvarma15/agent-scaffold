"""Effort presets — single source of truth for the ``--effort`` / ``/effort`` knob.

An effort level bundles the model id, token budgets, thinking budget,
strict-prompt flag, and the context-assembly limits behind one keyword.
``low`` / ``medium`` / ``high`` is the surface; the dict body is the
machinery. Lives in its own leaf module so both ``cli.py`` and
``repl/commands.py`` consume the same definition — previously they each
carried their own copy and the REPL's was missing the context-budget
fields, so ``/effort high`` in the REPL behaved differently from
``--effort high`` on the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EffortPreset:
    """Typed view of an effort preset.

    ``thinking`` is ``None`` for low (no extended thinking); medium and
    high request a budget. ``strict`` flips the system prompt to the
    lint-cleanliness variant.
    """

    model: str
    max_tokens: int
    thinking: int | None
    strict: bool
    max_context_tokens: int
    max_link_depth: int
    max_tokens_per_doc: int


EFFORT_PRESETS: dict[str, EffortPreset] = {
    "low": EffortPreset(
        model="claude-haiku-4-5",
        max_tokens=16_000,
        thinking=None,
        strict=False,
        max_context_tokens=30_000,
        max_link_depth=1,
        max_tokens_per_doc=4_000,
    ),
    "medium": EffortPreset(
        model="claude-sonnet-5",
        max_tokens=32_000,
        thinking=8_000,
        strict=False,
        max_context_tokens=60_000,
        max_link_depth=2,
        max_tokens_per_doc=8_000,
    ),
    "high": EffortPreset(
        model="claude-opus-4-8",
        max_tokens=64_000,
        thinking=16_000,
        strict=True,
        max_context_tokens=100_000,
        max_link_depth=3,
        max_tokens_per_doc=12_000,
    ),
}
