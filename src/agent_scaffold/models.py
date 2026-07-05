"""Anthropic model metadata — the single source of truth for the model ids the
scaffold knows about, their extended-thinking mode, and their prompt-cache
minimums.

Historically these facts were spread across ``config``, ``generator``,
``effort``, ``cli_interactive`` and the REPL, and drifted apart: the
adaptive-thinking gate matched only one model family, and the cache minimum was
a single global constant. Both are per-family, so they live here and every
consumer looks them up by id.

Pricing is deliberately *not* here — it lives in :mod:`agent_scaffold.costs`.
The Models API does not return prices, so that table is hand-maintained
regardless, and keeping money out of this module keeps it a pure leaf.

Lookups are substring-based so a dated or aliased id
(``claude-opus-4-8-20260514``) resolves to the same family as the bare alias.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ThinkingMode = Literal["adaptive", "legacy"]

DEFAULT_MODEL = "claude-opus-4-8"

# Conservative fallback for an unrecognized id: the highest current minimum, so
# we never attach a cache breakpoint the API would silently refuse to honor.
_FALLBACK_CACHE_MIN_TOKENS = 4096


@dataclass(frozen=True)
class ModelInfo:
    """What the scaffold needs to know about one model family.

    ``match`` is the substring that identifies the family within a (possibly
    dated) model id. ``cache_min_tokens`` is the minimum cacheable prefix for
    the family — blocks below it get no ``cache_control`` breakpoint because the
    API would ignore it. ``thinking`` picks the request shape: ``adaptive``
    models reject the legacy ``budget_tokens`` field with an HTTP 400.
    """

    match: str
    label: str
    cache_min_tokens: int
    thinking: ThinkingMode


# Ordered most-specific first; a lookup returns the first family whose ``match``
# substring appears in the queried id. Cache minimums follow Anthropic's
# published per-model figures (Sonnet 5 is unpublished, so it takes the
# conservative 4096-token floor).
_MODELS: tuple[ModelInfo, ...] = (
    ModelInfo("fable-5", "Fable 5 — most capable (slowest, most expensive)", 2048, "adaptive"),
    ModelInfo("opus-4-8", "Opus 4.8 — highest quality (slowest, most expensive)", 4096, "adaptive"),
    ModelInfo("opus-4-7", "Opus 4.7 — high quality", 4096, "adaptive"),
    ModelInfo("opus-4-6", "Opus 4.6", 4096, "legacy"),
    ModelInfo("opus-4-5", "Opus 4.5", 4096, "legacy"),
    ModelInfo("sonnet-5", "Sonnet 5 — balanced (recommended for most runs)", 4096, "adaptive"),
    ModelInfo("sonnet-4-6", "Sonnet 4.6 — balanced", 2048, "legacy"),
    ModelInfo("sonnet-4-5", "Sonnet 4.5", 1024, "legacy"),
    ModelInfo("haiku-4-5", "Haiku 4.5 — fast iteration (lowest quality)", 4096, "legacy"),
)

# Ids offered by the interactive picker, best-first. Uses the bare aliases (no
# date suffix) so they always resolve to the latest snapshot.
PICKER_MODELS: tuple[str, ...] = (
    "claude-opus-4-8",
    "claude-sonnet-5",
    "claude-haiku-4-5",
)


def _lookup(model: str) -> ModelInfo | None:
    for info in _MODELS:
        if info.match in model:
            return info
    return None


def uses_adaptive_thinking(model: str) -> bool:
    """Whether ``model`` requires the adaptive-thinking request shape.

    Adaptive-only models reject ``{"type": "enabled", "budget_tokens": N}`` with
    an HTTP 400; they must be sent ``{"type": "adaptive"}`` plus an
    ``output_config.effort``. Unknown ids default to ``False`` (the legacy shape,
    matching prior behavior) — the live drift-canary test flags a new family
    that lands here so this table gets updated.
    """
    info = _lookup(model)
    return info is not None and info.thinking == "adaptive"


def min_cache_tokens(model: str) -> int:
    """The minimum cacheable prefix, in tokens, for ``model``'s family."""
    info = _lookup(model)
    return info.cache_min_tokens if info else _FALLBACK_CACHE_MIN_TOKENS


def picker_choices() -> list[tuple[str, str]]:
    """``(id, label)`` pairs for the interactive model picker, best-first."""
    labels = {info.match: info.label for info in _MODELS}
    choices: list[tuple[str, str]] = []
    for model in PICKER_MODELS:
        label = next((lbl for key, lbl in labels.items() if key in model), model)
        choices.append((model, label))
    return choices
