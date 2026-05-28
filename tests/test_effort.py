"""Tests for ``agent_scaffold.effort`` — the shared effort-preset table.

This module is intentionally tiny — its job is to be the single source of
truth used by both the CLI and the REPL. The tests lock in the *contract*
(every preset has every field, low/medium/high progression makes sense)
rather than every numeric value, so the presets can be tuned without
hostile test churn.
"""

from __future__ import annotations

from agent_scaffold.effort import EFFORT_PRESETS, EffortPreset


def test_presets_cover_low_medium_high() -> None:
    assert set(EFFORT_PRESETS) == {"low", "medium", "high"}


def test_presets_are_typed_dataclasses() -> None:
    for preset in EFFORT_PRESETS.values():
        assert isinstance(preset, EffortPreset)


def test_preset_budgets_increase_monotonically_with_effort() -> None:
    """Higher effort buys more tokens, more context, deeper link recursion."""
    low = EFFORT_PRESETS["low"]
    medium = EFFORT_PRESETS["medium"]
    high = EFFORT_PRESETS["high"]
    assert low.max_tokens < medium.max_tokens < high.max_tokens
    assert low.max_context_tokens < medium.max_context_tokens < high.max_context_tokens
    assert low.max_link_depth <= medium.max_link_depth <= high.max_link_depth
    assert low.max_tokens_per_doc < medium.max_tokens_per_doc < high.max_tokens_per_doc


def test_only_high_is_strict() -> None:
    """Strict prompt is opt-in via the high preset."""
    assert EFFORT_PRESETS["low"].strict is False
    assert EFFORT_PRESETS["medium"].strict is False
    assert EFFORT_PRESETS["high"].strict is True


def test_low_skips_thinking() -> None:
    """Low effort runs Haiku without extended thinking."""
    assert EFFORT_PRESETS["low"].thinking is None
    assert EFFORT_PRESETS["medium"].thinking is not None
    assert EFFORT_PRESETS["high"].thinking is not None


def test_effort_preset_is_frozen() -> None:
    """Defensive: the table is the single source of truth — callers must
    not mutate a preset in place. ``model_copy``-style updates happen on
    the Config / SessionState that consumes the preset, not on the preset
    itself."""
    import dataclasses

    preset = EFFORT_PRESETS["low"]
    try:
        preset.model = "claude-sonnet-4-6"  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("EffortPreset should be frozen")
