"""Tests for the model-metadata module: the adaptive-thinking gate, per-model
cache minimums, and the picker list."""

from __future__ import annotations

import pytest

from agent_scaffold import models


@pytest.mark.parametrize(
    "model",
    ["claude-opus-4-8", "claude-opus-4-7", "claude-sonnet-5", "claude-fable-5"],
)
def test_current_models_use_adaptive_thinking(model: str) -> None:
    assert models.uses_adaptive_thinking(model) is True


@pytest.mark.parametrize(
    "model",
    ["claude-opus-4-6", "claude-opus-4-5", "claude-sonnet-4-6", "claude-haiku-4-5"],
)
def test_legacy_models_use_budget_thinking(model: str) -> None:
    assert models.uses_adaptive_thinking(model) is False


def test_adaptive_gate_resolves_dated_ids() -> None:
    # A dated snapshot must resolve to the same family as the bare alias.
    assert models.uses_adaptive_thinking("claude-opus-4-8-20260514") is True


def test_unknown_model_defaults_to_legacy_thinking() -> None:
    assert models.uses_adaptive_thinking("claude-mystery-9") is False


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("claude-opus-4-8", 4096),
        ("claude-opus-4-5", 4096),
        ("claude-haiku-4-5", 4096),
        ("claude-sonnet-5", 4096),  # unpublished — conservative floor
        ("claude-sonnet-4-6", 2048),
        ("claude-fable-5", 2048),
        ("claude-sonnet-4-5", 1024),
    ],
)
def test_min_cache_tokens_per_family(model: str, expected: int) -> None:
    assert models.min_cache_tokens(model) == expected


def test_unknown_model_takes_conservative_cache_floor() -> None:
    # The highest current minimum, so we never mark an uncacheable block cacheable.
    assert models.min_cache_tokens("claude-mystery-9") == 4096


def test_default_model_is_current_opus() -> None:
    assert models.DEFAULT_MODEL == "claude-opus-4-8"


def test_picker_choices_are_current_and_labeled() -> None:
    choices = models.picker_choices()
    ids = [mid for mid, _ in choices]
    assert ids == ["claude-opus-4-8", "claude-sonnet-5", "claude-haiku-4-5"]
    assert all(label for _, label in choices)
