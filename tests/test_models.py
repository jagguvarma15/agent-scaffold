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
    # The default (Opus) leads; Fable is second as an explicit opt-in.
    assert ids == ["claude-opus-4-8", "claude-fable-5", "claude-sonnet-5", "claude-haiku-4-5"]
    assert all(label for _, label in choices)


def test_fable_picker_label_carries_the_caveats() -> None:
    """Picking Fable must be an informed act: the label states the cost,
    refusal, and retention tradeoffs rather than only its capability."""
    label = dict(models.picker_choices())["claude-fable-5"]
    assert "2x Opus cost" in label
    assert "refuse" in label
    assert "retention" in label


def test_runtime_choices_include_fable() -> None:
    assert "claude-fable-5" in models.RUNTIME_MODEL_CHOICES


# ---- model-id validation ------------------------------------------------


def test_find_unknown_model_ids_flags_fabricated_date_suffix() -> None:
    text = "RESEARCH_MODEL: claude-sonnet-4-6-20250514"
    assert models.find_unknown_model_ids(text) == ["claude-sonnet-4-6-20250514"]


def test_find_unknown_model_ids_accepts_known_ids() -> None:
    text = (
        'model = "claude-sonnet-4-6"\n'
        "fallback = 'claude-opus-4-8'\n"
        "dated = 'claude-haiku-4-5-20251001'\n"
    )
    assert models.find_unknown_model_ids(text) == []


def test_find_unknown_model_ids_flags_retired_ids() -> None:
    assert models.find_unknown_model_ids("claude-3-5-sonnet-20241022") == [
        "claude-3-5-sonnet-20241022"
    ]


def test_find_unknown_model_ids_ignores_non_model_tokens() -> None:
    text = "claude-code and Claude Sonnet and anthropic-sdk and claude_docs"
    assert models.find_unknown_model_ids(text) == []


def test_find_unknown_model_ids_dedupes_in_order() -> None:
    text = "claude-sonnet-9 then claude-opus-9-20990101 then claude-sonnet-9"
    assert models.find_unknown_model_ids(text) == [
        "claude-sonnet-9",
        "claude-opus-9-20990101",
    ]


def test_runtime_choices_and_picker_are_known_ids() -> None:
    assert set(models.RUNTIME_MODEL_CHOICES) <= models.KNOWN_MODEL_IDS
    assert set(models.PICKER_MODELS) <= models.KNOWN_MODEL_IDS
    assert models.DEFAULT_MODEL in models.KNOWN_MODEL_IDS


def test_prompt_model_rule_matches_runtime_choices() -> None:
    # The prompt rule and the validator must never drift: every runtime choice
    # is named in both system prompts, and the only unknown-shaped id in the
    # prompt is the deliberate counter-example.
    from importlib import resources

    for filename in ("system.md", "system_strict.md"):
        text = resources.files("agent_scaffold.prompts").joinpath(filename).read_text("utf-8")
        for model_id in models.RUNTIME_MODEL_CHOICES:
            assert f"`{model_id}`" in text, f"{filename} does not name {model_id}"
        assert models.find_unknown_model_ids(text) == ["claude-sonnet-4-6-20250514"]
