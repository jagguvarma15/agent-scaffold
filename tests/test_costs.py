"""Tests for the model pricing table and per-call cost estimate.

Locks in the corrected Opus-tier pricing ($5/$25, not the retired $15/$75) and
that the current model ids resolve.
"""

from __future__ import annotations

import pytest

from agent_scaffold.costs import PRICING, estimate


@pytest.mark.parametrize(
    ("model", "input_per_mtok", "output_per_mtok"),
    [
        ("claude-opus-4-8", 5.00, 25.00),
        ("claude-opus-4-7", 5.00, 25.00),
        ("claude-sonnet-5", 3.00, 15.00),
        ("claude-haiku-4-5", 1.00, 5.00),
    ],
)
def test_current_models_priced(model: str, input_per_mtok: float, output_per_mtok: float) -> None:
    price = PRICING[model]
    assert price.input_per_mtok == input_per_mtok
    assert price.output_per_mtok == output_per_mtok


def test_opus_priced_at_current_tier_not_retired_rate() -> None:
    # The retired $15/$75 rate over-stated every Opus cost report by ~3x.
    assert PRICING["claude-opus-4-8"].input_per_mtok == 5.00
    assert PRICING["claude-opus-4-8"] != PRICING.get("__retired__")
    assert PRICING["claude-opus-4-8"].input_per_mtok != 15.00


def test_estimate_uses_corrected_opus_pricing() -> None:
    # 1M uncached input tokens at the corrected $5/MTok Opus rate.
    breakdown = estimate("claude-opus-4-8", input_tokens=1_000_000, output_tokens=0)
    assert breakdown is not None
    assert breakdown.input_uncached == pytest.approx(5.00)


def test_estimate_unknown_model_returns_none() -> None:
    assert estimate("claude-mystery-9", input_tokens=1_000, output_tokens=1_000) is None
