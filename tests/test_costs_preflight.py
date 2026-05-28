"""Tests for ``agent_scaffold.costs.estimate_preflight``.

Pre-generation cost estimation. Input cost is exact (token count is known
from the assembled context); output is a range bracket since the response
length isn't known until the LLM runs.
"""

from __future__ import annotations

import pytest

from agent_scaffold.costs import PreflightCost, estimate_preflight


def test_known_model_returns_exact_input_cost() -> None:
    pre = estimate_preflight("claude-sonnet-4-6", input_tokens=10_000, output_range=(8_000, 32_000))
    assert pre is not None
    # Sonnet input is $3/MTok → 10k tokens = $0.03.
    assert pre.input_cost == pytest.approx(0.03)
    # Output low = 8k × $15/MTok = $0.12; high = 32k × $15/MTok = $0.48.
    assert pre.output_cost_low == pytest.approx(0.12)
    assert pre.output_cost_high == pytest.approx(0.48)
    assert pre.total_low == pytest.approx(0.15)
    assert pre.total_high == pytest.approx(0.51)
    assert pre.total_low < pre.total_high


def test_unknown_model_returns_none() -> None:
    assert estimate_preflight("claude-mystery-7", input_tokens=1000) is None


def test_cache_read_tokens_reduce_input_cost() -> None:
    # 10k input tokens with 8k served from cache (Opus: $15/MTok fresh, $1.50 cached).
    pre = estimate_preflight("claude-opus-4-7", input_tokens=10_000, cache_read_tokens=8_000)
    assert pre is not None
    # Fresh portion (2k) → $0.03; cached (8k × $1.50/MTok) → $0.012.
    assert pre.input_cost == pytest.approx(0.042)
    # cache_savings = full price ($0.15) minus what we actually pay ($0.042) = $0.108.
    assert pre.cache_savings == pytest.approx(0.108)


def test_zero_input_tokens_produces_zero_input_cost() -> None:
    pre = estimate_preflight("claude-haiku-4-5-20251001", input_tokens=0)
    assert pre is not None
    assert pre.input_cost == pytest.approx(0.0)
    assert pre.total_low > 0  # output cost still nonzero


def test_output_range_swapped_is_normalized() -> None:
    """Caller passing (high, low) by accident shouldn't yield negative spread."""
    pre = estimate_preflight("claude-sonnet-4-6", input_tokens=1_000, output_range=(32_000, 8_000))
    assert pre is not None
    assert pre.output_cost_low < pre.output_cost_high


def test_format_one_liner_shape() -> None:
    pre = estimate_preflight("claude-sonnet-4-6", input_tokens=10_000, output_range=(8_000, 32_000))
    assert pre is not None
    rendered = pre.format()
    assert rendered.startswith("$")
    assert "input $" in rendered
    assert "output ~$" in rendered
    assert "±$" in rendered


def test_preflight_cost_is_namedtuple_with_model() -> None:
    pre = estimate_preflight("claude-sonnet-4-6", input_tokens=1_000)
    assert isinstance(pre, PreflightCost)
    assert pre is not None
    assert pre.model == "claude-sonnet-4-6"
    assert pre.input_tokens == 1_000
