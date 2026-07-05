"""Public-pricing cost estimation for Anthropic models.

Prices are dollars per million tokens. Update when Anthropic releases new
models or changes pricing. The estimator is best-effort — its only purpose is
to give the user a feel for spend after each run.
"""

from __future__ import annotations

from typing import NamedTuple


class ModelPricing(NamedTuple):
    input_per_mtok: float
    output_per_mtok: float
    cache_write_per_mtok: float
    cache_read_per_mtok: float


# Public Anthropic pricing as of 2026-07. Opus 4.5 dropped the Opus tier to
# $5/$25 (the old $15/$75 rate belongs to Opus 4.1 and earlier). Sonnet 5 is
# listed at sticker price, not the introductory rate that expires 2026-08-31.
PRICING: dict[str, ModelPricing] = {
    "claude-fable-5": ModelPricing(10.00, 50.00, 12.50, 1.00),
    "claude-opus-4-8": ModelPricing(5.00, 25.00, 6.25, 0.50),
    "claude-opus-4-7": ModelPricing(5.00, 25.00, 6.25, 0.50),
    "claude-opus-4-6": ModelPricing(5.00, 25.00, 6.25, 0.50),
    "claude-opus-4-5": ModelPricing(5.00, 25.00, 6.25, 0.50),
    "claude-sonnet-5": ModelPricing(3.00, 15.00, 3.75, 0.30),
    "claude-sonnet-4-6": ModelPricing(3.00, 15.00, 3.75, 0.30),
    "claude-sonnet-4-5": ModelPricing(3.00, 15.00, 3.75, 0.30),
    "claude-haiku-4-5-20251001": ModelPricing(1.00, 5.00, 1.25, 0.10),
    "claude-haiku-4-5": ModelPricing(1.00, 5.00, 1.25, 0.10),
}


def _resolve_pricing(model: str) -> ModelPricing | None:
    if model in PRICING:
        return PRICING[model]
    # Best-effort substring match for date-suffixed model ids.
    for key, price in PRICING.items():
        if model.startswith(key) or key.startswith(model):
            return price
    return None


class CostBreakdown(NamedTuple):
    total: float
    input_uncached: float
    output: float
    cache_write: float
    cache_read: float


def estimate(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> CostBreakdown | None:
    """Estimate USD cost for one Anthropic call. Returns ``None`` if model unknown."""
    price = _resolve_pricing(model)
    if price is None:
        return None
    # input_tokens is the *uncached* portion (cache reads/writes are separate).
    fresh_input = max(0, input_tokens - cache_read_tokens - cache_write_tokens)
    input_cost = fresh_input * price.input_per_mtok / 1_000_000
    output_cost = output_tokens * price.output_per_mtok / 1_000_000
    write_cost = cache_write_tokens * price.cache_write_per_mtok / 1_000_000
    read_cost = cache_read_tokens * price.cache_read_per_mtok / 1_000_000
    return CostBreakdown(
        total=input_cost + output_cost + write_cost + read_cost,
        input_uncached=input_cost,
        output=output_cost,
        cache_write=write_cost,
        cache_read=read_cost,
    )


# Range used when the caller has no specific output expectation. 8k is
# typical for "small project, no thinking budget"; 32k covers Opus +
# thinking on a chunky recipe. The user sees both bounds so they can
# anticipate the worst case.
_DEFAULT_OUTPUT_RANGE: tuple[int, int] = (8_000, 32_000)


class PreflightCost(NamedTuple):
    """Pre-generation cost estimate.

    Input cost is exact (token count is known from the assembled context);
    output is a range bracket since the LLM hasn't run yet. ``cache_savings``
    is the dollar amount saved by reading from the prompt cache rather than
    paying full input price.
    """

    model: str
    input_tokens: int
    input_cost: float
    output_cost_low: float
    output_cost_high: float
    cache_savings: float
    total_low: float
    total_high: float

    def format(self) -> str:
        """One-liner for the plan panel: ``$0.92 (input $0.42, output ~$0.50 ±20%)``."""
        midpoint = (self.total_low + self.total_high) / 2
        spread = (self.total_high - self.total_low) / 2
        output_mid = (self.output_cost_low + self.output_cost_high) / 2
        return (
            f"${midpoint:.2f} (input ${self.input_cost:.2f}, "
            f"output ~${output_mid:.2f} ±${spread:.2f})"
        )


def estimate_preflight(
    model: str,
    *,
    input_tokens: int,
    output_range: tuple[int, int] = _DEFAULT_OUTPUT_RANGE,
    cache_read_tokens: int = 0,
) -> PreflightCost | None:
    """Pre-generation cost estimate. Returns ``None`` for unknown models.

    ``input_tokens`` is the assembled context size (known before the call).
    ``output_range`` brackets the unknown response length; widen for runs
    that demand large outputs (e.g. ``--effort high`` with thinking).
    ``cache_read_tokens`` discounts the input cost for tokens we expect the
    prompt cache to hit.
    """
    price = _resolve_pricing(model)
    if price is None:
        return None
    low_out, high_out = output_range
    if low_out > high_out:
        low_out, high_out = high_out, low_out

    fresh_input = max(0, input_tokens - cache_read_tokens)
    input_cost = fresh_input * price.input_per_mtok / 1_000_000
    cached_cost = cache_read_tokens * price.cache_read_per_mtok / 1_000_000
    full_input_cost = input_tokens * price.input_per_mtok / 1_000_000
    cache_savings = max(0.0, full_input_cost - (input_cost + cached_cost))

    out_low = low_out * price.output_per_mtok / 1_000_000
    out_high = high_out * price.output_per_mtok / 1_000_000

    return PreflightCost(
        model=model,
        input_tokens=input_tokens,
        input_cost=input_cost + cached_cost,
        output_cost_low=out_low,
        output_cost_high=out_high,
        cache_savings=cache_savings,
        total_low=input_cost + cached_cost + out_low,
        total_high=input_cost + cached_cost + out_high,
    )
