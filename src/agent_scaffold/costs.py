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


# Public Anthropic pricing as of 2026-05.
PRICING: dict[str, ModelPricing] = {
    "claude-opus-4-7": ModelPricing(15.00, 75.00, 18.75, 1.50),
    "claude-opus-4-6": ModelPricing(15.00, 75.00, 18.75, 1.50),
    "claude-opus-4-5": ModelPricing(15.00, 75.00, 18.75, 1.50),
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
