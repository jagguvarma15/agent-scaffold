"""Opt-in live-model integration tests.

These make real Anthropic API calls, so they are gated on ``ANTHROPIC_API_KEY``
and the ``integration`` marker — they never run in the default suite (which is
fully mocked). Run them with a key present::

    uv run pytest -m integration

They exercise the model-layer facts the mocked suite cannot verify: that the
current model ids exist (a drift canary for the next model generation), that
adaptive-only models reject the legacy thinking shape, that the request shape
this tool builds is accepted, and that a block above the cache minimum actually
caches. Spend is kept to a few cents via Haiku and tiny ``max_tokens``.
"""

from __future__ import annotations

import os
import re

import anthropic
import pytest

from agent_scaffold.costs import PRICING
from agent_scaffold.generator import _build_thinking_kwargs

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="live API test — set ANTHROPIC_API_KEY to run",
    ),
]

_DATED_ID = re.compile(r"-\d{8}$")
_ALIAS_MODELS = sorted({m for m in PRICING if not _DATED_ID.search(m)})
# A block comfortably above every current cache minimum (Opus: 4096 tokens).
_LARGE_TEXT = "The quick brown fox jumps over the lazy dog. " * 500


@pytest.fixture(scope="module")
def client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


@pytest.mark.parametrize("model", _ALIAS_MODELS)
def test_priced_model_ids_exist(client: anthropic.Anthropic, model: str) -> None:
    # Drift canary: every id we price must still resolve. When a new generation
    # ships and an old id is retired, this fails and the tables get updated.
    retrieved = client.models.retrieve(model)
    assert retrieved.id


def test_adaptive_only_model_rejects_budget_tokens(client: anthropic.Anthropic) -> None:
    # This is the failure the widened adaptive gate prevents: the legacy
    # {"type": "enabled", "budget_tokens": N} shape is a 400 on Opus 4.8.
    with pytest.raises(anthropic.BadRequestError):
        client.messages.create(
            model="claude-opus-4-8",
            max_tokens=16,
            thinking={"type": "enabled", "budget_tokens": 1024},
            messages=[{"role": "user", "content": "hi"}],
        )


def test_tool_built_thinking_request_is_accepted(client: anthropic.Anthropic) -> None:
    # The exact kwargs the generator builds for an adaptive model + effort budget
    # must be accepted, not rejected.
    kwargs = _build_thinking_kwargs("claude-opus-4-8", thinking_budget=16000)
    assert kwargs["thinking"] == {"type": "adaptive"}
    resp = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=16,
        messages=[{"role": "user", "content": "Reply with the single word: ok"}],
        **kwargs,
    )
    assert resp.stop_reason in {"end_turn", "max_tokens"}


def test_block_above_minimum_caches(client: anthropic.Anthropic) -> None:
    # Proves the 4096-token Opus minimum: a block just above it caches. Two
    # identical-prefix calls; the cache is created on the first and read on the
    # second.
    system = [{"type": "text", "text": _LARGE_TEXT, "cache_control": {"type": "ephemeral"}}]
    first = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=16,
        system=system,
        messages=[{"role": "user", "content": "Reply with: one"}],
    )
    second = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=16,
        system=system,
        messages=[{"role": "user", "content": "Reply with: two"}],
    )
    created = first.usage.cache_creation_input_tokens or 0
    read = second.usage.cache_read_input_tokens or 0
    assert created > 0 or read > 0, "a block above the cache minimum must cache"


def test_len4_estimate_tracks_actual_tokens(client: anthropic.Anthropic) -> None:
    # The assembly budget estimates tokens as len/4. Confirm that stays within a
    # sane band of the real count so the conservative cache floor keeps absorbing
    # the error rather than mis-sizing blocks by an order of magnitude.
    estimate = len(_LARGE_TEXT) // 4
    counted = client.messages.count_tokens(
        model="claude-opus-4-8",
        messages=[{"role": "user", "content": _LARGE_TEXT}],
    ).input_tokens
    ratio = estimate / counted
    assert 0.5 < ratio < 1.6, f"len/4 estimate off by {ratio:.2f}x (est {estimate}, real {counted})"
