"""Tests for the ``prompt_to_raise_context_cap`` helper.

The helper is the bridge between ``ContextBudgetError`` and the
wizard/REPL: it decides whether a one-shot bump to high effort's 100k cap
is offered, captures the user's y/N, and returns the new cap pair (or
``None`` so the caller re-raises).
"""

from __future__ import annotations

from io import StringIO
from unittest.mock import patch

import pytest
from rich.console import Console

from agent_scaffold.cli_shared import prompt_to_raise_context_cap
from agent_scaffold.context import ContextBudgetError
from agent_scaffold.effort import EFFORT_PRESETS


def _err(essentials: int, cap: int = 60_000) -> ContextBudgetError:
    return ContextBudgetError(
        f"essentials are ~{essentials} tokens, exceeding cap {cap}",
        essentials_tokens=essentials,
        current_cap=cap,
    )


def _make_console() -> Console:
    """Console writing to an in-memory buffer so prints don't pollute pytest."""
    return Console(file=StringIO(), force_terminal=False, width=120)


def test_accept_returns_high_preset_cap_and_per_doc() -> None:
    """Default-yes (bare Enter) returns high preset's (100k, 12k)."""
    console = _make_console()
    high = EFFORT_PRESETS["high"]
    with patch.object(Console, "input", return_value=""):
        result = prompt_to_raise_context_cap(console, _err(essentials=66_000))
    assert result == (high.max_context_tokens, high.max_tokens_per_doc)


@pytest.mark.parametrize("answer", ["y", "Y", "yes", "YES"])
def test_explicit_yes_accepts(answer: str) -> None:
    console = _make_console()
    high = EFFORT_PRESETS["high"]
    with patch.object(Console, "input", return_value=answer):
        result = prompt_to_raise_context_cap(console, _err(essentials=66_000))
    assert result == (high.max_context_tokens, high.max_tokens_per_doc)


@pytest.mark.parametrize("answer", ["n", "N", "no", "anything-else"])
def test_decline_returns_none(answer: str) -> None:
    console = _make_console()
    with patch.object(Console, "input", return_value=answer):
        result = prompt_to_raise_context_cap(console, _err(essentials=66_000))
    assert result is None


def test_non_interactive_returns_none_without_prompting() -> None:
    """CI must not be silently bumped — caller has to pass --max-context-tokens."""
    console = _make_console()
    with patch.object(Console, "input") as mock_input:
        result = prompt_to_raise_context_cap(console, _err(essentials=66_000), non_interactive=True)
    assert result is None
    mock_input.assert_not_called()


def test_essentials_above_high_cap_skips_prompt() -> None:
    """If even high's 100k can't fit the essentials, no bump is offered."""
    console = _make_console()
    high = EFFORT_PRESETS["high"]
    with patch.object(Console, "input") as mock_input:
        result = prompt_to_raise_context_cap(console, _err(essentials=high.max_context_tokens + 1))
    assert result is None
    mock_input.assert_not_called()


def test_decline_message_includes_essentials_size() -> None:
    """User needs to see how big the recipe actually is to decide."""
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, width=120)
    with patch.object(Console, "input", return_value="n"):
        prompt_to_raise_context_cap(console, _err(essentials=66_769, cap=60_000))
    output = buffer.getvalue()
    assert "66,769" in output
    assert "60,000" in output
