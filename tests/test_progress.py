"""Tests for agent_scaffold.progress and agent_scaffold.costs."""

from __future__ import annotations

import io

from rich.console import Console

from agent_scaffold.costs import estimate
from agent_scaffold.progress import (
    NullProgressDisplay,
    ProgressEvent,
    RichProgressDisplay,
)


def test_null_progress_swallows_events() -> None:
    with NullProgressDisplay() as display:
        display.on_event(ProgressEvent("text_delta", "hi"))
        display.on_event(ProgressEvent("usage", {"input_tokens": 1, "output_tokens": 1}))


def _capturing_console() -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    return Console(file=buf, force_terminal=True, width=120), buf


def test_rich_progress_renders_model_and_counts() -> None:
    console, buf = _capturing_console()
    with RichProgressDisplay(console, "claude-opus-4-7", expected_files=3) as display:
        display.on_event(ProgressEvent("thinking_delta", "lots of thought" * 50))
        display.on_event(
            ProgressEvent(
                "text_delta",
                '{"files": [{"path": "src/a.py", "content": "x"}, {"path": "src/b.py"',
            )
        )
        display.on_event(
            ProgressEvent(
                "usage",
                {
                    "input_tokens": 1000,
                    "output_tokens": 200,
                    "cache_read_input_tokens": 800,
                    "cache_creation_input_tokens": 0,
                },
            )
        )
    output = buf.getvalue()
    assert "claude-opus-4-7" in output
    # Per-file detection should have surfaced both paths in the buffer.
    assert "2/3 files" in output or "2 files" in output


def test_rich_progress_emits_heartbeat_warning() -> None:
    console, buf = _capturing_console()
    with RichProgressDisplay(console, "claude-sonnet-4-6") as display:
        display.on_event(ProgressEvent("heartbeat", 45))
    assert "No streaming events for 45s" in buf.getvalue()


def test_cost_estimate_known_model() -> None:
    breakdown = estimate(
        "claude-sonnet-4-6",
        input_tokens=10_000,
        output_tokens=2_000,
        cache_read_tokens=8_000,
    )
    assert breakdown is not None
    # uncached input = 2000 tokens at $3/M, output = 2000 at $15/M, cache reads = 8000 at $0.30/M.
    assert round(breakdown.input_uncached, 4) == round(2_000 * 3.00 / 1_000_000, 4)
    assert round(breakdown.output, 4) == round(2_000 * 15.00 / 1_000_000, 4)
    assert round(breakdown.cache_read, 4) == round(8_000 * 0.30 / 1_000_000, 4)
    assert breakdown.total > 0


def test_cost_estimate_unknown_model_returns_none() -> None:
    assert estimate("not-a-real-model", input_tokens=10, output_tokens=10) is None
