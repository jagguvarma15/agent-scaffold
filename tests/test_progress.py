"""Tests for agent_scaffold.progress and agent_scaffold.costs."""

from __future__ import annotations

import io

from rich.console import Console

from agent_scaffold.costs import estimate
from agent_scaffold.progress import (
    NullProgressDisplay,
    ProgressEvent,
    RichProgressDisplay,
    _pre_fill_hint,
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


def test_rich_progress_renders_heartbeat_inside_panel_not_via_print() -> None:
    """B1 regression: heartbeats must render inside the Live panel.

    Calling ``console.print`` while Live is active flushes the current panel
    to scrollback and re-renders below, producing the stacked-panel artifact
    that trial run 2 hit. We exercise that exact sequence (four heartbeats
    arriving 30s apart) and assert the captured output contains the panel
    title exactly **once** — a stacked-panel bug would print it four times.
    """
    console, buf = _capturing_console()
    display = RichProgressDisplay(console, "claude-sonnet-4-6")
    with display:
        display.on_event(ProgressEvent("heartbeat", 30))
        display.on_event(ProgressEvent("heartbeat", 60))
        display.on_event(ProgressEvent("heartbeat", 90))
        display.on_event(ProgressEvent("heartbeat", 120))
    output = buf.getvalue()
    assert output.count("Generation progress") == 1
    # Only the latest heartbeat value is held in state — earlier ones were
    # overwritten, not appended.
    assert display._state.heartbeat_silence == 120


def test_rich_progress_clears_heartbeat_after_real_event() -> None:
    console, _buf = _capturing_console()
    display = RichProgressDisplay(console, "claude-opus-4-7")
    with display:
        display.on_event(ProgressEvent("heartbeat", 60))
        assert display._state.heartbeat_silence == 60
        display.on_event(ProgressEvent("thinking_delta", "thought"))
        assert display._state.heartbeat_silence is None


def test_rich_progress_error_deferred_until_exit() -> None:
    """B1: error events are captured and printed only after Live has stopped."""
    console, buf = _capturing_console()
    display = RichProgressDisplay(console, "claude-opus-4-7")
    with display:
        display.on_event(ProgressEvent("error", "boom"))
        # While Live is still active, nothing should hit stdout outside the
        # panel — the captured buffer must not yet contain the error string
        # as a standalone red print. We can't easily diff inner vs outer at
        # this point, so just assert the deferred print fires on __exit__.
        assert display._state.last_error == "boom"
    assert "boom" in buf.getvalue()


def test_pre_fill_hint_buckets() -> None:
    assert "~5s" in _pre_fill_hint(5_000, thinking_enabled=False)
    assert "~15s" in _pre_fill_hint(40_000, thinking_enabled=False)
    assert "~30s" in _pre_fill_hint(40_000, thinking_enabled=True)
    assert "60–180s" in _pre_fill_hint(99_000, thinking_enabled=True)
    assert "120–300s" in _pre_fill_hint(150_000, thinking_enabled=True)
    assert "max-context-tokens" in _pre_fill_hint(150_000, thinking_enabled=True)


def test_rich_progress_stream_started_renders_pre_fill_hint() -> None:
    console, buf = _capturing_console()
    with RichProgressDisplay(console, "claude-opus-4-7") as display:
        display.on_event(
            ProgressEvent(
                "stream_started",
                {
                    "input_tokens_estimate": 99_000,
                    "thinking_enabled": True,
                    "model": "claude-opus-4-7",
                },
            )
        )
    output = buf.getvalue()
    assert "Status:" in output
    assert "pre-fill" in output
    assert "60" in output  # bucket includes "60–180s typical"


def test_rich_progress_pre_fill_cleared_on_first_delta() -> None:
    console, _buf = _capturing_console()
    display = RichProgressDisplay(console, "claude-opus-4-7")
    with display:
        display.on_event(
            ProgressEvent(
                "stream_started",
                {"input_tokens_estimate": 80_000, "thinking_enabled": True},
            )
        )
        assert display._state.pre_fill_message is not None
        display.on_event(ProgressEvent("thinking_delta", "starting to think..."))
        # First delta arrived: pre-fill hint must be cleared so the panel
        # switches to live counter display.
        assert display._state.pre_fill_message is None
        assert display._state.first_delta_received is True


def test_rich_progress_verbose_renders_deltas_tail() -> None:
    console, buf = _capturing_console()
    with RichProgressDisplay(console, "claude-opus-4-7", verbose=True) as display:
        display.on_event(ProgressEvent("text_delta", "some emitted text payload here"))
    output = buf.getvalue()
    assert "emitted text payload" in output


def test_rich_progress_non_verbose_omits_deltas_tail() -> None:
    console, buf = _capturing_console()
    with RichProgressDisplay(console, "claude-opus-4-7", verbose=False) as display:
        display.on_event(ProgressEvent("text_delta", "uniqueXYZpayload"))
    assert "uniqueXYZpayload" not in buf.getvalue()


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
