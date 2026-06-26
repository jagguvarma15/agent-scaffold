"""Tests for agent_scaffold.progress and agent_scaffold.costs."""

from __future__ import annotations

import io
from typing import Any

import pytest
from rich.console import Console

from agent_scaffold.costs import estimate
from agent_scaffold.progress import (
    NullProgressDisplay,
    PlainProgressDisplay,
    ProgressEvent,
    RichProgressDisplay,
    _pre_fill_hint,
)


def test_null_progress_swallows_events() -> None:
    with NullProgressDisplay() as display:
        display.on_event(ProgressEvent("text_delta", "hi"))
        display.on_event(ProgressEvent("usage", {"input_tokens": 1, "output_tokens": 1}))


def test_null_progress_is_non_interactive_with_nullcontext_suspend() -> None:
    display = NullProgressDisplay()
    assert display.interactive is False
    # suspend() must be a usable (no-op) context manager.
    with display.suspend():
        pass


def test_rich_progress_suspend_pauses_live_for_a_prompt() -> None:
    """suspend() stops the Live panel so a prompt can print/read cleanly,
    then resumes — the fix for the diff/overwrite confirm deadlock."""
    console, buf = _capturing_console()
    with RichProgressDisplay(console, "claude-test") as display:
        display.on_event(ProgressEvent("file_detected", {"path": "a.py"}))
        with display.suspend():
            # Live is stopped here, so a direct print lands without the panel
            # fighting it for stdout (and a real prompt would own stdin).
            console.print("APPLY THESE CHANGES?")
        # Resumes cleanly; further events still render.
        display.on_event(ProgressEvent("file_detected", {"path": "b.py"}))
    assert "APPLY THESE CHANGES?" in buf.getvalue()


def test_rich_progress_interactive_requires_a_tty() -> None:
    # stdin is not a TTY under pytest, so the confirm-prompt gate stays off —
    # generation never blocks on input in CI / piped runs.
    console, _ = _capturing_console()
    assert RichProgressDisplay(console, "claude-test").interactive is False


def _capturing_console() -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    return Console(file=buf, force_terminal=True, width=120), buf


def _plain_console() -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    return Console(file=buf, force_terminal=False, width=200), buf


def test_plain_progress_prints_one_line_per_transition() -> None:
    console, buf = _plain_console()
    with PlainProgressDisplay(console) as display:
        display.on_event(
            ProgressEvent("operation_started", {"name": "generate", "hint": "model=m"})
        )
        display.on_event(ProgressEvent("text_delta", "noisy chunk that must not print"))
        display.on_event(ProgressEvent("heartbeat", 30))
        display.on_event(ProgressEvent("file_written", {"path": "src/a.py", "mode": "new"}))
        display.on_event(
            ProgressEvent("operation_done", {"name": "generate", "status": "ok", "summary": "ok"})
        )
    output = buf.getvalue()
    assert "generate: started (model=m)" in output
    assert "wrote src/a.py [new]" in output
    assert "generate: ok" in output
    assert "noisy chunk" not in output
    assert "30" not in output  # heartbeats are panel-only


def test_plain_progress_tracks_report_attributes() -> None:
    console, _buf = _plain_console()
    display = PlainProgressDisplay(console)
    display.on_event(ProgressEvent("operation_started", {"name": "validate"}))
    display.on_event(
        ProgressEvent("operation_done", {"name": "validate", "status": "warn", "summary": "lint"})
    )
    display.on_event(ProgressEvent("error", "stream broke"))
    assert "validate" in display.phase_durations
    assert display.warnings == ["validate: lint"]
    assert display.errors == ["stream broke"]


def test_plain_progress_redacts_secret_shaped_output() -> None:
    console, buf = _plain_console()
    display = PlainProgressDisplay(console)
    display.on_event(
        ProgressEvent(
            "operation_done",
            {"name": "wire", "status": "fail", "summary": "key sk-ant-api03-aaaabbbbcccc leaked"},
        )
    )
    output = buf.getvalue()
    assert "sk-ant-api03-aaaabbbbcccc" not in output
    assert "REDACTED" in output


def test_plain_progress_defaults_to_stderr_console() -> None:
    display = PlainProgressDisplay()
    assert display._console.stderr is True


def test_plain_progress_prints_bash_lines_redacted() -> None:
    console, buf = _plain_console()
    display = PlainProgressDisplay(console)
    display.on_event(ProgressEvent("bash_started", {"cmd": ["uv", "sync"]}))
    display.on_event(
        ProgressEvent(
            "bash_line",
            {"cmd": ["uv", "sync"], "line": "Installed 42 packages", "stream": "stdout"},
        )
    )
    display.on_event(
        ProgressEvent(
            "bash_line",
            {"cmd": ["uv", "sync"], "line": "token sk-ant-api03-aaaabbbbcccc", "stream": "stderr"},
        )
    )
    output = buf.getvalue()
    assert "| Installed 42 packages" in output
    assert "sk-ant-api03-aaaabbbbcccc" not in output
    assert "REDACTED" in output


def test_rich_progress_shows_bash_tail_under_active_op_and_clears_on_done() -> None:
    console, buf = _capturing_console()
    with RichProgressDisplay(console, "claude-test") as display:
        display.on_event(ProgressEvent("bash_started", {"cmd": ["uv", "sync"]}))
        for i in range(5):
            display.on_event(
                ProgressEvent(
                    "bash_line",
                    {"cmd": ["uv", "sync"], "line": f"step-{i}", "stream": "stdout"},
                )
            )
        mid_render = buf.getvalue()
        # Only the last 3 lines render while the command is active.
        assert "step-4" in mid_render
        assert "step-2" in mid_render
        display.on_event(ProgressEvent("bash_done", {"cmd": ["uv", "sync"], "exit_code": 0}))
    # After bash_done the tail is cleared from the final render.
    final_section = buf.getvalue().rsplit("Recent operations", 1)[-1]
    assert "exit 0" in final_section


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
    """B1 regression: heartbeats must NOT call ``console.print`` directly.

    Calling ``console.print`` while Rich Live is active flushes the current
    panel to scrollback and re-renders below — producing the four-stacked-
    panels artifact trial run 2 hit (one panel per heartbeat at 30/60/90/120s).
    The fix: heartbeat state is rendered into the panel via ``_render()``;
    no side-channel print happens.
    """
    console, _buf = _capturing_console()
    display = RichProgressDisplay(console, "claude-sonnet-4-6")
    # Capture *content* prints (strings / Text / Panel) and ignore Rich's
    # internal Control prints used for cursor movement during Live refresh.
    text_prints: list[Any] = []
    orig_print = console.print

    def _spy(*args: Any, **kwargs: Any) -> None:
        for a in args:
            if isinstance(a, str) and a:
                text_prints.append(a)
        orig_print(*args, **kwargs)

    console.print = _spy  # type: ignore[method-assign]
    with display:
        display.on_event(ProgressEvent("heartbeat", 30))
        display.on_event(ProgressEvent("heartbeat", 60))
        display.on_event(ProgressEvent("heartbeat", 90))
        display.on_event(ProgressEvent("heartbeat", 120))
    # Heartbeats must not have printed any text content directly — that's what
    # caused the stacked-panel artifact.
    assert not any("No streaming events" in p for p in text_prints)
    assert not any("heartbeat" in p.lower() for p in text_prints)
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


def test_rich_progress_two_panel_layout_shows_files_section() -> None:
    """P1: the right-hand panel shows the file count and per-file rows."""
    console, buf = _capturing_console()
    with RichProgressDisplay(console, "claude-opus-4-7", expected_files=3) as display:
        display.on_event(ProgressEvent("file_detected", "src/a.py"))
        display.on_event(ProgressEvent("file_detected", "src/b.py"))
        display.on_event(ProgressEvent("file_written", {"path": "src/a.py", "mode": "new"}))
    output = buf.getvalue()
    assert "Files" in output
    # File panel header reports detected/written counts.
    assert "2 detected" in output
    assert "1 written" in output


def test_rich_progress_file_written_event_flips_state_to_written() -> None:
    console, _buf = _capturing_console()
    display = RichProgressDisplay(console, "claude-opus-4-7")
    with display:
        display.on_event(ProgressEvent("file_detected", "src/x.py"))
        assert display._state.files["src/x.py"] == "detected"
        display.on_event(ProgressEvent("file_written", {"path": "src/x.py", "mode": "new"}))
        assert display._state.files["src/x.py"] == "written"
        display.on_event(ProgressEvent("file_written", {"path": "src/y.py", "mode": "overwrite"}))
        assert display._state.files["src/y.py"] == "overwritten"
        display.on_event(ProgressEvent("file_written", {"path": "src/z.py", "mode": "skip"}))
        assert display._state.files["src/z.py"] == "skipped"


def test_rich_progress_operations_log_and_phase_timings() -> None:
    console, buf = _capturing_console()
    display = RichProgressDisplay(console, "claude-opus-4-7")
    with display:
        display.on_event(ProgressEvent("operation_started", {"name": "generate"}))
        display.on_event(
            ProgressEvent(
                "operation_done",
                {"name": "generate", "status": "ok", "summary": "46 files"},
            )
        )
        display.on_event(ProgressEvent("operation_started", {"name": "validate"}))
        display.on_event(
            ProgressEvent(
                "operation_done",
                {"name": "validate", "status": "fail", "summary": "ruff exit 1"},
            )
        )
    output = buf.getvalue()
    assert "Recent operations" in output
    assert "generate" in output
    assert "46 files" in output
    assert "validate" in output
    # Phase timings populated for both ops; failed op surfaces in errors list.
    assert set(display.phase_durations.keys()) == {"generate", "validate"}
    assert any("validate" in e and "ruff exit 1" in e for e in display.errors)
    assert display.warnings == []


def test_rich_progress_operation_done_without_started_synthesizes_entry() -> None:
    """Defensive: operation_done arriving alone shouldn't crash and should still log."""
    console, _buf = _capturing_console()
    display = RichProgressDisplay(console, "claude-opus-4-7")
    with display:
        display.on_event(ProgressEvent("operation_done", {"name": "orphan", "status": "ok"}))
    assert "orphan" in display.phase_durations
    assert any(op.name == "orphan" for op in display._state.operations)


def test_rich_progress_bash_events_log_exit_status() -> None:
    console, buf = _capturing_console()
    display = RichProgressDisplay(console, "claude-opus-4-7")
    with display:
        display.on_event(ProgressEvent("bash_started", {"cmd": ["ruff", "check", "--fix"]}))
        display.on_event(
            ProgressEvent("bash_done", {"cmd": ["ruff", "check", "--fix"], "exit_code": 0})
        )
        display.on_event(ProgressEvent("bash_started", {"cmd": ["ruff", "format"]}))
        display.on_event(ProgressEvent("bash_done", {"cmd": ["ruff", "format"], "exit_code": 1}))
    output = buf.getvalue()
    assert "ruff check --fix" in output
    assert "exit 0" in output
    assert "exit 1" in output


def test_null_progress_display_exposes_empty_summaries() -> None:
    """The CLI reads phase_durations/warnings/errors from the display unconditionally."""
    display = NullProgressDisplay()
    assert display.phase_durations == {}
    assert display.warnings == []
    assert display.errors == []


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


def test_quiet_terminal_input_noop_without_tty(monkeypatch: Any) -> None:
    from agent_scaffold.progress import _quiet_terminal_input

    class _NoTTY:
        def isatty(self) -> bool:
            return False

    monkeypatch.setattr("sys.stdin", _NoTTY())
    # No terminal -> the guard must do nothing and never touch termios.
    with _quiet_terminal_input():
        pass


def test_quiet_terminal_input_mutes_echo_keeps_signals_and_restores(monkeypatch: Any) -> None:
    termios = pytest.importorskip("termios")
    from agent_scaffold.progress import _quiet_terminal_input

    class _TTY:
        def isatty(self) -> bool:
            return True

        def fileno(self) -> int:
            return 0

    saved_attrs = [0, 0, 0, termios.ECHO | termios.ICANON | termios.ISIG, 0, 0, []]
    set_calls: list[list[Any]] = []
    flushed: list[int] = []

    monkeypatch.setattr("sys.stdin", _TTY())
    monkeypatch.setattr(termios, "tcgetattr", lambda fd: list(saved_attrs))
    monkeypatch.setattr(termios, "tcsetattr", lambda fd, when, attrs: set_calls.append(list(attrs)))
    monkeypatch.setattr(termios, "tcflush", lambda fd, queue: flushed.append(queue))

    with _quiet_terminal_input():
        muted = set_calls[0]
        assert not muted[3] & termios.ECHO  # echo off — keystrokes don't print
        assert not muted[3] & termios.ICANON  # line editing off — no injected newlines
        assert muted[3] & termios.ISIG  # signals kept — Ctrl-C still aborts

    assert flushed == [termios.TCIFLUSH]  # buffered keystrokes discarded on exit
    assert set_calls[-1] == saved_attrs  # original terminal mode restored
