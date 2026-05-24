"""Real-time progress display for generation runs.

The generator emits ``ProgressEvent`` instances as it iterates the Anthropic
stream. A display (``RichProgressDisplay``, ``NullProgressDisplay``) consumes
them and updates the user-facing output. Splitting events from display keeps
the generator I/O-free and unit-testable without a TTY.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

EventKind = Literal[
    "thinking_delta",
    "text_delta",
    "usage",
    "file_emitted",
    "heartbeat",
    "stream_started",
    "done",
    "error",
]


@dataclass
class ProgressEvent:
    kind: EventKind
    payload: Any = None


class ProgressSink(Protocol):
    def on_event(self, event: ProgressEvent) -> None: ...


class NullProgressDisplay:
    """No-op sink used by tests and non-interactive runs."""

    def __enter__(self) -> NullProgressDisplay:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def on_event(self, event: ProgressEvent) -> None:  # noqa: ARG002
        return None


# Cheap incremental detector for ``"path": "<file>"`` fields in the JSON
# output contract. We match on the *files* array entries; the contract layer
# does the real parsing. This is just a UX signal.
_FILE_PATH_RE = re.compile(r'"path"\s*:\s*"((?:[^"\\]|\\.)+)"')


@dataclass
class _State:
    text_buffer: str = ""
    text_tokens: int = 0
    thinking_tokens: int = 0
    files_seen: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    last_event_at: float = field(default_factory=time.monotonic)
    started_at: float = field(default_factory=time.monotonic)
    # Heartbeat silence (set on heartbeat, cleared on any other event). Rendered
    # into the panel rather than printed separately so Rich Live keeps exclusive
    # ownership of stdout.
    heartbeat_silence: int | None = None
    # Pre-fill hint state, populated by a one-shot ``stream_started`` event from
    # the generator. Cleared as soon as the first thinking/text delta arrives.
    pre_fill_message: str | None = None
    first_delta_received: bool = False
    # Final error string, captured during stream and printed in ``__exit__``
    # (after Live has stopped, so it doesn't fight the panel).
    last_error: str | None = None


# A rough chars/token approximation matching context.py.
_CHARS_PER_TOKEN = 4


def _pre_fill_hint(input_tokens_estimate: int, thinking_enabled: bool) -> str:
    """Return a human-readable wait estimate for the model's pre-fill phase.

    Buckets are coarse — the goal is just to tell the user "this is normal,
    not stuck" while the model chews through input + adaptive thinking before
    emitting the first ``content_block_delta``.
    """
    n = input_tokens_estimate
    if n < 20_000 and not thinking_enabled:
        return "pre-fill (first event in ~5s)"
    if n < 60_000 and not thinking_enabled:
        return "pre-fill (first event in ~15s)"
    if n < 60_000:
        return "pre-fill (first event in ~30s)"
    if n <= 100_000:
        return "pre-fill (first event in 60–180s typical)"
    return "pre-fill (first event in 120–300s; consider lowering --max-context-tokens)"


class RichProgressDisplay:
    """Drive a Rich Live panel while the model streams output."""

    def __init__(
        self,
        console: Console,
        model: str,
        *,
        verbose: bool = False,
        expected_files: int | None = None,
        refresh_per_second: float = 4.0,
    ) -> None:
        self._console = console
        self._model = model
        self._verbose = verbose
        self._expected_files = expected_files
        self._state = _State()
        self._live = Live(
            self._render(),
            console=console,
            refresh_per_second=refresh_per_second,
            transient=False,
        )

    def __enter__(self) -> RichProgressDisplay:
        self._live.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        # Final render so the last state is what the user sees.
        try:
            self._live.update(self._render(), refresh=True)
        finally:
            self._live.__exit__(*args)
        # Now that Live has released stdout, surface any error captured during
        # the stream. Doing this inside __exit__ would tear Live's panel.
        if self._state.last_error is not None:
            self._console.print(f"[red]stream error:[/] {self._state.last_error}")

    def on_event(self, event: ProgressEvent) -> None:
        state = self._state
        now = time.monotonic()
        # Any event other than the heartbeat itself means the stream is alive;
        # clear the silence warning so the panel reflects current state.
        if event.kind != "heartbeat":
            state.last_event_at = now
            state.heartbeat_silence = None
        if event.kind == "text_delta":
            text = str(event.payload or "")
            state.text_buffer += text
            state.text_tokens += max(1, len(text) // _CHARS_PER_TOKEN)
            state.first_delta_received = True
            state.pre_fill_message = None
            self._scan_for_new_files()
        elif event.kind == "thinking_delta":
            text = str(event.payload or "")
            state.thinking_tokens += max(1, len(text) // _CHARS_PER_TOKEN)
            state.first_delta_received = True
            state.pre_fill_message = None
        elif event.kind == "usage":
            payload = event.payload or {}
            state.input_tokens = int(payload.get("input_tokens", state.input_tokens) or 0)
            state.output_tokens = int(payload.get("output_tokens", state.output_tokens) or 0)
            state.cache_read_tokens = int(
                payload.get("cache_read_input_tokens", state.cache_read_tokens) or 0
            )
            state.cache_write_tokens = int(
                payload.get("cache_creation_input_tokens", state.cache_write_tokens) or 0
            )
        elif event.kind == "heartbeat":
            # Render into the panel rather than calling console.print: a direct
            # print while Live is active forces Live to flush its panel to
            # scrollback and re-render below, which produced the stacked-panel
            # artifact in trial run 2.
            state.heartbeat_silence = int(event.payload or 0)
        elif event.kind == "stream_started":
            payload = event.payload or {}
            input_estimate = int(payload.get("input_tokens_estimate", 0) or 0)
            thinking_enabled = bool(payload.get("thinking_enabled", False))
            state.pre_fill_message = _pre_fill_hint(input_estimate, thinking_enabled)
        elif event.kind == "file_emitted":
            path = str(event.payload or "")
            if path and path not in state.files_seen:
                state.files_seen.append(path)
        elif event.kind == "error":
            # Defer the actual print to __exit__ so we don't break Live's
            # exclusive ownership of stdout. Keep the latest error.
            state.last_error = str(event.payload)
        # done events fall through; the final render in __exit__ handles them.
        self._live.update(self._render(), refresh=True)

    def _scan_for_new_files(self) -> None:
        seen = set(self._state.files_seen)
        for match in _FILE_PATH_RE.finditer(self._state.text_buffer):
            path = match.group(1)
            if path in seen:
                continue
            seen.add(path)
            self._state.files_seen.append(path)

    def _render(self) -> Panel:
        s = self._state
        elapsed = int(time.monotonic() - s.started_at)
        mins, secs = divmod(elapsed, 60)
        elapsed_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"

        cache_total = s.cache_read_tokens + s.cache_write_tokens
        if s.input_tokens or cache_total:
            denom = max(1, s.input_tokens + cache_total)
            pct = int(100 * s.cache_read_tokens / denom)
            cache_line = (
                f"Cache:    {s.cache_read_tokens:,} read / "
                f"{s.cache_write_tokens:,} write ({pct}% hit)"
            )
        else:
            cache_line = "Cache:    (waiting for first usage event)"

        lines: list[Text] = [
            Text.from_markup(f"[bold]Generating[/] with {self._model}  [elapsed {elapsed_str}]"),
            Text(""),
        ]
        if s.pre_fill_message is not None and not s.first_delta_received:
            # Pre-fill phase: model is processing input + thinking but has not
            # yet emitted any content_block_delta. Show a contextual hint that
            # tells the user this is normal, not a hang.
            lines.append(Text.from_markup(f"Status:   [yellow]{s.pre_fill_message}[/]"))
            lines.append(Text("Thinking: not yet"))
            lines.append(Text("Output:   not yet"))
            lines.append(Text(cache_line))
        else:
            thinking_line = f"Thinking: ~{s.thinking_tokens:,} tokens"
            if self._expected_files:
                files_part = f"  ({len(s.files_seen)}/{self._expected_files} files)"
            elif s.files_seen:
                files_part = f"  ({len(s.files_seen)} files)"
            else:
                files_part = ""
            output_line = f"Output:   ~{s.text_tokens:,} tokens{files_part}"
            lines.append(Text(thinking_line))
            lines.append(Text(output_line))
            lines.append(Text(cache_line))

        if s.heartbeat_silence is not None:
            lines.append(
                Text.from_markup(
                    f"[yellow]⚠ No streaming events for {s.heartbeat_silence}s — "
                    "model may be in pre-fill phase[/]"
                )
            )
        if self._verbose and s.text_buffer:
            tail = s.text_buffer[-300:].replace("\n", " ")
            lines.append(Text(""))
            lines.append(Text.from_markup(f"[dim]…{tail}[/]"))

        body = Text("\n").join(lines)
        return Panel(body, title="Generation progress", expand=False)

    @property
    def seconds_since_last_event(self) -> float:
        return time.monotonic() - self._state.last_event_at
