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


# A rough chars/token approximation matching context.py.
_CHARS_PER_TOKEN = 4


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

    def on_event(self, event: ProgressEvent) -> None:
        state = self._state
        state.last_event_at = time.monotonic()
        if event.kind == "text_delta":
            text = str(event.payload or "")
            state.text_buffer += text
            state.text_tokens += max(1, len(text) // _CHARS_PER_TOKEN)
            self._scan_for_new_files()
        elif event.kind == "thinking_delta":
            text = str(event.payload or "")
            state.thinking_tokens += max(1, len(text) // _CHARS_PER_TOKEN)
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
            # Surface a transient warning beneath the panel.
            self._console.print(
                f"[yellow]No streaming events for {int(event.payload or 0)}s — "
                "Opus may still be processing the input prompt[/]"
            )
        elif event.kind == "file_emitted":
            path = str(event.payload or "")
            if path and path not in state.files_seen:
                state.files_seen.append(path)
        elif event.kind == "error":
            self._console.print(f"[red]stream error:[/] {event.payload}")
        # done events fall through; the final render in __exit__ handles them.
        self._live.update(self._render())

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

        thinking_line = f"Thinking: ~{s.thinking_tokens:,} tokens"
        files_part = ""
        if self._expected_files:
            files_part = f"  ({len(s.files_seen)}/{self._expected_files} files)"
        else:
            files_part = f"  ({len(s.files_seen)} files)" if s.files_seen else ""
        output_line = f"Output:   ~{s.text_tokens:,} tokens{files_part}"

        lines = [
            Text.from_markup(f"[bold]Generating[/] with {self._model}  [elapsed {elapsed_str}]"),
            Text(""),
            Text(thinking_line),
            Text(output_line),
            Text(cache_line),
        ]
        if self._verbose and s.text_buffer:
            tail = s.text_buffer[-300:].replace("\n", " ")
            lines.append(Text(""))
            lines.append(Text.from_markup(f"[dim]…{tail}[/]"))

        body = Text("\n").join(lines)
        return Panel(body, title="Generation progress", expand=False)

    @property
    def seconds_since_last_event(self) -> float:
        return time.monotonic() - self._state.last_event_at
