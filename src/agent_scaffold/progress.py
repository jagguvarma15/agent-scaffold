"""Real-time progress display for generation runs.

The generator emits ``ProgressEvent`` instances as it iterates the Anthropic
stream. A display (``RichProgressDisplay``, ``NullProgressDisplay``) consumes
them and updates the user-facing output. Splitting events from display keeps
the generator I/O-free and unit-testable without a TTY.

P1 extends the display from a single panel into a two-column layout:
generation status + recent operations log on the left, per-file tracking on
the right, optional verbose deltas panel below. Non-LLM phases (write,
format, validate) emit ``operation_started`` / ``operation_done`` /
``bash_started`` / ``bash_done`` events so the user sees post-generation
steps live instead of a stalled spinner.
"""

from __future__ import annotations

import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from rich.columns import Columns
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

EventKind = Literal[
    "thinking_delta",
    "text_delta",
    "usage",
    "file_emitted",
    "file_detected",
    "file_written",
    "operation_started",
    "operation_done",
    "bash_started",
    "bash_done",
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

    def __init__(self) -> None:
        self.phase_durations: dict[str, float] = {}
        self.warnings: list[str] = []
        self.errors: list[str] = []

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


# A rough chars/token approximation matching context.py.
_CHARS_PER_TOKEN = 4


# File state → (symbol, rich style).
_FILE_SYMBOL: dict[str, tuple[str, str]] = {
    "detected": ("⠋", "yellow"),
    "written": ("✓", "green"),
    "overwritten": ("✓", "cyan"),
    "skipped": ("↷", "dim"),
    "modified": ("↻", "magenta"),
    "warning": ("⚠", "yellow"),
    "failed": ("✗", "red"),
}

# Operation state → (symbol, rich style).
_OP_SYMBOL: dict[str, tuple[str, str]] = {
    "active": ("⠋", "yellow"),
    "ok": ("✓", "green"),
    "warn": ("⚠", "yellow"),
    "fail": ("✗", "red"),
}


@dataclass
class _OperationEntry:
    name: str
    started_at: float
    finished_at: float | None = None
    status: str | None = None  # None = active; "ok"|"warn"|"fail" once done
    summary: str | None = None
    hint: str | None = None


@dataclass
class _State:
    text_buffer: str = ""
    text_tokens: int = 0
    thinking_tokens: int = 0
    # path -> state (see _FILE_SYMBOL keys). Insertion order preserved so the
    # right panel reads chronologically.
    files: OrderedDict[str, str] = field(default_factory=OrderedDict)
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
    # Operations log: every operation_started/bash_started appends here.
    operations: list[_OperationEntry] = field(default_factory=list)
    active_operations: dict[str, _OperationEntry] = field(default_factory=dict)
    phase_durations: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


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


def _format_cmd(payload: Any) -> str:
    """Best-effort stringify of a bash event's cmd payload."""
    if isinstance(payload, list):
        return " ".join(str(p) for p in payload)
    return str(payload)


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

    @property
    def phase_durations(self) -> dict[str, float]:
        return self._state.phase_durations

    @property
    def warnings(self) -> list[str]:
        return self._state.warnings

    @property
    def errors(self) -> list[str]:
        return self._state.errors

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
        elif event.kind in ("file_emitted", "file_detected"):
            path = self._extract_path(event.payload)
            if path:
                state.files.setdefault(path, "detected")
        elif event.kind == "file_written":
            payload = event.payload if isinstance(event.payload, dict) else {}
            path = str(payload.get("path", "") or "")
            mode = str(payload.get("mode", "new") or "new")
            if path:
                new_state = {
                    "new": "written",
                    "overwrite": "overwritten",
                    "skip": "skipped",
                    "modified": "modified",
                    "warn": "warning",
                    "fail": "failed",
                }.get(mode, "written")
                state.files[path] = new_state
        elif event.kind == "operation_started":
            payload = event.payload if isinstance(event.payload, dict) else {}
            name = str(payload.get("name", "") or "")
            if name:
                op = _OperationEntry(
                    name=name,
                    started_at=now,
                    hint=str(payload["hint"]) if payload.get("hint") else None,
                )
                state.operations.append(op)
                state.active_operations[name] = op
        elif event.kind == "operation_done":
            payload = event.payload if isinstance(event.payload, dict) else {}
            name = str(payload.get("name", "") or "")
            status = str(payload.get("status", "ok") or "ok")
            summary = payload.get("summary")
            done_op: _OperationEntry | None = state.active_operations.pop(name, None)
            if done_op is None:
                done_op = _OperationEntry(name=name, started_at=now)
                state.operations.append(done_op)
            done_op.finished_at = now
            done_op.status = status
            if summary is not None:
                done_op.summary = str(summary)
            state.phase_durations[name] = done_op.finished_at - done_op.started_at
            if status == "warn":
                state.warnings.append(f"{name}: {done_op.summary or 'warning'}")
            elif status == "fail":
                state.errors.append(f"{name}: {done_op.summary or 'failed'}")
        elif event.kind == "bash_started":
            payload = event.payload if isinstance(event.payload, dict) else {}
            cmd = _format_cmd(payload.get("cmd", ""))
            if cmd:
                op_name = f"$ {cmd}"
                op = _OperationEntry(name=op_name, started_at=now)
                state.operations.append(op)
                state.active_operations[op_name] = op
        elif event.kind == "bash_done":
            payload = event.payload if isinstance(event.payload, dict) else {}
            cmd = _format_cmd(payload.get("cmd", ""))
            exit_code = int(payload.get("exit_code", 0) or 0)
            op_name = f"$ {cmd}"
            bash_op: _OperationEntry | None = state.active_operations.pop(op_name, None)
            if bash_op is None:
                bash_op = _OperationEntry(name=op_name, started_at=now)
                state.operations.append(bash_op)
            bash_op.finished_at = now
            bash_op.status = "ok" if exit_code == 0 else "warn"
            bash_op.summary = f"exit {exit_code}"
        elif event.kind == "error":
            # Defer the actual print to __exit__ so we don't break Live's
            # exclusive ownership of stdout. Keep the latest error.
            state.last_error = str(event.payload)
        # done events fall through; the final render in __exit__ handles them.
        self._live.update(self._render(), refresh=True)

    def _extract_path(self, payload: Any) -> str:
        if isinstance(payload, str):
            return payload
        if isinstance(payload, dict):
            return str(payload.get("path", "") or "")
        return ""

    def _scan_for_new_files(self) -> None:
        for match in _FILE_PATH_RE.finditer(self._state.text_buffer):
            path = match.group(1)
            self._state.files.setdefault(path, "detected")

    def _render(self) -> RenderableType:
        left = self._render_status_panel()
        right = self._render_files_panel()
        parts: list[RenderableType] = [Columns([left, right], equal=True, expand=True)]
        if self._verbose:
            verbose = self._render_verbose_panel()
            if verbose is not None:
                parts.append(verbose)
        return Group(*parts)

    def _render_status_panel(self) -> Panel:
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
            lines.append(Text.from_markup(f"Status:   [yellow]{s.pre_fill_message}[/]"))
            lines.append(Text("Thinking: not yet"))
            lines.append(Text("Output:   not yet"))
            lines.append(Text(cache_line))
        else:
            thinking_line = f"Thinking: ~{s.thinking_tokens:,} tokens"
            if self._expected_files:
                files_part = f"  ({len(s.files)}/{self._expected_files} files)"
            elif s.files:
                files_part = f"  ({len(s.files)} files)"
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

        lines.append(Text(""))
        lines.append(Text.from_markup("[bold]Recent operations:[/]"))
        for op_line in self._render_operations():
            lines.append(op_line)

        body = Text("\n").join(lines)
        return Panel(body, title="Generation progress", expand=True)

    def _render_operations(self) -> list[Text]:
        ops = self._state.operations[-5:]
        if not ops:
            return [Text("  (no operations yet)", style="dim")]
        rendered: list[Text] = []
        for op in ops:
            key = op.status if op.status is not None else "active"
            sym, style = _OP_SYMBOL.get(key, ("•", "white"))
            if op.finished_at is not None:
                d = op.finished_at - op.started_at
                duration = f" ({d:.1f}s)"
            else:
                duration = " ..."
            summary = f" — {op.summary}" if op.summary else ""
            line = Text("  ")
            line.append(f"{sym} ", style=style)
            line.append(f"{op.name}{summary}{duration}")
            rendered.append(line)
        return rendered

    def _render_files_panel(self) -> Panel:
        items = list(self._state.files.items())
        written_states = {"written", "overwritten", "modified"}
        count_written = sum(1 for _, s in items if s in written_states)
        title = f"Files ({len(items)} detected / {count_written} written)"
        if not items:
            body: RenderableType = Text("(waiting for files...)", style="dim")
            return Panel(body, title=title, expand=True)
        max_rows = 20
        truncated = max(0, len(items) - max_rows)
        visible = items[-max_rows:]
        lines: list[Text] = []
        if truncated:
            lines.append(Text(f"... ({truncated} earlier)", style="dim"))
        for path, status in visible:
            sym, style = _FILE_SYMBOL.get(status, ("•", "white"))
            line = Text()
            line.append(f"{sym} ", style=style)
            line.append(path)
            lines.append(line)
        body = Text("\n").join(lines)
        return Panel(body, title=title, expand=True)

    def _render_verbose_panel(self) -> Panel | None:
        if not self._state.text_buffer:
            return None
        tail = self._state.text_buffer[-1200:]
        # Trim to the last ~20 non-empty lines for readability.
        recent_lines = [line for line in tail.splitlines() if line.strip()][-20:]
        if not recent_lines:
            return None
        body = Text("\n".join(recent_lines), style="dim")
        return Panel(body, title="Verbose: recent stream deltas", expand=True)

    @property
    def seconds_since_last_event(self) -> float:
        return time.monotonic() - self._state.last_event_at
