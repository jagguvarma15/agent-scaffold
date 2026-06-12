"""Persistent per-run log artifacts for generation + provisioning runs.

Every ``agent-scaffold new`` invocation gets a directory under
``<cache_dir>/runs/<run_id>/`` holding two sinks:

- ``run.log`` — human-readable, one timestamped line per event. The file a
  user opens when a run failed and the panel has already scrolled away.
- ``events.jsonl`` — one JSON object per event (``{"ts", "kind", "payload"}``)
  for tooling: replay, diffing two runs, or piping into ``jq``.

Both sinks pass every string through :mod:`agent_scaffold._redact` before
it touches disk, so a credential that leaks into a subprocess tail or an
exception ``repr()`` never lands in a log file.

High-frequency stream events (``text_delta`` / ``thinking_delta`` /
``heartbeat``) are counted but not persisted per-event — a single run emits
thousands of deltas and the per-chunk text is reconstructable from the
written files anyway. The counts appear in the ``run_closed`` summary event.

Logging must never break a run: every disk write is wrapped so an
``OSError`` (disk full, permissions) degrades to dropping the log line,
not aborting generation.
"""

from __future__ import annotations

import json
import secrets
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

from agent_scaffold._redact import redact, redact_obj
from agent_scaffold.orchestrator import (
    StepEvent,
    StepFinished,
    StepLog,
    StepProgress,
    StepStarted,
)
from agent_scaffold.progress import ProgressEvent

RUNS_DIR_NAME = "runs"
RUN_LOG_FILENAME = "run.log"
EVENTS_FILENAME = "events.jsonl"
# How many run directories to keep before pruning oldest-first. Generous
# enough to debug "it worked last week", small enough to stay out of the way.
MAX_RUN_DIRS = 20

# Stream events too chatty to persist individually (see module docstring).
_COUNTED_KINDS = frozenset({"text_delta", "thinking_delta", "heartbeat"})


def runs_root(cache_dir: Path) -> Path:
    return cache_dir / RUNS_DIR_NAME


def prune_runs(root: Path, *, keep: int = MAX_RUN_DIRS) -> list[Path]:
    """Delete all but the ``keep`` most-recent run directories.

    Mirrors ``template_snapshot.prune_snapshots``: sort by mtime, drop the
    oldest. Returns the removed paths. Never raises — a prune failure is
    not worth failing a run over.
    """
    if not root.is_dir():
        return []
    run_dirs = [p for p in root.iterdir() if p.is_dir()]
    run_dirs.sort(key=lambda p: p.stat().st_mtime)
    excess = max(0, len(run_dirs) - keep)
    removed: list[Path] = []
    for path in run_dirs[:excess]:
        shutil.rmtree(path, ignore_errors=True)
        removed.append(path)
    return removed


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _serialize_step_event(event: StepEvent) -> dict[str, Any]:
    """Flatten a StepEvent dataclass into a JSON-shaped payload."""
    payload: dict[str, Any] = {"step_id": event.step_id, "event": type(event).__name__}
    if isinstance(event, StepProgress):
        payload["message"] = event.message
        if event.percent is not None:
            payload["percent"] = event.percent
    elif isinstance(event, StepLog):
        payload["line"] = event.line
        payload["stream"] = event.stream
    elif isinstance(event, StepFinished) and event.result is not None:
        payload["result"] = asdict(event.result)
        payload["result"]["status"] = event.result.status.value
    return payload


class RunLogger:
    """Owns one run directory and its two sinks. Safe to close twice."""

    def __init__(self, cache_dir: Path, *, command: str = "new") -> None:
        stamp = _utc_now().strftime("%Y%m%dT%H%M%SZ")
        self.run_id = f"{stamp}-{secrets.token_hex(3)}"
        self.run_dir = runs_root(cache_dir) / self.run_id
        self._command = command
        self._started_at = _utc_now()
        self._delta_counts: dict[str, int] = {}
        self._closed = False
        self._log: IO[str] | None = None
        self._events: IO[str] | None = None
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # Line-buffered so a crash mid-run still leaves a useful tail.
        self._log = (self.run_dir / RUN_LOG_FILENAME).open("a", buffering=1, encoding="utf-8")
        self._events = (self.run_dir / EVENTS_FILENAME).open("a", buffering=1, encoding="utf-8")
        self._write_line(f"agent-scaffold {command} — run {self.run_id}")
        self.log_event("run_started", {"command": command})
        prune_runs(runs_root(cache_dir))

    @property
    def log_path(self) -> Path:
        return self.run_dir / RUN_LOG_FILENAME

    @property
    def events_path(self) -> Path:
        return self.run_dir / EVENTS_FILENAME

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close(status="failed" if exc_type is not None else "completed")

    # -- sinks --------------------------------------------------------------

    def _write_line(self, text: str) -> None:
        if self._log is None or self._closed:
            return
        ts = _utc_now().strftime("%H:%M:%S")
        try:
            self._log.write(f"{ts}  {redact(text)}\n")
        except OSError:
            pass

    def _write_event(self, kind: str, payload: Any) -> None:
        if self._events is None or self._closed:
            return
        record = {
            "ts": _utc_now().isoformat(timespec="milliseconds"),
            "kind": kind,
            "payload": redact_obj(payload),
        }
        try:
            self._events.write(json.dumps(record, default=str) + "\n")
        except (OSError, TypeError, ValueError):
            pass

    # -- public API ----------------------------------------------------------

    def log_event(self, kind: str, payload: Any = None) -> None:
        """Record one generation-pipeline event in both sinks."""
        if kind in _COUNTED_KINDS:
            self._delta_counts[kind] = self._delta_counts.get(kind, 0) + 1
            return
        self._write_event(kind, payload)
        self._write_line(self._human_line(kind, payload))

    def log_progress_event(self, event: ProgressEvent) -> None:
        self.log_event(event.kind, event.payload)

    def log_step_event(self, event: StepEvent) -> None:
        """Record one orchestrator (``up``) event in both sinks."""
        payload = _serialize_step_event(event)
        self._write_event("step", payload)
        if isinstance(event, StepStarted):
            self._write_line(f"[{event.step_id}] started")
        elif isinstance(event, StepProgress):
            self._write_line(f"[{event.step_id}] {event.message}")
        elif isinstance(event, StepLog):
            tag = "log" if event.stream == "stdout" else "err"
            self._write_line(f"[{event.step_id}] {tag}: {event.line.rstrip()}")
        elif isinstance(event, StepFinished):
            status = event.result.status.value if event.result is not None else "done"
            detail = ""
            if event.result is not None:
                detail_text = event.result.detail or event.result.error or ""
                if detail_text:
                    detail = f" — {detail_text}"
            self._write_line(f"[{event.step_id}] {status}{detail}")

    def note(self, text: str) -> None:
        """Free-form human-log line (also mirrored to JSONL as a ``note``)."""
        self._write_event("note", text)
        self._write_line(text)

    def close(self, *, status: str = "completed") -> None:
        """Write the closing summary and release file handles. Idempotent."""
        if self._closed:
            return
        duration = (_utc_now() - self._started_at).total_seconds()
        self._write_event(
            "run_closed",
            {
                "status": status,
                "duration_seconds": round(duration, 1),
                "suppressed_stream_events": dict(self._delta_counts),
            },
        )
        self._write_line(f"run {status} after {duration:.1f}s")
        self._closed = True
        for handle in (self._log, self._events):
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass
        self._log = None
        self._events = None

    # -- human formatting ------------------------------------------------------

    def _human_line(self, kind: str, payload: Any) -> str:
        p = payload if isinstance(payload, dict) else {}
        if kind == "operation_started":
            hint = f" ({p['hint']})" if p.get("hint") else ""
            return f"▶ {p.get('name', '?')}{hint}"
        if kind == "operation_done":
            summary = f" — {p['summary']}" if p.get("summary") else ""
            return f"{p.get('status', 'ok')}: {p.get('name', '?')}{summary}"
        if kind == "bash_started":
            return f"$ {_cmd_str(p.get('cmd'))}"
        if kind == "bash_done":
            return f"$ {_cmd_str(p.get('cmd'))} → exit {p.get('exit_code', '?')}"
        if kind == "file_written":
            return f"wrote {p.get('path', '?')} [{p.get('mode', 'new')}]"
        if kind in ("file_emitted", "file_detected"):
            path = payload if isinstance(payload, str) else p.get("path", "?")
            return f"detected {path}"
        if kind == "usage":
            return (
                f"usage: in={p.get('input_tokens', 0)} out={p.get('output_tokens', 0)} "
                f"cache_read={p.get('cache_read_input_tokens', 0)}"
            )
        if kind == "error":
            return f"ERROR: {payload}"
        if payload is None:
            return kind
        return f"{kind}: {payload}"


def _cmd_str(cmd: Any) -> str:
    if isinstance(cmd, list):
        return " ".join(str(c) for c in cmd)
    return str(cmd or "")


class TeeProgressSink:
    """Forward generation events to a display *and* a :class:`RunLogger`.

    Drop-in for the displays ``pipeline.run_generation`` accepts: forwards
    the context-manager protocol and ``on_event``, and delegates the
    report-facing attributes (``phase_durations`` / ``warnings`` /
    ``errors``) so ``_emit_generation_report``'s ``getattr`` calls see the
    underlying display's state.
    """

    def __init__(self, display: Any, run_logger: RunLogger) -> None:
        self._display = display
        self._run_logger = run_logger

    def __enter__(self) -> TeeProgressSink:
        self._display.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        self._display.__exit__(*args)

    def on_event(self, event: ProgressEvent) -> None:
        self._display.on_event(event)
        self._run_logger.log_progress_event(event)

    @property
    def phase_durations(self) -> dict[str, float]:
        return dict(getattr(self._display, "phase_durations", {}))

    @property
    def warnings(self) -> list[str]:
        return list(getattr(self._display, "warnings", []))

    @property
    def errors(self) -> list[str]:
        return list(getattr(self._display, "errors", []))

    @property
    def run_log_dir(self) -> str:
        return str(self._run_logger.run_dir)


__all__ = [
    "EVENTS_FILENAME",
    "MAX_RUN_DIRS",
    "RUN_LOG_FILENAME",
    "RUNS_DIR_NAME",
    "RunLogger",
    "TeeProgressSink",
    "prune_runs",
    "runs_root",
]
