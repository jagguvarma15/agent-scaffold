"""Tests for agent_scaffold.run_log — per-run log artifacts + tee sink."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from agent_scaffold._redact import contains_secret_shape
from agent_scaffold.orchestrator import (
    StepFinished,
    StepLog,
    StepResult,
    StepStarted,
    StepStatus,
)
from agent_scaffold.progress import NullProgressDisplay, ProgressEvent
from agent_scaffold.run_log import (
    MAX_RUN_DIRS,
    RunLogger,
    TeeProgressSink,
    prune_runs,
    runs_root,
)

_PLANTED_KEY = "sk-ant-api03-aaaabbbbccccddddeeee"


def _read_events(logger: RunLogger) -> list[dict[str, object]]:
    lines = logger.events_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_run_logger_creates_artifacts(tmp_path: Path) -> None:
    logger = RunLogger(tmp_path, command="new")
    logger.log_event("operation_started", {"name": "generate", "hint": "model=m"})
    logger.log_event("operation_done", {"name": "generate", "status": "ok", "summary": "done"})
    logger.close()

    assert logger.run_dir.is_dir()
    assert logger.log_path.is_file()
    assert logger.events_path.is_file()

    events = _read_events(logger)
    kinds = [e["kind"] for e in events]
    assert kinds[0] == "run_started"
    assert "operation_started" in kinds
    assert kinds[-1] == "run_closed"
    # Every event carries an ISO timestamp.
    assert all(isinstance(e["ts"], str) and "T" in e["ts"] for e in events)

    human = logger.log_path.read_text(encoding="utf-8")
    assert "generate" in human
    assert "run completed" in human


def test_run_logger_redacts_both_sinks(tmp_path: Path) -> None:
    logger = RunLogger(tmp_path)
    logger.log_event("error", f"auth failed for {_PLANTED_KEY}")
    logger.log_event("bash_done", {"cmd": ["echo", _PLANTED_KEY], "exit_code": 1})
    logger.note(f"postgres://user:hunter2@localhost/db and {_PLANTED_KEY}")
    logger.close()

    for path in (logger.log_path, logger.events_path):
        text = path.read_text(encoding="utf-8")
        assert _PLANTED_KEY not in text
        assert "hunter2" not in text
        assert not contains_secret_shape(text)


def test_run_logger_counts_noisy_kinds_instead_of_persisting(tmp_path: Path) -> None:
    logger = RunLogger(tmp_path)
    for _ in range(50):
        logger.log_event("text_delta", "chunk")
        logger.log_event("thinking_delta", "hmm")
    logger.log_event("heartbeat", 30)
    logger.close()

    events = _read_events(logger)
    assert all(e["kind"] not in ("text_delta", "thinking_delta", "heartbeat") for e in events)
    closed = events[-1]
    assert closed["kind"] == "run_closed"
    payload = closed["payload"]
    assert isinstance(payload, dict)
    suppressed = payload["suppressed_stream_events"]
    assert suppressed == {"text_delta": 50, "thinking_delta": 50, "heartbeat": 1}


def test_run_logger_close_is_idempotent(tmp_path: Path) -> None:
    logger = RunLogger(tmp_path)
    logger.close(status="failed")
    logger.close()  # second close must not raise or append
    events = _read_events(logger)
    assert [e["kind"] for e in events].count("run_closed") == 1
    payload = events[-1]["payload"]
    assert isinstance(payload, dict)
    assert payload["status"] == "failed"


def test_run_logger_context_manager_records_failure(tmp_path: Path) -> None:
    try:
        with RunLogger(tmp_path) as logger:
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    events = _read_events(logger)
    payload = events[-1]["payload"]
    assert isinstance(payload, dict)
    assert payload["status"] == "failed"


def test_step_events_serialize_into_both_sinks(tmp_path: Path) -> None:
    logger = RunLogger(tmp_path)
    logger.log_step_event(StepStarted(step_id="docker_up"))
    logger.log_step_event(StepLog(step_id="docker_up", line="pulling image...", stream="stdout"))
    logger.log_step_event(
        StepFinished(
            step_id="docker_up",
            result=StepResult(status=StepStatus.DONE, detail="3 services healthy"),
        )
    )
    logger.close()

    events = [e for e in _read_events(logger) if e["kind"] == "step"]
    assert len(events) == 3
    finished = events[-1]["payload"]
    assert isinstance(finished, dict)
    assert finished["event"] == "StepFinished"
    result = finished["result"]
    assert isinstance(result, dict)
    assert result["status"] == "done"

    human = logger.log_path.read_text(encoding="utf-8")
    assert "[docker_up] started" in human
    assert "pulling image" in human
    assert "3 services healthy" in human


def test_prune_runs_keeps_newest(tmp_path: Path) -> None:
    root = runs_root(tmp_path)
    root.mkdir(parents=True)
    for i in range(MAX_RUN_DIRS + 4):
        d = root / f"run-{i:03d}"
        d.mkdir()
        stamp = time.time() - (MAX_RUN_DIRS + 4 - i) * 60
        os.utime(d, (stamp, stamp))
    removed = prune_runs(root)
    assert len(removed) == 4
    survivors = sorted(p.name for p in root.iterdir())
    assert survivors[0] == "run-004"
    assert len(survivors) == MAX_RUN_DIRS


class _RecordingDisplay(NullProgressDisplay):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[ProgressEvent] = []
        self.phase_durations = {"generate": 1.5}
        self.warnings = ["w1"]
        self.errors = ["e1"]

    def on_event(self, event: ProgressEvent) -> None:
        self.events.append(event)


def test_tee_progress_sink_forwards_and_delegates(tmp_path: Path) -> None:
    inner = _RecordingDisplay()
    logger = RunLogger(tmp_path)
    with TeeProgressSink(inner, logger) as tee:
        tee.on_event(ProgressEvent("file_written", {"path": "src/a.py", "mode": "new"}))
    logger.close()

    assert [e.kind for e in inner.events] == ["file_written"]
    assert any(e["kind"] == "file_written" for e in _read_events(logger))
    assert tee.phase_durations == {"generate": 1.5}
    assert tee.warnings == ["w1"]
    assert tee.errors == ["e1"]
    assert tee.run_log_dir == str(logger.run_dir)
