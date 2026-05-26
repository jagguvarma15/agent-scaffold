"""Tests for ``agent_scaffold.steps._subprocess.stream_subprocess``.

Cover the two non-obvious correctness properties:

1. We don't deadlock on stderr-heavy output (selectors-based read).
2. We kill the process on timeout (no orphan, no infinite wait).
"""

from __future__ import annotations

import sys
from pathlib import Path

from agent_scaffold.orchestrator import StepLog
from agent_scaffold.steps._subprocess import stream_subprocess


def test_streams_stdout_lines(tmp_path: Path) -> None:
    events: list[StepLog] = []
    result = stream_subprocess(
        [sys.executable, "-c", "print('hello'); print('world')"],
        cwd=tmp_path,
        step_id="t",
        callback=events.append,
        timeout=10.0,
    )
    assert result.exit_code == 0
    lines = [e.line for e in events if e.stream == "stdout"]
    assert "hello" in lines
    assert "world" in lines


def test_does_not_deadlock_on_heavy_stderr(tmp_path: Path) -> None:
    # 2 MiB of stderr — easily exceeds the 64 KiB pipe buffer that would
    # deadlock a naive read-stdout-first implementation.
    program = "import sys\nsys.stderr.write('x' * (2 * 1024 * 1024))\nsys.stderr.flush()\n"
    result = stream_subprocess(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        step_id="t",
        callback=None,
        timeout=15.0,
    )
    assert result.exit_code == 0


def test_timeout_kills_long_running_process(tmp_path: Path) -> None:
    result = stream_subprocess(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=tmp_path,
        step_id="t",
        callback=None,
        timeout=1.0,
    )
    assert result.timed_out is True
    assert result.exit_code == -1
    assert result.duration < 5.0  # killed promptly


def test_nonzero_exit_captured_in_stderr_tail(tmp_path: Path) -> None:
    program = "import sys\nprint('out', flush=True)\nprint('err line', file=sys.stderr, flush=True)\nsys.exit(3)\n"
    result = stream_subprocess(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        step_id="t",
        callback=None,
        timeout=10.0,
    )
    assert result.exit_code == 3
    assert "err line" in result.stderr_tail
