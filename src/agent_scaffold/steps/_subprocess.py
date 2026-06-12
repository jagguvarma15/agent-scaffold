"""Streaming subprocess runner shared by every concrete step.

Steps shell out to ``uv``, ``docker``, ``alembic``, ``pytest`` etc. and want
to surface each line as a ``StepLog`` event so the user sees pull progress,
test output, etc. in real time rather than after the process exits.

Two implementation notes that matter for correctness:

- We multiplex stdout + stderr with :mod:`selectors`. Reading stdout to EOF
  before touching stderr (the naive pattern) deadlocks any subprocess that
  fills its 64 KiB stderr pipe buffer — `pytest` and `docker compose pull`
  both do this in the field.
- Timeouts ``kill()`` the process group rather than ``terminate()``. A few
  subprocesses ignore SIGTERM during long network reads (notably
  ``docker compose pull``); SIGKILL is the only signal that's guaranteed to
  unblock the wait.

Use ``shell=False`` everywhere. Callers pass list-form ``cmd``.
"""

from __future__ import annotations

import os
import selectors
import signal
import subprocess
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Literal

from agent_scaffold.orchestrator import StepLog

_READ_CHUNK = 4096
_STDERR_TAIL_LINES = 10


@dataclass(frozen=True)
class SubprocessResult:
    exit_code: int
    """Process exit code; ``-signal`` on a signal, ``-1`` on our own timeout kill."""

    stderr_tail: str
    """Last ``_STDERR_TAIL_LINES`` lines of stderr, newline-joined, for failure panels."""

    timed_out: bool
    """``True`` iff we killed the process because it exceeded ``timeout``."""

    duration: float
    """Wall-clock seconds from launch to last byte read."""


def stream_subprocess(
    cmd: list[str],
    cwd: Path,
    *,
    step_id: str,
    callback: Callable[[StepLog], None] | None = None,
    line_callback: Callable[[str, str], None] | None = None,
    timeout: float = 600.0,
    env: dict[str, str] | None = None,
) -> SubprocessResult:
    """Run ``cmd`` under ``cwd``, streaming each line through ``callback``.

    ``line_callback(stream, line)`` is the orchestrator-free sibling of
    ``callback`` — callers outside the step framework (the validator) get
    per-line streaming without constructing ``StepLog`` events.

    Stops reading as soon as both pipes hit EOF AND the process has exited.
    On timeout, kills the process group and returns whatever was buffered
    plus ``timed_out=True`` so the caller can render a useful failure panel.
    """
    started = time.monotonic()
    # start_new_session so we can kill the whole group on timeout — child
    # commands that spawn helpers (npm-style) otherwise survive the parent kill.
    proc = subprocess.Popen(  # noqa: S603 — cmd is list-form, shell=False, callers control input
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
        start_new_session=True,
    )

    stderr_tail: deque[str] = deque(maxlen=_STDERR_TAIL_LINES)
    sel = selectors.DefaultSelector()
    if proc.stdout is not None:
        sel.register(proc.stdout, selectors.EVENT_READ, data="stdout")
    if proc.stderr is not None:
        sel.register(proc.stderr, selectors.EVENT_READ, data="stderr")

    timed_out = False
    deadline = started + timeout
    try:
        while sel.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                _kill_group(proc)
                break
            for key, _events in sel.select(timeout=min(remaining, 1.0)):
                stream = key.data
                fileobj: IO[str] = key.fileobj  # type: ignore[assignment]
                line = fileobj.readline()
                if not line:
                    sel.unregister(fileobj)
                    continue
                text = line.rstrip("\n")
                if stream == "stderr":
                    stderr_tail.append(text)
                if callback is not None:
                    callback(StepLog(step_id=step_id, line=text, stream=stream))
                if line_callback is not None:
                    line_callback(stream, text)
    finally:
        sel.close()
        if proc.poll() is None:
            # Either we broke on timeout above, or an exception unwound the
            # loop with the child still alive. Either way: ensure no orphan.
            _kill_group(proc)
        # ``wait`` collects the exit status. Always after the kill so the
        # group has actually torn down.
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            # Best-effort. The OS will reap eventually.
            pass
        # Drain anything still buffered after the process exited (stdout/stderr
        # may have been partially read mid-line above).
        for fh in (proc.stdout, proc.stderr):
            if fh is None:
                continue
            try:
                remainder = fh.read()
            except (ValueError, OSError):
                continue
            if not remainder:
                continue
            stream_name: Literal["stdout", "stderr"] = (
                "stderr" if fh is proc.stderr else "stdout"
            )
            for text in remainder.splitlines():
                if fh is proc.stderr:
                    stderr_tail.append(text)
                if callback is not None:
                    callback(StepLog(step_id=step_id, line=text, stream=stream_name))
                if line_callback is not None:
                    line_callback(stream_name, text)
            try:
                fh.close()
            except (ValueError, OSError):
                pass

    exit_code = -1 if timed_out else (proc.returncode if proc.returncode is not None else -1)
    return SubprocessResult(
        exit_code=exit_code,
        stderr_tail="\n".join(stderr_tail),
        timed_out=timed_out,
        duration=time.monotonic() - started,
    )


def _kill_group(proc: subprocess.Popen[str]) -> None:
    """SIGKILL the whole process group; fall back to ``proc.kill()`` on Windows."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, AttributeError, OSError):
        # AttributeError: Windows has no killpg; OSError covers race on already-exited proc.
        try:
            proc.kill()
        except (ProcessLookupError, OSError):
            pass


__all__ = [
    "SubprocessResult",
    "stream_subprocess",
]
