"""Post-generation validation tiers.

Run lightweight static checks, full builds, or the smoke check as subprocesses
inside the generated project's directory. Each tier captures stdout+stderr.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from agent_scaffold.progress import ProgressEvent


class ValidationTier(str, Enum):
    static = "static"
    build = "build"
    smoke = "smoke"


class ValidationResult(BaseModel):
    tier: ValidationTier
    passed: bool
    output: str


def verify_required_files_on_disk(dest: Path, required_files: list[str]) -> list[str]:
    """Return the subset of ``required_files`` that are not present at ``dest``.

    ``validate_required_files`` in ``contract.py`` checks the LLM's response
    body. This is the post-write counterpart: it confirms the writer actually
    persisted each required path. Failures here typically mean ``--write-mode
    skip`` collided with a stray pre-existing file, a parent-path
    sanitisation dropped the entry, or the filesystem rejected the write
    (permissions, disk full, etc.).
    """
    missing: list[str] = []
    for rel in required_files:
        target = dest / rel
        if not target.is_file():
            missing.append(rel)
    return missing


def _emit(on_event: Callable[[ProgressEvent], None] | None, event: ProgressEvent) -> None:
    if on_event is not None:
        on_event(event)


def _run(
    cmd: list[str],
    cwd: Path,
    on_event: Callable[[ProgressEvent], None] | None = None,
) -> tuple[bool, str]:
    _emit(on_event, ProgressEvent(kind="bash_started", payload={"cmd": cmd, "cwd": str(cwd)}))
    if shutil.which(cmd[0]) is None:
        msg = f"command not found on PATH: {cmd[0]}"
        _emit(
            on_event,
            ProgressEvent(
                kind="bash_done",
                payload={"cmd": cmd, "exit_code": 127, "stderr_tail": msg},
            ),
        )
        return False, msg
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        msg = f"timeout: {exc}"
        _emit(
            on_event,
            ProgressEvent(
                kind="bash_done",
                payload={"cmd": cmd, "exit_code": -1, "stderr_tail": msg},
            ),
        )
        return False, msg
    except OSError as exc:
        msg = f"failed to launch {cmd[0]}: {exc}"
        _emit(
            on_event,
            ProgressEvent(
                kind="bash_done",
                payload={"cmd": cmd, "exit_code": -1, "stderr_tail": msg},
            ),
        )
        return False, msg
    output = (proc.stdout or "") + (proc.stderr or "")
    _emit(
        on_event,
        ProgressEvent(
            kind="bash_done",
            payload={
                "cmd": cmd,
                "exit_code": proc.returncode,
                "stdout_tail": (proc.stdout or "")[-200:],
                "stderr_tail": (proc.stderr or "")[-200:],
            },
        ),
    )
    return proc.returncode == 0, output


def _run_shell(
    cmd: str,
    cwd: Path,
    on_event: Callable[[ProgressEvent], None] | None = None,
) -> tuple[bool, str]:
    _emit(on_event, ProgressEvent(kind="bash_started", payload={"cmd": cmd, "cwd": str(cwd)}))
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            shell=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        msg = f"timeout: {exc}"
        _emit(
            on_event,
            ProgressEvent(
                kind="bash_done",
                payload={"cmd": cmd, "exit_code": -1, "stderr_tail": msg},
            ),
        )
        return False, msg
    output = (proc.stdout or "") + (proc.stderr or "")
    _emit(
        on_event,
        ProgressEvent(
            kind="bash_done",
            payload={
                "cmd": cmd,
                "exit_code": proc.returncode,
                "stdout_tail": (proc.stdout or "")[-200:],
                "stderr_tail": (proc.stderr or "")[-200:],
            },
        ),
    )
    return proc.returncode == 0, output


def _static_command(language: str) -> list[str] | None:
    if language == "python":
        return ["ruff", "check", "."]
    if language == "typescript":
        return ["pnpm", "exec", "tsc", "--noEmit"]
    return None


def _build_command(language: str) -> list[str] | None:
    if language == "python":
        return ["uv", "sync"]
    if language == "typescript":
        return ["pnpm", "install"]
    return None


def validate(
    dest: Path,
    hints: dict[str, Any],
    smoke_check: str,
    tiers: list[ValidationTier],
    continue_on_failure: bool = False,
    on_event: Callable[[ProgressEvent], None] | None = None,
) -> list[ValidationResult]:
    """Run requested validation tiers in order and return their results.

    When ``on_event`` is supplied, each tier emits ``bash_started`` and
    ``bash_done`` events through the underlying ``_run`` / ``_run_shell``
    helpers so a progress display can surface subprocess activity in real
    time.
    """
    results: list[ValidationResult] = []
    language = str(hints.get("language", "python"))
    for tier in tiers:
        if tier is ValidationTier.static:
            cmd = _static_command(language)
            if cmd is None:
                results.append(
                    ValidationResult(
                        tier=tier,
                        passed=True,
                        output=f"no static check defined for language={language}",
                    )
                )
                continue
            passed, output = _run(cmd, dest, on_event=on_event)
        elif tier is ValidationTier.build:
            cmd = _build_command(language)
            if cmd is None:
                results.append(
                    ValidationResult(
                        tier=tier,
                        passed=True,
                        output=f"no build command defined for language={language}",
                    )
                )
                continue
            passed, output = _run(cmd, dest, on_event=on_event)
        elif tier is ValidationTier.smoke:
            if not smoke_check:
                results.append(
                    ValidationResult(tier=tier, passed=True, output="no smoke_check supplied")
                )
                continue
            passed, output = _run_shell(smoke_check, dest, on_event=on_event)
        else:  # pragma: no cover - exhaustive
            raise ValueError(f"Unknown tier: {tier}")

        results.append(ValidationResult(tier=tier, passed=passed, output=output))
        if not passed and not continue_on_failure:
            break
    return results
