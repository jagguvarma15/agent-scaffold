"""Post-generation validation tiers.

Run lightweight static checks, full builds, or the smoke check as subprocesses
inside the generated project's directory. Output streams line-by-line through
``bash_line`` progress events (a ``uv sync`` can take minutes — the user
should see pip-style progress, not a frozen spinner) while the full combined
output is still captured and returned for the repair loop.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from agent_scaffold.progress import ProgressEvent
from agent_scaffold.steps._subprocess import stream_subprocess

_TIER_TIMEOUT_SECONDS = 300.0


class ValidationTier(str, Enum):
    static = "static"
    build = "build"
    compile = "compile"
    docker_up = "docker_up"
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


def _stream(
    argv: list[str],
    display_cmd: list[str] | str,
    cwd: Path,
    on_event: Callable[[ProgressEvent], None] | None,
) -> tuple[bool, str]:
    """Run ``argv`` streaming each output line as a ``bash_line`` event.

    ``display_cmd`` is what event payloads carry (the original command, not
    the ``/bin/sh -c`` wrapper). Lines are captured chronologically
    interleaved (stdout + stderr) — better for repair-loop diagnostics than
    the old stdout-then-stderr concatenation.
    """
    captured: list[str] = []

    def _line(stream: str, line: str) -> None:
        captured.append(line)
        _emit(
            on_event,
            ProgressEvent(
                kind="bash_line",
                payload={"cmd": display_cmd, "line": line, "stream": stream},
            ),
        )

    try:
        result = stream_subprocess(
            argv,
            cwd,
            step_id="validate",
            line_callback=_line,
            timeout=_TIER_TIMEOUT_SECONDS,
        )
    except OSError as exc:
        msg = f"failed to launch {argv[0]}: {exc}"
        _emit(
            on_event,
            ProgressEvent(
                kind="bash_done",
                payload={"cmd": display_cmd, "exit_code": -1, "stderr_tail": msg},
            ),
        )
        return False, msg

    output = "\n".join(captured) + ("\n" if captured else "")
    if result.timed_out:
        msg = f"timeout after {result.duration:.0f}s"
        _emit(
            on_event,
            ProgressEvent(
                kind="bash_done",
                payload={"cmd": display_cmd, "exit_code": -1, "stderr_tail": msg},
            ),
        )
        return False, output + msg
    _emit(
        on_event,
        ProgressEvent(
            kind="bash_done",
            payload={
                "cmd": display_cmd,
                "exit_code": result.exit_code,
                "stderr_tail": result.stderr_tail[-200:],
            },
        ),
    )
    return result.exit_code == 0, output


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
    return _stream(cmd, cmd, cwd, on_event)


def _run_shell(
    cmd: str,
    cwd: Path,
    on_event: Callable[[ProgressEvent], None] | None = None,
) -> tuple[bool, str]:
    _emit(on_event, ProgressEvent(kind="bash_started", payload={"cmd": cmd, "cwd": str(cwd)}))
    if os.name == "nt":  # pragma: no cover — POSIX-first; keep Windows working
        return _run_shell_buffered(cmd, cwd, on_event)
    return _stream(["/bin/sh", "-c", cmd], cmd, cwd, on_event)


_DOCKER_UP_DISPLAY = "docker compose up -d --build --wait"


def _docker_up(
    dest: Path,
    on_event: Callable[[ProgressEvent], None] | None = None,
) -> tuple[bool, str]:
    """The ``docker_up`` validation tier — bring the generated stack up.

    Fail-soft by self-skip: with no ``docker-compose.yml`` or no usable Docker
    (not installed / daemon down), the tier *passes* with an explanatory note so
    generation never regresses on a laptop or CI box without Docker. When Docker
    *is* usable, a failed bring-up is a real (repairable) failure.

    Shares the one compose implementation (:func:`steps.docker_up.bring_up`) with
    the ``up`` command, and manages its own 600s budget there (vs the 300s tier
    default) since image pulls + builds are slow.
    """
    from agent_scaffold.steps.docker_up import bring_up, docker_available

    _emit(
        on_event,
        ProgressEvent(kind="bash_started", payload={"cmd": _DOCKER_UP_DISPLAY, "cwd": str(dest)}),
    )

    def _done(passed: bool, output: str) -> tuple[bool, str]:
        _emit(
            on_event,
            ProgressEvent(
                kind="bash_done",
                payload={
                    "cmd": _DOCKER_UP_DISPLAY,
                    "exit_code": 0 if passed else 1,
                    "stderr_tail": output[-200:],
                },
            ),
        )
        return passed, output

    if not (dest / "docker-compose.yml").is_file():
        return _done(True, "no docker-compose.yml — docker_up tier skipped")
    usable, reason = docker_available()
    if not usable:
        return _done(True, f"docker not usable ({reason}) — docker_up tier skipped")

    def _line(stream: str, line: str) -> None:
        _emit(
            on_event,
            ProgressEvent(
                kind="bash_line",
                payload={"cmd": _DOCKER_UP_DISPLAY, "line": line, "stream": stream},
            ),
        )

    ok, output = bring_up(dest, line_callback=_line)
    return _done(ok, output)


def _run_shell_buffered(
    cmd: str,
    cwd: Path,
    on_event: Callable[[ProgressEvent], None] | None = None,
) -> tuple[bool, str]:  # pragma: no cover — Windows fallback
    try:
        proc = subprocess.run(  # noqa: S602 — sanctioned: the recipe-author smoke-check string is composed shell (see docs/design/security.md rule 4); Windows-only buffered fallback
            cmd,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            shell=True,
            timeout=_TIER_TIMEOUT_SECONDS,
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


# Directories never worth byte-compiling: the virtualenv ``uv sync`` just
# populated (it holds half of PyPI), vendored JS deps, VCS metadata, and tool
# caches. The compile tier runs after the build tier, so ``.venv`` is on disk
# by the time we look — keep ``compileall`` away from it.
_COMPILE_SKIP_DIRS = frozenset(
    {
        ".venv",
        "venv",
        ".git",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
    }
)


def _compile_targets(dest: Path, hints: dict[str, Any]) -> list[str]:
    """Relative paths to byte-compile for a Python project.

    Prefers the recipe's declared package roots — the ``project_layout``
    directory and the top-level directory of ``entry_point`` (typically
    ``app/`` or ``src/``) — keeping only those that exist on disk. When none
    resolve (e.g. a flat single-file layout), falls back to every top-level
    ``.py`` file and package directory minus the virtualenv / cache dirs.

    Returns an empty list when there is nothing project-owned to compile (an
    unreadable tree, or one holding only the virtualenv / cache dirs). The
    caller skips the tier in that case rather than handing ``compileall`` the
    whole tree — pointing it at ``.``/``.venv`` would byte-compile the
    installed dependencies and fail on any Py2-only file a wheel happens to
    ship.
    """
    candidates: list[str] = []
    layout = str(hints.get("project_layout", "")).replace("\\", "/").strip("/")
    if layout:
        candidates.append(layout)
    entry = str(hints.get("entry_point", "")).replace("\\", "/")
    if "/" in entry:
        candidates.append(entry.split("/", 1)[0])
    roots: list[str] = []
    for name in candidates:
        if name and name not in roots and (dest / name).is_dir():
            roots.append(name)
    if roots:
        return roots
    # Fallback: enumerate top-level entries so compileall never descends into
    # the virtualenv the build tier populated.
    targets: list[str] = []
    try:
        names = sorted(p.name for p in dest.iterdir())
    except OSError:
        return []
    for name in names:
        path = dest / name
        if path.is_dir():
            if name not in _COMPILE_SKIP_DIRS and not name.startswith("."):
                targets.append(name)
        elif name.endswith(".py"):
            targets.append(name)
    return targets


def _compile_command(language: str, dest: Path, hints: dict[str, Any]) -> list[str] | None:
    """Byte-compile command for the compile tier, or ``None`` to skip.

    Python: ``uv run --no-sync python -m compileall -q <roots>`` — a fast,
    network-free syntax check across the package that catches ``SyntaxError``
    in files the static tier's linter may exclude. ``--no-sync`` guarantees
    the call never triggers a dependency download: compilation only needs the
    interpreter, and on the standalone ``validate --tier compile`` path the
    project may not have been built yet. Returns ``None`` for non-Python
    languages (TS is already covered by ``tsc --noEmit`` in the static tier)
    and when there is nothing project-owned to compile.
    """
    if language != "python":
        return None
    targets = _compile_targets(dest, hints)
    if not targets:
        return None
    return ["uv", "run", "--no-sync", "python", "-m", "compileall", "-q", *targets]


def tier_command(
    tier: ValidationTier,
    language: str,
    smoke_check: str = "",
    *,
    dest: Path | None = None,
    hints: dict[str, Any] | None = None,
) -> str:
    """Human-readable command string for a tier — used by repair prompts.

    ``dest`` / ``hints`` are only consulted for the compile tier (whose
    command depends on the on-disk package layout); the other tiers ignore
    them.
    """
    if tier is ValidationTier.static:
        cmd = _static_command(language)
        return " ".join(cmd) if cmd else ""
    if tier is ValidationTier.build:
        cmd = _build_command(language)
        return " ".join(cmd) if cmd else ""
    if tier is ValidationTier.compile:
        cmd = _compile_command(language, dest, hints or {}) if dest is not None else None
        return " ".join(cmd) if cmd else "python -m compileall"
    if tier is ValidationTier.docker_up:
        return _DOCKER_UP_DISPLAY
    return smoke_check


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
        elif tier is ValidationTier.compile:
            cmd = _compile_command(language, dest, hints)
            if cmd is None:
                reason = (
                    "nothing to compile"
                    if language == "python"
                    else f"no compile check defined for language={language}"
                )
                results.append(ValidationResult(tier=tier, passed=True, output=reason))
                continue
            passed, output = _run(cmd, dest, on_event=on_event)
        elif tier is ValidationTier.docker_up:
            passed, output = _docker_up(dest, on_event=on_event)
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
