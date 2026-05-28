"""Atomic project writer.

Files are first written to a temp directory next to ``dest`` so that any
failure during staging leaves the destination untouched. Once all files
are staged we ``os.replace`` them into place one by one.
"""

from __future__ import annotations

import difflib
import os
import shutil
import stat
import tempfile
from collections.abc import Callable
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from agent_scaffold._scaffold_dir import SCAFFOLD_DIR
from agent_scaffold.contract import GeneratedFile, GenerationResult
from agent_scaffold.progress import ProgressEvent

# Default ``.gitignore`` lines we ensure live on every generated project.
# Rule 8 of the Q9 9-point checklist: secret-bearing files (`.env.local`,
# `credentials`) and machine state (`.scaffold/`) must never make it into
# a commit. We append (not overwrite) so user-authored entries survive.
DEFAULT_GITIGNORE_ENTRIES: tuple[str, ...] = (
    f"{SCAFFOLD_DIR}/",
    ".env",
    ".env.local",
    ".env.*.local",
    "credentials",
    ".DS_Store",
    "__pycache__/",
    "*.pyc",
)

_GITIGNORE_HEADER = "# Added by agent-scaffold for secret safety"


class WriteMode(str, Enum):
    abort = "abort"
    skip = "skip"
    diff = "diff"
    overwrite = "overwrite"


class DestinationExistsError(Exception):
    """Raised when ``dest`` is non-empty and ``WriteMode.abort`` is requested."""


class WriteReport(BaseModel):
    written: list[str] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)
    overwritten: list[str] = Field(default_factory=list)


def _is_non_empty(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def _normalize(path: str) -> str:
    return path.replace("\\", "/")


def _set_exec_bit(path: Path) -> None:
    if path.suffix == ".sh":
        mode = path.stat().st_mode
        path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _confirm_diff_default(rel_path: str, diff_text: str) -> bool:
    """Default per-file prompt for ``diff`` mode."""
    import questionary

    print(diff_text)
    answer = questionary.confirm(f"Overwrite {rel_path}?", default=False).ask()
    return bool(answer)


def write_project(
    result: GenerationResult,
    dest: Path,
    mode: WriteMode,
    confirm_diff: Callable[[str, str], bool] | None = None,
    on_event: Callable[[ProgressEvent], None] | None = None,
) -> WriteReport:
    """Write ``result`` into ``dest`` honoring ``mode``.

    The optional ``confirm_diff`` callback is only used in ``WriteMode.diff``.
    It receives ``(relative_path, unified_diff_text)`` and must return ``True``
    to overwrite the existing file.

    ``on_event`` receives a ``file_written`` ``ProgressEvent`` per file with
    ``{path, mode: "new"|"overwrite"|"skip", bytes}`` once that file lands.
    Failures are reported as ``mode="fail"`` before the exception propagates.
    """
    dest = dest.resolve()
    confirm = confirm_diff or _confirm_diff_default

    if _is_non_empty(dest) and mode is WriteMode.abort:
        raise DestinationExistsError(
            f"Destination {dest} is not empty. Re-run with --write-mode "
            "skip|diff|overwrite or pick an empty path."
        )

    dest.mkdir(parents=True, exist_ok=True)

    plan = _plan_writes(result.files, dest, mode, confirm)
    planned_rels = {_normalize(entry.path) for entry, _ in plan}

    parent = dest.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".agent-scaffold-stage-", dir=parent))
    try:
        for entry, _decision in plan:
            staged_path = staging_root / _normalize(entry.path)
            staged_path.parent.mkdir(parents=True, exist_ok=True)
            staged_path.write_text(entry.content, encoding="utf-8")
            _set_exec_bit(staged_path)

        report = WriteReport()
        for entry, decision in plan:
            rel = _normalize(entry.path)
            staged_path = staging_root / rel
            final_path = dest / rel
            final_path.parent.mkdir(parents=True, exist_ok=True)
            existed_before = final_path.exists()
            try:
                os.replace(staged_path, final_path)
            except OSError:
                if on_event is not None:
                    on_event(
                        ProgressEvent(
                            kind="file_written",
                            payload={
                                "path": rel,
                                "mode": "fail",
                                "bytes": len(entry.content),
                            },
                        )
                    )
                raise
            _set_exec_bit(final_path)
            if decision == "overwrite" and existed_before:
                report.overwritten.append(rel)
                event_mode = "overwrite"
            else:
                report.written.append(rel)
                event_mode = "new"
            if on_event is not None:
                on_event(
                    ProgressEvent(
                        kind="file_written",
                        payload={
                            "path": rel,
                            "mode": event_mode,
                            "bytes": len(entry.content),
                        },
                    )
                )

        # Track skips for the report (planned, but never written).
        skipped: list[str] = []
        for f in result.files:
            rel = _normalize(f.path)
            if rel in planned_rels:
                continue
            skipped.append(rel)
            if on_event is not None:
                on_event(
                    ProgressEvent(
                        kind="file_written",
                        payload={"path": rel, "mode": "skip", "bytes": len(f.content)},
                    )
                )
        report.skipped = skipped
        return report
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def _plan_writes(
    files: list[GeneratedFile],
    dest: Path,
    mode: WriteMode,
    confirm: Callable[[str, str], bool],
) -> list[tuple[GeneratedFile, str]]:
    """Decide which files to actually write, based on ``mode``.

    Returns a list of ``(file, decision)`` tuples where ``decision`` is one of
    ``"create"``, ``"overwrite"``. Files that are filtered out (skip mode or
    user declined a diff) are simply omitted from the returned list.
    """
    plan: list[tuple[GeneratedFile, str]] = []
    for entry in files:
        rel = _normalize(entry.path)
        final_path = dest / rel
        exists = final_path.exists()

        if not exists:
            plan.append((entry, "create"))
            continue

        if mode is WriteMode.abort:
            # Should never reach here because we already raised, but be safe.
            raise DestinationExistsError(f"File {rel} already exists at {final_path}")
        if mode is WriteMode.skip:
            continue
        if mode is WriteMode.overwrite:
            plan.append((entry, "overwrite"))
            continue
        if mode is WriteMode.diff:
            existing_text = final_path.read_text(encoding="utf-8").splitlines(keepends=True)
            new_text = entry.content.splitlines(keepends=True)
            diff = "".join(difflib.unified_diff(existing_text, new_text, fromfile=rel, tofile=rel))
            if not diff:
                continue
            if confirm(rel, diff):
                plan.append((entry, "overwrite"))
            continue

    return plan


def ensure_gitignore_defaults(project_dir: Path, *, extra: tuple[str, ...] = ()) -> list[str]:
    """Append :data:`DEFAULT_GITIGNORE_ENTRIES` to ``.gitignore`` as needed.

    - Creates ``.gitignore`` if missing, with the full default list under
      the ``# Added by agent-scaffold for secret safety`` header.
    - If ``.gitignore`` exists: appends only the entries that aren't already
      present (exact line match, ignoring leading/trailing whitespace).
      Existing user-authored lines are preserved verbatim.

    Returns the list of entries that were actually appended (empty if
    everything was already present). Honoured by both ``cmd_new`` after
    project generation and ``wire_credentials`` apply().
    """
    gitignore = project_dir / ".gitignore"
    want = list(DEFAULT_GITIGNORE_ENTRIES) + list(extra)
    existing_lines: list[str] = []
    existing_set: set[str] = set()
    if gitignore.is_file():
        existing_lines = gitignore.read_text(encoding="utf-8").splitlines()
        existing_set = {line.strip() for line in existing_lines if line.strip()}
    to_append = [entry for entry in want if entry not in existing_set]
    if not to_append:
        return []
    out_lines = list(existing_lines)
    if out_lines and out_lines[-1].strip():
        out_lines.append("")  # blank separator before our block
    out_lines.append(_GITIGNORE_HEADER)
    out_lines.extend(to_append)
    gitignore.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return to_append
