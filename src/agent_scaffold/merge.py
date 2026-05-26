"""Three-way file merge for ``agent-scaffold update``.

Copier-style template evolution: given three versions of a file —

- ``base``   — what the template produced at the *previous* generation
- ``ours``   — what's on disk now (the user's edits applied to base)
- ``theirs`` — what the template produces *now* (the fresh generation)

…produce a merged file. When the user and the template touched different
lines, both edits land cleanly. When they touched the same lines, emit
``<<<<<<< user / ======= / >>>>>>> template`` conflict markers and let the
user resolve in their editor.

We deliberately use stdlib ``difflib`` rather than pulling in ``mergetools``
or a real diff3 binary — the cost of a new dep wouldn't buy us much for the
text shapes scaffold output produces (markdown, Python, YAML, TOML). The
algorithm is straightforward:

    1. Run ``SequenceMatcher`` between base and ours → opcodes describing
       the user's edits as ranges of line indices.
    2. Same between base and theirs.
    3. Walk the two opcode streams over the base index in lockstep. For
       any base region:
         - if neither side changed it → keep base
         - if only one side changed it → take that side's text
         - if both changed it → emit the marker block
    4. Reassemble into a single text.

Binary files (those with a NUL byte in the first 8 KB) are returned as
``ours`` with the ``binary=True`` flag — the caller surfaces a warning.

Line endings: we normalise to ``\n`` for diffing but preserve the *user's*
detected ending on write (CRLF input stays CRLF). Template-driven endings
don't churn the user's repo.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal

MARKER_USER = "<<<<<<< user"
MARKER_SEP = "======="
MARKER_TEMPLATE = ">>>>>>> template"

_BINARY_PROBE_BYTES = 8000

LineEnding = Literal["\n", "\r\n", "\r"]


@dataclass(frozen=True)
class MergeResult:
    text: str
    """The merged file content, ready to write."""

    conflicted: bool
    """``True`` iff at least one hunk could not be resolved automatically."""

    hunks_clean: int
    """Number of regions merged without conflict (added/removed/modified-by-one-side)."""

    hunks_conflicted: int
    """Number of regions emitted with conflict markers."""

    binary: bool = False
    """``True`` iff we declined to merge a binary file and returned ``ours``."""


# ---------------------------------------------------------------------------
# Text/binary detection + line-ending handling
# ---------------------------------------------------------------------------


def _is_binary(content: bytes | str) -> bool:
    """Heuristic: NUL byte in the first 8 KB → binary.

    Cheap and matches what ``git`` does. Anything we can't decode as utf-8
    also counts as binary.
    """
    if isinstance(content, str):
        sample = content.encode("utf-8", errors="replace")[:_BINARY_PROBE_BYTES]
    else:
        sample = content[:_BINARY_PROBE_BYTES]
    return b"\0" in sample


def _detect_line_ending(text: str) -> LineEnding:
    """Pick the dominant newline. Falls back to ``\\n`` for empty / single-line text."""
    if "\r\n" in text:
        return "\r\n"
    if "\r" in text and "\n" not in text:
        return "\r"
    return "\n"


def _normalise(text: str) -> str:
    """Collapse CRLF / CR endings to ``\\n`` for stable diffing."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _restore_line_ending(text: str, ending: LineEnding) -> str:
    if ending == "\n":
        return text
    return text.replace("\n", ending)


# ---------------------------------------------------------------------------
# Three-way merge
# ---------------------------------------------------------------------------


def three_way_merge(
    base: str | bytes,
    ours: str | bytes,
    theirs: str | bytes,
) -> MergeResult:
    """Merge ``theirs`` onto ``ours`` using ``base`` as the common ancestor.

    Always returns a ``MergeResult`` — never raises for content reasons.
    """
    if _is_binary(base) or _is_binary(ours) or _is_binary(theirs):
        ours_text = ours if isinstance(ours, str) else ours.decode("utf-8", errors="replace")
        return MergeResult(
            text=ours_text, conflicted=False, hunks_clean=0, hunks_conflicted=0, binary=True
        )

    base_text = base if isinstance(base, str) else base.decode("utf-8")
    ours_text = ours if isinstance(ours, str) else ours.decode("utf-8")
    theirs_text = theirs if isinstance(theirs, str) else theirs.decode("utf-8")

    target_ending = _detect_line_ending(ours_text)

    base_lines = _normalise(base_text).splitlines(keepends=True)
    ours_lines = _normalise(ours_text).splitlines(keepends=True)
    theirs_lines = _normalise(theirs_text).splitlines(keepends=True)

    if base_lines == ours_lines == theirs_lines:
        return MergeResult(
            text=_restore_line_ending("".join(ours_lines), target_ending),
            conflicted=False,
            hunks_clean=0,
            hunks_conflicted=0,
        )
    if base_lines == ours_lines:
        # User didn't touch the file — take the template wholesale.
        return MergeResult(
            text=_restore_line_ending("".join(theirs_lines), target_ending),
            conflicted=False,
            hunks_clean=1,
            hunks_conflicted=0,
        )
    if base_lines == theirs_lines:
        # Template didn't change the file — keep the user's edits.
        return MergeResult(
            text=_restore_line_ending("".join(ours_lines), target_ending),
            conflicted=False,
            hunks_clean=0,
            hunks_conflicted=0,
        )

    merged_lines, clean, conflicts = _merge_hunks(base_lines, ours_lines, theirs_lines)
    return MergeResult(
        text=_restore_line_ending("".join(merged_lines), target_ending),
        conflicted=conflicts > 0,
        hunks_clean=clean,
        hunks_conflicted=conflicts,
    )


# ---------------------------------------------------------------------------
# Hunk-level merge engine
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Edit:
    """One opcode from base → side. ``tag`` is difflib's ``equal/replace/insert/delete``."""

    tag: str
    base_lo: int
    base_hi: int
    side_lo: int
    side_hi: int


def _edits(base: list[str], side: list[str]) -> list[_Edit]:
    matcher = SequenceMatcher(a=base, b=side, autojunk=False)
    return [
        _Edit(tag=tag, base_lo=i1, base_hi=i2, side_lo=j1, side_hi=j2)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
    ]


def _merge_hunks(base: list[str], ours: list[str], theirs: list[str]) -> tuple[list[str], int, int]:
    """Walk the base index in lockstep across the two opcode streams.

    Returns ``(merged_lines, clean_hunks, conflicted_hunks)``.

    Algorithm: maintain a base cursor ``i`` and an index into each change
    list. At each step:

    - If a change on either side has ``base_lo <= i``, it's "live" at the
      current cursor. Both live → conflict (or dedupe if identical). Only
      one side live → take that side's text and advance.
    - If no change is live, copy unchanged base lines forward until the
      nearest upcoming change.

    The key invariant: insertions at *different* base positions are never
    grouped together (the previous bug). Each side's index advances
    independently.
    """
    us = [e for e in _edits(base, ours) if e.tag != "equal"]
    them = [e for e in _edits(base, theirs) if e.tag != "equal"]

    out: list[str] = []
    clean = 0
    conflicted = 0
    i = 0
    n = len(base)
    ui = 0  # index into us
    ti = 0  # index into them

    while ui < len(us) or ti < len(them) or i < n:
        u = us[ui] if ui < len(us) else None
        t = them[ti] if ti < len(them) else None

        us_live = u is not None and u.base_lo <= i
        them_live = t is not None and t.base_lo <= i

        if us_live and them_live:
            assert u is not None and t is not None  # for type narrowing
            us_text = ours[u.side_lo : u.side_hi]
            them_text = theirs[t.side_lo : t.side_hi]
            if us_text == them_text:
                out.extend(us_text)
                clean += 1
            else:
                out.extend(_conflict_block(us_text, them_text))
                conflicted += 1
            i = max(i, u.base_hi, t.base_hi)
            ui += 1
            ti += 1
            continue

        if us_live:
            assert u is not None
            out.extend(ours[u.side_lo : u.side_hi])
            clean += 1
            i = max(i, u.base_hi)
            ui += 1
            continue

        if them_live:
            assert t is not None
            out.extend(theirs[t.side_lo : t.side_hi])
            clean += 1
            i = max(i, t.base_hi)
            ti += 1
            continue

        # Neither side has a live change at i. Copy unchanged base lines up
        # to the next pending change (or to end-of-base).
        next_us = u.base_lo if u is not None else n
        next_them = t.base_lo if t is not None else n
        next_change = min(next_us, next_them)
        if next_change > i:
            out.extend(base[i:next_change])
            i = next_change
        else:
            # No more work — both sides exhausted and i == n.
            break

    return out, clean, conflicted


def _find_change_starting_at_or_after(changes: list[_Edit], i: int) -> _Edit | None:
    """Retained for clarity / future use; not called by the rewritten merge loop."""
    for e in changes:
        if e.base_hi <= i:
            continue
        return e
    return None


def _consume_overlapping(changes: list[_Edit], start: int) -> list[_Edit]:
    """Pop and return the contiguous chain of changes from this side starting at or near ``start``.

    Changes are 'overlapping' if their base ranges touch or interleave with
    the chain so far. We mutate ``changes`` so subsequent calls don't revisit.
    """
    if not changes:
        return []
    region: list[_Edit] = []
    end = start
    while changes:
        head = changes[0]
        if head.base_lo > end:
            break
        # The head touches the region — include it and extend ``end``.
        region.append(head)
        end = max(end, head.base_hi)
        changes.pop(0)
    return region


def _gather_side(region: list[_Edit], side_lines: list[str]) -> list[str]:
    out: list[str] = []
    for e in region:
        out.extend(side_lines[e.side_lo : e.side_hi])
    return out


def _conflict_block(us_text: list[str], them_text: list[str]) -> list[str]:
    """Wrap the two sides in conflict markers (each marker on its own line)."""
    block = [MARKER_USER + "\n"]
    block.extend(_ensure_trailing_newline(us_text))
    block.append(MARKER_SEP + "\n")
    block.extend(_ensure_trailing_newline(them_text))
    block.append(MARKER_TEMPLATE + "\n")
    return block


def _ensure_trailing_newline(lines: list[str]) -> list[str]:
    """Make sure each chunk ends in ``\\n`` so the marker that follows starts at col 0."""
    if not lines:
        return lines
    if lines[-1].endswith("\n"):
        return lines
    return [*lines[:-1], lines[-1] + "\n"]


def _any_overlapping(us: list[_Edit], them: list[_Edit], i: int) -> bool:
    """Have we exhausted both change lists past position ``i``?"""
    return any(e.base_hi > i for e in us) or any(e.base_hi > i for e in them)


# ---------------------------------------------------------------------------
# Marker-resolution check (used by ``update --continue``)
# ---------------------------------------------------------------------------


def has_unresolved_markers(text: str) -> bool:
    """``True`` if any line begins with one of our merge markers.

    Anchored at start-of-line so an embedded string literal like
    ``"<<<<<<< user"`` inside source code doesn't trip the check.
    """
    needles = (MARKER_USER, MARKER_SEP, MARKER_TEMPLATE)
    for line in text.splitlines():
        if line.startswith(needles):
            return True
    return False


def count_unresolved_markers(text: str) -> int:
    """Total count of marker-starting lines (sum across all three marker kinds)."""
    needles = (MARKER_USER, MARKER_SEP, MARKER_TEMPLATE)
    return sum(1 for line in text.splitlines() if line.startswith(needles))


__all__ = [
    "MARKER_SEP",
    "MARKER_TEMPLATE",
    "MARKER_USER",
    "MergeResult",
    "count_unresolved_markers",
    "has_unresolved_markers",
    "three_way_merge",
]
