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
    """
    us = _edits(base, ours)
    them = _edits(base, theirs)

    out: list[str] = []
    clean = 0
    conflicted = 0
    i = 0  # base index
    n = len(base)

    # Map a base position to "is this position changed by us / them?"
    us_changes = [e for e in us if e.tag != "equal"]
    them_changes = [e for e in them if e.tag != "equal"]

    while i < n or _any_overlapping(us_changes, them_changes, i):
        u = _find_change_starting_at_or_after(us_changes, i)
        t = _find_change_starting_at_or_after(them_changes, i)

        # If neither side has any remaining change at/after i, copy the rest of base.
        if u is None and t is None:
            out.extend(base[i:n])
            break

        # Pick the nearest upcoming change. If they overlap on the base range,
        # merge them as one region (potentially conflicted).
        next_us = u.base_lo if u is not None else n
        next_them = t.base_lo if t is not None else n
        next_change = min(next_us, next_them)

        # Copy through any unchanged base before the next change.
        if next_change > i:
            out.extend(base[i:next_change])
            i = next_change

        # Now i is at the start of a change region. Find the *end* of the merged
        # region: union of overlapping change ranges from both sides.
        region_us = _consume_overlapping(us_changes, i)
        region_them = _consume_overlapping(them_changes, i)

        # If only one side has a change in this region, take it.
        if region_us and not region_them:
            for e in region_us:
                out.extend(ours[e.side_lo : e.side_hi])
            clean += 1
            i = max((e.base_hi for e in region_us), default=i)
        elif region_them and not region_us:
            for e in region_them:
                out.extend(theirs[e.side_lo : e.side_hi])
            clean += 1
            i = max((e.base_hi for e in region_them), default=i)
        else:
            # Both sides changed overlapping regions.
            us_text = _gather_side(region_us, ours)
            them_text = _gather_side(region_them, theirs)
            if us_text == them_text:
                # Same edit on both sides — accept once.
                out.extend(us_text)
                clean += 1
            else:
                out.extend(_conflict_block(us_text, them_text))
                conflicted += 1
            i = max(
                max((e.base_hi for e in region_us), default=i),
                max((e.base_hi for e in region_them), default=i),
            )

    return out, clean, conflicted


def _find_change_starting_at_or_after(changes: list[_Edit], i: int) -> _Edit | None:
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
