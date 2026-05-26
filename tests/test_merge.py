"""Tests for ``agent_scaffold.merge``.

These pin the 3-way merge semantics the ``update`` subcommand relies on.
Each test names the scenario in human terms so a regression is easy to
diagnose without re-reading the algorithm.
"""

from __future__ import annotations

from agent_scaffold.merge import (
    MARKER_SEP,
    MARKER_TEMPLATE,
    MARKER_USER,
    count_unresolved_markers,
    has_unresolved_markers,
    three_way_merge,
)


def test_identical_base_ours_theirs_returns_unchanged() -> None:
    text = "a\nb\nc\n"
    result = three_way_merge(text, text, text)
    assert result.text == text
    assert result.conflicted is False
    assert result.hunks_conflicted == 0


def test_only_ours_changed_keeps_ours() -> None:
    base = "a\nb\nc\n"
    ours = "a\nB\nc\n"
    theirs = base
    result = three_way_merge(base, ours, theirs)
    assert result.text == ours
    assert not result.conflicted


def test_only_theirs_changed_takes_theirs() -> None:
    base = "a\nb\nc\n"
    ours = base
    theirs = "a\nB\nc\n"
    result = three_way_merge(base, ours, theirs)
    assert result.text == theirs
    assert not result.conflicted


def test_disjoint_changes_both_applied_without_conflict() -> None:
    base = "a\nb\nc\nd\ne\n"
    ours = "A\nb\nc\nd\ne\n"  # changed line 1
    theirs = "a\nb\nc\nd\nE\n"  # changed line 5
    result = three_way_merge(base, ours, theirs)
    assert not result.conflicted
    assert "A\n" in result.text
    assert "E\n" in result.text


def test_overlapping_changes_emit_conflict_markers() -> None:
    base = "a\nb\nc\n"
    ours = "a\nUSER\nc\n"
    theirs = "a\nTEMPLATE\nc\n"
    result = three_way_merge(base, ours, theirs)
    assert result.conflicted is True
    assert MARKER_USER in result.text
    assert MARKER_SEP in result.text
    assert MARKER_TEMPLATE in result.text
    assert "USER" in result.text
    assert "TEMPLATE" in result.text


def test_same_edit_on_both_sides_dedupes() -> None:
    """If user and template happen to make the same change, no conflict."""
    base = "a\nb\nc\n"
    ours = "a\nBOTH\nc\n"
    theirs = "a\nBOTH\nc\n"
    result = three_way_merge(base, ours, theirs)
    assert not result.conflicted
    assert result.text.count("BOTH") == 1


def test_binary_file_falls_back_to_ours() -> None:
    base = b"hello\x00world"
    ours = b"hello\x00local-edit"
    theirs = b"hello\x00template-edit"
    result = three_way_merge(base, ours, theirs)
    assert result.binary is True
    assert "local-edit" in result.text
    assert "template-edit" not in result.text


def test_crlf_line_endings_preserved() -> None:
    base = "a\r\nb\r\nc\r\n"
    ours = "A\r\nb\r\nc\r\n"
    theirs = "a\r\nb\r\nC\r\n"
    result = three_way_merge(base, ours, theirs)
    assert "\r\n" in result.text
    assert "A\r\n" in result.text
    assert "C\r\n" in result.text


def test_marker_detection_anchored_at_start_of_line() -> None:
    """A string literal containing the marker must not trip the check."""
    safe = 'x = "<<<<<<< user inside a string"\n'
    assert not has_unresolved_markers(safe)
    marked = "ok\n<<<<<<< user\nA\n=======\nB\n>>>>>>> template\n"
    assert has_unresolved_markers(marked)
    assert count_unresolved_markers(marked) == 3


def test_added_lines_in_both_at_different_positions() -> None:
    """User inserts a line at the top; template inserts a line at the bottom."""
    base = "b\nc\n"
    ours = "header\nb\nc\n"
    theirs = "b\nc\nfooter\n"
    result = three_way_merge(base, ours, theirs)
    assert not result.conflicted
    assert "header\n" in result.text
    assert "footer\n" in result.text


def test_empty_base_with_both_sides_changed() -> None:
    """Edge case: brand-new file the user edited before the template caught up."""
    result = three_way_merge("", "user line\n", "template line\n")
    # Both sides "changed" from empty → conflict region.
    assert result.conflicted
