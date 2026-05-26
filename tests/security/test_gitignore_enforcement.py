"""Tests for ``writer.ensure_gitignore_defaults``.

Behaviour we lock in:

- New project (no ``.gitignore``): full default list is written under the
  ``# Added by agent-scaffold for secret safety`` header.
- Existing ``.gitignore`` missing some entries: just the missing ones are
  appended; existing lines (user-authored or not) are preserved verbatim.
- Existing ``.gitignore`` with all default entries: file is unchanged.
"""

from __future__ import annotations

from pathlib import Path

from agent_scaffold.writer import DEFAULT_GITIGNORE_ENTRIES, ensure_gitignore_defaults


def test_creates_gitignore_with_full_default_list(tmp_path: Path) -> None:
    appended = ensure_gitignore_defaults(tmp_path)
    assert set(appended) == set(DEFAULT_GITIGNORE_ENTRIES)
    text = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    for entry in DEFAULT_GITIGNORE_ENTRIES:
        assert entry in text, f"missing default entry: {entry}"
    assert "# Added by agent-scaffold for secret safety" in text


def test_existing_gitignore_only_missing_entries_appended(tmp_path: Path) -> None:
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text(
        "# user-authored block\n__pycache__/\n*.pyc\n",
        encoding="utf-8",
    )
    appended = ensure_gitignore_defaults(tmp_path)
    assert ".scaffold/" in appended
    assert ".env.local" in appended
    # Existing entries were not duplicated.
    assert "__pycache__/" not in appended
    text = gitignore.read_text(encoding="utf-8")
    # User block preserved at top.
    assert text.startswith("# user-authored block\n__pycache__/\n*.pyc\n")
    # Single occurrence of __pycache__/ even though it's in the default list.
    assert text.count("__pycache__/") == 1


def test_idempotent_when_all_entries_present(tmp_path: Path) -> None:
    gitignore = tmp_path / ".gitignore"
    body = "\n".join(DEFAULT_GITIGNORE_ENTRIES) + "\n"
    gitignore.write_text(body, encoding="utf-8")
    appended = ensure_gitignore_defaults(tmp_path)
    assert appended == []
    assert gitignore.read_text(encoding="utf-8") == body


def test_extra_entries_appended_alongside_defaults(tmp_path: Path) -> None:
    appended = ensure_gitignore_defaults(tmp_path, extra=(".myapp.cache",))
    assert ".myapp.cache" in appended
    text = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    assert ".myapp.cache" in text


def test_preserves_existing_trailing_blank_line_handling(tmp_path: Path) -> None:
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("foo\n", encoding="utf-8")
    ensure_gitignore_defaults(tmp_path)
    text = gitignore.read_text(encoding="utf-8")
    # A blank-line separator separates user content from our block.
    assert "foo\n\n# Added by agent-scaffold" in text
