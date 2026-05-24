"""Unit tests for agent_scaffold.manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_scaffold.manifest import (
    Manifest,
    ManifestFile,
    ManifestNotFoundError,
    build_file_entries,
    hash_file,
    manifest_path,
    read_manifest,
    update_file_entry,
    write_manifest,
)


def _make_manifest(tmp_path: Path) -> Manifest:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# demo\n", encoding="utf-8")
    return Manifest(
        recipe="demo-recipe",
        language="python",
        framework="langgraph",
        topology="single",
        roles=[{"name": "intake", "description": "..."}],
        model="claude-sonnet-4-6",
        generated_at="2026-05-24T00:00:00+00:00",
        files=build_file_entries(tmp_path, ["src/main.py", "README.md", "missing.txt"]),
    )


def test_hash_file_counts_lines_and_hashes(tmp_path: Path) -> None:
    p = tmp_path / "f.txt"
    p.write_text("a\nb\nc\n", encoding="utf-8")
    lines, sha = hash_file(p)
    assert lines == 3
    assert len(sha) == 64


def test_hash_file_counts_unterminated_final_line(tmp_path: Path) -> None:
    p = tmp_path / "f.txt"
    p.write_text("a\nb", encoding="utf-8")
    lines, _sha = hash_file(p)
    assert lines == 2


def test_build_file_entries_skips_missing(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hi\n", encoding="utf-8")
    entries = build_file_entries(tmp_path, ["a.txt", "missing.txt"])
    assert [e.path for e in entries] == ["a.txt"]


def test_write_read_roundtrip(tmp_path: Path) -> None:
    manifest = _make_manifest(tmp_path)
    target = write_manifest(tmp_path, manifest)
    assert target == manifest_path(tmp_path)
    assert target.is_file()
    loaded = read_manifest(tmp_path)
    assert loaded.recipe == "demo-recipe"
    assert loaded.language == "python"
    assert loaded.framework == "langgraph"
    assert loaded.model == "claude-sonnet-4-6"
    assert {f.path for f in loaded.files} == {"src/main.py", "README.md"}


def test_read_manifest_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(ManifestNotFoundError):
        read_manifest(tmp_path)


def test_read_manifest_invalid_json_raises(tmp_path: Path) -> None:
    target = manifest_path(tmp_path)
    target.parent.mkdir(parents=True)
    target.write_text("{not json", encoding="utf-8")
    with pytest.raises(ManifestNotFoundError):
        read_manifest(tmp_path)


def test_update_file_entry_replaces_existing(tmp_path: Path) -> None:
    manifest = _make_manifest(tmp_path)
    # Change the file's contents on disk.
    (tmp_path / "src" / "main.py").write_text("x = 2\ny = 3\n", encoding="utf-8")
    updated = update_file_entry(manifest, tmp_path, "src/main.py")
    by_path = {f.path: f for f in updated.files}
    assert by_path["src/main.py"].lines == 2
    # README untouched.
    assert by_path["README.md"].lines == manifest.files[1].lines


def test_update_file_entry_drops_when_file_removed(tmp_path: Path) -> None:
    manifest = _make_manifest(tmp_path)
    (tmp_path / "README.md").unlink()
    updated = update_file_entry(manifest, tmp_path, "README.md")
    assert all(f.path != "README.md" for f in updated.files)


def test_update_file_entry_adds_when_new(tmp_path: Path) -> None:
    manifest = _make_manifest(tmp_path)
    (tmp_path / "new.txt").write_text("hi\n", encoding="utf-8")
    updated = update_file_entry(manifest, tmp_path, "new.txt")
    assert any(f.path == "new.txt" for f in updated.files)


def test_manifest_file_schema_explicit() -> None:
    """ManifestFile must stay a stable Pydantic model — regen on disk reads this shape."""
    raw = json.dumps({"path": "a.txt", "lines": 1, "sha256": "deadbeef"})
    parsed = ManifestFile.model_validate_json(raw)
    assert parsed.path == "a.txt"
    assert parsed.sha256 == "deadbeef"
