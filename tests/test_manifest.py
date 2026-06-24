"""Unit tests for agent_scaffold.manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_scaffold.manifest import (
    SCHEMA_VERSION,
    Manifest,
    ManifestFile,
    ManifestNotFoundError,
    UpdateEntry,
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


# ---------------------------------------------------------------------------
# Q8 — v1 → v2 migration + new fields
# ---------------------------------------------------------------------------


def _write_v1_manifest(tmp_path: Path) -> Path:
    target = manifest_path(tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "recipe": "demo",
                "language": "python",
                "framework": "none",
                "topology": None,
                "roles": [],
                "model": "claude-test",
                "generated_at": "2026-05-24T00:00:00+00:00",
                "files": [],
            }
        ),
        encoding="utf-8",
    )
    return target


def test_v1_manifest_migrates_to_v2_with_new_fields(tmp_path: Path) -> None:
    _write_v1_manifest(tmp_path)
    manifest = read_manifest(tmp_path)
    assert manifest.schema_version == SCHEMA_VERSION
    # New fields populated with defaults.
    assert manifest.template_snapshot_sha is None
    assert manifest.answers == {}
    assert manifest.update_history == []
    # Existing fields preserved.
    assert manifest.recipe == "demo"
    assert manifest.language == "python"


def test_v1_migration_persists_to_disk(tmp_path: Path) -> None:
    target = _write_v1_manifest(tmp_path)
    read_manifest(tmp_path)
    saved = json.loads(target.read_text(encoding="utf-8"))
    assert saved["schema_version"] == SCHEMA_VERSION
    assert "update_history" in saved


def test_entry_point_and_smoke_check_round_trip(tmp_path: Path) -> None:
    """The entry-point + smoke contract persists and reloads verbatim."""
    manifest = _make_manifest(tmp_path).model_copy(
        update={
            "entry_point": "app/main.py",
            "smoke_check": "uv run python -c 'from app.main import agent; print(\"ok\")'",
        }
    )
    write_manifest(tmp_path, manifest)
    loaded = read_manifest(tmp_path)
    assert loaded.entry_point == "app/main.py"
    assert loaded.smoke_check == "uv run python -c 'from app.main import agent; print(\"ok\")'"


def test_older_manifest_loads_entry_point_as_none(tmp_path: Path) -> None:
    """A v1 manifest (no entry_point/smoke_check) migrates + loads with the new
    fields defaulting to None — consumers fall back to heuristics."""
    _write_v1_manifest(tmp_path)
    manifest = read_manifest(tmp_path)
    assert manifest.entry_point is None
    assert manifest.smoke_check is None


def test_update_entry_round_trips_through_json() -> None:
    entry = UpdateEntry(
        timestamp="2026-05-26T00:00:00+00:00",
        from_schema=1,
        to_schema=2,
        from_template_sha="abc",
        to_template_sha="def",
        model="claude-opus-4-7",
        files_added=["src/new.py"],
        files_modified=["src/main.py"],
        files_removed=[],
        files_conflicted=["src/conflict.py"],
    )
    raw = entry.model_dump_json()
    rehydrated = UpdateEntry.model_validate_json(raw)
    assert rehydrated == entry


def test_manifest_with_update_history_roundtrips(tmp_path: Path) -> None:
    manifest = Manifest(
        recipe="demo",
        language="python",
        framework="none",
        model="claude-test",
        generated_at="2026-05-24T00:00:00+00:00",
        template_snapshot_sha="abc",
        answers={"project_name": "demo"},
        update_history=[
            UpdateEntry(
                timestamp="2026-05-26T00:00:00+00:00",
                from_schema=1,
                to_schema=2,
                to_template_sha="def",
                model="claude-test",
                files_added=["a.py"],
            )
        ],
    )
    write_manifest(tmp_path, manifest)
    rehydrated = read_manifest(tmp_path)
    assert rehydrated.template_snapshot_sha == "abc"
    assert rehydrated.answers == {"project_name": "demo"}
    assert len(rehydrated.update_history) == 1
    assert rehydrated.update_history[0].files_added == ["a.py"]


def test_unknown_schema_version_raises(tmp_path: Path) -> None:
    target = manifest_path(tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({"schema_version": 99, "recipe": "x"}),
        encoding="utf-8",
    )
    with pytest.raises(ManifestNotFoundError):
        read_manifest(tmp_path)
