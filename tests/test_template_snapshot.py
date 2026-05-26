"""Tests for ``agent_scaffold.template_snapshot``."""

from __future__ import annotations

import os
import tarfile
import time
from pathlib import Path

import pytest

from agent_scaffold.template_snapshot import (
    MAX_SNAPSHOTS,
    cleanup_tempdir,
    compute_template_sha,
    has_snapshot,
    list_snapshots,
    load_generation_snapshot,
    prune_snapshots,
    save_generation_snapshot,
    short_sha,
    snapshot_path,
)


def _make_deployments_tree(root: Path) -> None:
    (root / "docs" / "recipes").mkdir(parents=True)
    (root / "docs" / "recipes" / "demo.md").write_text("# demo\n", encoding="utf-8")
    (root / "docs" / "patterns").mkdir(parents=True)
    (root / "docs" / "patterns" / "react.md").write_text("# react\n", encoding="utf-8")


def test_compute_template_sha_deterministic(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    _make_deployments_tree(a)
    _make_deployments_tree(b)
    assert compute_template_sha(a) == compute_template_sha(b)


def test_compute_template_sha_changes_on_edit(tmp_path: Path) -> None:
    deployments = tmp_path / "d"
    _make_deployments_tree(deployments)
    before = compute_template_sha(deployments)
    (deployments / "docs" / "recipes" / "demo.md").write_text("# demo v2\n", encoding="utf-8")
    after = compute_template_sha(deployments)
    assert before != after


def test_compute_template_sha_ignores_hidden_files(tmp_path: Path) -> None:
    deployments = tmp_path / "d"
    _make_deployments_tree(deployments)
    before = compute_template_sha(deployments)
    (deployments / "docs" / ".DS_Store").write_text("junk", encoding="utf-8")
    (deployments / ".git").mkdir()
    (deployments / ".git" / "HEAD").write_text("ref: x", encoding="utf-8")
    after = compute_template_sha(deployments)
    assert before == after, "hidden files / dotdirs must not influence the sha"


def test_save_and_load_generation_snapshot_round_trip(tmp_path: Path) -> None:
    files = {
        "src/main.py": "print('hi')\n",
        "README.md": "# project\n",
        "pyproject.toml": "[project]\nname='x'\n",
    }
    project = tmp_path / "proj"
    project.mkdir()
    info = save_generation_snapshot(project, "deadbeef" * 8, files)
    assert info.path.is_file()
    assert info.bytes > 0
    extract_dir = load_generation_snapshot(project, info.sha)
    try:
        for rel, expected in files.items():
            actual = (extract_dir / rel).read_text(encoding="utf-8")
            assert actual == expected, f"round-trip mismatch for {rel}"
    finally:
        cleanup_tempdir(extract_dir)


def test_save_generation_snapshot_uses_short_sha_in_filename(tmp_path: Path) -> None:
    sha = "feedface" * 8
    project = tmp_path / "proj"
    project.mkdir()
    info = save_generation_snapshot(project, sha, {"a.txt": "x"})
    assert info.path.name == f"{short_sha(sha)}.tgz"


def test_has_snapshot_reports_existence(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    assert not has_snapshot(project, "abc")
    save_generation_snapshot(project, "abc", {"f.txt": "x"})
    assert has_snapshot(project, "abc")


def test_prune_snapshots_keeps_last_n(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    shas = [f"sha-{i:04d}" + "0" * 12 for i in range(MAX_SNAPSHOTS + 2)]
    for sha in shas:
        save_generation_snapshot(project, sha, {"a": str(sha)})
        # Spread out mtimes so the LRU ordering is deterministic.
        time.sleep(0.01)
    removed = prune_snapshots(project)
    assert len(removed) == 2
    remaining = list_snapshots(project)
    assert len(remaining) == MAX_SNAPSHOTS


def test_prune_snapshots_no_op_when_under_limit(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    save_generation_snapshot(project, "abc", {"a": "x"})
    assert prune_snapshots(project) == []
    assert len(list_snapshots(project)) == 1


def test_load_rejects_unsafe_tar_member(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    sha = "evil0000" + "0" * 16
    target = snapshot_path(project, sha)
    target.parent.mkdir(parents=True)
    # Hand-craft a tarball with a path traversal entry.
    with tarfile.open(target, "w:gz") as tar:
        info = tarfile.TarInfo(name="../escape.txt")
        info.size = 4
        import io

        tar.addfile(info, io.BytesIO(b"boom"))
    with pytest.raises(ValueError):
        load_generation_snapshot(project, sha)


def test_snapshot_is_idempotent_for_same_sha(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    sha = "cafebabe" * 8
    info1 = save_generation_snapshot(project, sha, {"a": "v1"})
    # Give the FS a moment so a second save with newer content is distinguishable.
    time.sleep(0.05)
    info2 = save_generation_snapshot(project, sha, {"a": "v2"})
    assert info2.sha == sha
    # The same path is reused (overwrite), so the count stays at 1.
    assert len(list_snapshots(project)) == 1
    assert os.path.exists(info1.path)
