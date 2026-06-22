"""Template + generation snapshots for ``agent-scaffold update``.

Two things share this module because they're keyed by the same identity (a
sha256 over the deployments tree the recipe was generated against):

- :func:`compute_template_sha` — content-addressed hash over the canonical
  ``(path, sha)`` list of the deployments ``docs/`` tree. Cheap; stable
  across cosmetic repo reshuffles. Recorded on ``manifest.template_snapshot_sha``
  so :func:`compute_template_sha` on the live tree can detect "the recipe
  changed since last generation".

- :func:`save_generation_snapshot` — a tar+gz of the **generated project
  files** at that generation, written to
  ``.scaffold/template-snapshots/<short_sha>.tgz``. This is the 3-way merge
  *base*: ``update`` extracts it and merges the user's on-disk version
  (``ours``) against the fresh re-generation (``theirs``) using this
  snapshot as the common ancestor.

Snapshots are small (Python source — typically < 100 KB per project) and
LRU-pruned to the last :data:`MAX_SNAPSHOTS` so the project doesn't
accumulate them. The directory is git-ignorable; users don't need to
commit it.
"""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from agent_scaffold._scaffold_dir import SCAFFOLD_DIR

SNAPSHOT_DIR = f"{SCAFFOLD_DIR}/template-snapshots"
SNAPSHOT_SUFFIX = ".tgz"
# How many snapshots to keep before LRU-pruning. 3 covers "previous + current +
# one for safety" without bloating the project.
MAX_SNAPSHOTS = 3
# Prefix length when shortening the sha for filenames.
_SHORT_SHA_LEN = 16


@dataclass(frozen=True)
class SnapshotInfo:
    sha: str
    path: Path
    bytes: int


# ---------------------------------------------------------------------------
# Template-tree hash (the change-detection key)
# ---------------------------------------------------------------------------


def _canonical_file_list(deployments_path: Path) -> list[tuple[str, str]]:
    """Return ``[(relative_posix_path, content_sha256)]`` sorted by path.

    Only includes the docs the LLM sees (``docs/`` subtree). Hidden files,
    build artifacts, and non-text helpers are skipped — we want the sha to
    be stable across cosmetic repo reshuffles.
    """
    docs_root = deployments_path / "docs"
    if not docs_root.is_dir():
        # Some test fixtures only have the top-level dir; fall back to all files.
        docs_root = deployments_path

    entries: list[tuple[str, str]] = []
    for path in sorted(docs_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(deployments_path).as_posix()
        # Skip hidden directories / files (.git, .DS_Store).
        if any(part.startswith(".") for part in path.relative_to(deployments_path).parts):
            continue
        if rel.endswith((".pyc", ".swp")):
            continue
        content_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append((rel, content_sha))
    return entries


def compute_template_sha(deployments_path: Path) -> str:
    """SHA-256 over the canonical ``(path, sha)`` list of the deployments tree.

    The same tree (regardless of disk path, mtime, etc.) always produces the
    same sha. A single byte change in any included file changes the sha.
    """
    digest = hashlib.sha256()
    for rel, content_sha in _canonical_file_list(deployments_path):
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content_sha.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def short_sha(sha: str) -> str:
    return sha[:_SHORT_SHA_LEN]


# ---------------------------------------------------------------------------
# Per-generation snapshots (the 3-way merge base)
# ---------------------------------------------------------------------------


def snapshot_dir(project_dir: Path) -> Path:
    return project_dir / SNAPSHOT_DIR


def snapshot_path(project_dir: Path, sha: str) -> Path:
    return snapshot_dir(project_dir) / f"{short_sha(sha)}{SNAPSHOT_SUFFIX}"


def has_snapshot(project_dir: Path, sha: str) -> bool:
    return snapshot_path(project_dir, sha).is_file()


def save_generation_snapshot(
    project_dir: Path,
    sha: str,
    files: Mapping[str, str],
) -> SnapshotInfo:
    """Tar+gz the ``{relative_path: text}`` file map, keyed by ``sha``.

    Returns the :class:`SnapshotInfo`. Idempotent: a second call with the
    same sha overwrites (so test re-runs don't accumulate).
    """
    target = snapshot_path(project_dir, sha)
    target.parent.mkdir(parents=True, exist_ok=True)

    # Write to a sibling tmp file then atomically rename, so a crashed save
    # doesn't leave a half-written ``.tgz`` that callers might trust.
    with tempfile.NamedTemporaryFile(delete=False, dir=str(target.parent), suffix=".tmp") as tmp:
        tmp_path = Path(tmp.name)
    try:
        with tarfile.open(tmp_path, "w:gz") as tar:
            for rel in sorted(files):
                content = files[rel].encode("utf-8")
                info = tarfile.TarInfo(name=rel)
                info.size = len(content)
                info.mtime = 0  # stable across runs so the tarball is reproducible
                import io

                tar.addfile(info, io.BytesIO(content))
        tmp_path.replace(target)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return SnapshotInfo(sha=sha, path=target, bytes=target.stat().st_size)


def load_generation_snapshot(project_dir: Path, sha: str, *, dest: Path | None = None) -> Path:
    """Extract the snapshot to ``dest`` (or a fresh tempdir). Returns the extracted root."""
    source = snapshot_path(project_dir, sha)
    if not source.is_file():
        raise FileNotFoundError(f"No generation snapshot found at {source}")
    out = dest if dest is not None else Path(tempfile.mkdtemp(prefix="agent-scaffold-snap-"))
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(source, "r:gz") as tar:
        members = list(tar.getmembers())
        for m in members:
            _reject_unsafe_member(m, out)
        # ``filter="data"`` is the safe default on 3.12+; falls back to no
        # filter on older Pythons (we already vetted paths above).
        try:
            tar.extractall(out, members=members, filter="data")  # noqa: S202 — members vetted by _reject_unsafe_member above
        except TypeError:
            tar.extractall(out, members=members)  # noqa: S202 — members vetted by _reject_unsafe_member above (pre-3.12 fallback)
    return out


def _reject_unsafe_member(member: tarfile.TarInfo, dest: Path) -> None:
    name = member.name
    if name.startswith("/") or ".." in Path(name).parts:
        raise ValueError(f"Refusing unsafe tar member: {name!r}")
    target = (dest / name).resolve()
    dest_resolved = dest.resolve()
    try:
        target.relative_to(dest_resolved)
    except ValueError as exc:
        raise ValueError(f"Tar member escapes dest: {name!r}") from exc


# ---------------------------------------------------------------------------
# LRU pruning + helpers
# ---------------------------------------------------------------------------


def list_snapshots(project_dir: Path) -> list[Path]:
    """Snapshot files sorted oldest-first (by mtime)."""
    root = snapshot_dir(project_dir)
    if not root.is_dir():
        return []
    snapshots = [p for p in root.iterdir() if p.is_file() and p.suffix == SNAPSHOT_SUFFIX]
    snapshots.sort(key=lambda p: p.stat().st_mtime)
    return snapshots


def prune_snapshots(project_dir: Path, *, keep: int = MAX_SNAPSHOTS) -> list[Path]:
    """Delete all but the ``keep`` most-recent snapshots. Returns the removed paths."""
    if keep < 0:
        raise ValueError("keep must be >= 0")
    snapshots = list_snapshots(project_dir)
    excess = max(0, len(snapshots) - keep)
    removed: list[Path] = []
    for path in snapshots[:excess]:
        try:
            path.unlink()
            removed.append(path)
        except FileNotFoundError:
            pass
    return removed


def cleanup_tempdir(path: Path) -> None:
    """Convenience: remove the tempdir ``load_generation_snapshot`` created."""
    shutil.rmtree(path, ignore_errors=True)


def iter_snapshot_files(extracted_root: Path) -> Iterable[tuple[str, Path]]:
    """Yield ``(relative_posix_path, absolute_path)`` for every file under the extract."""
    for path in sorted(extracted_root.rglob("*")):
        if path.is_file():
            yield path.relative_to(extracted_root).as_posix(), path


__all__ = [
    "MAX_SNAPSHOTS",
    "SNAPSHOT_DIR",
    "SnapshotInfo",
    "cleanup_tempdir",
    "compute_template_sha",
    "has_snapshot",
    "iter_snapshot_files",
    "list_snapshots",
    "load_generation_snapshot",
    "prune_snapshots",
    "save_generation_snapshot",
    "short_sha",
    "snapshot_dir",
    "snapshot_path",
]
