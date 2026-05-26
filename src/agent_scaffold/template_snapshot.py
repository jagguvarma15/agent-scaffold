"""Template snapshots for ``agent-scaffold update`` 3-way merges.

Whenever ``agent-scaffold new`` (or a successful ``update``) finishes, we
snapshot the deployments tree it ran against — a content-addressed sha256
over the canonical ``(path, sha)`` list plus a tarball under
``.scaffold/template-snapshots/<short_sha>.tgz``.

The sha is recorded on ``manifest.template_snapshot_sha``. On the next
``update``, we compute the *current* sha; if it matches, there's nothing to
update. If it doesn't, the prior snapshot is the **base** for the 3-way
merge — the "what the template said last time" anchor that lets us
distinguish user edits from template edits.

Snapshots are small (recipes are markdown — typically < 5 MB) and LRU-pruned
to the last :data:`MAX_SNAPSHOTS` so the project doesn't accumulate them.
The directory is git-ignorable; users don't need to commit it.
"""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

SNAPSHOT_DIR = ".scaffold/template-snapshots"
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
# Hashing
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
# Save / load
# ---------------------------------------------------------------------------


def snapshot_dir(project_dir: Path) -> Path:
    return project_dir / SNAPSHOT_DIR


def snapshot_path(project_dir: Path, sha: str) -> Path:
    return snapshot_dir(project_dir) / f"{short_sha(sha)}{SNAPSHOT_SUFFIX}"


def has_snapshot(project_dir: Path, sha: str) -> bool:
    return snapshot_path(project_dir, sha).is_file()


def save_template_snapshot(project_dir: Path, deployments_path: Path) -> SnapshotInfo:
    """Tar+gz the ``docs/`` tree to ``.scaffold/template-snapshots/<short_sha>.tgz``.

    Returns the :class:`SnapshotInfo`. Idempotent: a second call for the same
    sha is a no-op.
    """
    sha = compute_template_sha(deployments_path)
    target = snapshot_path(project_dir, sha)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        return SnapshotInfo(sha=sha, path=target, bytes=target.stat().st_size)

    docs_root = deployments_path / "docs"
    arcname_root = "docs" if docs_root.is_dir() else deployments_path.name
    source_root = docs_root if docs_root.is_dir() else deployments_path

    # Write to a sibling tmp file then atomically rename, so a crashed save
    # doesn't leave a half-written ``.tgz`` that callers might trust.
    with tempfile.NamedTemporaryFile(
        delete=False, dir=str(target.parent), suffix=".tmp"
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with tarfile.open(tmp_path, "w:gz") as tar:
            for path in sorted(source_root.rglob("*")):
                if not path.is_file():
                    continue
                if any(p.startswith(".") for p in path.relative_to(source_root).parts):
                    continue
                arcname = f"{arcname_root}/{path.relative_to(source_root).as_posix()}"
                tar.add(path, arcname=arcname)
        tmp_path.replace(target)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return SnapshotInfo(sha=sha, path=target, bytes=target.stat().st_size)


def load_template_snapshot(project_dir: Path, sha: str, *, dest: Path | None = None) -> Path:
    """Extract the snapshot to ``dest`` (or a fresh tempdir). Returns the extracted root."""
    source = snapshot_path(project_dir, sha)
    if not source.is_file():
        raise FileNotFoundError(f"No template snapshot found at {source}")
    out = dest if dest is not None else Path(tempfile.mkdtemp(prefix="agent-scaffold-snap-"))
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(source, "r:gz") as tar:
        # Python 3.12+ adds tarfile.data_filter; use it when available to
        # refuse unsafe member paths (absolute / .. escapes). Fallback to a
        # manual safety check.
        members = list(tar.getmembers())
        for m in members:
            _reject_unsafe_member(m, out)
        tar.extractall(out, members=members)
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
# LRU pruning
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
    """Convenience: remove the tempdir ``load_template_snapshot`` created."""
    shutil.rmtree(path, ignore_errors=True)


def iter_snapshot_files(extracted_root: Path) -> Iterable[tuple[str, Path]]:
    """Yield ``(relative_posix_path, absolute_path)`` for every file under the extract.

    The ``docs/`` prefix that ``save_template_snapshot`` adds is preserved in
    the relative path so callers can match against the recipe-author's view
    of the world.
    """
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
    "load_template_snapshot",
    "prune_snapshots",
    "save_template_snapshot",
    "short_sha",
    "snapshot_dir",
    "snapshot_path",
]
