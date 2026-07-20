"""Copy capability template files into the generated project.

Each capability may declare ``emit_files: [{source, dest}, ...]`` in its
frontmatter. ``source`` is relative to the capability's directory under
``docs/capabilities/<kind>/``; ``dest`` is relative to the generated
project root. A trailing ``/**`` on ``source`` glob-expands.

The copier runs after the LLM has written its files (``write_project``)
and before the post-gen formatter. It NEVER overwrites a file the model
emitted in the same path — collisions log a warning and SKIP that file
(the LLM's specialization wins). Within the writer's own modes, ``skip``
preserves any pre-existing dest, ``overwrite`` replaces it.

Path-safety mirrors :func:`contract.validate_paths`: dest paths must
resolve inside ``project_dir``, no ``..`` segments, no absolutes.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from agent_scaffold.capabilities import Capability, EmitFile, ResolvedStack
from agent_scaffold.writer import WriteMode

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmitResult:
    """Outcome of one ``copy_capability_templates`` call."""

    written: list[Path] = field(default_factory=list)
    """Dest paths newly created by the copier."""

    overwritten: list[Path] = field(default_factory=list)
    """Dest paths that existed and were replaced (only when ``mode="overwrite"``)."""

    skipped_existing: list[Path] = field(default_factory=list)
    """Dest paths preserved because they pre-existed (``mode="skip"`` or
    a model-emitted file lives there)."""

    skipped_unsafe: list[str] = field(default_factory=list)
    """``source -> dest`` entries the copier refused on path-safety grounds."""

    missing_source: list[str] = field(default_factory=list)
    """``source`` paths the capability declared that don't exist on disk."""

    def total_actions(self) -> int:
        return (
            len(self.written)
            + len(self.overwritten)
            + len(self.skipped_existing)
            + len(self.skipped_unsafe)
            + len(self.missing_source)
        )


class CapabilityEmitError(Exception):
    """Raised when an ``emit_files`` entry is structurally invalid.

    Only used for cases that would have been caught at catalog-load time
    if the loader didn't trust the frontmatter (it does, deliberately —
    this is the defence-in-depth layer at write time).
    """


def copy_capability_templates(
    stack: ResolvedStack,
    capabilities_root: Path,
    project_dir: Path,
    write_mode: WriteMode = WriteMode.skip,
    *,
    model_paths: set[str] | None = None,
) -> EmitResult:
    """Copy every capability's ``emit_files`` into ``project_dir``.

    Walks ``stack.capabilities`` in declaration order. For each
    :class:`EmitFile` entry, resolves ``source`` against the capability's
    directory under ``capabilities_root``, expands ``**`` globs, and writes
    each file to ``project_dir / dest`` atomically.

    ``model_paths`` (optional) is the set of paths the LLM emitted in the
    current run; any capability dest that lives in that set is logged as a
    conflict and SKIPped. This is the "model output wins" rule.
    """
    project_dir_resolved = project_dir.resolve()
    capabilities_root_resolved = capabilities_root.resolve()
    normalized_model = {_norm(p) for p in (model_paths or set())}

    written: list[Path] = []
    overwritten: list[Path] = []
    skipped_existing: list[Path] = []
    skipped_unsafe: list[str] = []
    missing_source: list[str] = []

    for capability in stack.capabilities:
        cap_dir = _capability_dir(capability, capabilities_root_resolved)
        if cap_dir is None:
            # Capability path doesn't sit under the catalog root — refuse
            # to follow it. Surfaces as missing_source for visibility.
            for entry in capability.emit_files:
                missing_source.append(f"{capability.id}:{entry.source}")
            continue
        for entry in capability.emit_files:
            pairs = _expand_entry(entry, cap_dir)
            if not pairs:
                missing_source.append(f"{capability.id}:{entry.source}")
                continue
            for source_path, dest_rel in pairs:
                # Path-safety on the dest.
                if dest_rel.startswith(("/", "\\")):
                    skipped_unsafe.append(f"{capability.id}:{entry.source}->{dest_rel}")
                    log.warning(
                        "capability_emit: absolute dest not allowed (%s); skipping",
                        dest_rel,
                    )
                    continue
                normalized_dest = dest_rel.replace("\\", "/")
                if any(part == ".." for part in normalized_dest.split("/")):
                    skipped_unsafe.append(f"{capability.id}:{entry.source}->{dest_rel}")
                    log.warning(
                        "capability_emit: '..' segment not allowed in dest (%s); skipping",
                        dest_rel,
                    )
                    continue
                dest_full = (project_dir_resolved / normalized_dest).resolve()
                try:
                    dest_full.relative_to(project_dir_resolved)
                except ValueError:
                    skipped_unsafe.append(f"{capability.id}:{entry.source}->{dest_rel}")
                    log.warning("capability_emit: dest %s escapes project_dir; skipping", dest_full)
                    continue

                # Path-safety on the source: must resolve inside the capability dir.
                try:
                    source_path.resolve().relative_to(cap_dir)
                except ValueError:
                    skipped_unsafe.append(f"{capability.id}:{entry.source}->{dest_rel}")
                    log.warning(
                        "capability_emit: source %s escapes capability dir; skipping",
                        source_path,
                    )
                    continue

                # Conflict with model output: never overwrite.
                if _norm(normalized_dest) in normalized_model:
                    skipped_existing.append(dest_full)
                    log.warning(
                        "capability_emit: capability %s would overwrite model-emitted "
                        "%s; capability copy SKIPPED",
                        capability.id,
                        normalized_dest,
                    )
                    continue

                # Existing on-disk: honor write_mode.
                if dest_full.exists():
                    if write_mode == WriteMode.overwrite:
                        overwritten.append(dest_full)
                        _atomic_copy(source_path, dest_full)
                        continue
                    # skip / abort / diff: preserve the existing file.
                    # (The pipeline's write step has already enforced the
                    # destination-not-empty policy if mode was abort, so we
                    # never reach here with abort and unwritten existing
                    # files we'd want to clobber.)
                    skipped_existing.append(dest_full)
                    continue

                _atomic_copy(source_path, dest_full)
                written.append(dest_full)

    return EmitResult(
        written=written,
        overwritten=overwritten,
        skipped_existing=skipped_existing,
        skipped_unsafe=skipped_unsafe,
        missing_source=missing_source,
    )


def _norm(path: str) -> str:
    """Normalize for set-comparison: forward slashes, no leading ./, no trailing /."""
    n = path.replace("\\", "/").lstrip("./").rstrip("/")
    return n


def _capability_dir(capability: Capability, capabilities_root: Path) -> Path | None:
    """Resolve the capability's containing directory under the catalog root."""
    cap_path = capability.path.resolve()
    try:
        cap_path.relative_to(capabilities_root)
    except ValueError:
        return None
    return cap_path.parent


def _expand_entry(entry: EmitFile, cap_dir: Path) -> list[tuple[Path, str]]:
    """Expand one ``EmitFile`` into a list of ``(source_path, dest_rel)`` pairs.

    Rules:
    - ``source`` ends with ``**`` → recursive glob; dest is treated as a
      directory (its trailing ``/`` is optional but encouraged) and the
      relative tree under the glob root is preserved.
    - ``source`` ends with ``*`` (single-level) → flat glob; dest is a dir.
    - Otherwise ``source`` is a single file; dest is the file path.
    """
    source_spec = entry.source.replace("\\", "/")
    dest_spec = entry.dest.replace("\\", "/")

    # Recursive glob: foo/** or **
    if source_spec.endswith("**"):
        root_rel = source_spec[: -len("**")].rstrip("/")
        glob_root = (cap_dir / root_rel).resolve() if root_rel else cap_dir
        if not glob_root.is_dir():
            return []
        dest_dir = dest_spec.rstrip("/")
        pairs: list[tuple[Path, str]] = []
        for path in sorted(glob_root.rglob("*")):
            if not path.is_file():
                continue
            try:
                rel = path.relative_to(glob_root).as_posix()
            except ValueError:
                continue
            dest_rel = f"{dest_dir}/{rel}" if dest_dir else rel
            pairs.append((path, dest_rel))
        return pairs

    # Single-level glob: foo/*
    if source_spec.endswith("/*") or source_spec == "*":
        root_rel = source_spec[: -len("*")].rstrip("/")
        glob_root = (cap_dir / root_rel).resolve() if root_rel else cap_dir
        if not glob_root.is_dir():
            return []
        dest_dir = dest_spec.rstrip("/")
        pairs = []
        for path in sorted(glob_root.iterdir()):
            if not path.is_file():
                continue
            dest_rel = f"{dest_dir}/{path.name}" if dest_dir else path.name
            pairs.append((path, dest_rel))
        return pairs

    # Single file.
    source_path = (cap_dir / source_spec).resolve()
    if not source_path.is_file():
        return []
    # If dest ends with / treat as a directory and append the source filename.
    if dest_spec.endswith("/"):
        return [(source_path, f"{dest_spec.rstrip('/')}/{source_path.name}")]
    return [(source_path, dest_spec)]


def _atomic_copy(source: Path, dest: Path) -> None:
    """Copy ``source`` to ``dest`` via a temp file + ``os.replace``.

    Mirrors :mod:`agent_scaffold.writer`'s atomicity guarantee: partial
    failures never leave a half-written dest.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=dest.parent, delete=False) as tmp:
        with source.open("rb") as src:
            shutil.copyfileobj(src, tmp)
        tmp_path = Path(tmp.name)
    try:
        os.replace(tmp_path, dest)
    except OSError:
        tmp_path.unlink(missing_ok=True)
        raise


SKILLS_DIR = Path(".claude") / "skills"


def write_capability_skills(stack: ResolvedStack, project_dir: Path) -> list[str]:
    """Emit ``.claude/skills/<name>/SKILL.md`` for every skill-declaring capability.

    Agent Skills are the open packaging standard coding agents auto-load; a
    capability that declares a ``skill:`` block travels with the generated
    project as one. Existing files are never overwritten (the user may have
    tuned a skill) — same discipline as the template copier's skip mode.
    Returns the project-relative paths written.
    """
    written: list[str] = []
    for cap in stack.capabilities:
        if cap.skill is None:
            continue
        dest = project_dir / SKILLS_DIR / cap.skill.name / "SKILL.md"
        if dest.exists():
            continue
        body = cap.skill.body.strip()
        text = f"---\nname: {cap.skill.name}\ndescription: {cap.skill.description}\n---\n" + (
            f"\n{body}\n" if body else ""
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        written.append(dest.relative_to(project_dir).as_posix())
    return written


__all__ = [
    "CapabilityEmitError",
    "EmitResult",
    "copy_capability_templates",
    "write_capability_skills",
]
