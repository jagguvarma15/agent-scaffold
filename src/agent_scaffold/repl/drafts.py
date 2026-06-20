"""Named, locally-cached selection drafts for the REPL.

A user's recipe / language / framework / capability picks live in an in-memory
:class:`~agent_scaffold.repl.session.SessionState` that's lost the moment the
shell exits. This module persists a serializable *projection* of those selections
under ``<cache_dir>/drafts/<name>.json`` so an accidental exit loses nothing and
a half-built project can be resumed by name — the same atomic-write +
schema-versioned + migration-on-read shape as ``orchestrator.state.json``.

Drafts hold **selections, never secrets**: the Anthropic key and service creds
live in the keyring/vault, not here. Resume re-resolves the ``Recipe`` from the
stored slug against the *current* deployments, so a draft survives a deployments
update (and degrades gracefully if the recipe was removed).

At most :data:`MAX_DRAFTS` drafts are kept — saving a new one beyond the cap
evicts the oldest (LRU by ``saved_at``), mirroring the snapshot/run-log pruning.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from agent_scaffold._filesec import MODE_PUBLIC, secure_write
from agent_scaffold.discovery import Recipe
from agent_scaffold.repl.session import SessionState, StackMode
from agent_scaffold.writer import WriteMode

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
# Keep at most this many drafts; a new save beyond the cap evicts the oldest.
MAX_DRAFTS = 3
_DRAFTS_SUBDIR = "drafts"
_SUFFIX = ".json"
_NAME_RE = re.compile(r"[^a-z0-9._-]+")


class DraftSelections(BaseModel):
    """Serializable projection of a REPL ``SessionState``'s user selections.

    Session-scope inputs (cfg / deployments / blueprints) and the transient
    ``dirty_since_plan`` flag are intentionally excluded — only the user's
    choices are persisted.
    """

    schema_version: int = SCHEMA_VERSION
    name: str
    saved_at: str = ""
    # required selections (recipe stored by slug; re-resolved on load)
    recipe_slug: str | None = None
    language: str | None = None
    framework: str | None = None
    project_name: str | None = None
    dest: str | None = None
    # optional overrides
    model: str | None = None
    effort: str | None = None
    max_tokens: int | None = None
    thinking_budget: int | None = None
    strict: bool = False
    write_mode: str = WriteMode.abort.value
    # behaviour toggles
    autorun: bool = True
    use_docker: bool | None = None  # tri-state: None=auto, True=on, False=off
    stack_mode: str = "quick"
    # accumulators
    extra_dependencies: dict[str, dict[str, str]] = Field(default_factory=dict)
    extra_steps: list[str] = Field(default_factory=list)
    removed_steps: list[str] = Field(default_factory=list)
    removed_roles: list[str] = Field(default_factory=list)
    add_capabilities: list[str] = Field(default_factory=list)
    remove_capabilities: list[str] = Field(default_factory=list)
    refinement_notes: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class DraftMeta:
    """Lightweight listing entry — what ``/drafts`` renders per row."""

    name: str
    saved_at: str
    recipe_slug: str | None


# ---------------------------------------------------------------------------
# Naming + paths
# ---------------------------------------------------------------------------


def sanitize_name(name: str) -> str:
    """Filesystem-safe draft name: lowercase, ``[a-z0-9._-]`` only."""
    cleaned = _NAME_RE.sub("-", name.strip().lower()).strip("-._")
    return cleaned or "draft"


def default_draft_name(state: SessionState) -> str | None:
    """Stable auto-save name: the project name, else the recipe slug, else None."""
    if state.project_name:
        return sanitize_name(state.project_name)
    if state.recipe is not None:
        return sanitize_name(state.recipe.slug)
    return None


def drafts_dir(cache_dir: Path) -> Path:
    return cache_dir / _DRAFTS_SUBDIR


def draft_path(cache_dir: Path, name: str) -> Path:
    return drafts_dir(cache_dir) / f"{sanitize_name(name)}{_SUFFIX}"


# ---------------------------------------------------------------------------
# Project ↔ rehydrate
# ---------------------------------------------------------------------------


def from_state(state: SessionState, name: str) -> DraftSelections:
    """Project a ``SessionState``'s selections into a serializable draft."""
    return DraftSelections(
        name=sanitize_name(name),
        saved_at=datetime.now(UTC).isoformat(),
        recipe_slug=state.recipe.slug if state.recipe else None,
        language=state.language,
        framework=state.framework,
        project_name=state.project_name,
        dest=str(state.dest) if state.dest else None,
        model=state.model,
        effort=state.effort,
        max_tokens=state.max_tokens,
        thinking_budget=state.thinking_budget,
        strict=state.strict,
        write_mode=state.write_mode.value,
        autorun=state.autorun,
        use_docker=state.use_docker,
        stack_mode=state.stack_mode,
        extra_dependencies={k: dict(v) for k, v in state.extra_dependencies.items()},
        extra_steps=list(state.extra_steps),
        removed_steps=sorted(state.removed_steps),
        removed_roles=sorted(state.removed_roles),
        add_capabilities=list(state.add_capabilities),
        remove_capabilities=sorted(state.remove_capabilities),
        refinement_notes=list(state.refinement_notes),
    )


def apply_to_state(
    draft: DraftSelections, state: SessionState, recipes: dict[str, Recipe]
) -> SessionState:
    """Rehydrate ``draft``'s selections onto ``state``.

    The recipe is re-resolved from its slug against the *current* ``recipes`` so
    a draft survives a deployments update; an unknown slug degrades to ``None``
    (the caller surfaces a warning). ``dirty_since_plan`` is forced on so the
    plan/gate re-render before a resumed draft can generate.
    """
    recipe = recipes.get(draft.recipe_slug) if draft.recipe_slug else None
    try:
        write_mode = WriteMode(draft.write_mode)
    except ValueError:
        write_mode = WriteMode.abort
    stack_mode: StackMode = "customize" if draft.stack_mode == "customize" else "quick"
    return replace(
        state,
        recipe=recipe,
        language=draft.language,
        framework=draft.framework,
        project_name=draft.project_name,
        dest=Path(draft.dest) if draft.dest else None,
        model=draft.model,
        effort=draft.effort,
        max_tokens=draft.max_tokens,
        thinking_budget=draft.thinking_budget,
        strict=draft.strict,
        write_mode=write_mode,
        autorun=draft.autorun,
        use_docker=draft.use_docker,
        stack_mode=stack_mode,
        extra_dependencies={k: dict(v) for k, v in draft.extra_dependencies.items()},
        extra_steps=list(draft.extra_steps),
        removed_steps=set(draft.removed_steps),
        removed_roles=set(draft.removed_roles),
        add_capabilities=list(draft.add_capabilities),
        remove_capabilities=set(draft.remove_capabilities),
        refinement_notes=list(draft.refinement_notes),
        dirty_since_plan=True,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_draft(cache_dir: Path, draft: DraftSelections) -> Path:
    """Atomically write ``draft`` to ``<cache_dir>/drafts/<name>.json`` (0644).

    Then prune to :data:`MAX_DRAFTS`: the just-saved draft carries the newest
    ``saved_at`` so it's always retained; any drafts beyond the cap (the oldest)
    are evicted.
    """
    path = draft_path(cache_dir, draft.name)
    written = secure_write(path, draft.model_dump_json(indent=2) + "\n", mode=MODE_PUBLIC)
    _prune_drafts(cache_dir, keep=MAX_DRAFTS)
    return written


def _prune_drafts(cache_dir: Path, *, keep: int) -> list[str]:
    """Delete all but the ``keep`` most-recent drafts. Returns evicted names."""
    metas = list_drafts(cache_dir)  # most-recent first
    evicted = [meta.name for meta in metas[keep:]]
    for name in evicted:
        delete_draft(cache_dir, name)
    return evicted


def load_draft(cache_dir: Path, name: str) -> DraftSelections | None:
    """Read a draft by name, or ``None`` if absent / unreadable / too new.

    Never raises: a corrupt or future-schema draft is logged and skipped so a
    bad file can't crash the REPL on open or on `/drafts`.
    """
    path = draft_path(cache_dir, name)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("draft %r unreadable (%s); ignoring", name, exc)
        return None
    version = data.get("schema_version", 1) if isinstance(data, dict) else None
    if not isinstance(version, int) or version > SCHEMA_VERSION:
        log.warning("draft %r schema_version %r unsupported; ignoring", name, version)
        return None
    try:
        return DraftSelections.model_validate(data)
    except ValidationError as exc:
        log.warning("draft %r invalid (%s); ignoring", name, exc)
        return None


def list_drafts(cache_dir: Path) -> list[DraftMeta]:
    """Every readable draft, most-recently-saved first."""
    root = drafts_dir(cache_dir)
    if not root.is_dir():
        return []
    metas: list[DraftMeta] = []
    for path in root.glob(f"*{_SUFFIX}"):
        draft = load_draft(cache_dir, path.stem)
        if draft is not None:
            metas.append(
                DraftMeta(name=draft.name, saved_at=draft.saved_at, recipe_slug=draft.recipe_slug)
            )
    metas.sort(key=lambda m: m.saved_at, reverse=True)
    return metas


def delete_draft(cache_dir: Path, name: str) -> bool:
    """Remove a draft. ``True`` if a file was deleted."""
    path = draft_path(cache_dir, name)
    if path.is_file():
        path.unlink()
        return True
    return False


def relative_time(iso: str) -> str:
    """``2026-06-20T17:00:00+00:00`` → ``3h ago`` (best-effort; raw on parse error)."""
    try:
        then = datetime.fromisoformat(iso)
    except ValueError:
        return iso or "—"
    secs = int((datetime.now(UTC) - then).total_seconds())
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"


__all__ = [
    "MAX_DRAFTS",
    "DraftMeta",
    "DraftSelections",
    "apply_to_state",
    "default_draft_name",
    "delete_draft",
    "draft_path",
    "drafts_dir",
    "from_state",
    "list_drafts",
    "load_draft",
    "relative_time",
    "sanitize_name",
    "save_draft",
]
