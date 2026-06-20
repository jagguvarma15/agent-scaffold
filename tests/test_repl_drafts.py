"""Tests for named REPL selection drafts: persistence, LRU cap, resume."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_scaffold.config import Config
from agent_scaffold.discovery import Recipe
from agent_scaffold.repl import drafts
from agent_scaffold.repl.commands import CommandError, CommandHandler
from agent_scaffold.repl.drafts import DraftSelections
from agent_scaffold.repl.session import SessionState
from agent_scaffold.sources import DEPLOYMENTS_SPEC, ResolvedSource
from agent_scaffold.writer import WriteMode

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _cfg(tmp_path: Path) -> Config:
    return Config(
        anthropic_api_key="test-key",
        cache_dir=tmp_path / "cache",
        failures_dir=tmp_path / "cache" / "failures",
    )


def _src(tmp_path: Path) -> ResolvedSource:
    return ResolvedSource(
        spec=DEPLOYMENTS_SPEC,
        path=tmp_path / "deployments",
        label="test",
        kind="explicit-path",
        commit_sha=None,
    )


@pytest.fixture
def recipe(tmp_path: Path) -> Recipe:
    md = tmp_path / "demo.md"
    md.write_text("# Demo\n", encoding="utf-8")
    return Recipe(slug="demo", title="Demo", path=md, status="blueprint")


def _blank_state(tmp_path: Path) -> SessionState:
    src = _src(tmp_path)
    return SessionState(cfg=_cfg(tmp_path), deployments=src, blueprints=src)


def _selected_state(tmp_path: Path, recipe: Recipe) -> SessionState:
    state = _blank_state(tmp_path)
    state.recipe = recipe
    state.language = "python"
    state.framework = "langgraph"
    state.project_name = "my-proj"
    state.dest = tmp_path / "out" / "my-proj"
    state.use_docker = True
    state.write_mode = WriteMode.overwrite
    state.add_capabilities = ["obs.langfuse"]
    state.refinement_notes = ["prefer sonnet"]
    return state


# ---------------------------------------------------------------------------
# Round-trip + resume
# ---------------------------------------------------------------------------


def test_round_trip_preserves_selections(tmp_path: Path, recipe: Recipe) -> None:
    state = _selected_state(tmp_path, recipe)
    cache = state.cfg.cache_dir
    drafts.save_draft(cache, drafts.from_state(state, "my-proj"))

    loaded = drafts.load_draft(cache, "my-proj")
    assert loaded is not None
    restored = drafts.apply_to_state(loaded, _blank_state(tmp_path), {recipe.slug: recipe})

    assert restored.recipe is recipe  # re-resolved from slug against current recipes
    assert restored.language == "python"
    assert restored.framework == "langgraph"
    assert restored.project_name == "my-proj"
    assert restored.dest == state.dest
    assert restored.use_docker is True
    assert restored.write_mode is WriteMode.overwrite
    assert restored.add_capabilities == ["obs.langfuse"]
    assert restored.refinement_notes == ["prefer sonnet"]
    assert restored.dirty_since_plan is True  # forces a plan/gate re-render on resume


def test_resume_unknown_recipe_degrades_to_none(tmp_path: Path) -> None:
    # A draft whose recipe was removed from deployments resumes the rest, recipe=None.
    draft = DraftSelections(name="x", recipe_slug="ghost", language="python")
    restored = drafts.apply_to_state(draft, _blank_state(tmp_path), {})
    assert restored.recipe is None
    assert restored.language == "python"


# ---------------------------------------------------------------------------
# LRU cap (max 3; a 4th save evicts the oldest)
# ---------------------------------------------------------------------------


def test_lru_cap_keeps_three_newest(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    for i, name in enumerate(["a", "b", "c", "d"]):
        # Explicit increasing timestamps: a oldest … d newest.
        drafts.save_draft(
            cache,
            DraftSelections(name=name, saved_at=f"2026-06-20T10:0{i}:00+00:00", recipe_slug="r"),
        )
    names = {m.name for m in drafts.list_drafts(cache)}
    assert names == {"b", "c", "d"}  # "a", the oldest, was evicted
    assert len(names) == 3


def test_list_drafts_most_recent_first(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    drafts.save_draft(cache, DraftSelections(name="old", saved_at="2026-06-20T09:00:00+00:00"))
    drafts.save_draft(cache, DraftSelections(name="new", saved_at="2026-06-20T18:00:00+00:00"))
    assert [m.name for m in drafts.list_drafts(cache)] == ["new", "old"]


def test_resave_same_name_overwrites_not_grows(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    drafts.save_draft(cache, DraftSelections(name="p", saved_at="2026-06-20T10:00:00+00:00"))
    drafts.save_draft(cache, DraftSelections(name="p", saved_at="2026-06-20T11:00:00+00:00"))
    metas = drafts.list_drafts(cache)
    assert len(metas) == 1 and metas[0].name == "p"


# ---------------------------------------------------------------------------
# Resilience + helpers
# ---------------------------------------------------------------------------


def test_delete_draft(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    drafts.save_draft(cache, DraftSelections(name="gone"))
    assert drafts.delete_draft(cache, "gone") is True
    assert drafts.delete_draft(cache, "gone") is False  # already gone
    assert drafts.list_drafts(cache) == []


def test_corrupt_draft_is_ignored_not_crashing(tmp_path: Path) -> None:
    path = drafts.draft_path(tmp_path, "bad")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert drafts.load_draft(tmp_path, "bad") is None
    assert drafts.list_drafts(tmp_path) == []  # survives a corrupt file


def test_future_schema_draft_is_ignored(tmp_path: Path) -> None:
    path = drafts.draft_path(tmp_path, "future")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 99, "name": "future"}), encoding="utf-8")
    assert drafts.load_draft(tmp_path, "future") is None


def test_sanitize_name() -> None:
    assert drafts.sanitize_name("My Project!") == "my-project"
    assert drafts.sanitize_name("research_assistant") == "research_assistant"
    assert drafts.sanitize_name("   ") == "draft"


def test_default_draft_name_prefers_project_then_slug(tmp_path: Path, recipe: Recipe) -> None:
    state = _blank_state(tmp_path)
    assert drafts.default_draft_name(state) is None
    state.recipe = recipe
    assert drafts.default_draft_name(state) == "demo"
    state.project_name = "Cool Bot"
    assert drafts.default_draft_name(state) == "cool-bot"


# ---------------------------------------------------------------------------
# REPL commands
# ---------------------------------------------------------------------------


def _text(messages: list[object]) -> str:
    from rich.console import Console

    console = Console(record=True, color_system=None, width=120)
    for msg in messages:
        console.print(msg)
    return console.export_text()


def test_cmd_draft_save_and_drafts_list(tmp_path: Path, recipe: Recipe) -> None:
    handler = CommandHandler(recipes=[recipe])
    state = _selected_state(tmp_path, recipe)
    save = handler.cmd_draft(["save"], state)  # default name = project name
    assert "saved draft" in _text(save.messages) and "my-proj" in _text(save.messages)
    listing = _text(handler.cmd_drafts([], state).messages)
    assert "my-proj" in listing and "demo" in listing  # name + recipe slug rendered


def test_cmd_draft_load_rehydrates_state(tmp_path: Path, recipe: Recipe) -> None:
    handler = CommandHandler(recipes=[recipe])
    drafts.save_draft(
        (tmp_path / "cache"),
        drafts.from_state(_selected_state(tmp_path, recipe), "my-proj"),
    )
    result = handler.cmd_draft(["load", "my-proj"], _blank_state(tmp_path))
    assert result.new_state is not None
    assert result.new_state.recipe is recipe
    assert result.new_state.project_name == "my-proj"
    assert "resumed draft" in _text(result.messages)


def test_cmd_draft_load_unknown_raises(tmp_path: Path) -> None:
    handler = CommandHandler(recipes=[])
    with pytest.raises(CommandError):
        handler.cmd_draft(["load", "nope"], _blank_state(tmp_path))


# ---------------------------------------------------------------------------
# Shell auto-save
# ---------------------------------------------------------------------------


def test_autosave_writes_named_draft(tmp_path: Path, recipe: Recipe) -> None:
    from agent_scaffold.repl import shell

    state = _selected_state(tmp_path, recipe)
    shell._maybe_autosave_draft(state)
    names = {m.name for m in drafts.list_drafts(state.cfg.cache_dir)}
    assert "my-proj" in names


def test_autosave_noop_without_any_selection(tmp_path: Path) -> None:
    from agent_scaffold.repl import shell

    state = _blank_state(tmp_path)  # no recipe, no project name
    shell._maybe_autosave_draft(state)
    assert drafts.list_drafts(state.cfg.cache_dir) == []
