"""Tests for /open: attaching the REPL session to an existing generated project."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rich.console import Console

from agent_scaffold.config import Config
from agent_scaffold.discovery import Recipe
from agent_scaffold.manifest import Manifest, write_manifest
from agent_scaffold.repl.commands import CommandError, CommandHandler, CommandResult
from agent_scaffold.repl.session import SessionState
from agent_scaffold.sources import DEPLOYMENTS_SPEC, ResolvedSource

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


def _blank_state(tmp_path: Path) -> SessionState:
    src = _src(tmp_path)
    return SessionState(cfg=_cfg(tmp_path), deployments=src, blueprints=src)


@pytest.fixture
def recipe(tmp_path: Path) -> Recipe:
    md = tmp_path / "demo.md"
    md.write_text("# Demo\n", encoding="utf-8")
    return Recipe(slug="demo", title="Demo", path=md, status="blueprint")


def _generated_project(
    tmp_path: Path,
    *,
    recipe: str = "demo",
    answers: dict[str, str] | None = None,
    capabilities: list[str] | None = None,
) -> Path:
    project = tmp_path / "out" / "my-proj"
    project.mkdir(parents=True)
    write_manifest(
        project,
        Manifest(
            recipe=recipe,
            language="python",
            framework="langgraph",
            model="claude-test",
            generated_at="2026-01-01T00:00:00+00:00",
            answers=answers if answers is not None else {"project_name": "my-proj"},
            capabilities=capabilities or [],
        ),
    )
    return project


def _messages_text(result: CommandResult) -> str:
    console = Console(record=True, width=200, no_color=True)
    for message in result.messages:
        console.print(message)
    return console.export_text()


# ---------------------------------------------------------------------------
# Registration + argument handling
# ---------------------------------------------------------------------------


def test_cmd_open_registered_and_in_help(tmp_path: Path, recipe: Recipe) -> None:
    handler = CommandHandler(recipes=[recipe])
    assert "open" in handler.commands
    help_text = _messages_text(handler.cmd_help([], _blank_state(tmp_path)))
    assert "/open" in help_text


def test_load_alias_dispatches_to_open(tmp_path: Path, recipe: Recipe) -> None:
    handler = CommandHandler(recipes=[recipe])
    project = _generated_project(tmp_path)
    result = handler.dispatch(f"/load {project}", _blank_state(tmp_path))
    assert result.new_state is not None
    assert result.new_state.dest == project


def test_cmd_open_requires_path_arg(tmp_path: Path, recipe: Recipe) -> None:
    handler = CommandHandler(recipes=[recipe])
    with pytest.raises(CommandError, match="usage: /open"):
        handler.cmd_open([], _blank_state(tmp_path))


def test_cmd_open_nonexistent_dir_is_command_error(tmp_path: Path, recipe: Recipe) -> None:
    handler = CommandHandler(recipes=[recipe])
    with pytest.raises(CommandError, match="not a directory"):
        handler.cmd_open([str(tmp_path / "nope")], _blank_state(tmp_path))


def test_cmd_open_missing_manifest_is_command_error(tmp_path: Path, recipe: Recipe) -> None:
    handler = CommandHandler(recipes=[recipe])
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(CommandError, match="No manifest"):
        handler.cmd_open([str(empty)], _blank_state(tmp_path))


def test_cmd_open_newer_schema_manifest_is_command_error(tmp_path: Path, recipe: Recipe) -> None:
    handler = CommandHandler(recipes=[recipe])
    project = tmp_path / "future"
    (project / ".scaffold").mkdir(parents=True)
    (project / ".scaffold" / "manifest.json").write_text(
        json.dumps({"schema_version": 99}), encoding="utf-8"
    )
    with pytest.raises(CommandError):
        handler.cmd_open([str(project)], _blank_state(tmp_path))


# ---------------------------------------------------------------------------
# Hydration
# ---------------------------------------------------------------------------


def test_cmd_open_hydrates_state_from_manifest(tmp_path: Path, recipe: Recipe) -> None:
    handler = CommandHandler(recipes=[recipe])
    project = _generated_project(tmp_path)
    result = handler.cmd_open([str(project)], _blank_state(tmp_path))

    state = result.new_state
    assert state is not None
    assert state.dest == project
    assert state.recipe is recipe  # re-resolved from the manifest slug
    assert state.language == "python"
    assert state.framework == "langgraph"
    assert state.project_name == "my-proj"
    assert state.model == "claude-test"
    assert state.dirty_since_plan is True
    assert "attached" in _messages_text(result)


def test_cmd_open_unknown_recipe_attaches_with_warning(tmp_path: Path, recipe: Recipe) -> None:
    handler = CommandHandler(recipes=[recipe])
    project = _generated_project(tmp_path, recipe="ghost")
    result = handler.cmd_open([str(project)], _blank_state(tmp_path))

    assert result.new_state is not None
    assert result.new_state.recipe is None
    assert result.new_state.dest == project
    assert "not in current deployments" in _messages_text(result)


def test_cmd_open_project_name_falls_back_to_dir_name(tmp_path: Path, recipe: Recipe) -> None:
    handler = CommandHandler(recipes=[recipe])
    project = _generated_project(tmp_path, answers={})
    result = handler.cmd_open([str(project)], _blank_state(tmp_path))
    assert result.new_state is not None
    assert result.new_state.project_name == "my-proj"


def test_cmd_open_resets_accumulators(tmp_path: Path, recipe: Recipe) -> None:
    handler = CommandHandler(recipes=[recipe])
    project = _generated_project(tmp_path)
    state = _blank_state(tmp_path)
    state.add_capabilities = ["obs.langfuse"]
    state.remove_capabilities = {"cache.redis"}
    state.extra_steps = ["extra"]
    state.refinement_notes = ["prefer sonnet"]

    result = handler.cmd_open([str(project)], state)
    attached = result.new_state
    assert attached is not None
    assert attached.add_capabilities == []
    assert attached.remove_capabilities == set()
    assert attached.extra_steps == []
    assert attached.refinement_notes == []


def test_cmd_open_carries_session_toggles(tmp_path: Path, recipe: Recipe) -> None:
    handler = CommandHandler(recipes=[recipe])
    project = _generated_project(tmp_path)
    state = _blank_state(tmp_path)
    state.autorun = True
    state.use_docker = False

    result = handler.cmd_open([str(project)], state)
    assert result.new_state is not None
    assert result.new_state.autorun is True
    assert result.new_state.use_docker is False


def test_cmd_open_renders_stack_line(tmp_path: Path, recipe: Recipe) -> None:
    handler = CommandHandler(recipes=[recipe])
    project = _generated_project(tmp_path, capabilities=["cache.redis"])
    result = handler.cmd_open([str(project)], _blank_state(tmp_path))
    assert "Stack" in _messages_text(result)


# ---------------------------------------------------------------------------
# Attach unlocks lifecycle commands + guard messages point at /open
# ---------------------------------------------------------------------------


def test_cmd_open_then_up_signals_run(tmp_path: Path, recipe: Recipe) -> None:
    handler = CommandHandler(recipes=[recipe])
    project = _generated_project(tmp_path)
    attached = handler.cmd_open([str(project)], _blank_state(tmp_path)).new_state
    assert attached is not None
    result = handler.cmd_up([], attached)
    assert result.next_action == "up"


def test_guard_messages_mention_open(tmp_path: Path, recipe: Recipe) -> None:
    handler = CommandHandler(recipes=[recipe])
    state = _blank_state(tmp_path)
    with pytest.raises(CommandError, match="/open"):
        handler.cmd_up([], state)
    with pytest.raises(CommandError, match="/open"):
        handler.cmd_down([], state)
    with pytest.raises(CommandError, match="/open"):
        handler.cmd_connect([], state)
    with pytest.raises(CommandError, match="/open"):
        handler.cmd_logs(["backend"], state)
