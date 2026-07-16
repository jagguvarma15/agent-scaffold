"""Tests for the /customize and /layer slash commands."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_scaffold.config import Config
from agent_scaffold.discovery import Recipe
from agent_scaffold.repl.commands import CommandHandler, CommandResult
from agent_scaffold.repl.session import SessionState
from agent_scaffold.sources import DEPLOYMENTS_SPEC, ResolvedSource


@pytest.fixture
def demo_recipe(tmp_path: Path) -> Recipe:
    md = tmp_path / "demo.md"
    md.write_text("# Demo\n", encoding="utf-8")
    return Recipe(
        slug="demo",
        title="Demo",
        path=md,
        capabilities=["cache.redis", "obs.langsmith", "vector_db.qdrant"],
    )


@pytest.fixture
def base_state(tmp_path: Path, mock_deployments_path: Path) -> SessionState:
    cfg = Config(
        anthropic_api_key="test-key",
        cache_dir=tmp_path / "cache",
        failures_dir=tmp_path / "cache" / "failures",
    )
    src = ResolvedSource(
        spec=DEPLOYMENTS_SPEC,
        path=mock_deployments_path,
        label="mock",
        kind="explicit-path",
        commit_sha=None,
    )
    return SessionState(cfg=cfg, deployments=src, blueprints=src)


@pytest.fixture
def handler(demo_recipe: Recipe) -> CommandHandler:
    return CommandHandler(recipes=[demo_recipe])


def _messages_text(result: CommandResult) -> str:
    from io import StringIO

    from rich.console import Console

    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=120)
    for m in result.messages:
        console.print(m)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# /customize
# ---------------------------------------------------------------------------


def test_cmd_customize_on_sets_customize(handler: CommandHandler, base_state: SessionState) -> None:
    result = handler.dispatch("/customize on", base_state)
    assert result.new_state is not None
    assert result.new_state.stack_mode == "customize"


def test_cmd_customize_off_sets_quick(handler: CommandHandler, base_state: SessionState) -> None:
    base_state.stack_mode = "customize"
    result = handler.dispatch("/customize off", base_state)
    assert result.new_state is not None
    assert result.new_state.stack_mode == "quick"


def test_cmd_customize_toggle_flips(handler: CommandHandler, base_state: SessionState) -> None:
    # Default is quick → toggle to customize.
    result = handler.dispatch("/customize", base_state)
    assert result.new_state is not None
    assert result.new_state.stack_mode == "customize"
    # Toggle again → back to quick.
    result2 = handler.dispatch("/customize", result.new_state)
    assert result2.new_state is not None
    assert result2.new_state.stack_mode == "quick"


def test_cmd_customize_unknown_arg_errors(
    handler: CommandHandler, base_state: SessionState
) -> None:
    result = handler.dispatch("/customize whenever", base_state)
    assert "usage" in _messages_text(result)


# ---------------------------------------------------------------------------
# /layer
# ---------------------------------------------------------------------------


def test_cmd_layer_no_args_lists_all_layers(
    handler: CommandHandler, base_state: SessionState, demo_recipe: Recipe
) -> None:
    base_state.recipe = demo_recipe
    result = handler.dispatch("/layer", base_state)
    text = _messages_text(result)
    assert "memory" in text
    assert "observability" in text
    assert "eval" in text
    assert "interface" in text
    # Recipe-declared caps surface in the layers they belong to.
    assert "cache.redis" in text
    assert "obs.langsmith" in text


def test_cmd_layer_unknown_layer_errors(handler: CommandHandler, base_state: SessionState) -> None:
    result = handler.dispatch("/layer mystery", base_state)
    assert "unknown layer" in _messages_text(result)


def test_cmd_layer_describe_one_layer(
    handler: CommandHandler, base_state: SessionState, demo_recipe: Recipe
) -> None:
    base_state.recipe = demo_recipe
    result = handler.dispatch("/layer memory", base_state)
    text = _messages_text(result)
    assert "layer memory" in text
    # Effective caps in the memory layer surface.
    assert "cache.redis" in text or "vector_db.qdrant" in text


def test_cmd_layer_set_replaces_layer(
    handler: CommandHandler, base_state: SessionState, demo_recipe: Recipe
) -> None:
    base_state.recipe = demo_recipe
    # mock_deployments ships cache.redis + vector_db.qdrant under those kinds.
    result = handler.dispatch("/layer memory cache.redis", base_state)
    assert result.new_state is not None
    # vector_db.qdrant was in the recipe + in-layer → must land in removes.
    assert "vector_db.qdrant" in result.new_state.remove_capabilities


def test_cmd_layer_rejects_out_of_layer_ids(
    handler: CommandHandler, base_state: SessionState, demo_recipe: Recipe
) -> None:
    base_state.recipe = demo_recipe
    # obs.langsmith is in the catalog but belongs to the obs layer, not memory.
    result = handler.dispatch("/layer memory obs.langsmith", base_state)
    text = _messages_text(result)
    assert "not in layer" in text


def test_cmd_layer_rejects_unknown_capability(
    handler: CommandHandler, base_state: SessionState, demo_recipe: Recipe
) -> None:
    base_state.recipe = demo_recipe
    result = handler.dispatch("/layer memory vector_db.totally-fake", base_state)
    assert "unknown capability" in _messages_text(result)


# ---------------------------------------------------------------------------
# Extended layer map: tools / infrastructure / memory_store coverage
# ---------------------------------------------------------------------------


def test_cmd_layer_tools_lists_agent_tier_capabilities(
    handler: CommandHandler, base_state: SessionState
) -> None:
    result = handler.dispatch("/layer tools", base_state)
    text = _messages_text(result)
    assert "live_data.tavily" in text
    assert "sandbox.e2b" in text


def test_cmd_layer_infrastructure_and_memory_extended(
    handler: CommandHandler, base_state: SessionState
) -> None:
    result = handler.dispatch("/layer infrastructure queue.kafka", base_state)
    assert result.new_state is not None
    assert "queue.kafka" in result.new_state.add_capabilities

    result2 = handler.dispatch("/layer memory memory_store.zep", result.new_state)
    assert result2.new_state is not None
    assert "memory_store.zep" in result2.new_state.add_capabilities


def test_format_all_layers_lists_new_groups(
    handler: CommandHandler, base_state: SessionState
) -> None:
    text = _messages_text(handler.dispatch("/layer", base_state))
    for key in ("memory", "infrastructure", "tools", "hosting", "auth"):
        assert key in text


def test_layer_groups_mirror_wizard_groups() -> None:
    """The slash-command map and the wizard walk must agree on kinds per key."""
    from agent_scaffold.repl.commands import _LAYER_GROUPS_BY_KEY
    from agent_scaffold.repl.shell import _LAYER_GROUPS

    for key, _label, kinds in _LAYER_GROUPS:
        assert _LAYER_GROUPS_BY_KEY.get(key) == kinds, key
