"""Tests for the /tier and /layers slash commands."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from agent_scaffold.config import Config
from agent_scaffold.discovery import Recipe
from agent_scaffold.repl.commands import CommandHandler, CommandResult
from agent_scaffold.repl.session import SessionState
from agent_scaffold.sources import DEPLOYMENTS_SPEC, ResolvedSource


@pytest.fixture
def basic_recipe(tmp_path: Path) -> Recipe:
    md = tmp_path / "basic.md"
    md.write_text("# basic\n", encoding="utf-8")
    return Recipe(
        slug="simple-chat",
        title="Simple Chat",
        path=md,
        agent_pattern="react",
        capabilities=["cache.redis"],
    )


@pytest.fixture
def mid_recipe(tmp_path: Path) -> Recipe:
    md = tmp_path / "mid.md"
    md.write_text("# mid\n", encoding="utf-8")
    return Recipe(
        slug="triage-bot",
        title="Triage Bot",
        path=md,
        topology="multi-agent-flat",
        agent_pattern="supervisor",
        capabilities=["relational.postgres", "cache.redis", "obs.langfuse"],
    )


@pytest.fixture
def complex_recipe(tmp_path: Path) -> Recipe:
    md = tmp_path / "complex.md"
    md.write_text("# complex\n", encoding="utf-8")
    return Recipe(
        slug="rebooking",
        title="Restaurant Rebooking",
        path=md,
        topology="multi-agent-flat",
        agent_pattern="event-driven",
        capabilities=["host.vercel", "queue.kafka", "frontend.nextjs-chat"],
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
def handler(basic_recipe: Recipe, mid_recipe: Recipe, complex_recipe: Recipe) -> CommandHandler:
    return CommandHandler(recipes=[basic_recipe, mid_recipe, complex_recipe])


def _messages_text(result: CommandResult) -> str:
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=120)
    for m in result.messages:
        console.print(m)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# /tier
# ---------------------------------------------------------------------------


def test_cmd_tier_with_no_recipe_errors(
    handler: CommandHandler, base_state: SessionState
) -> None:
    result = handler.dispatch("/tier", base_state)
    text = _messages_text(result)
    assert "no recipe selected" in text


def test_cmd_tier_inspects_current_recipe(
    handler: CommandHandler, base_state: SessionState, mid_recipe: Recipe
) -> None:
    base_state.recipe = mid_recipe
    result = handler.dispatch("/tier", base_state)
    text = _messages_text(result)
    assert "tier: mid" in text
    # Peers within the same tier surface — at minimum mid_recipe itself.
    assert "triage-bot" in text
    # Other-tier recipes don't surface.
    assert "rebooking" not in text
    assert "simple-chat" not in text


def test_cmd_tier_with_explicit_tier_arg(
    handler: CommandHandler, base_state: SessionState
) -> None:
    result = handler.dispatch("/tier complex", base_state)
    text = _messages_text(result)
    assert "tier: complex" in text
    assert "rebooking" in text
    assert "triage-bot" not in text


def test_cmd_tier_unknown_value_errors(
    handler: CommandHandler, base_state: SessionState
) -> None:
    result = handler.dispatch("/tier impossible", base_state)
    text = _messages_text(result)
    assert "must be one of" in text


def test_cmd_tier_includes_agent_pattern_hint(
    handler: CommandHandler, base_state: SessionState
) -> None:
    result = handler.dispatch("/tier basic", base_state)
    text = _messages_text(result)
    # Recipe has agent_pattern="react" → should surface as a hint.
    assert "react" in text


# ---------------------------------------------------------------------------
# /layers (plural)
# ---------------------------------------------------------------------------


def test_cmd_layers_lists_all_layers(
    handler: CommandHandler, base_state: SessionState, mid_recipe: Recipe
) -> None:
    base_state.recipe = mid_recipe
    result = handler.dispatch("/layers", base_state)
    text = _messages_text(result)
    assert "memory" in text
    assert "tools" in text
    assert "observability" in text
    # Recipe-declared caps in the right buckets.
    assert "relational.postgres" in text
    assert "obs.langfuse" in text


def test_cmd_layers_works_with_no_recipe(
    handler: CommandHandler, base_state: SessionState
) -> None:
    # No crash; just shows "(none)" for every layer.
    result = handler.dispatch("/layers", base_state)
    text = _messages_text(result)
    assert "(none)" in text
