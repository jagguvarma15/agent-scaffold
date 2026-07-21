"""Regression tests for the REPL output audit.

Pins the fixes from the output-hygiene pass:

- Dynamic values (user input, exception text, probe details) render
  literally instead of being parsed as Rich markup.
- The ``/plan`` cap-bump branch keeps the cost line the normal branch has.
- Patches that only touch hosting / rag_preset / optional_features render
  a delta instead of "No changes." right after "applied refinement".
- The session panel's value column doesn't drift between the always-on
  rows and the conditional ones.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from rich.console import Console, RenderableType

from agent_scaffold.config import Config
from agent_scaffold.context import ContextBudgetError
from agent_scaffold.discovery import Recipe
from agent_scaffold.plan import GenerationPlan
from agent_scaffold.repl import commands as commands_module
from agent_scaffold.repl.commands import CommandHandler, CommandResult
from agent_scaffold.repl.render import render_patch_delta, render_state_summary
from agent_scaffold.repl.session import SessionState
from agent_scaffold.sources import DEPLOYMENTS_SPEC, ResolvedSource
from agent_scaffold.topology import Topology


@pytest.fixture
def demo_recipe(tmp_path: Path) -> Recipe:
    recipe_md = tmp_path / "demo.md"
    recipe_md.write_text("# Demo\n", encoding="utf-8")
    return Recipe(slug="demo", title="Demo", path=recipe_md, status="blueprint")


@pytest.fixture
def base_state(tmp_path: Path) -> SessionState:
    cfg = Config(
        anthropic_api_key="test-key",
        cache_dir=tmp_path / "cache",
        failures_dir=tmp_path / "cache" / "failures",
    )
    src = ResolvedSource(
        spec=DEPLOYMENTS_SPEC,
        path=tmp_path / "deployments",
        label="test",
        kind="explicit-path",
        commit_sha=None,
    )
    return SessionState(cfg=cfg, deployments=src, blueprints=src)


@pytest.fixture
def handler(demo_recipe: Recipe) -> CommandHandler:
    return CommandHandler(recipes=[demo_recipe])


def _render_text(*renderables: RenderableType) -> str:
    console = Console(width=120, no_color=True, force_terminal=False)
    with console.capture() as capture:
        for renderable in renderables:
            console.print(renderable)
    return capture.get()


def _messages_text(result: CommandResult) -> str:
    return _render_text(*result.messages)


# ---------------------------------------------------------------------------
# Markup injection
# ---------------------------------------------------------------------------


def test_bracketed_name_renders_literally(base_state: SessionState) -> None:
    state = replace(base_state, project_name="a[b]c")
    assert "a[b]c" in _render_text(render_state_summary(state))


def test_bracketed_dest_survives_the_plan_panel(tmp_path: Path) -> None:
    plan = GenerationPlan(
        recipe_slug="demo",
        recipe_status="blueprint",
        language="python",
        framework="none",
        project_name="demo",
        dest=tmp_path / "out[1]",
        topology=Topology.SINGLE,
        model="claude-opus-4-8",
        max_tokens=1000,
    )
    assert "out[1]" in _render_text(plan.render())


def test_invalid_observability_lists_the_options(
    handler: CommandHandler, base_state: SessionState
) -> None:
    text = _messages_text(handler.dispatch("/observability bogus", base_state))
    for option in ("langsmith", "langfuse", "grafana-stack"):
        assert option in text


def test_command_error_with_markup_input_renders_literally(
    handler: CommandHandler, base_state: SessionState
) -> None:
    text = _messages_text(handler.dispatch("/recipe [bold]x[/]", base_state))
    assert "[bold]x[/]" in text


def test_unknown_command_with_markup_token_is_safe(
    handler: CommandHandler, base_state: SessionState
) -> None:
    text = _messages_text(handler.dispatch("/[red]nope", base_state))
    assert "[red]nope" in text
    assert "Unknown command" in text


# ---------------------------------------------------------------------------
# /plan cap-bump branch
# ---------------------------------------------------------------------------


def test_cap_bump_plan_keeps_the_cost_line(
    monkeypatch: pytest.MonkeyPatch,
    handler: CommandHandler,
    base_state: SessionState,
    demo_recipe: Recipe,
    tmp_path: Path,
) -> None:
    state = replace(
        base_state,
        recipe=demo_recipe,
        language="python",
        framework="none",
        project_name="demo",
        dest=tmp_path / "out",
    )
    fake_plan = GenerationPlan(
        recipe_slug="demo",
        recipe_status="blueprint",
        language="python",
        framework="none",
        project_name="demo",
        dest=tmp_path / "out",
        topology=Topology.SINGLE,
        model="claude-opus-4-8",
        max_tokens=1000,
    )
    calls = {"n": 0}

    def fake_build_plan(_state: SessionState) -> GenerationPlan:
        if calls["n"] == 0:
            calls["n"] += 1
            raise ContextBudgetError("too big", essentials_tokens=999, current_cap=100)
        return fake_plan

    monkeypatch.setattr(commands_module, "_build_plan", fake_build_plan)
    monkeypatch.setattr(
        commands_module,
        "prompt_to_raise_context_cap",
        lambda _console, _exc: (200_000, 20_000),
    )
    text = _messages_text(handler.dispatch("/plan", state))
    assert "Context cap raised" in text
    assert "Est. cost" in text


# ---------------------------------------------------------------------------
# Patch-delta coverage
# ---------------------------------------------------------------------------


def test_hosting_only_patch_renders_a_delta(base_state: SessionState) -> None:
    after = replace(base_state, hosting_overrides={"obs.langfuse": "cloud"})
    text = _render_text(render_patch_delta(base_state, after))
    assert "hosting" in text
    assert "obs.langfuse=cloud" in text
    assert "No changes" not in text


def test_rag_preset_only_patch_renders_a_delta(base_state: SessionState) -> None:
    after = replace(base_state, rag_preset="simple")
    text = _render_text(render_patch_delta(base_state, after))
    assert "rag preset" in text
    assert "No changes" not in text


def test_optional_features_only_patch_renders_a_delta(base_state: SessionState) -> None:
    after = replace(base_state, optional_features=["rag", "observability"])
    text = _render_text(render_patch_delta(base_state, after))
    assert "optional features" in text
    assert "No changes" not in text


def test_agent_description_only_patch_renders_a_delta(base_state: SessionState) -> None:
    after = replace(base_state, agent_description="A research helper")
    text = _render_text(render_patch_delta(base_state, after))
    assert "description" in text
    assert "No changes" not in text


# ---------------------------------------------------------------------------
# Session panel alignment
# ---------------------------------------------------------------------------


def test_state_summary_value_columns_align(base_state: SessionState) -> None:
    state = replace(base_state, project_name="demo", tier="T2")
    lines = _render_text(render_state_summary(state)).splitlines()
    name_line = next(line for line in lines if "demo" in line)
    tier_line = next(line for line in lines if "T2" in line)
    assert name_line.index("demo") == tier_line.index("T2")
