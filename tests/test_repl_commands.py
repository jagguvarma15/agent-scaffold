"""Tests for ``agent_scaffold.repl.commands`` — slash-command dispatcher.

Each cmd_* is a small pure function. These tests exercise:

- State mutation: scalar fields move; effort preset bundles model + tokens
  + thinking + strict.
- Validation: missing args / unknown slugs raise ``CommandError``, which
  the dispatcher turns into a message rather than a stack trace.
- Routing: ``dispatch`` recognizes slash, bare slug, and free-text.
- Discoverability: ``/help`` lists every cmd_*.

Tests skip the network-heavy /plan + /cost paths (those exercise
context.assemble which has its own coverage in test_context.py); the
state contract is what matters here.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from agent_scaffold.config import Config
from agent_scaffold.discovery import Recipe
from agent_scaffold.repl.commands import (
    CommandHandler,
    CommandResult,
)
from agent_scaffold.repl.session import SessionState
from agent_scaffold.sources import DEPLOYMENTS_SPEC, ResolvedSource
from agent_scaffold.writer import WriteMode

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def demo_recipe(tmp_path: Path) -> Recipe:
    recipe_md = tmp_path / "demo.md"
    recipe_md.write_text("# Demo\n", encoding="utf-8")
    return Recipe(slug="demo", title="Demo", path=recipe_md, status="blueprint")


@pytest.fixture
def other_recipe(tmp_path: Path) -> Recipe:
    recipe_md = tmp_path / "other.md"
    recipe_md.write_text("# Other\n", encoding="utf-8")
    return Recipe(slug="customer-support-triage", title="Other", path=recipe_md)


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
def handler(demo_recipe: Recipe, other_recipe: Recipe) -> CommandHandler:
    return CommandHandler(recipes=[demo_recipe, other_recipe])


def _messages_text(result: CommandResult) -> str:
    """Render every message through a capturing Console and join the text.

    Rich Tables, Panels, and Texts each render differently, so we go
    through the real Console (with color stripped) to get plain text the
    tests can grep on.
    """
    from rich.console import Console

    console = Console(record=True, color_system=None, width=120)
    for msg in result.messages:
        console.print(msg)
    return console.export_text()


# ---------------------------------------------------------------------------
# Discovery + /help
# ---------------------------------------------------------------------------


def test_handler_discovers_commands_by_prefix(handler: CommandHandler) -> None:
    expected = {
        "help",
        "recipe",
        "language",
        "framework",
        "name",
        "dest",
        "model",
        "effort",
        "reset",
        "plan",
        "generate",
        "exit",
    }
    assert expected.issubset(set(handler.commands))
    # /cost (folded into /plan) and /go (kept for muscle memory) are aliases,
    # not discovered commands. They still dispatch via the _aliases mapping.
    assert "cost" not in set(handler.commands)
    assert "go" not in set(handler.commands)


def test_cmd_help_lists_every_command(handler: CommandHandler, base_state: SessionState) -> None:
    result = handler.dispatch("/help", base_state)
    text = _messages_text(result)
    for cmd in handler.commands:
        assert f"/{cmd}" in text


def test_cmd_help_points_at_refinement_subcommand(
    handler: CommandHandler, base_state: SessionState
) -> None:
    """Plain /help must nudge users toward /help refine so the free-text
    refinement surface is discoverable."""
    result = handler.dispatch("/help", base_state)
    text = _messages_text(result)
    assert "/help refine" in text


def test_cmd_help_refine_lists_every_refinement_key(
    handler: CommandHandler, base_state: SessionState
) -> None:
    """/help refine renders the REFINEMENT_KEYS registry — every key and
    its description must show up."""
    from agent_scaffold.repl.refine import REFINEMENT_KEYS

    result = handler.dispatch("/help refine", base_state)
    text = _messages_text(result)
    for key in REFINEMENT_KEYS:
        assert key in text, f"/help refine omitted refinement key {key!r}"


# ---------------------------------------------------------------------------
# /plan + /cost convergence
# ---------------------------------------------------------------------------


def test_cost_is_an_alias_for_plan(handler: CommandHandler, base_state: SessionState) -> None:
    """Typing /cost dispatches as /plan — the alias preserves muscle memory
    after the methods were merged. base_state isn't ready, so both fall into
    the "Plan needs:" pre-check and produce identical output."""
    plan_text = _messages_text(handler.dispatch("/plan", base_state))
    cost_text = _messages_text(handler.dispatch("/cost", base_state))
    assert plan_text == cost_text


def test_build_cost_renderable_uses_set_model(base_state: SessionState) -> None:
    """The shared cost helper used by /plan + /cost falls through to
    render_cost when a model is set."""
    from dataclasses import replace as dc_replace

    from agent_scaffold.repl.commands import _build_cost_renderable

    # Sonnet is in the pricing table, so this exercises the happy path.
    state_with_model = dc_replace(base_state, model="claude-sonnet-4-6")
    rendered = _build_cost_renderable(state_with_model)
    text = str(rendered)
    # render_cost returns "Est. cost ..." when pricing is known, or
    # "Est. cost unavailable" otherwise — either is acceptable; we just
    # assert it's the cost helper output, not the no-model hint.
    assert "Est. cost" in text
    assert "/model" not in text


# ---------------------------------------------------------------------------
# /recipe
# ---------------------------------------------------------------------------


def test_cmd_recipe_with_known_slug_mutates_state(
    handler: CommandHandler, base_state: SessionState, demo_recipe: Recipe
) -> None:
    result = handler.dispatch("/recipe demo", base_state)
    assert result.new_state is not None
    assert result.new_state.recipe == demo_recipe
    assert result.next_action == "continue"


def test_cmd_recipe_with_unknown_slug_returns_error_message(
    handler: CommandHandler, base_state: SessionState
) -> None:
    result = handler.dispatch("/recipe nonexistent", base_state)
    assert result.new_state is None  # no state change
    text = _messages_text(result)
    assert "unknown recipe" in text


def test_cmd_recipe_suggests_close_match_on_typo(
    handler: CommandHandler, base_state: SessionState
) -> None:
    result = handler.dispatch("/recipe demmo", base_state)
    text = _messages_text(result)
    assert "demo" in text


def test_cmd_recipe_shows_service_readiness_for_recipes_with_external_services(
    base_state: SessionState,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """/recipe <slug> for a recipe declaring external_services renders the
    readiness one-liner produced by ``probe_external_services``."""
    from agent_scaffold.discovery import ExternalService
    from agent_scaffold.doctor import CheckResult, CheckStatus

    recipe_md = tmp_path / "with-services.md"
    recipe_md.write_text("# With services\n", encoding="utf-8")
    recipe_with_services = Recipe(
        slug="with-services",
        title="With services",
        path=recipe_md,
        external_services=[
            ExternalService(id="postgres", required=True, probe="postgres_select_one"),
            ExternalService(id="qdrant", required=True, probe="qdrant_collections"),
        ],
    )

    fixed_results = [
        CheckResult(
            id="postgres",
            category="service",
            status=CheckStatus.OK,
            title="postgres: ok",
            detail="12ms",
        ),
        CheckResult(
            id="qdrant",
            category="service",
            status=CheckStatus.FAIL,
            title="qdrant: connect refused",
            detail="ConnectionRefusedError on localhost:6333",
        ),
    ]
    monkeypatch.setattr(
        "agent_scaffold.probes.probe_external_services",
        lambda services, timeout=1.0, max_workers=4: fixed_results,
    )

    handler_with_services = CommandHandler(recipes=[recipe_with_services])
    result = handler_with_services.dispatch("/recipe with-services", base_state)
    text = _messages_text(result)
    assert "Services" in text
    assert "ok postgres" in text
    assert "fail qdrant" in text
    # The recipe selection still succeeded — the readiness line is non-blocking.
    assert result.new_state is not None
    assert result.new_state.recipe == recipe_with_services


def test_cmd_recipe_no_external_services_skips_readiness_line(
    handler: CommandHandler, base_state: SessionState
) -> None:
    """The demo fixture recipe has no external_services — the readiness line
    must not appear."""
    result = handler.dispatch("/recipe demo", base_state)
    text = _messages_text(result)
    assert "Services" not in text


def test_cmd_recipe_probe_runner_crash_does_not_block_selection(
    base_state: SessionState,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the probe runner raises, the readiness line falls back to a dim
    warning and recipe selection still succeeds."""
    from agent_scaffold.discovery import ExternalService

    recipe_md = tmp_path / "boom.md"
    recipe_md.write_text("# Boom\n", encoding="utf-8")
    recipe = Recipe(
        slug="boom",
        title="Boom",
        path=recipe_md,
        external_services=[ExternalService(id="anything", probe="postgres_select_one")],
    )

    def explode(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("simulated runner crash")

    monkeypatch.setattr("agent_scaffold.probes.probe_external_services", explode)

    handler = CommandHandler(recipes=[recipe])
    result = handler.dispatch("/recipe boom", base_state)
    text = _messages_text(result)
    assert "probe runner failed" in text
    assert result.new_state is not None
    assert result.new_state.recipe == recipe


def test_cmd_recipe_no_args_lists_available(
    handler: CommandHandler, base_state: SessionState
) -> None:
    result = handler.dispatch("/recipe", base_state)
    text = _messages_text(result)
    assert "demo" in text
    assert "customer-support-triage" in text


def test_bare_slug_shortcuts_to_recipe(
    handler: CommandHandler, base_state: SessionState, demo_recipe: Recipe
) -> None:
    """Typing the slug alone (no slash) is equivalent to /recipe <slug>."""
    result = handler.dispatch("demo", base_state)
    assert result.new_state is not None
    assert result.new_state.recipe == demo_recipe


# ---------------------------------------------------------------------------
# /language /framework /name /dest
# ---------------------------------------------------------------------------


def test_cmd_language_accepts_python(handler: CommandHandler, base_state: SessionState) -> None:
    result = handler.dispatch("/language python", base_state)
    assert result.new_state is not None
    assert result.new_state.language == "python"


def test_cmd_language_rejects_unknown(handler: CommandHandler, base_state: SessionState) -> None:
    result = handler.dispatch("/language go", base_state)
    assert result.new_state is None
    assert "must be one of" in _messages_text(result)


def test_cmd_language_no_args_errors(handler: CommandHandler, base_state: SessionState) -> None:
    result = handler.dispatch("/language", base_state)
    assert "usage" in _messages_text(result)


def test_cmd_framework_sets_freeform(handler: CommandHandler, base_state: SessionState) -> None:
    result = handler.dispatch("/framework langgraph", base_state)
    assert result.new_state is not None
    assert result.new_state.framework == "langgraph"


def _write_framework_docs(root: Path) -> None:
    """Two python framework docs: one the test recipes declare, one they don't."""
    fw = root / "docs" / "frameworks"
    fw.mkdir(parents=True, exist_ok=True)
    (fw / "pydantic-ai.md").write_text(
        "---\nid: pydantic_ai\nlanguage: python\npackage: pydantic-ai\n"
        'versions:\n  minimum: ">=0.1.0"\n---\n\nBody.\n',
        encoding="utf-8",
    )
    (fw / "crewai.md").write_text(
        "---\nid: crewai\nlanguage: python\npackage: crewai\n"
        'versions:\n  minimum: ">=0.100.0"\n---\n\nBody.\n',
        encoding="utf-8",
    )


def _pydantic_ai_recipe(demo_recipe: Recipe) -> Recipe:
    return demo_recipe.model_copy(
        update={"recipe_dependencies": {"python": {"pydantic-ai": ">=0.1.0"}}}
    )


def test_cmd_framework_blocks_undeclared_for_recipe(
    base_state: SessionState, demo_recipe: Recipe
) -> None:
    """The generated code follows the recipe's blueprints — a framework the
    recipe never declares must be rejected, not silently recorded."""
    _write_framework_docs(base_state.deployments.path)
    recipe = _pydantic_ai_recipe(demo_recipe)
    state = SessionState(
        cfg=base_state.cfg,
        deployments=base_state.deployments,
        blueprints=base_state.blueprints,
        recipe=recipe,
        language="python",
    )
    result = CommandHandler(recipes=[recipe]).dispatch("/framework crewai", state)
    text = _messages_text(result)
    assert "pydantic_ai" in text
    assert result.new_state is None


def test_cmd_framework_allows_declared_and_normalizes(
    base_state: SessionState, demo_recipe: Recipe
) -> None:
    _write_framework_docs(base_state.deployments.path)
    recipe = _pydantic_ai_recipe(demo_recipe)
    state = SessionState(
        cfg=base_state.cfg,
        deployments=base_state.deployments,
        blueprints=base_state.blueprints,
        recipe=recipe,
        language="python",
    )
    result = CommandHandler(recipes=[recipe]).dispatch("/framework pydantic-ai", state)
    assert result.new_state is not None
    assert result.new_state.framework == "pydantic_ai"


def test_cmd_framework_always_allows_none(base_state: SessionState, demo_recipe: Recipe) -> None:
    _write_framework_docs(base_state.deployments.path)
    recipe = _pydantic_ai_recipe(demo_recipe)
    state = SessionState(
        cfg=base_state.cfg,
        deployments=base_state.deployments,
        blueprints=base_state.blueprints,
        recipe=recipe,
        language="python",
    )
    result = CommandHandler(recipes=[recipe]).dispatch("/framework none", state)
    assert result.new_state is not None
    assert result.new_state.framework == "none"


def test_cmd_framework_defers_validation_without_recipe(
    handler: CommandHandler, base_state: SessionState
) -> None:
    result = handler.dispatch("/framework crewai", base_state)
    assert result.new_state is not None
    assert result.new_state.framework == "crewai"
    assert "validated once a recipe is chosen" in _messages_text(result)


def test_cmd_recipe_warns_when_framework_no_longer_supported(
    base_state: SessionState, demo_recipe: Recipe
) -> None:
    """Changing the recipe under an incompatible framework warns instead of
    silently keeping a pick the new recipe cannot generate."""
    _write_framework_docs(base_state.deployments.path)
    recipe = _pydantic_ai_recipe(demo_recipe)
    state = SessionState(
        cfg=base_state.cfg,
        deployments=base_state.deployments,
        blueprints=base_state.blueprints,
        framework="crewai",
        language="python",
    )
    result = CommandHandler(recipes=[recipe]).dispatch("/recipe demo", state)
    assert result.new_state is not None
    text = _messages_text(result)
    assert "/framework" in text
    assert "crewai" in text


def test_cmd_observability_langfuse_swaps_obs(
    handler: CommandHandler, base_state: SessionState
) -> None:
    result = handler.dispatch("/observability langfuse", base_state)
    assert result.new_state is not None
    assert result.new_state.add_capabilities == ["obs.langfuse"]
    assert result.new_state.remove_capabilities == {"obs.langsmith", "obs.grafana-stack"}


def test_cmd_observability_langsmith_swaps_obs(
    handler: CommandHandler, base_state: SessionState
) -> None:
    result = handler.dispatch("/observability langsmith", base_state)
    assert result.new_state is not None
    assert result.new_state.add_capabilities == ["obs.langsmith"]
    assert result.new_state.remove_capabilities == {"obs.langfuse", "obs.grafana-stack"}


def test_cmd_observability_none_removes_all(
    handler: CommandHandler, base_state: SessionState
) -> None:
    result = handler.dispatch("/observability none", base_state)
    assert result.new_state is not None
    assert result.new_state.add_capabilities == []
    assert result.new_state.remove_capabilities == {
        "obs.langsmith",
        "obs.langfuse",
        "obs.grafana-stack",
    }


def test_cmd_observability_unknown_rejected(
    handler: CommandHandler, base_state: SessionState
) -> None:
    result = handler.dispatch("/observability newrelic", base_state)
    assert result.new_state is None
    assert "must be one of" in _messages_text(result)


def test_cmd_observability_no_args_errors(
    handler: CommandHandler, base_state: SessionState
) -> None:
    result = handler.dispatch("/observability", base_state)
    assert "usage" in _messages_text(result)


def test_cmd_name_auto_derives_dest(handler: CommandHandler, base_state: SessionState) -> None:
    result = handler.dispatch("/name demo-project", base_state)
    assert result.new_state is not None
    assert result.new_state.project_name == "demo-project"
    assert result.new_state.dest is not None
    assert result.new_state.dest.name == "demo-project"


def test_cmd_name_preserves_user_dest_when_already_set(
    handler: CommandHandler, base_state: SessionState, tmp_path: Path
) -> None:
    explicit = tmp_path / "already-here"
    pre = handler.dispatch(f"/dest {explicit}", base_state)
    assert pre.new_state is not None
    after = handler.dispatch("/name demo-project", pre.new_state)
    assert after.new_state is not None
    # dest stays where the user put it; doesn't get clobbered to cwd/<name>.
    assert after.new_state.dest == explicit.resolve()


def test_cmd_dest_expands_user(handler: CommandHandler, base_state: SessionState) -> None:
    result = handler.dispatch("/dest ~/foo", base_state)
    assert result.new_state is not None
    assert str(result.new_state.dest).startswith(str(Path.home()))


# ---------------------------------------------------------------------------
# /model and /effort
# ---------------------------------------------------------------------------


def test_cmd_model_sets_string(handler: CommandHandler, base_state: SessionState) -> None:
    result = handler.dispatch("/model claude-sonnet-4-6", base_state)
    assert result.new_state is not None
    assert result.new_state.model == "claude-sonnet-4-6"


def test_cmd_effort_low_bundles_haiku_and_no_thinking(
    handler: CommandHandler, base_state: SessionState
) -> None:
    result = handler.dispatch("/effort low", base_state)
    assert result.new_state is not None
    s = result.new_state
    assert s.effort == "low"
    assert s.model == "claude-haiku-4-5"
    assert s.thinking_budget is None
    assert s.strict is False


def test_cmd_effort_high_bundles_opus_strict_thinking(
    handler: CommandHandler, base_state: SessionState
) -> None:
    result = handler.dispatch("/effort high", base_state)
    assert result.new_state is not None
    s = result.new_state
    assert s.model == "claude-opus-4-8"
    assert s.thinking_budget == 16_000
    assert s.strict is True


def test_cmd_effort_unknown_level_errors(handler: CommandHandler, base_state: SessionState) -> None:
    result = handler.dispatch("/effort extreme", base_state)
    assert result.new_state is None
    assert "must be one of" in _messages_text(result)


def test_explicit_model_after_effort_overrides_preset(
    handler: CommandHandler, base_state: SessionState
) -> None:
    r1 = handler.dispatch("/effort high", base_state)
    assert r1.new_state is not None
    r2 = handler.dispatch("/model claude-haiku-4-5-20251001", r1.new_state)
    assert r2.new_state is not None
    # /model overrides; the other preset fields stick around.
    assert r2.new_state.model == "claude-haiku-4-5-20251001"
    assert r2.new_state.thinking_budget == 16_000


# ---------------------------------------------------------------------------
# /reset /go /exit
# ---------------------------------------------------------------------------


def test_cmd_reset_clears_selections_keeps_session_inputs(
    handler: CommandHandler, base_state: SessionState
) -> None:
    selected = handler.dispatch("/language python", base_state).new_state
    assert selected is not None
    after = handler.dispatch("/reset", selected)
    assert after.new_state is not None
    assert after.new_state.language is None
    # cfg / deployments / blueprints preserved
    assert after.new_state.cfg is base_state.cfg
    assert after.new_state.deployments is base_state.deployments
    assert after.new_state.write_mode == WriteMode.abort


def test_cmd_go_with_incomplete_state_reports_missing(
    handler: CommandHandler, base_state: SessionState
) -> None:
    result = handler.dispatch("/go", base_state)
    assert result.next_action == "continue"
    text = _messages_text(result)
    assert "missing" in text.lower()


def test_cmd_go_with_complete_state_signals_generate(
    handler: CommandHandler, base_state: SessionState
) -> None:
    state = base_state
    for line in [
        "/recipe demo",
        "/language python",
        "/framework langgraph",
        "/name demo-project",
    ]:
        result = handler.dispatch(line, state)
        assert result.new_state is not None
        state = result.new_state
    # The field-setters above all dirty the plan. Clear directly so this
    # test stays focused on the "ready + clean" path; the dirty path has
    # its own test below.
    state = replace(state, dirty_since_plan=False)
    result = handler.dispatch("/go", state)
    assert result.next_action == "generate"


def test_cmd_go_with_dirty_state_asks_for_confirmation(
    handler: CommandHandler, base_state: SessionState
) -> None:
    state = base_state
    for line in [
        "/recipe demo",
        "/language python",
        "/framework langgraph",
        "/name demo-project",
    ]:
        result = handler.dispatch(line, state)
        assert result.new_state is not None
        state = result.new_state
    assert state.dirty_since_plan is True
    result = handler.dispatch("/go", state)
    assert result.next_action == "confirm_generate"
    # The plan panel is rendered as part of the messages so the user sees
    # what they're about to ship before answering the confirm.
    text = _messages_text(result)
    assert "Plan" in text or "plan" in text or "recipe" in text.lower()


def test_cmd_go_does_not_clear_dirty_itself(
    handler: CommandHandler, base_state: SessionState
) -> None:
    """The confirm flow lives in the shell. cmd_go just signals it; the
    flag is only cleared on a confirmed /go (in the shell) or on /plan.
    """
    state = base_state
    for line in [
        "/recipe demo",
        "/language python",
        "/framework langgraph",
        "/name demo-project",
    ]:
        result = handler.dispatch(line, state)
        assert result.new_state is not None
        state = result.new_state
    result = handler.dispatch("/go", state)
    # If cmd_go returned new_state, dirty must still be True (the user
    # hasn't actually confirmed yet).
    if result.new_state is not None:
        assert result.new_state.dirty_since_plan is True


def test_cmd_plan_clears_dirty_flag(
    handler: CommandHandler,
    base_state: SessionState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful /plan render means the user has seen the resolved
    stack, so the dirty flag clears."""
    from tests.test_plan import _plan as _stub_plan

    monkeypatch.setattr("agent_scaffold.repl.commands._build_plan", lambda state: _stub_plan())
    state = base_state
    for line in [
        "/recipe demo",
        "/language python",
        "/framework langgraph",
        "/name demo-project",
    ]:
        result = handler.dispatch(line, state)
        assert result.new_state is not None
        state = result.new_state
    assert state.dirty_since_plan is True

    plan_result = handler.dispatch("/plan", state)
    assert plan_result.new_state is not None
    assert plan_result.new_state.dirty_since_plan is False


def test_cmd_plan_idempotent_when_already_clean(
    handler: CommandHandler,
    base_state: SessionState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running /plan when not dirty doesn't churn state — the command
    returns no new_state so the shell carries the existing one forward."""
    from tests.test_plan import _plan as _stub_plan

    monkeypatch.setattr("agent_scaffold.repl.commands._build_plan", lambda state: _stub_plan())
    state = base_state
    for line in [
        "/recipe demo",
        "/language python",
        "/framework langgraph",
        "/name demo-project",
    ]:
        result = handler.dispatch(line, state)
        assert result.new_state is not None
        state = result.new_state
    state = replace(state, dirty_since_plan=False)
    plan_result = handler.dispatch("/plan", state)
    assert plan_result.new_state is None


# ---------------------------------------------------------------------------
# /context
# ---------------------------------------------------------------------------


def test_cmd_context_with_incomplete_state_reports_missing(
    handler: CommandHandler, base_state: SessionState
) -> None:
    """/context refuses cleanly when required fields are missing —
    same UX shape as /plan, but the message names the right verb."""
    result = handler.dispatch("/context", base_state)
    assert result.new_state is None
    text = _messages_text(result)
    assert "Context needs" in text


def test_cmd_context_renders_summary_with_dropped_doc(
    handler: CommandHandler,
    base_state: SessionState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path renders the full ContextSummary — including the dropped
    doc name, which the plan panel never shows."""
    from agent_scaffold.context import AssembledContext, ContextSummary, TierStats
    from agent_scaffold.repl import commands as commands_module

    summary = ContextSummary(
        total_tokens=12_000,
        cap=80_000,
        tiers=[
            TierStats(tier=1, label="Recipe", docs=1, tokens=2_000),
            TierStats(tier=2, label="Composes / Load as Context", docs=2, tokens=8_000),
        ],
        dropped=["cross-cutting/observability.md"],
        truncated=[],
    )
    fake_ctx = AssembledContext(
        recipe_path=Path("/tmp/recipe.md"),
        referenced_paths=[],
        body="",
        token_estimate=12_000,
        summary=summary,
    )
    monkeypatch.setattr(commands_module, "_assemble_for_state", lambda _state: fake_ctx)

    state = base_state
    for line in [
        "/recipe demo",
        "/language python",
        "/framework langgraph",
        "/name demo-project",
    ]:
        result = handler.dispatch(line, state)
        assert result.new_state is not None
        state = result.new_state

    ctx_result = handler.dispatch("/context", state)
    text = _messages_text(ctx_result)
    assert "Recipe" in text
    assert "Dropped to fit budget" in text
    assert "cross-cutting/observability.md" in text


def test_help_lists_context_command(handler: CommandHandler, base_state: SessionState) -> None:
    """/help auto-discovers /context once cmd_context is defined."""
    result = handler.dispatch("/help", base_state)
    text = _messages_text(result)
    assert "/context" in text


# ---------------------------------------------------------------------------
# /write-mode
# ---------------------------------------------------------------------------


def test_cmd_write_mode_no_args_shows_current(
    handler: CommandHandler, base_state: SessionState
) -> None:
    result = handler.dispatch("/write-mode", base_state)
    assert result.new_state is None
    text = _messages_text(result).lower()
    # Default is WriteMode.abort.
    assert "abort" in text
    assert "options" in text


def test_cmd_write_mode_sets_state(handler: CommandHandler, base_state: SessionState) -> None:
    for mode in ("skip", "overwrite", "merge", "abort"):
        result = handler.dispatch(f"/write-mode {mode}", base_state)
        assert result.new_state is not None, f"/write-mode {mode} produced no new_state"
        assert result.new_state.write_mode.value == mode


def test_cmd_write_mode_rejects_unknown_value(
    handler: CommandHandler, base_state: SessionState
) -> None:
    result = handler.dispatch("/write-mode bogus", base_state)
    # CommandError gets caught and surfaced as a message; state should not change.
    assert result.new_state is None
    text = _messages_text(result).lower()
    assert "unknown" in text or "bogus" in text


def test_cmd_help_lists_write_mode(handler: CommandHandler, base_state: SessionState) -> None:
    result = handler.dispatch("/help", base_state)
    text = _messages_text(result)
    assert "/write_mode" in text or "/write-mode" in text


def test_cmd_exit_signals_exit(handler: CommandHandler, base_state: SessionState) -> None:
    result = handler.dispatch("/exit", base_state)
    assert result.next_action == "exit"


def test_quit_aliases_to_exit(handler: CommandHandler, base_state: SessionState) -> None:
    assert handler.dispatch("/quit", base_state).next_action == "exit"
    assert handler.dispatch("/q", base_state).next_action == "exit"


def test_cmd_new_signals_wizard(handler: CommandHandler, base_state: SessionState) -> None:
    """/new should hand the shell loop the signal to enter the guided wizard."""
    result = handler.dispatch("/new", base_state)
    assert result.next_action == "wizard"
    # State is carried through unchanged at this point — the wizard mutates it.
    assert result.new_state is base_state


def test_generate_runs_pipeline(handler: CommandHandler, base_state: SessionState) -> None:
    """/generate is the canonical verb that confirms + runs the pipeline."""
    # With incomplete state, /generate reports missing fields (continue).
    incomplete = handler.dispatch("/generate", base_state)
    assert incomplete.next_action == "continue"
    text = _messages_text(incomplete)
    assert "missing" in text.lower()

    # With a complete state, /generate signals generate (after dirty cleared).
    state = base_state
    for line in [
        "/recipe demo",
        "/language python",
        "/framework langgraph",
        "/name demo-project",
    ]:
        result = handler.dispatch(line, state)
        assert result.new_state is not None
        state = result.new_state
    state = replace(state, dirty_since_plan=False)
    ready = handler.dispatch("/generate", state)
    assert ready.next_action == "generate"


def test_gen_short_alias_routes_to_generate(
    handler: CommandHandler, base_state: SessionState
) -> None:
    """/gen and /go are aliases that route to the canonical /generate."""
    # Each alias produces the same 'missing fields' continue result as /generate.
    for alias in ("/gen", "/go"):
        result = handler.dispatch(alias, base_state)
        assert result.next_action == "continue"
        assert "missing" in _messages_text(result).lower()


# ---------------------------------------------------------------------------
# Dispatcher edge cases
# ---------------------------------------------------------------------------


def test_empty_input_is_noop(handler: CommandHandler, base_state: SessionState) -> None:
    result = handler.dispatch("", base_state)
    assert result.messages == []
    assert result.new_state is None
    assert result.next_action == "continue"


def test_unknown_slash_command_suggests_close_match(
    handler: CommandHandler, base_state: SessionState
) -> None:
    result = handler.dispatch("/recip demo", base_state)
    text = _messages_text(result)
    assert "Unknown command" in text
    assert "/recipe" in text


def test_unknown_slash_command_without_close_match(
    handler: CommandHandler, base_state: SessionState
) -> None:
    result = handler.dispatch("/zzz", base_state)
    text = _messages_text(result)
    assert "Unknown command" in text
    assert "/help" in text


def test_free_text_destructive_patch_returns_pending_for_confirmation(
    handler: CommandHandler,
    base_state: SessionState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A patch that overwrites the model is destructive; the dispatcher
    must defer the apply to the shell loop (so the user can confirm)."""
    from agent_scaffold.repl.session import StatePatch

    def fake_interpret(state, text, cfg):  # type: ignore[no-untyped-def]
        return StatePatch(model="claude-sonnet-4-6", notes="swapping for cost")

    monkeypatch.setattr("agent_scaffold.repl.commands.interpret_refinement", fake_interpret)
    result = handler.dispatch("swap to sonnet", base_state)
    assert result.new_state is None, "destructive patches must not auto-apply"
    assert result.pending_patch is not None
    assert result.pending_patch.model == "claude-sonnet-4-6"
    assert result.pending_patch.notes == "swapping for cost"
    text = _messages_text(result)
    assert "Interpreted refinement" in text
    assert "model" in text


def test_free_text_additive_patch_applies_inline_without_pending(
    handler: CommandHandler,
    base_state: SessionState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A purely additive patch (add_dependencies, notes) applies inline —
    no confirmation needed, no pending_patch set."""
    from agent_scaffold.repl.session import StatePatch

    def fake_interpret(state, text, cfg):  # type: ignore[no-untyped-def]
        return StatePatch(
            add_dependencies={"python": {"redis": ">=5"}},
            notes="cache layer requested",
        )

    monkeypatch.setattr("agent_scaffold.repl.commands.interpret_refinement", fake_interpret)
    result = handler.dispatch("add redis", base_state)
    assert result.pending_patch is None
    assert result.new_state is not None
    assert result.new_state.extra_dependencies == {"python": {"redis": ">=5"}}
    assert result.new_state.refinement_notes == ["cache layer requested"]


def test_free_text_failure_warns_and_leaves_state_intact(
    handler: CommandHandler,
    base_state: SessionState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RefinementError surfaces as a yellow warning, state untouched."""
    from agent_scaffold.repl.refine import RefinementError

    def fake_interpret(state, text, cfg):  # type: ignore[no-untyped-def]
        raise RefinementError("network down")

    monkeypatch.setattr("agent_scaffold.repl.commands.interpret_refinement", fake_interpret)
    result = handler.dispatch("swap to sonnet", base_state)
    assert result.new_state is None  # unchanged
    text = _messages_text(result)
    assert "Couldn't interpret" in text


def test_free_text_empty_patch_reports_no_change(
    handler: CommandHandler,
    base_state: SessionState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty patch (LLM understood but nothing to change) doesn't pretend to mutate."""
    from agent_scaffold.repl.session import StatePatch

    monkeypatch.setattr(
        "agent_scaffold.repl.commands.interpret_refinement",
        lambda *_a, **_kw: StatePatch(),
    )
    result = handler.dispatch("hmm", base_state)
    assert result.new_state is None
    assert "No changes" in _messages_text(result)


# ---------------------------------------------------------------------------
# _assemble_for_state cache
# ---------------------------------------------------------------------------


def test_assemble_for_state_caches_identical_inputs(
    base_state: SessionState,
    demo_recipe: Recipe,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated calls with the same state hit the cache — one assemble() invocation.

    Mirrors the /plan → /cost flow where both call sites want the same
    AssembledContext and previously walked the blueprints tree twice.
    """
    from agent_scaffold.repl import commands as commands_module

    commands_module._clear_assemble_cache()

    calls: list[tuple] = []

    class _StubContext:
        body = "stub"
        token_estimate = 100
        summary = None
        referenced_paths: list[Path] = []

    def fake_assemble(recipe, language, framework, deployments_path, **kwargs):  # type: ignore[no-untyped-def]
        calls.append((recipe.slug, language, framework, str(deployments_path)))
        return _StubContext()

    monkeypatch.setattr(commands_module, "assemble", fake_assemble)

    state = SessionState(
        cfg=base_state.cfg,
        deployments=base_state.deployments,
        blueprints=base_state.blueprints,
        recipe=demo_recipe,
        language="python",
        framework="langgraph",
    )

    commands_module._assemble_for_state(state)
    commands_module._assemble_for_state(state)
    commands_module._assemble_for_state(state)

    assert len(calls) == 1, "cache should dedupe identical inputs"


def test_assemble_for_state_cache_invalidates_on_recipe_change(
    base_state: SessionState,
    demo_recipe: Recipe,
    other_recipe: Recipe,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Different recipes must produce distinct cache entries — otherwise
    switching recipes mid-session would render the previous recipe's plan."""
    from agent_scaffold.repl import commands as commands_module

    commands_module._clear_assemble_cache()

    calls: list[str] = []

    class _StubContext:
        body = ""
        token_estimate = 0
        summary = None
        referenced_paths: list[Path] = []

    def fake_assemble(recipe, *_a, **_kw):  # type: ignore[no-untyped-def]
        calls.append(recipe.slug)
        return _StubContext()

    monkeypatch.setattr(commands_module, "assemble", fake_assemble)

    base_args: dict = {
        "cfg": base_state.cfg,
        "deployments": base_state.deployments,
        "blueprints": base_state.blueprints,
        "language": "python",
        "framework": "langgraph",
    }
    state_a = SessionState(recipe=demo_recipe, **base_args)
    state_b = SessionState(recipe=other_recipe, **base_args)

    commands_module._assemble_for_state(state_a)
    commands_module._assemble_for_state(state_b)
    commands_module._assemble_for_state(state_a)

    assert calls == [
        "demo",
        "customer-support-triage",
    ], "a → b → a should hit assemble for a and b once each, then cache for a"


def test_assemble_for_state_passes_resolved_stack(
    monkeypatch: pytest.MonkeyPatch,
    mock_deployments_path: Path,
) -> None:
    """``resolve_stack_for_session`` flows into ``context.assemble`` as
    ``resolved_stack=...`` so the load_list predicate evaluator sees the
    user's effective capability set, not just the recipe frontmatter."""
    from agent_scaffold.discovery import discover_recipes
    from agent_scaffold.repl import commands as commands_module

    commands_module._clear_assemble_cache()

    captured: dict[str, object] = {}

    class _StubContext:
        body = ""
        token_estimate = 0
        summary = None
        referenced_paths: list[Path] = []

    def fake_assemble(recipe, language, framework, deployments_path, **kwargs):  # type: ignore[no-untyped-def]
        captured["resolved_stack"] = kwargs.get("resolved_stack")
        return _StubContext()

    monkeypatch.setattr(commands_module, "assemble", fake_assemble)

    recipes = discover_recipes(mock_deployments_path)
    recipe = next(r for r in recipes if r.slug == "with-capabilities")
    cfg = Config(
        anthropic_api_key="test-key",
        cache_dir=mock_deployments_path / ".cache",
        failures_dir=mock_deployments_path / ".cache" / "failures",
    )
    src = ResolvedSource(
        spec=DEPLOYMENTS_SPEC,
        path=mock_deployments_path,
        label="test",
        kind="explicit-path",
        commit_sha=None,
    )
    state = SessionState(
        cfg=cfg,
        deployments=src,
        blueprints=src,
        recipe=recipe,
        language="python",
        framework="langgraph",
        add_capabilities=["obs.langfuse"],
    )

    commands_module._assemble_for_state(state)
    stack = captured["resolved_stack"]
    assert stack is not None
    assert "obs.langfuse" in stack.ids()  # type: ignore[union-attr]
    assert "cache.redis" in stack.ids()  # type: ignore[union-attr]


def test_assemble_for_state_cache_busts_on_capability_change(
    monkeypatch: pytest.MonkeyPatch,
    mock_deployments_path: Path,
) -> None:
    """Same recipe + different capability overrides must produce distinct
    cache entries — otherwise the first /plan after /observability langfuse
    would render with the pre-override context."""
    from agent_scaffold.discovery import discover_recipes
    from agent_scaffold.repl import commands as commands_module

    commands_module._clear_assemble_cache()

    calls: list[tuple[str, ...]] = []

    class _StubContext:
        body = ""
        token_estimate = 0
        summary = None
        referenced_paths: list[Path] = []

    def fake_assemble(recipe, language, framework, deployments_path, **kwargs):  # type: ignore[no-untyped-def]
        stack = kwargs.get("resolved_stack")
        ids = tuple(stack.ids()) if stack is not None else ()
        calls.append(ids)
        return _StubContext()

    monkeypatch.setattr(commands_module, "assemble", fake_assemble)

    recipes = discover_recipes(mock_deployments_path)
    recipe = next(r for r in recipes if r.slug == "with-capabilities")
    cfg = Config(
        anthropic_api_key="test-key",
        cache_dir=mock_deployments_path / ".cache",
        failures_dir=mock_deployments_path / ".cache" / "failures",
    )
    src = ResolvedSource(
        spec=DEPLOYMENTS_SPEC,
        path=mock_deployments_path,
        label="test",
        kind="explicit-path",
        commit_sha=None,
    )
    base_args: dict = {
        "cfg": cfg,
        "deployments": src,
        "blueprints": src,
        "recipe": recipe,
        "language": "python",
        "framework": "langgraph",
    }
    state_a = SessionState(**base_args)
    state_b = SessionState(**base_args, add_capabilities=["obs.langfuse"])

    commands_module._assemble_for_state(state_a)
    commands_module._assemble_for_state(state_b)
    commands_module._assemble_for_state(state_a)

    assert len(calls) == 2, "cache should bust on capability change but rehit for a"
    assert calls[0] != calls[1], "different override sets must reach assemble distinctly"


def test_cmd_docker_toggles_use_docker(handler: CommandHandler, base_state: SessionState) -> None:
    """/docker on|off|<bare> sets the tri-state use_docker (default None = auto)."""
    assert base_state.use_docker is None  # auto: containers when Docker is usable
    on = handler.dispatch("/docker on", base_state)
    assert on.new_state is not None and on.new_state.use_docker is True
    off = handler.dispatch("/docker off", on.new_state)
    assert off.new_state is not None and off.new_state.use_docker is False
    # Bare /docker from the auto default flips to an explicit on.
    toggled = handler.dispatch("/docker", base_state)
    assert toggled.new_state is not None and toggled.new_state.use_docker is True


def test_cmd_observability_notes_cloud_vs_docker(
    handler: CommandHandler, base_state: SessionState
) -> None:
    """The pick's delivery mode is spelled out so the post-generation path is clear."""
    result = handler.dispatch("/observability langsmith", base_state)
    text = " ".join(str(m) for m in result.messages)
    assert "/connect langsmith" in text
    result = handler.dispatch("/observability langfuse", base_state)
    text = " ".join(str(m) for m in result.messages)
    assert "docker" in text


# ---------------------------------------------------------------------------
# /stack — the full-catalog browser
# ---------------------------------------------------------------------------


def _fixture_catalog() -> Any:
    from types import SimpleNamespace

    from agent_scaffold.catalog import CapabilityCard, CapabilityEntry, VerificationEntry

    caps = [
        CapabilityEntry(
            id="cache.redis",
            kind="cache",
            path="docs/capabilities/cache/redis.md",
            env_vars=["REDIS_URL"],
            docker_service="redis",
            probe="redis_ping",
            card=CapabilityCard(name="Redis", description="Cache and queues."),
            cost_tier="free",
            provisioning_time="instant",
            verification=VerificationEntry(tier="T1", delivery="self-hosted"),
        ),
        CapabilityEntry(
            id="sandbox.e2b",
            kind="sandbox",
            path="docs/capabilities/sandbox/e2b.md",
            env_vars=["E2B_API_KEY"],
            probe="e2b_session_open",
            card=CapabilityCard(
                name="E2B",
                description="Hosted code sandbox.",
                required_credentials=["E2B_API_KEY"],
            ),
            cost_tier="per-call",
            provisioning_time="~5s",
            verification=VerificationEntry(tier="T1", delivery="managed"),
        ),
        CapabilityEntry(
            id="core.prompts",
            kind="core",
            path="docs/capabilities/core/prompts.md",
            card=CapabilityCard(name="Owned prompts", description="Prompt files."),
        ),
    ]
    return SimpleNamespace(capabilities=caps)


@pytest.fixture
def stack_catalog(monkeypatch: pytest.MonkeyPatch) -> Any:
    catalog = _fixture_catalog()
    monkeypatch.setattr("agent_scaffold.catalog.load_catalog_for_config", lambda _cfg: catalog)
    return catalog


def test_cmd_stack_registered_and_in_help(handler: CommandHandler) -> None:
    assert "stack" in handler.commands


def test_cmd_stack_lists_groups_and_marks_picked(
    handler: CommandHandler, base_state: SessionState, demo_recipe: Recipe, stack_catalog: Any
) -> None:
    base_state.recipe = demo_recipe
    base_state.add_capabilities = ["cache.redis"]
    result = handler.dispatch("/stack", base_state)
    text = _messages_text(result)
    assert "memory" in text
    assert "tools" in text
    assert "core (always included)" in text
    assert "cache.redis" in text
    assert "sandbox.e2b" in text
    assert "yes" in text  # cache.redis is picked via the recipe
    assert "docker + cloud override" in text or "docker" in text
    assert "cloud hosted" in text


def test_cmd_stack_layer_filter(
    handler: CommandHandler, base_state: SessionState, stack_catalog: Any
) -> None:
    result = handler.dispatch("/stack tools", base_state)
    text = _messages_text(result)
    assert "sandbox.e2b" in text
    assert "cache.redis" not in text


def test_cmd_stack_detail_card_shows_env_vars_and_connect_handle(
    handler: CommandHandler, base_state: SessionState, stack_catalog: Any
) -> None:
    result = handler.dispatch("/stack sandbox.e2b", base_state)
    text = _messages_text(result)
    assert "E2B_API_KEY" in text
    assert "Hosted code sandbox." in text
    assert "/connect e2b" in text


def test_cmd_stack_unknown_id_suggests_close_match(
    handler: CommandHandler, base_state: SessionState, stack_catalog: Any
) -> None:
    result = handler.dispatch("/stack sandbox.e2", base_state)
    text = _messages_text(result)
    assert "unknown layer or capability id" in text
    assert "sandbox.e2b" in text


def test_cmd_stack_catalog_unavailable_is_command_error(
    handler: CommandHandler, base_state: SessionState, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_scaffold.catalog import CatalogError

    def boom(_cfg: Any) -> Any:
        raise CatalogError("offline and no cache")

    monkeypatch.setattr("agent_scaffold.catalog.load_catalog_for_config", boom)
    result = handler.dispatch("/stack", base_state)
    assert "catalog unavailable" in _messages_text(result)


def test_cmd_stack_too_many_args_errors(
    handler: CommandHandler, base_state: SessionState, stack_catalog: Any
) -> None:
    result = handler.dispatch("/stack tools extra", base_state)
    assert "usage: /stack" in _messages_text(result)


def test_cmd_observability_hosting_argument(
    handler: CommandHandler, base_state: SessionState, monkeypatch: pytest.MonkeyPatch
) -> None:
    """/observability langfuse cloud lands the hosting override with the swap."""
    monkeypatch.setattr(
        CommandHandler, "_hosting_modes", lambda _self, _state, _cap: ["cloud", "docker"]
    )
    result = handler.dispatch("/observability langfuse cloud", base_state)
    assert result.new_state is not None
    assert result.new_state.add_capabilities == ["obs.langfuse"]
    assert result.new_state.hosting_overrides == {"obs.langfuse": "cloud"}
    assert "hosted on cloud" in _messages_text(result)


def test_cmd_observability_rejects_unsupported_hosting(
    handler: CommandHandler, base_state: SessionState, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(CommandHandler, "_hosting_modes", lambda _self, _state, _cap: ["cloud"])
    result = handler.dispatch("/observability langsmith docker", base_state)
    assert result.new_state is None
    assert "supports hosting cloud" in _messages_text(result)


def test_cmd_observability_grafana_stack_accepted(
    handler: CommandHandler, base_state: SessionState
) -> None:
    result = handler.dispatch("/observability grafana-stack", base_state)
    assert result.new_state is not None
    assert result.new_state.add_capabilities == ["obs.grafana-stack"]
    assert result.new_state.remove_capabilities == {"obs.langsmith", "obs.langfuse"}


# ---------------------------------------------------------------------------
# Command consolidation: /drafts -> /draft list, deprecation hints
# ---------------------------------------------------------------------------


def test_draft_list_subcommand_lists(handler: CommandHandler, base_state: SessionState) -> None:
    """Bare /draft and /draft list both render the saved-draft view."""
    bare = _messages_text(handler.dispatch("/draft", base_state))
    listed = _messages_text(handler.dispatch("/draft list", base_state))
    assert "No saved drafts" in bare
    assert "No saved drafts" in listed


def test_drafts_is_deprecated_alias_with_hint(
    handler: CommandHandler, base_state: SessionState
) -> None:
    """/drafts still works but prints the migration hint to /draft list."""
    result = handler.dispatch("/drafts", base_state)
    text = _messages_text(result)
    assert "/draft list" in text
    assert "No saved drafts" in text


def test_customize_is_deprecated_but_still_functional(
    handler: CommandHandler, base_state: SessionState
) -> None:
    """/customize keeps setting stack_mode (no functionality loss) but nudges
    toward the /new features menu and /layer."""
    result = handler.dispatch("/customize on", base_state)
    assert result.new_state is not None
    assert result.new_state.stack_mode == "customize"
    assert "retiring" in _messages_text(result)


def test_help_omits_deprecated_commands(handler: CommandHandler, base_state: SessionState) -> None:
    """Deprecated names drop out of /help + the discovered command list, but
    still dispatch."""
    assert "drafts" not in handler.commands
    assert "customize" not in handler.commands
    help_text = _messages_text(handler.dispatch("/help", base_state))
    assert "/drafts" not in help_text
    assert "/customize" not in help_text
    # Still dispatchable.
    assert "unknown command" not in _messages_text(handler.dispatch("/drafts", base_state)).lower()
