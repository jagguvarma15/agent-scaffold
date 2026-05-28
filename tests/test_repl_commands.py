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

from pathlib import Path

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
        "cost",
        "go",
        "exit",
    }
    assert expected.issubset(set(handler.commands))


def test_cmd_help_lists_every_command(handler: CommandHandler, base_state: SessionState) -> None:
    result = handler.dispatch("/help", base_state)
    text = _messages_text(result)
    for cmd in handler.commands:
        assert f"/{cmd}" in text


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
    assert s.model == "claude-haiku-4-5-20251001"
    assert s.thinking_budget is None
    assert s.strict is False


def test_cmd_effort_high_bundles_opus_strict_thinking(
    handler: CommandHandler, base_state: SessionState
) -> None:
    result = handler.dispatch("/effort high", base_state)
    assert result.new_state is not None
    s = result.new_state
    assert s.model == "claude-opus-4-7"
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
    result = handler.dispatch("/go", state)
    assert result.next_action == "generate"


def test_cmd_exit_signals_exit(handler: CommandHandler, base_state: SessionState) -> None:
    result = handler.dispatch("/exit", base_state)
    assert result.next_action == "exit"


def test_quit_aliases_to_exit(handler: CommandHandler, base_state: SessionState) -> None:
    assert handler.dispatch("/quit", base_state).next_action == "exit"
    assert handler.dispatch("/q", base_state).next_action == "exit"


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


def test_free_text_hands_off_to_refinement_interpreter(
    handler: CommandHandler,
    base_state: SessionState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Free text routes through the Haiku interpreter; state moves on success."""
    from agent_scaffold.repl.session import StatePatch

    def fake_interpret(state, text, cfg):  # type: ignore[no-untyped-def]
        return StatePatch(model="claude-sonnet-4-6", notes="swapping for cost")

    monkeypatch.setattr("agent_scaffold.repl.commands.interpret_refinement", fake_interpret)
    result = handler.dispatch("swap to sonnet", base_state)
    assert result.new_state is not None
    assert result.new_state.model == "claude-sonnet-4-6"
    assert result.new_state.refinement_notes == ["swapping for cost"]


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
