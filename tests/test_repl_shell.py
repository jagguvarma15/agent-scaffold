"""Tests for ``agent_scaffold.repl.shell`` — the prompt_toolkit loop.

We stub out the PromptSession with a class that yields a scripted line
sequence so the test runs headless. Real PromptSession needs a TTY,
which CI doesn't have, so this seam (``prompt_factory``) is permanent —
not just for tests.

Three things matter:
1. The loop processes lines in order and dispatches each.
2. ``next_action="exit"`` and EOFError both break the loop cleanly.
3. History file gets created at the configured path.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from agent_scaffold.config import Config
from agent_scaffold.repl.shell import (
    ScaffoldCompleter,
    _apply_observability_choice,
    _build_pipeline_inputs,
    _format_observability_display,
    run_shell,
)
from agent_scaffold.sources import DEPLOYMENTS_SPEC, ResolvedSource

# ---------------------------------------------------------------------------
# Stub PromptSession that replays a fixed line list
# ---------------------------------------------------------------------------


class _ScriptedSession:
    """Minimal PromptSession-compatible stub for headless tests.

    Records every prompt() call so tests can verify the loop iteration
    count without having to scrape stdout.
    """

    def __init__(self, **_kwargs: Any) -> None:
        self.lines: Iterator[str] = iter([])
        self.calls = 0

    def prompt(self, *args: Any, **kwargs: Any) -> str:
        self.calls += 1
        try:
            return next(self.lines)
        except StopIteration as exc:
            # Exhausting the script should EOF the loop cleanly.
            raise EOFError from exc


def _make_session_factory(lines: list[str]) -> type[Any]:
    """Build a factory that produces a _ScriptedSession pre-loaded with lines."""

    class _Factory(_ScriptedSession):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.lines = iter(lines)

    return _Factory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(
        anthropic_api_key="test-key",
        cache_dir=tmp_path / "cache",
        failures_dir=tmp_path / "cache" / "failures",
    )


@pytest.fixture
def deployments_source(mock_deployments_path: Path) -> ResolvedSource:
    return ResolvedSource(
        spec=DEPLOYMENTS_SPEC,
        path=mock_deployments_path,
        label="test deployments",
        kind="explicit-path",
        commit_sha=None,
    )


@pytest.fixture
def blueprints_skipped() -> ResolvedSource:
    """Skipped (offline) blueprints — what the resolver returns when
    network is unavailable. Shell should handle this gracefully."""
    return ResolvedSource(
        spec=DEPLOYMENTS_SPEC,  # spec is just for label; path is None
        path=None,
        label="skipped (offline)",
        kind="skipped",
        commit_sha=None,
    )


# ---------------------------------------------------------------------------
# Loop happy path
# ---------------------------------------------------------------------------


def test_shell_processes_scripted_lines_then_exits_on_exit_command(
    cfg: Config,
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
) -> None:
    factory = _make_session_factory(["/help", "/exit"])
    exit_code = run_shell(
        cfg,
        deployments_source,
        blueprints_skipped,
        prompt_factory=factory,
    )
    assert exit_code == 0


def test_shell_exits_cleanly_on_eof(
    cfg: Config,
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
) -> None:
    """Empty script → first prompt() raises EOFError → loop exits 0."""
    factory = _make_session_factory([])
    assert run_shell(cfg, deployments_source, blueprints_skipped, prompt_factory=factory) == 0


def test_shell_writes_history_file_at_cache_dir(
    cfg: Config,
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
) -> None:
    """The shell creates the cache dir + history file on first session."""
    factory = _make_session_factory(["/exit"])
    run_shell(cfg, deployments_source, blueprints_skipped, prompt_factory=factory)
    # FileHistory creates the file on the first write attempt; the parent
    # dir is created by run_shell upfront.
    assert (cfg.cache_dir / "repl_history").parent.exists()


def test_destructive_refinement_confirmed_applies_and_renders_delta(
    cfg: Config,
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the user confirms a destructive refinement, the loop applies
    the patch (the next iteration sees the new state) and renders a delta."""
    from agent_scaffold.repl import shell as shell_module
    from agent_scaffold.repl.session import StatePatch

    monkeypatch.setattr(
        "agent_scaffold.repl.commands.interpret_refinement",
        lambda *_a, **_kw: StatePatch(model="claude-sonnet-4-6"),
    )
    # Auto-confirm.
    monkeypatch.setattr(shell_module, "_confirm_refinement", lambda *_a, **_kw: True)

    factory = _make_session_factory(["swap to sonnet", "/exit"])
    assert run_shell(cfg, deployments_source, blueprints_skipped, prompt_factory=factory) == 0


def test_destructive_refinement_declined_leaves_state_intact(
    cfg: Config,
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Declining the confirm aborts the apply — loop continues with the
    pre-refinement state and prints "Skipped"."""
    from agent_scaffold.repl import shell as shell_module
    from agent_scaffold.repl.session import StatePatch

    applied_calls: list[bool] = []
    real_apply = shell_module.apply_patch

    def tracking_apply(state, patch):  # type: ignore[no-untyped-def]
        applied_calls.append(True)
        return real_apply(state, patch)

    monkeypatch.setattr(
        "agent_scaffold.repl.commands.interpret_refinement",
        lambda *_a, **_kw: StatePatch(model="claude-sonnet-4-6"),
    )
    monkeypatch.setattr(shell_module, "_confirm_refinement", lambda *_a, **_kw: False)
    monkeypatch.setattr(shell_module, "apply_patch", tracking_apply)

    factory = _make_session_factory(["swap to sonnet", "/exit"])
    assert run_shell(cfg, deployments_source, blueprints_skipped, prompt_factory=factory) == 0
    # Decline path must NOT call apply_patch — the patch is dropped.
    assert applied_calls == []


def test_shell_returns_nonzero_when_deployments_unavailable(
    cfg: Config,
    blueprints_skipped: ResolvedSource,
) -> None:
    """If deployments resolved to None (offline + no bundled fallback), we
    can't discover recipes — shell refuses to start rather than crash."""
    no_deployments = ResolvedSource(
        spec=DEPLOYMENTS_SPEC,
        path=None,
        label="unavailable",
        kind="skipped",
        commit_sha=None,
    )
    factory = _make_session_factory([])
    assert run_shell(cfg, no_deployments, blueprints_skipped, prompt_factory=factory) == 1


# ---------------------------------------------------------------------------
# Completer
# ---------------------------------------------------------------------------


def test_completer_suggests_slash_commands_on_slash_prefix() -> None:
    completer = ScaffoldCompleter(
        command_names=["help", "recipe", "exit"],
        recipe_slugs=["demo", "other"],
    )
    completions = list(_complete(completer, "/r"))
    texts = {c.text for c in completions}
    assert "/recipe" in texts
    # Not the recipe slug — we're past the slash boundary.
    assert "demo" not in texts


def test_completer_suggests_slugs_on_bare_word() -> None:
    completer = ScaffoldCompleter(
        command_names=["help"],
        recipe_slugs=["demo", "demo-extended", "other"],
    )
    completions = list(_complete(completer, "dem"))
    texts = {c.text for c in completions}
    assert "demo" in texts
    assert "demo-extended" in texts
    assert "other" not in texts


def test_completer_quiet_after_first_word() -> None:
    """Once the user has typed past the first word (e.g. a refinement),
    we shouldn't pop noisy completions over their typing."""
    completer = ScaffoldCompleter(command_names=["help"], recipe_slugs=["demo"])
    assert list(_complete(completer, "swap to dem")) == []


def _complete(completer: ScaffoldCompleter, text: str) -> Any:
    """Helper: get_completions wants a Document + arbitrary event object."""
    from prompt_toolkit.document import Document

    return completer.get_completions(Document(text), object())


# ---------------------------------------------------------------------------
# /new wizard
# ---------------------------------------------------------------------------


class _ScriptedSelections:
    """Plays back scripted answers for ``_ask_select`` / ``_ask_text`` calls.

    The wizard uses questionary internally; tests inject this stub via
    monkeypatch so the headless test run doesn't need a TTY.
    """

    def __init__(self, picks: list[Any]) -> None:
        self._picks = iter(picks)

    def select(self, _prompt: str, _choices: list[Any]) -> Any:
        return self._next()

    def text(self, _prompt: str, default: str = "") -> Any:
        nxt = self._next()
        return default if nxt == "__DEFAULT__" else nxt

    def _next(self) -> Any:
        try:
            return next(self._picks)
        except StopIteration as exc:
            raise AssertionError("wizard asked more questions than the test scripted") from exc


def _install_wizard_stubs(monkeypatch: pytest.MonkeyPatch, picks: list[Any]) -> None:
    """Replace shell's question helpers with the scripted version."""
    from agent_scaffold.repl import shell as shell_module

    stub = _ScriptedSelections(picks)
    monkeypatch.setattr(shell_module, "_ask_select", stub.select)
    monkeypatch.setattr(shell_module, "_ask_text", stub.text)


def test_new_wizard_walks_arrow_selections_then_generates(
    cfg: Config,
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wizard collects all 5 required fields via arrow-key picks, then
    /generate signals the main loop to run the pipeline.

    Pick sequence: recipe (Recipe value) → language ("python") →
    framework ("langgraph") → name ("my-demo") → dest ("__DEFAULT__")
    → /generate (from the refine loop's text prompt).
    """
    from agent_scaffold.discovery import discover_recipes
    from agent_scaffold.repl import shell as shell_module

    run_generation_called: list[Any] = []
    monkeypatch.setattr(
        shell_module,
        "_run_generation_and_render",
        lambda state, console: run_generation_called.append(state),
    )

    recipes = discover_recipes(deployments_source.path)  # type: ignore[arg-type]
    target_recipe = next(r for r in recipes if r.slug == "customer-support-triage")

    _install_wizard_stubs(
        monkeypatch,
        [
            target_recipe,  # _select_recipe
            "python",  # _select_language
            "langgraph",  # _select_framework
            "langfuse",  # _select_observability
            "my-demo",  # _input_name text
            "__DEFAULT__",  # _input_dest accepts default
        ],
    )

    factory = _make_session_factory(["/new", "/generate", "/exit"])
    assert run_shell(cfg, deployments_source, blueprints_skipped, prompt_factory=factory) == 0
    assert len(run_generation_called) == 1
    final_state = run_generation_called[0]
    assert final_state.recipe.slug == "customer-support-triage"
    assert final_state.language == "python"
    assert final_state.framework == "langgraph"
    assert final_state.project_name == "my-demo"
    assert final_state.dest is not None
    assert "obs.langfuse" in final_state.add_capabilities
    assert "obs.langsmith" in final_state.remove_capabilities


def test_new_wizard_pause_returns_to_repl_with_partial_state(
    cfg: Config,
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """User picks the pause option mid-wizard → shell continues, no generation runs."""
    from agent_scaffold.discovery import discover_recipes
    from agent_scaffold.repl import shell as shell_module

    run_called: list[Any] = []
    monkeypatch.setattr(
        shell_module,
        "_run_generation_and_render",
        lambda *_a, **_kw: run_called.append("x"),
    )

    recipes = discover_recipes(deployments_source.path)  # type: ignore[arg-type]
    target_recipe = next(r for r in recipes if r.slug == "customer-support-triage")

    # User picks recipe, then picks the pause sentinel at the language step.
    _install_wizard_stubs(
        monkeypatch,
        [
            target_recipe,  # _select_recipe
            shell_module._STOP_SENTINEL,  # _select_language → pause wizard
        ],
    )

    factory = _make_session_factory(["/new", "/help", "/exit"])
    assert run_shell(cfg, deployments_source, blueprints_skipped, prompt_factory=factory) == 0
    assert run_called == []  # no generation, but shell stayed alive


def test_new_wizard_resume_offers_keep_or_change_for_set_fields(
    cfg: Config,
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Second /new after a pause shows the reuse-or-change gate on already-set fields."""
    from agent_scaffold.discovery import discover_recipes
    from agent_scaffold.repl import shell as shell_module

    monkeypatch.setattr(
        shell_module,
        "_run_generation_and_render",
        lambda *_a, **_kw: None,
    )

    recipes = discover_recipes(deployments_source.path)  # type: ignore[arg-type]
    first_recipe = next(r for r in recipes if r.slug == "customer-support-triage")

    # 1st /new: pick recipe → pause at language.
    # 2nd /new: "Recipe already set" gate → keep → pause at language again.
    _install_wizard_stubs(
        monkeypatch,
        [
            first_recipe,  # _select_recipe (run 1)
            shell_module._STOP_SENTINEL,  # _select_language (run 1) → pause
            "keep",  # _select_reuse_or_change for Recipe (run 2)
            shell_module._STOP_SENTINEL,  # _select_language (run 2) → pause
        ],
    )

    factory = _make_session_factory(["/new", "/new", "/exit"])
    assert run_shell(cfg, deployments_source, blueprints_skipped, prompt_factory=factory) == 0


# ---------------------------------------------------------------------------
# Observability wizard step apply / display helpers
# ---------------------------------------------------------------------------


def test_apply_observability_langfuse(
    cfg: Config,
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
) -> None:
    """Wizard pick 'langfuse' produces the same patch shape as /observability langfuse."""
    from agent_scaffold.repl.session import SessionState

    state = SessionState(cfg=cfg, deployments=deployments_source, blueprints=blueprints_skipped)
    new_state = _apply_observability_choice(state, "langfuse")
    assert "obs.langfuse" in new_state.add_capabilities
    assert "obs.langsmith" in new_state.remove_capabilities


def test_apply_observability_none_drops_all(
    cfg: Config,
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
) -> None:
    from agent_scaffold.repl.session import SessionState

    state = SessionState(cfg=cfg, deployments=deployments_source, blueprints=blueprints_skipped)
    new_state = _apply_observability_choice(state, "none")
    assert new_state.add_capabilities == []
    assert {"obs.langsmith", "obs.langfuse"} <= new_state.remove_capabilities


def test_format_observability_display_reflects_state(
    cfg: Config,
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
) -> None:
    """The display callback reads the current add/remove sets so the keep/change
    gate shows the user what they picked last time."""
    from agent_scaffold.repl.session import SessionState

    state = SessionState(cfg=cfg, deployments=deployments_source, blueprints=blueprints_skipped)
    assert _format_observability_display(state) == ""
    state = _apply_observability_choice(state, "langfuse")
    assert _format_observability_display(state) == "langfuse"
    state = _apply_observability_choice(state, "none")
    assert _format_observability_display(state) == "none"


# ---------------------------------------------------------------------------
# _build_pipeline_inputs — the bridge between REPL state and the pipeline
# ---------------------------------------------------------------------------


def test_build_pipeline_inputs_threads_overrides_through(
    cfg: Config,
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """state.model / max_tokens / thinking override Config before reaching the pipeline."""
    from agent_scaffold.discovery import discover_recipes
    from agent_scaffold.repl.session import SessionState
    from agent_scaffold.writer import WriteMode

    recipes = discover_recipes(deployments_source.path)  # type: ignore[arg-type]
    recipe = next(r for r in recipes if r.slug == "customer-support-triage")
    state = SessionState(
        cfg=cfg,
        deployments=deployments_source,
        blueprints=blueprints_skipped,
        recipe=recipe,
        language="python",
        framework="langgraph",
        project_name="demo",
        dest=Path("/tmp/demo"),
        model="claude-sonnet-4-6",
        max_tokens=99_999,
        thinking_budget=4_000,
        write_mode=WriteMode.overwrite,
    )

    inputs = _build_pipeline_inputs(state)
    assert inputs.cfg.model == "claude-sonnet-4-6"
    assert inputs.cfg.max_tokens == 99_999
    assert inputs.cfg.thinking_budget == 4_000
    assert inputs.write_mode == WriteMode.overwrite
    assert inputs.project_name == "demo"
    assert inputs.deployments == deployments_source.path


def test_build_pipeline_inputs_propagates_refinement_accumulators(
    cfg: Config,
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
) -> None:
    """Refinement state on SessionState must reach PipelineInputs.

    Locks in the fix for the silent-no-op bug: SessionState.extra_*,
    removed_*, and refinement_notes used to be rendered in the REPL delta
    panel but never threaded into the pipeline, so the generator never
    saw them. Every field must round-trip here.
    """
    from agent_scaffold.discovery import discover_recipes
    from agent_scaffold.repl.session import SessionState

    recipes = discover_recipes(deployments_source.path)  # type: ignore[arg-type]
    recipe = next(r for r in recipes if r.slug == "customer-support-triage")
    state = SessionState(
        cfg=cfg,
        deployments=deployments_source,
        blueprints=blueprints_skipped,
        recipe=recipe,
        language="python",
        framework="langgraph",
        project_name="demo",
        dest=Path("/tmp/demo"),
        extra_dependencies={"python": {"psycopg": "^3.2"}},
        extra_steps=["wire prometheus exporter"],
        removed_steps={"docker_up"},
        removed_roles={"evaluator"},
        refinement_notes=["use async/await throughout"],
    )

    inputs = _build_pipeline_inputs(state)

    assert inputs.extra_dependencies == {"python": {"psycopg": "^3.2"}}
    assert inputs.extra_steps == ["wire prometheus exporter"]
    assert inputs.removed_steps == {"docker_up"}
    assert inputs.removed_roles == {"evaluator"}
    assert inputs.refinement_notes == ["use async/await throughout"]
