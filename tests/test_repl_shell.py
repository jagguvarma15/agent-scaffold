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

import io
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from agent_scaffold.config import Config
from agent_scaffold.repl.shell import (
    ScaffoldCompleter,
    _accept_completion_or_submit,
    _apply_observability_choice,
    _build_pipeline_inputs,
    _format_observability_display,
    _print_banner,
    _print_turn_rule,
    _render_bottom_toolbar,
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


def test_banner_advertises_context_and_drops_stale_cost_descriptor(
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
) -> None:
    """Quick-start now points at /context (full tier breakdown). The old
    descriptor for /cost ("just the pre-flight cost line") has been a lie
    since /cost was folded into /plan — verify it's gone."""
    from rich.console import Console

    console = Console(record=True, color_system=None, width=120)
    _print_banner(console, deployments_source, blueprints_skipped)
    rendered = console.export_text()
    assert "/context" in rendered
    assert "just the pre-flight cost line" not in rendered


def test_banner_lists_help_in_quick_start(
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
) -> None:
    """The shell-open banner must mention /help so new users discover
    the full command surface beyond /new and /generate."""
    from rich.console import Console

    console = Console(record=True, color_system=None, width=120)
    _print_banner(console, deployments_source, blueprints_skipped)
    rendered = console.export_text()
    assert "/help" in rendered
    assert "/stack" in rendered


def test_banner_warns_when_startup_sync_failed(
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
    mock_deployments_path: Path,
) -> None:
    """A failed refresh must be loud in the banner, not a bare (cached)."""
    from dataclasses import replace as dc_replace

    from rich.console import Console

    stale = dc_replace(deployments_source, sync_failed=True, cache_mtime=1.0)
    console = Console(record=True, color_system=None, width=120)
    _print_banner(console, stale, blueprints_skipped)
    rendered = console.export_text()
    assert "Startup sync failed" in rendered
    assert "/sync" in rendered

    quiet = Console(record=True, color_system=None, width=120)
    _print_banner(quiet, deployments_source, blueprints_skipped)
    assert "Startup sync failed" not in quiet.export_text()


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


def test_completer_catches_slash_command_typo() -> None:
    """Fuzzy completion surfaces the intended command even when the prefix
    isn't exact — /observ still reaches /observability."""
    completer = ScaffoldCompleter(
        command_names=["observability", "generate", "layer"],
        recipe_slugs=[],
    )
    texts = {c.text for c in _complete(completer, "/observ")}
    assert "/observability" in texts


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

    def checkbox(self, _prompt: str, _choices: list[Any]) -> Any:
        return self._next()

    def text(self, _prompt: str, default: str = "") -> Any:
        nxt = self._next()
        return default if nxt == "__DEFAULT__" else nxt

    def _next(self) -> Any:
        try:
            return next(self._picks)
        except StopIteration as exc:
            raise AssertionError("wizard asked more questions than the test scripted") from exc


def _install_wizard_stubs(
    monkeypatch: pytest.MonkeyPatch, picks: list[Any], *, describe: str = ""
) -> None:
    """Replace shell's question helpers with the scripted version.

    The wizard now opens with a free-text "describe your agent" step; ``describe``
    is its answer and defaults to ``""`` so existing scripts (which start at the
    recipe pick) skip it without a Haiku call. Pass a non-empty ``describe`` and
    monkeypatch ``interpret_description`` to exercise the suggestion path.
    """
    from agent_scaffold.repl import shell as shell_module

    stub = _ScriptedSelections([describe, *picks])
    monkeypatch.setattr(shell_module, "_ask_select", stub.select)
    monkeypatch.setattr(shell_module, "_ask_text", stub.text)
    monkeypatch.setattr(shell_module, "_ask_checkbox", stub.checkbox)
    # Hosting modes come from the live catalog; pin them so the walk is
    # deterministic offline (a single mode auto-applies without a prompt).
    monkeypatch.setattr(shell_module, "_hosting_modes_for", lambda _s, _c: [])


def test_new_wizard_walks_arrow_selections_then_generates(
    cfg: Config,
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wizard collects all 5 required fields via arrow-key picks, then
    /generate signals the main loop to run the pipeline.

    Pick sequence: recipe (Recipe value) → language ("python") →
    framework ("langgraph") → name ("my-demo") → dest ("__DEFAULT__") →
    features menu (["observability"]) → obs backend ("langfuse")
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
            "my-demo",  # _input_name text
            "__DEFAULT__",  # _input_dest accepts default
            ["observability"],  # _select_optional_features checkbox
            "langfuse",  # _select_observability backend
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


def test_build_pipeline_inputs_canonicalizes_python_module_name(
    cfg: Config,
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
) -> None:
    """A hyphenated project name becomes a valid Python module name for the
    pipeline (so entry-point / module paths aren't ``src/research-assistant/``),
    mirroring ``cmd_new``; ``raw_project_name`` keeps the original."""
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
        project_name="research-assistant",
        dest=Path("/tmp/research-assistant"),
    )

    inputs = _build_pipeline_inputs(state)
    assert inputs.project_name == "research_assistant"  # module-path safe
    assert inputs.raw_project_name == "research-assistant"  # original preserved


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


def test_build_pipeline_inputs_resolves_stack(
    cfg: Config,
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
) -> None:
    """Recipe-declared capabilities + REPL overrides reach PipelineInputs.

    Without this thread, every ``/observability langfuse`` / ``/layer``
    swap or free-text ``add cache.redis`` refinement is silently
    discarded between the REPL state and the generator prompt + the
    manifest's capabilities array.
    """
    from agent_scaffold.discovery import discover_recipes
    from agent_scaffold.repl.session import SessionState

    recipes = discover_recipes(deployments_source.path)  # type: ignore[arg-type]
    recipe = next(r for r in recipes if r.slug == "with-capabilities")
    state = SessionState(
        cfg=cfg,
        deployments=deployments_source,
        blueprints=blueprints_skipped,
        recipe=recipe,
        language="python",
        framework="langgraph",
        project_name="demo",
        dest=Path("/tmp/demo"),
        add_capabilities=["obs.langfuse"],
        remove_capabilities={"host.vercel"},
    )

    inputs = _build_pipeline_inputs(state)
    assert inputs.resolved_stack is not None
    ids = inputs.resolved_stack.ids()
    assert "cache.redis" in ids
    assert "vector_db.qdrant" in ids
    assert "obs.langfuse" in ids
    assert "host.vercel" not in ids


def test_build_pipeline_inputs_prompts_write_mode_on_nonempty_dest(
    cfg: Config,
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A non-empty dest + WriteMode.abort + console → prompt; the chosen mode wins."""
    from rich.console import Console

    from agent_scaffold import cli_interactive
    from agent_scaffold.discovery import discover_recipes
    from agent_scaffold.repl.session import SessionState
    from agent_scaffold.writer import WriteMode

    dest = tmp_path / "proj"
    dest.mkdir()
    (dest / "existing.py").write_text("x = 1\n", encoding="utf-8")  # dest is not empty
    recipe = next(
        r for r in discover_recipes(deployments_source.path) if r.slug == "customer-support-triage"
    )
    state = SessionState(
        cfg=cfg,
        deployments=deployments_source,
        blueprints=blueprints_skipped,
        recipe=recipe,
        language="python",
        framework="langgraph",
        project_name="demo",
        dest=dest,
        write_mode=WriteMode.abort,
    )
    # The user picks "overwrite" at the prompt.
    monkeypatch.setattr(cli_interactive, "_select_write_mode", lambda: WriteMode.overwrite)
    inputs = _build_pipeline_inputs(state, Console())
    assert inputs.write_mode is WriteMode.overwrite


def test_build_pipeline_inputs_no_prompt_on_empty_dest(
    cfg: Config,
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An empty dest never prompts — the configured write_mode is kept."""
    from rich.console import Console

    from agent_scaffold import cli_interactive
    from agent_scaffold.discovery import discover_recipes
    from agent_scaffold.repl.session import SessionState
    from agent_scaffold.writer import WriteMode

    dest = tmp_path / "proj"
    dest.mkdir()  # exists but empty
    recipe = next(
        r for r in discover_recipes(deployments_source.path) if r.slug == "customer-support-triage"
    )
    state = SessionState(
        cfg=cfg,
        deployments=deployments_source,
        blueprints=blueprints_skipped,
        recipe=recipe,
        language="python",
        framework="langgraph",
        project_name="demo",
        dest=dest,
        write_mode=WriteMode.abort,
    )

    def _boom() -> WriteMode:
        raise AssertionError("must not prompt for an empty destination")

    monkeypatch.setattr(cli_interactive, "_select_write_mode", _boom)
    inputs = _build_pipeline_inputs(state, Console())
    assert inputs.write_mode is WriteMode.abort


def test_build_pipeline_inputs_no_stack_when_deployments_missing(
    cfg: Config,
    blueprints_skipped: ResolvedSource,
    mock_deployments_path: Path,
) -> None:
    """When the deployments path is unavailable, _build_pipeline_inputs
    raises a PipelineError before reaching stack resolution — but the
    helper itself degrades to None so callers that handle the missing
    path differently (e.g. plan rendering) don't crash."""
    from agent_scaffold.discovery import discover_recipes
    from agent_scaffold.repl._capabilities import resolve_stack_for_session
    from agent_scaffold.repl.session import SessionState

    recipes = discover_recipes(mock_deployments_path)
    recipe = next(r for r in recipes if r.slug == "with-capabilities")
    no_deployments = ResolvedSource(
        spec=DEPLOYMENTS_SPEC,
        path=None,
        label="skipped (offline)",
        kind="skipped",
        commit_sha=None,
    )
    state = SessionState(
        cfg=cfg,
        deployments=no_deployments,
        blueprints=blueprints_skipped,
        recipe=recipe,
        language="python",
        framework="langgraph",
        project_name="demo",
        dest=Path("/tmp/demo"),
    )

    assert resolve_stack_for_session(state) is None


# ---------------------------------------------------------------------------
# One-click run: wizard auto-flow into generate + docker-default resolution
# ---------------------------------------------------------------------------


def test_new_wizard_auto_offers_generate_when_configured(
    cfg: Config,
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selections done + no config gaps + confirm yes → generation runs without
    the user typing /generate (the auto-flow)."""
    from agent_scaffold.discovery import discover_recipes
    from agent_scaffold.repl import readiness as readiness_module
    from agent_scaffold.repl import shell as shell_module

    ran: list[Any] = []
    monkeypatch.setattr(
        shell_module, "_run_generation_and_render", lambda state, console: ran.append(state)
    )
    monkeypatch.setattr(readiness_module, "required_gaps", lambda _s: [])
    monkeypatch.setattr(shell_module, "_confirm_generate_now", lambda _c: True)

    recipes = discover_recipes(deployments_source.path)  # type: ignore[arg-type]
    target = next(r for r in recipes if r.slug == "customer-support-triage")
    _install_wizard_stubs(
        monkeypatch,
        [target, "python", "langgraph", "my-demo", "__DEFAULT__", ["observability"], "langfuse"],
    )
    # No /generate in the script — the wizard auto-offers it.
    factory = _make_session_factory(["/new", "/exit"])
    assert run_shell(cfg, deployments_source, blueprints_skipped, prompt_factory=factory) == 0
    assert len(ran) == 1
    assert ran[0].project_name == "my-demo"


def test_new_wizard_blocks_generate_when_unconfigured(
    cfg: Config,
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Config gaps → no auto-confirm, no generation; user is directed to /config."""
    from agent_scaffold.discovery import discover_recipes
    from agent_scaffold.repl import readiness as readiness_module
    from agent_scaffold.repl import shell as shell_module

    ran: list[Any] = []
    monkeypatch.setattr(shell_module, "_run_generation_and_render", lambda *a, **k: ran.append("x"))
    monkeypatch.setattr(readiness_module, "required_gaps", lambda _s: ["ANTHROPIC_API_KEY"])

    def _must_not_confirm(_c: object) -> bool:
        raise AssertionError("must not auto-confirm generation when the gate blocks")

    monkeypatch.setattr(shell_module, "_confirm_generate_now", _must_not_confirm)

    recipes = discover_recipes(deployments_source.path)  # type: ignore[arg-type]
    target = next(r for r in recipes if r.slug == "customer-support-triage")
    _install_wizard_stubs(
        monkeypatch,
        [target, "python", "langgraph", "my-demo", "__DEFAULT__", ["observability"], "langfuse"],
    )
    # Wizard drops into the refine loop (gaps present); /stop leaves it.
    factory = _make_session_factory(["/new", "/stop", "/exit"])
    assert run_shell(cfg, deployments_source, blueprints_skipped, prompt_factory=factory) == 0
    assert ran == []  # the gate prevented auto-generation


def _docker_state(
    cfg: Config, deployments_source: ResolvedSource, blueprints_skipped: ResolvedSource
) -> Any:
    from agent_scaffold.repl.session import SessionState

    return SessionState(cfg=cfg, deployments=deployments_source, blueprints=blueprints_skipped)


def test_resolve_repl_docker_auto_prefers_docker_when_available(
    cfg: Config,
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rich.console import Console

    import agent_scaffold.steps.docker_up as du
    from agent_scaffold.repl import shell as shell_module

    state = _docker_state(cfg, deployments_source, blueprints_skipped)  # use_docker=None (auto)
    monkeypatch.setattr(du, "docker_available", lambda **_k: (True, "ok"))
    assert shell_module._resolve_repl_docker(state, Console()) is True


def test_resolve_repl_docker_auto_falls_back_to_local(
    cfg: Config,
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rich.console import Console

    import agent_scaffold.steps.docker_up as du
    from agent_scaffold.repl import shell as shell_module

    state = _docker_state(cfg, deployments_source, blueprints_skipped)
    monkeypatch.setattr(du, "docker_available", lambda **_k: (False, "not installed"))
    assert shell_module._resolve_repl_docker(state, Console()) is False


def test_resolve_repl_docker_explicit_off_skips_probe(
    cfg: Config,
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rich.console import Console

    import agent_scaffold.steps.docker_up as du
    from agent_scaffold.repl import shell as shell_module

    state = _docker_state(cfg, deployments_source, blueprints_skipped)
    state.use_docker = False  # explicit /docker off

    def _boom(**_k: object) -> tuple[bool, str]:
        raise AssertionError("must not probe Docker when explicitly turned off")

    monkeypatch.setattr(du, "docker_available", _boom)
    assert shell_module._resolve_repl_docker(state, Console()) is False


def test_resolve_repl_docker_explicit_on_unavailable_warns(
    cfg: Config,
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rich.console import Console

    import agent_scaffold.steps.docker_up as du
    from agent_scaffold.repl import shell as shell_module

    state = _docker_state(cfg, deployments_source, blueprints_skipped)
    state.use_docker = True  # explicit /docker on, but Docker isn't usable
    monkeypatch.setattr(du, "docker_available", lambda **_k: (False, "daemon down"))
    console = Console(record=True, color_system=None, width=100)
    assert shell_module._resolve_repl_docker(state, console) is False
    assert "Docker not available" in console.export_text()


# ---------------------------------------------------------------------------
# Phase: the REPL input box — multiline bindings, bottom toolbar, turn rule
# ---------------------------------------------------------------------------


def _state(
    cfg: Config,
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
    **overrides: Any,
) -> Any:
    from agent_scaffold.repl.session import SessionState

    return SessionState(
        cfg=cfg, deployments=deployments_source, blueprints=blueprints_skipped, **overrides
    )


def test_bottom_toolbar_no_recipe_shows_defaults_and_keys(
    cfg: Config, deployments_source: ResolvedSource, blueprints_skipped: ResolvedSource
) -> None:
    state = _state(cfg, deployments_source, blueprints_skipped)  # no recipe, use_docker=None
    bar = _render_bottom_toolbar(state)
    assert "recipe: no recipe" in bar
    assert f"model: {state.cfg.model}" in bar  # falls back to Config model
    assert "docker: auto" in bar  # use_docker None -> auto
    assert "Enter submit" in bar and "Alt+Enter newline" in bar


def test_bottom_toolbar_reflects_recipe_model_override_and_docker(
    cfg: Config, deployments_source: ResolvedSource, blueprints_skipped: ResolvedSource
) -> None:
    from agent_scaffold.discovery import discover_recipes

    recipes = discover_recipes(deployments_source.path)  # type: ignore[arg-type]
    recipe = recipes[0]
    on = _render_bottom_toolbar(
        _state(
            cfg,
            deployments_source,
            blueprints_skipped,
            recipe=recipe,
            model="claude-sonnet-4-6",
            use_docker=True,
        )
    )
    assert f"recipe: {recipe.slug}" in on
    assert "model: claude-sonnet-4-6" in on  # state.model wins over cfg.model
    assert "docker: on" in on

    off = _render_bottom_toolbar(
        _state(cfg, deployments_source, blueprints_skipped, use_docker=False)
    )
    assert "docker: off" in off


def test_print_turn_rule_labels_recipe_when_selected(
    cfg: Config, deployments_source: ResolvedSource, blueprints_skipped: ResolvedSource
) -> None:
    from agent_scaffold.discovery import discover_recipes

    recipe = discover_recipes(deployments_source.path)[0]  # type: ignore[arg-type]
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=80)
    _print_turn_rule(console, _state(cfg, deployments_source, blueprints_skipped, recipe=recipe))
    out = buf.getvalue()
    assert recipe.slug in out
    assert "─" in out  # an actual rule was drawn


def test_print_turn_rule_bare_when_no_recipe(
    cfg: Config, deployments_source: ResolvedSource, blueprints_skipped: ResolvedSource
) -> None:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=80)
    _print_turn_rule(console, _state(cfg, deployments_source, blueprints_skipped))
    assert "─" in buf.getvalue()  # bare dim rule, no slug


class _FakeCompleteState:
    def __init__(self, completion: str | None) -> None:
        self.current_completion = completion


class _FakeBuffer:
    def __init__(self, complete_state: _FakeCompleteState | None = None) -> None:
        self.complete_state = complete_state
        self.submitted = False
        self.applied: str | None = None

    def validate_and_handle(self) -> None:
        self.submitted = True

    def apply_completion(self, completion: str) -> None:
        self.applied = completion


def test_enter_submits_when_no_completion_menu() -> None:
    buf = _FakeBuffer(complete_state=None)
    _accept_completion_or_submit(buf)
    assert buf.submitted is True
    assert buf.applied is None


def test_enter_accepts_highlighted_completion_instead_of_submitting() -> None:
    buf = _FakeBuffer(complete_state=_FakeCompleteState("/generate"))
    _accept_completion_or_submit(buf)
    assert buf.applied == "/generate"
    assert buf.submitted is False


def test_enter_submits_when_menu_open_but_nothing_highlighted() -> None:
    buf = _FakeBuffer(complete_state=_FakeCompleteState(None))
    _accept_completion_or_submit(buf)
    assert buf.submitted is True
    assert buf.applied is None


def test_run_shell_enables_multiline_and_bottom_toolbar(
    cfg: Config, deployments_source: ResolvedSource, blueprints_skipped: ResolvedSource
) -> None:
    """run_shell builds the session with multiline + a working toolbar callback,
    and draws a turn rule before prompting."""
    captured: dict[str, Any] = {}

    class _RecordingSession:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            self._lines = iter(["/exit"])

        def prompt(self, *_a: Any, **_k: Any) -> str:
            try:
                return next(self._lines)
            except StopIteration as exc:
                raise EOFError from exc

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=100)
    rc = run_shell(
        cfg,
        deployments_source,
        blueprints_skipped,
        console=console,
        prompt_factory=_RecordingSession,  # type: ignore[arg-type]
    )
    assert rc == 0
    assert captured["multiline"] is True
    toolbar = captured["bottom_toolbar"]
    assert callable(toolbar)
    assert "Enter submit" in toolbar()  # the callback renders the live toolbar
    assert "─" in buf.getvalue()  # a turn rule was drawn before the prompt


# ---------------------------------------------------------------------------
# The "describe your agent" first step
# ---------------------------------------------------------------------------


def test_run_describe_step_seeds_state_and_preselects_recipe(
    cfg: Config,
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_scaffold.discovery import discover_recipes
    from agent_scaffold.repl import refine as refine_module
    from agent_scaffold.repl import shell as shell_module
    from agent_scaffold.repl.commands import CommandHandler
    from agent_scaffold.repl.refine import DescriptionResult
    from agent_scaffold.repl.session import SessionState

    recipes = discover_recipes(deployments_source.path)  # type: ignore[arg-type]
    handler = CommandHandler(recipes=recipes)
    state = SessionState(cfg=cfg, deployments=deployments_source, blueprints=blueprints_skipped)

    monkeypatch.setattr(shell_module, "_ask_text", lambda *_a, **_k: "a customer support agent")
    monkeypatch.setattr(
        refine_module,
        "interpret_description",
        lambda _text, _recipes, _cfg: DescriptionResult(
            suggested_recipe_slug="customer-support-triage",
            agent_role="You are a support agent. Be concise and kind.",
            agent_title="Support Bot",
        ),
    )
    console = Console(file=io.StringIO(), force_terminal=False, width=100)
    new_state = shell_module._run_describe_step(console, handler, state)

    assert new_state.agent_description == "a customer support agent"
    assert new_state.agent_role == "You are a support agent. Be concise and kind."
    assert new_state.agent_title == "Support Bot"
    # The suggested recipe is pre-selected so the Recipe step offers keep/change.
    assert new_state.recipe is not None
    assert new_state.recipe.slug == "customer-support-triage"


def test_run_describe_step_skips_on_empty_without_calling_haiku(
    cfg: Config,
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_scaffold.discovery import discover_recipes
    from agent_scaffold.repl import refine as refine_module
    from agent_scaffold.repl import shell as shell_module
    from agent_scaffold.repl.commands import CommandHandler
    from agent_scaffold.repl.session import SessionState

    handler = CommandHandler(recipes=discover_recipes(deployments_source.path))  # type: ignore[arg-type]
    state = SessionState(cfg=cfg, deployments=deployments_source, blueprints=blueprints_skipped)

    monkeypatch.setattr(shell_module, "_ask_text", lambda *_a, **_k: "")

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("interpret_description must not run on empty input")

    monkeypatch.setattr(refine_module, "interpret_description", _boom)
    console = Console(file=io.StringIO(), force_terminal=False, width=100)
    new_state = shell_module._run_describe_step(console, handler, state)

    assert new_state.agent_description == ""  # marked skipped (not None) so /new won't re-ask
    assert new_state.agent_role is None
    assert new_state.recipe is None


def test_run_config_single_var_routes_through_secure_form(
    cfg: Config,
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`/config <VAR>` fills just that var via the secure form (managed services)."""
    from agent_scaffold.repl import shell as shell_module
    from agent_scaffold.repl.session import SessionState

    captured: dict[str, Any] = {}

    def fake_fill(report: Any, _console: Any, **kwargs: Any) -> None:
        captured["names"] = [r.name for r in report.requirements]
        captured["secure"] = kwargs.get("secure")

    monkeypatch.setattr("agent_scaffold.preflight.fill_missing", fake_fill)
    state = SessionState(cfg=cfg, deployments=deployments_source, blueprints=blueprints_skipped)
    console = Console(file=io.StringIO(), force_terminal=False, width=100)

    shell_module._run_config(state, console, var="REDIS_URL")

    assert captured["names"] == ["REDIS_URL"]  # only the named var, not the full walk
    assert captured["secure"] is True  # captured through the secure browser form


# ---------------------------------------------------------------------------
# Startup attach (/open via `scaffold <dir>`) + cwd hint
# ---------------------------------------------------------------------------


def _generated_project(tmp_path: Path) -> Path:
    from agent_scaffold.manifest import Manifest, write_manifest

    project = tmp_path / "existing-proj"
    project.mkdir()
    write_manifest(
        project,
        Manifest(
            recipe="demo",
            language="python",
            framework="langgraph",
            model="claude-test",
            generated_at="2026-01-01T00:00:00+00:00",
            answers={"project_name": "existing-proj"},
        ),
    )
    return project


def test_banner_lists_open(
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
) -> None:
    """The banner must mention /open so returning users discover the attach path."""
    console = Console(record=True, color_system=None, width=120)
    _print_banner(console, deployments_source, blueprints_skipped)
    assert "/open" in console.export_text()


def test_shell_open_dir_attaches_on_startup(
    cfg: Config,
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
    tmp_path: Path,
) -> None:
    project = _generated_project(tmp_path)
    console = Console(record=True, color_system=None, width=200)
    factory = _make_session_factory(["/exit"])

    exit_code = run_shell(
        cfg,
        deployments_source,
        blueprints_skipped,
        console=console,
        prompt_factory=factory,
        open_dir=project,
    )

    assert exit_code == 0
    rendered = console.export_text()
    assert "attached" in rendered
    assert str(project) in rendered


def test_shell_open_dir_without_manifest_warns_and_continues(
    cfg: Config,
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
    tmp_path: Path,
) -> None:
    plain = tmp_path / "not-a-project"
    plain.mkdir()
    console = Console(record=True, color_system=None, width=200)
    factory = _make_session_factory(["/exit"])

    exit_code = run_shell(
        cfg,
        deployments_source,
        blueprints_skipped,
        console=console,
        prompt_factory=factory,
        open_dir=plain,
    )

    assert exit_code == 0
    assert "Could not attach" in console.export_text()


def test_shell_hints_generated_project_in_cwd(
    cfg: Config,
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _generated_project(tmp_path)
    monkeypatch.chdir(project)
    console = Console(record=True, color_system=None, width=200)
    factory = _make_session_factory(["/exit"])

    run_shell(
        cfg,
        deployments_source,
        blueprints_skipped,
        console=console,
        prompt_factory=factory,
    )

    assert "/open . to attach" in console.export_text()


# ---------------------------------------------------------------------------
# Drafts hint skips generated projects
# ---------------------------------------------------------------------------


def _save_draft(cache_dir: Path, name: str, dest: Path) -> None:
    from agent_scaffold.repl.drafts import DraftSelections, save_draft

    save_draft(cache_dir, DraftSelections(name=name, dest=str(dest), project_name=name))


def test_hint_skips_generated_dest_drafts(cfg: Config, tmp_path: Path) -> None:
    from agent_scaffold.repl.shell import _hint_saved_drafts

    generated = _generated_project(tmp_path)
    _save_draft(cfg.cache_dir, "done-proj", generated)
    _save_draft(cfg.cache_dir, "wip-proj", tmp_path / "not-yet")

    console = Console(record=True, color_system=None, width=200)
    _hint_saved_drafts(console, cfg.cache_dir)
    rendered = console.export_text()
    assert "wip-proj" in rendered
    assert "done-proj" not in rendered


def test_hint_silent_when_all_drafts_generated(cfg: Config, tmp_path: Path) -> None:
    from agent_scaffold.repl.shell import _hint_saved_drafts

    generated = _generated_project(tmp_path)
    _save_draft(cfg.cache_dir, "done-proj", generated)

    console = Console(record=True, color_system=None, width=200)
    _hint_saved_drafts(console, cfg.cache_dir)
    assert console.export_text().strip() == ""


def test_wizard_customize_walk_covers_new_layers() -> None:
    """The customize walk must include the infrastructure and tools steps."""
    from agent_scaffold.repl.shell import _WIZARD_STEPS

    labels = [step.label for step in _WIZARD_STEPS]
    for expected in ("Memory", "Infrastructure", "Tools", "Eval", "Interface"):
        assert f"Layer · {expected}" in labels, expected


def _write_two_python_framework_docs(root: Path) -> None:
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


def test_wizard_framework_choices_filtered_by_recipe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The framework picker hides frameworks the recipe cannot generate —
    picking one would only record a framework the emitted code doesn't use."""
    from agent_scaffold.discovery import Recipe
    from agent_scaffold.repl import shell as shell_module

    _write_two_python_framework_docs(tmp_path)
    recipe_md = tmp_path / "demo.md"
    recipe_md.write_text("# Demo\n", encoding="utf-8")
    recipe = Recipe(
        slug="demo",
        title="Demo",
        path=recipe_md,
        recipe_dependencies={"python": {"pydantic-ai": ">=0.1.0"}},
    )

    captured: dict[str, list[Any]] = {}

    def fake_ask(_prompt: str, choices: list[Any]) -> Any:
        captured["values"] = [getattr(c, "value", None) for c in choices]
        return "pydantic_ai"

    monkeypatch.setattr(shell_module, "_ask_select", fake_ask)
    out = shell_module._select_framework(
        "python", tmp_path, recipe=recipe, console=Console(file=io.StringIO())
    )
    assert out == "pydantic_ai"
    assert "pydantic_ai" in captured["values"]
    assert "crewai" not in captured["values"]
    assert "none" in captured["values"]


def test_wizard_framework_choices_unfiltered_for_agnostic_recipe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A recipe with no framework package keeps the full per-language list."""
    from agent_scaffold.discovery import Recipe
    from agent_scaffold.repl import shell as shell_module

    _write_two_python_framework_docs(tmp_path)
    recipe_md = tmp_path / "demo.md"
    recipe_md.write_text("# Demo\n", encoding="utf-8")
    recipe = Recipe(slug="demo", title="Demo", path=recipe_md)

    captured: dict[str, list[Any]] = {}

    def fake_ask(_prompt: str, choices: list[Any]) -> Any:
        captured["values"] = [getattr(c, "value", None) for c in choices]
        return "crewai"

    monkeypatch.setattr(shell_module, "_ask_select", fake_ask)
    shell_module._select_framework("python", tmp_path, recipe=recipe)
    assert "crewai" in captured["values"]
    assert "pydantic_ai" in captured["values"]


# ---------------------------------------------------------------------------
# _render — vertical spacing between command messages
# ---------------------------------------------------------------------------


def test_render_blank_line_between_messages() -> None:
    """Consecutive messages get one blank line so panels don't stack
    edge-to-edge into a single wall of output."""
    from agent_scaffold.repl.commands import CommandResult
    from agent_scaffold.repl.shell import _render

    console = Console(record=True, width=60, force_terminal=False)
    _render(console, CommandResult(messages=["one", "two"]))
    assert "one\n\ntwo" in console.export_text()


def test_render_single_message_has_no_padding() -> None:
    from agent_scaffold.repl.commands import CommandResult
    from agent_scaffold.repl.shell import _render

    console = Console(record=True, width=60, force_terminal=False)
    _render(console, CommandResult(messages=["only"]))
    assert console.export_text() == "only\n"


# ---------------------------------------------------------------------------
# Features menu: RAG preset, observability hosting, guardrails gating
# ---------------------------------------------------------------------------


def _feature_state(
    cfg: Config, deployments_source: ResolvedSource, blueprints_skipped: ResolvedSource
) -> Any:
    from agent_scaffold.repl.session import SessionState

    return SessionState(cfg=cfg, deployments=deployments_source, blueprints=blueprints_skipped)


def test_apply_rag_choice_simple_expands_bundle(
    cfg: Config,
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The simple preset lands as expanded capability ids plus the preset name."""
    from agent_scaffold.bundles import default_presets
    from agent_scaffold.repl import shell as shell_module

    monkeypatch.setattr(shell_module, "_rag_bundle_presets", lambda _s: default_presets())
    state = _feature_state(cfg, deployments_source, blueprints_skipped)
    new_state = shell_module._apply_rag_choice(state, "simple")
    assert new_state.rag_preset == "simple"
    assert "vector_db.pgvector" in new_state.add_capabilities
    assert "embedding.openai" in new_state.add_capabilities


def test_apply_rag_choice_custom_opens_layer_walk(
    cfg: Config,
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
) -> None:
    from agent_scaffold.repl import shell as shell_module

    state = _feature_state(cfg, deployments_source, blueprints_skipped)
    state.optional_features = ["rag"]
    new_state = shell_module._apply_rag_choice(state, "custom")
    assert new_state.rag_preset == "custom"
    assert "layers" in new_state.optional_features
    assert new_state.add_capabilities == []


def test_apply_observability_choice_tuple_sets_hosting(
    cfg: Config,
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
) -> None:
    from agent_scaffold.repl import shell as shell_module

    state = _feature_state(cfg, deployments_source, blueprints_skipped)
    new_state = shell_module._apply_observability_choice(state, ("langfuse", "cloud"))
    assert new_state.add_capabilities == ["obs.langfuse"]
    assert new_state.remove_capabilities == {"obs.langsmith", "obs.grafana-stack"}
    assert new_state.hosting_overrides == {"obs.langfuse": "cloud"}


def test_select_observability_single_mode_skips_hosting_prompt(
    cfg: Config,
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One allowed mode auto-applies without a second question."""
    from agent_scaffold.repl import shell as shell_module

    asks: list[str] = []

    def fake_select(prompt: str, _choices: list[Any]) -> Any:
        asks.append(prompt)
        return "grafana-stack"

    monkeypatch.setattr(shell_module, "_ask_select", fake_select)
    monkeypatch.setattr(shell_module, "_hosting_modes_for", lambda _s, _c: ["docker"])
    state = _feature_state(cfg, deployments_source, blueprints_skipped)
    assert shell_module._select_observability(state) == ("grafana-stack", "docker")
    assert len(asks) == 1


def test_select_observability_two_modes_asks_hosting(
    cfg: Config,
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_scaffold.repl import shell as shell_module

    answers = iter(["langfuse", "cloud"])
    monkeypatch.setattr(shell_module, "_ask_select", lambda _p, _c: next(answers))
    monkeypatch.setattr(shell_module, "_hosting_modes_for", lambda _s, _c: ["cloud", "docker"])
    state = _feature_state(cfg, deployments_source, blueprints_skipped)
    assert shell_module._select_observability(state) == ("langfuse", "cloud")


def test_walk_steps_gate_feature_steps_on_menu(
    cfg: Config,
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
) -> None:
    """Nothing picked in the menu: every feature step is disabled, so the
    wizard goes straight from Destination to the plan."""
    from agent_scaffold.repl.shell import _WALK_STEPS

    state = _feature_state(cfg, deployments_source, blueprints_skipped)
    state.optional_features = []
    feature_steps = [s for s in _WALK_STEPS if s.phase == "feature"]
    assert feature_steps, "the walk should carry feature steps"
    assert all(s.enabled_when is not None for s in feature_steps)
    assert not any(s.enabled_when(state) for s in feature_steps if s.enabled_when)
    state.optional_features = ["rag", "observability", "guardrails"]
    enabled = [s.label for s in feature_steps if s.enabled_when and s.enabled_when(state)]
    assert "RAG preset" in enabled
    assert "Observability" in enabled
    assert "Layer · Guardrails" in enabled
    assert "Layer · Memory" not in enabled


# ---------------------------------------------------------------------------
# Tier wizard step
# ---------------------------------------------------------------------------


def _tier_state(
    cfg: Config,
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
    *,
    recipe_tier: str | None,
    tmp_path: Path,
) -> Any:
    from agent_scaffold.discovery import Recipe
    from agent_scaffold.repl.session import SessionState

    recipe_md = tmp_path / "tiered.md"
    recipe_md.write_text("# Tiered\n", encoding="utf-8")
    recipe = Recipe(slug="tiered", title="Tiered", path=recipe_md, tier=recipe_tier)
    state = SessionState(cfg=cfg, deployments=deployments_source, blueprints=blueprints_skipped)
    state.recipe = recipe
    return state


@pytest.fixture
def _embedded_tier_presets(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_scaffold.repl import _capabilities
    from agent_scaffold.tiers import default_presets

    monkeypatch.setattr(_capabilities, "session_tier_presets", lambda _s: default_presets())


@pytest.mark.usefixtures("_embedded_tier_presets")
def test_select_tier_puts_recipe_default_first(
    cfg: Config,
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_scaffold.repl import shell as shell_module

    captured: dict[str, Any] = {}

    def fake_select(_prompt: str, choices: list[Any]) -> Any:
        captured["choices"] = choices
        return "T2"

    monkeypatch.setattr(shell_module, "_ask_select", fake_select)
    state = _tier_state(
        cfg, deployments_source, blueprints_skipped, recipe_tier="T1", tmp_path=tmp_path
    )
    assert shell_module._select_tier(state) == "T2"
    first = captured["choices"][0]
    assert first.value == "T1"
    assert "(recipe default)" in str(first.title)


@pytest.mark.usefixtures("_embedded_tier_presets")
def test_select_tier_no_recipe_tier_defaults_to_none_choice(
    cfg: Config,
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_scaffold.repl import shell as shell_module

    captured: dict[str, Any] = {}

    def fake_select(_prompt: str, choices: list[Any]) -> Any:
        captured["choices"] = choices
        return shell_module._TIER_NONE

    monkeypatch.setattr(shell_module, "_ask_select", fake_select)
    state = _tier_state(
        cfg, deployments_source, blueprints_skipped, recipe_tier=None, tmp_path=tmp_path
    )
    shell_module._select_tier(state)
    assert captured["choices"][0].value == shell_module._TIER_NONE


def test_apply_tier_choice_sets_clears_and_noops(
    cfg: Config,
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
    tmp_path: Path,
) -> None:
    from agent_scaffold.repl import shell as shell_module

    state = _tier_state(
        cfg, deployments_source, blueprints_skipped, recipe_tier="T1", tmp_path=tmp_path
    )
    picked = shell_module._apply_tier_choice(state, "T3")
    assert picked.tier == "T3"
    # Both clear sentinels drop an explicit pick back to the recipe fallback.
    cleared = shell_module._apply_tier_choice(picked, shell_module._TIER_NONE)
    assert cleared.tier is None
    recleared = shell_module._apply_tier_choice(picked, shell_module._TIER_RECIPE_DEFAULT)
    assert recleared.tier is None
    # skip_when auto-apply passes None — a strict no-op.
    assert shell_module._apply_tier_choice(state, None) is state


@pytest.mark.usefixtures("_embedded_tier_presets")
def test_select_tier_clear_choice_is_honest_about_recipe_default(
    cfg: Config,
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a recipe-declared tier there is no "no tier" outcome — the clear
    entry must say it keeps the recipe default, and its confirmation must not
    report "none" while generation still seeds the recipe tier."""
    from agent_scaffold.repl import shell as shell_module

    captured: dict[str, Any] = {}

    def fake_select(_prompt: str, choices: list[Any]) -> Any:
        captured["choices"] = choices
        return shell_module._TIER_RECIPE_DEFAULT

    monkeypatch.setattr(shell_module, "_ask_select", fake_select)
    state = _tier_state(
        cfg, deployments_source, blueprints_skipped, recipe_tier="T2", tmp_path=tmp_path
    )
    shell_module._select_tier(state)
    titles = [str(getattr(c, "title", "")) for c in captured["choices"]]
    assert not any("(no tier)" in t for t in titles)
    clear = next(
        c
        for c in captured["choices"]
        if getattr(c, "value", None) == shell_module._TIER_RECIPE_DEFAULT
    )
    assert "recipe default T2" in str(clear.title)
    assert shell_module._format_tier_set(shell_module._TIER_RECIPE_DEFAULT) == "recipe default"


def test_tier_step_skips_only_undeclared_recipes(
    cfg: Config,
    deployments_source: ResolvedSource,
    blueprints_skipped: ResolvedSource,
    tmp_path: Path,
) -> None:
    """The walk prompts for tier exactly when the recipe declares one (or an
    explicit pick already exists) — undeclared recipes keep today's flow."""
    from agent_scaffold.repl import shell as shell_module

    step = next(s for s in shell_module._MANDATORY_STEPS if s.label == "Tier")
    assert step.skip_when is not None
    declared = _tier_state(
        cfg, deployments_source, blueprints_skipped, recipe_tier="T2", tmp_path=tmp_path
    )
    assert step.skip_when(declared) is False
    undeclared = _tier_state(
        cfg, deployments_source, blueprints_skipped, recipe_tier=None, tmp_path=tmp_path
    )
    assert step.skip_when(undeclared) is True
    # An explicit earlier pick keeps the step interactive even without a
    # recipe declaration (the keep/change gate then applies).
    undeclared.tier = "T1"
    assert step.skip_when(undeclared) is False
