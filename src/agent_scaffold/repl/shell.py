"""Interactive shell loop for ``agent-scaffold scaffold``.

The shell is a thin glue layer: it owns the input loop (prompt_toolkit
``PromptSession``), renders messages through Rich, and calls
:func:`agent_scaffold.pipeline.run_generation` when the dispatcher signals
``next_action="generate"``. Everything else — state mutation, slash
commands, refinement interpretation — lives in the sibling modules.

Design choices:

- **prompt_toolkit for input.** Already transitively available via
  questionary; the same library powers aider / ipython / ptpython. Gives
  history navigation, tab completion, multi-line input, and the
  ``patch_stdout`` trick that keeps Rich output from corrupting the
  prompt line.
- **One PromptSession per shell.** History persists at
  ``~/.cache/agent-scaffold/repl_history`` via ``FileHistory``; the
  session also wires the completer and key bindings.
- **Generation runs synchronously in the loop thread.** The existing
  two-panel Live UI from :class:`RichProgressDisplay` works fine — Rich's
  Live re-takes the terminal during generation, then prompt_toolkit
  resumes when it returns.
- **Errors surface in the loop, not as crashes.** Any
  :class:`PipelineError` from a generation run is rendered to the REPL
  and the loop returns to the prompt; the user can fix and retry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console

from agent_scaffold import __version__
from agent_scaffold.branding import print_banner as render_banner
from agent_scaffold.config import Config
from agent_scaffold.context import ContextBudgetError, assemble
from agent_scaffold.discovery import DiscoveryError, Recipe, discover_recipes
from agent_scaffold.language_hints import load_language_hints
from agent_scaffold.manifest import ManifestNotFoundError, read_manifest
from agent_scaffold.pipeline import (
    PipelineError,
    PipelineInputs,
    print_next_steps,
    print_phase_summary,
    run_generation,
)
from agent_scaffold.progress import RichProgressDisplay
from agent_scaffold.repl.commands import CommandHandler, CommandResult
from agent_scaffold.repl.session import SessionState, StatePatch, apply_patch
from agent_scaffold.sources import ResolvedSource
from agent_scaffold.topology import resolve as resolve_topology

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

# Filename for prompt_toolkit's command history. Lives under the existing
# cache dir so it shares the user's "scaffold writes here" policy.
_HISTORY_FILENAME = "repl_history"

# The prompt string. Trailing space ensures the cursor doesn't hug "›".
_PROMPT = "scaffold › "


class ScaffoldCompleter(Completer):
    """Tab-completion for slash commands and recipe slugs.

    Recipe slugs are cached at construction (the shell builds the
    completer once per session). Slash completions trigger as soon as the
    user types ``/``; bare-slug completions trigger when the cursor is at
    the start of the line so a free-text refinement that mentions a slug
    name doesn't generate noise.
    """

    def __init__(self, command_names: list[str], recipe_slugs: list[str]) -> None:
        self._commands = sorted(command_names)
        self._slugs = sorted(recipe_slugs)

    def get_completions(self, document: Document, complete_event: object) -> Iterable[Completion]:
        text = document.text_before_cursor
        # Only complete on the first word of the line — multi-word completion
        # would compete with the LLM refinement input flow.
        if " " in text.lstrip():
            return
        word = document.get_word_before_cursor(WORD=True)
        if word.startswith("/"):
            prefix = word[1:]
            for name in self._commands:
                if name.startswith(prefix):
                    yield Completion(
                        f"/{name}",
                        start_position=-len(word),
                        display=f"/{name}",
                    )
            return
        # Bare slug completion only at start-of-line.
        for slug in self._slugs:
            if slug.startswith(word):
                yield Completion(slug, start_position=-len(word), display=slug)


def _build_key_bindings() -> KeyBindings:
    """Ctrl-D exits cleanly; Ctrl-L clears the screen.

    Ctrl-C is left to prompt_toolkit's default (clear current input line)
    rather than killing the shell — that matches the REPL contract the
    user expects from aider / ipython.
    """
    kb = KeyBindings()

    @kb.add(Keys.ControlD)
    def _ctrl_d(event: object) -> None:
        # Raising EOFError mirrors readline's behavior; the shell catches it.
        raise EOFError

    @kb.add(Keys.ControlL)
    def _ctrl_l(event: object) -> None:
        # prompt_toolkit exposes the running application via event.app.
        event.app.renderer.clear()  # type: ignore[attr-defined]

    return kb


def _print_banner(
    console: Console, deployments: ResolvedSource, blueprints: ResolvedSource
) -> None:
    """One-time welcome at shell open — gradient figlet logo + REPL hints.

    Uses the same orange→red figlet wordmark as the top-level ``agent-scaffold``
    banner so the two surfaces stay visually consistent. The body lines below
    the logo are REPL-specific (slash commands, not CLI subcommands).
    """
    body_lines = [
        f"[bold]Agent Scaffold[/]  [dim]v{__version__}[/]  [dim]interactive shell[/]",
        "",
        "[dim]Recommended flow:[/]  [bold]/new[/] for guided setup → [bold]/generate[/] to run",
        "",
        "[bold]Quick start:[/]",
        "  [#FFA500]/new[/]        wizard: recipe → language → framework → name → plan",
        "  [#FF8C00]/plan[/]       re-render the plan + cost with current selections",
        "  [#FF6347]/cost[/]       just the pre-flight cost line",
        "  [#FF4500]/generate[/]   confirm + run the pipeline ([dim]alias:[/] [bold]/go[/])",
        "  [#DC143C]/exit[/]       leave the shell ([dim]Ctrl-D works too[/])",
        "",
        f"[dim]Deployments:[/] {deployments.label}",
        f"[dim]Blueprints: [/] {blueprints.label}",
        "",
        '[dim]Type any free text to refine the plan ([bold]"swap to sonnet, add postgres"[/]).[/]',
    ]
    render_banner(console, body_lines)


def _build_pipeline_inputs(state: SessionState) -> PipelineInputs:
    """Translate the REPL's SessionState into the pipeline's frozen inputs.

    Mirrors what cmd_new does: assembles context (which the REPL
    intentionally re-does on each generate so refinements between runs
    pick up cleanly), infers topology, threads through all the override
    fields. Raises :class:`PipelineError` early if assemble blows the
    context budget so the shell can render the failure inline.
    """
    assert state.recipe is not None  # caller verified is_ready
    assert state.language is not None
    assert state.framework is not None
    assert state.project_name is not None
    assert state.dest is not None
    deployments_path = state.deployments.path
    if deployments_path is None:
        raise PipelineError(
            "deployments source unavailable",
            phase="context",
            hint="restart the shell with --deployments-path",
        )

    try:
        ctx = assemble(
            state.recipe,
            state.language,
            state.framework,
            deployments_path,
            blueprints_path=state.blueprints.path,
            max_context_tokens=state.cfg.max_context_tokens,
            max_link_depth=state.cfg.max_link_depth,
            max_tokens_per_doc=state.cfg.max_tokens_per_doc,
        )
    except ContextBudgetError as exc:
        raise PipelineError(str(exc), phase="context") from exc

    topology, roles = resolve_topology(state.recipe, ctx.body)

    cfg = state.cfg
    # Selection-vs-default precedence: explicit overrides on state win,
    # otherwise fall back to the Config values load_config produced.
    if state.model:
        cfg = cfg.model_copy(update={"model": state.model})
    if state.max_tokens is not None:
        cfg = cfg.model_copy(update={"max_tokens": state.max_tokens})
    if state.thinking_budget is not None:
        cfg = cfg.model_copy(update={"thinking_budget": state.thinking_budget})

    return PipelineInputs(
        cfg=cfg,
        recipe=state.recipe,
        language=state.language,
        framework=state.framework,
        project_name=state.project_name,
        raw_project_name=state.project_name,
        dest=state.dest,
        deployments=deployments_path,
        ctx=ctx,
        hints=_load_hints_for(state.language),
        topology=topology,
        roles=roles,
        write_mode=state.write_mode,
        strict=state.strict,
        format_output=True,
        skip_validation=False,
        no_cache=False,
        # Pipe the REPL refinement accumulators through so the generator
        # actually honours "swap to sonnet, add postgres, skip docker_up".
        extra_dependencies=state.extra_dependencies,
        extra_steps=state.extra_steps,
        removed_steps=state.removed_steps,
        removed_roles=state.removed_roles,
        refinement_notes=state.refinement_notes,
    )


def _load_hints_for(language: str) -> dict[str, object]:
    """Load language hints YAML. Thin wrapper around the leaf module so the
    REPL doesn't have to translate ``UnknownLanguageError`` — at this call
    site the language has already been validated by ``cmd_language`` or
    the wizard, so a missing YAML would be a real bug, not user error."""
    return load_language_hints(language)


def _run_generation_and_render(state: SessionState, console: Console) -> None:
    """Build inputs, run the pipeline, render the trailing summaries.

    Failures (any :class:`PipelineError`) print to the REPL but don't kill
    the loop — the user can fix the underlying issue and retry.
    """
    try:
        inputs = _build_pipeline_inputs(state)
    except PipelineError as exc:
        console.print(f"[red]{exc.phase or 'context'} failed:[/] {exc.message}")
        if exc.hint:
            console.print(exc.hint)
        return

    display = RichProgressDisplay(
        console,
        inputs.cfg.model,
        verbose=False,
        expected_files=len(state.recipe.required_files) or None if state.recipe else None,
    )
    try:
        report = run_generation(inputs, display=display)
    except PipelineError as exc:
        console.print(f"[red]{exc.phase or 'pipeline'} failed:[/] {exc.message}")
        if exc.hint:
            console.print(exc.hint)
        return

    if report.result is not None:
        console.print(f"[green]Generated[/] {len(report.result.files)} files.")
    if report.report is not None:
        console.print(
            f"[green]Wrote[/] {len(report.report.written)} new, "
            f"{len(report.report.overwritten)} overwritten, "
            f"{len(report.report.skipped)} skipped."
        )
    print_phase_summary(
        getattr(display, "phase_durations", {}),
        getattr(display, "warnings", []),
        getattr(display, "errors", []),
    )
    if report.result is None or state.dest is None or state.language is None:
        return

    if state.autorun:
        _autorun_after_repl_generate(state.dest, console)
    else:
        print_next_steps(
            state.dest, state.language, report.result.smoke_check, report.result.post_install
        )


def _autorun_after_repl_generate(project_dir: Path, console: Console) -> None:
    """REPL mirror of ``cmd_new``'s autorun chain.

    The REPL never raises ``typer.Exit`` on autorun failure — it prints the
    exit-code-as-warning and returns control to the prompt so the user can
    retry, inspect, or just keep going.
    """
    from agent_scaffold.cli import (
        _autorun_after_new,
        _resolve_capability_stack_silently,
        _resolve_recipe_silently,
    )

    try:
        manifest = read_manifest(project_dir)
    except ManifestNotFoundError as exc:
        console.print(f"[yellow]Autorun skipped:[/] {exc}")
        return
    recipe = _resolve_recipe_silently(manifest.recipe)
    resolved_stack = _resolve_capability_stack_silently(recipe)
    rc = _autorun_after_new(
        project_dir=project_dir,
        recipe=recipe,
        resolved_stack=resolved_stack,
        open_browser=True,
    )
    if rc != 0:
        console.print(f"[yellow]autorun finished with exit code {rc}[/]")


def _render(console: Console, result: CommandResult) -> None:
    for msg in result.messages:
        console.print(msg)


# ---------------------------------------------------------------------------
# /new wizard — guided sub-loop with arrow-key selection
# ---------------------------------------------------------------------------

# Sentinel returned by selection helpers when the user picks the "stop" /
# "pause wizard" option. Distinct from ``None`` so callers can tell pause
# apart from "Ctrl-C / no choice".
_STOP_SENTINEL: Any = object()

# Tokens that exit the post-selection refine loop. ``/stop`` matches the
# in-wizard pause vocabulary so the same word works at every prompt.
_WIZARD_QUIT_TOKENS = {"/quit", "/exit", "/cancel", "/q", "/stop"}


def _ask_select(prompt: str, choices: list[Any]) -> Any:
    """Ask a questionary select; returns the chosen value or ``None`` on cancel.

    Test seam: tests monkeypatch this with a deterministic stub so the
    wizard can be driven headlessly without a TTY. Real code path goes
    through ``questionary.select`` (already a project dep; uses
    prompt_toolkit underneath, so it composes with our shell).
    """
    import questionary

    return questionary.select(prompt, choices=choices, qmark="›").ask()


def _ask_text(prompt: str, default: str = "") -> Any:
    """Ask a questionary text input; ``None`` on Ctrl-C, str otherwise.

    Typed ``Any`` because questionary's stubs return ``Any`` and pinning
    a stricter return type would chain a ``cast`` through every caller.
    """
    import questionary

    return questionary.text(prompt, default=default, qmark="›").ask()


def _pause_choice() -> Any:
    import questionary

    return questionary.Choice("⏸  pause wizard (selections preserved)", value=_STOP_SENTINEL)


def _separator() -> Any:
    import questionary

    return questionary.Separator()


def _select_recipe(console: Console, recipes: dict[str, Recipe]) -> Any:
    """Arrow-key recipe pick. Returns Recipe, ``_STOP_SENTINEL``, or ``None`` on Ctrl-C."""
    if not recipes:
        console.print("[yellow]No recipes available; cancelling wizard.[/]")
        return _STOP_SENTINEL
    import questionary

    sorted_recipes = sorted(recipes.values(), key=lambda r: r.slug)
    longest_slug = max(len(r.slug) for r in sorted_recipes)
    choices: list[Any] = [
        questionary.Choice(
            f"{r.slug:<{longest_slug}}  [{r.status}]  {r.title}",
            value=r,
        )
        for r in sorted_recipes
    ]
    choices.append(_separator())
    choices.append(_pause_choice())
    return _ask_select("Pick a recipe (↑/↓ + Enter)", choices)


def _select_language() -> Any:
    import questionary

    choices = [
        questionary.Choice("python", value="python"),
        questionary.Choice("typescript", value="typescript"),
        _separator(),
        _pause_choice(),
    ]
    return _ask_select("Target language?", choices)


def _select_framework(language: str) -> Any:
    """Frameworks come from the language hints YAML (``framework_dependencies`` keys)."""
    import questionary

    hints = _load_hints_for(language)
    fw_deps = hints.get("framework_dependencies") or {}
    frameworks: list[str] = sorted(fw_deps.keys()) if isinstance(fw_deps, dict) else []
    choices: list[Any] = [questionary.Choice(name, value=name) for name in frameworks]
    choices.append(questionary.Choice("none (no specific framework)", value="none"))
    choices.append(_separator())
    choices.append(_pause_choice())
    return _ask_select(f"Framework for {language}?", choices)


def _input_name(default: str = "") -> Any:
    """Project name. Free-text — selection menus don't fit. Blank → pause."""
    raw = _ask_text("Project name?", default=default)
    if raw is None:  # Ctrl-C
        return None
    cleaned = raw.strip()
    return _STOP_SENTINEL if not cleaned else cleaned


def _input_dest(project_name: str, current: Path | None) -> Any:
    default = str(current) if current else str(Path.cwd() / project_name)
    raw = _ask_text("Destination?", default=default)
    if raw is None:
        return None
    cleaned = raw.strip()
    if not cleaned:
        return _STOP_SENTINEL
    return Path(cleaned).expanduser().resolve()


def _select_reuse_or_change(field_name: str, current_value: str) -> Any:
    """When a field is already set, ask: keep it, change it, or pause."""
    import questionary

    choices = [
        questionary.Choice(f"keep current: {current_value}", value="keep"),
        questionary.Choice("change it", value="change"),
        _separator(),
        questionary.Choice("⏸  pause wizard", value="stop"),
    ]
    return _ask_select(f"{field_name} already set — what now?", choices)


def _resolve_field(
    name: str,
    current: Any,
    display: str,
    picker: Callable[[], Any],
) -> tuple[Any, str]:
    """Run the reuse-or-change gate, then either return current or call ``picker``.

    Returns ``(value, action)`` where ``action`` is:
    - ``"keep"``   — current value retained, picker not run
    - ``"set"``    — picker returned a new value
    - ``"stop"``   — user chose to pause; caller should exit the wizard
    - ``"cancel"`` — Ctrl-C / EOF; treated the same as stop by callers
    """
    if current is not None:
        decision = _select_reuse_or_change(name, display)
        if decision == "keep":
            return current, "keep"
        if decision in (None, "stop"):
            return current, "stop"
    picked = picker()
    if picked is None:
        return current, "cancel"
    if picked is _STOP_SENTINEL:
        return current, "stop"
    return picked, "set"


def _refine_loop(
    session: PromptSession[str],
    console: Console,
    handler: CommandHandler,
    state: SessionState,
) -> tuple[SessionState, str]:
    """After selections are collected, show plan + ask for refinement or /generate.

    Returns ``(final_state, terminal_action)`` where terminal_action is:
    - ``"generate"`` — user typed /generate or /go → main loop runs pipeline
    - ``"quit"``     — user typed /quit / /stop → main loop continues with state
    """
    _render(console, handler.dispatch("/plan", state))
    while True:
        console.print(
            "[bold #FFB347]›[/] Refine with free text, "
            "[bold]/generate[/] to run, [bold]/stop[/] to leave wizard."
        )
        try:
            with patch_stdout():
                raw = session.prompt(" › ").strip()
        except (EOFError, KeyboardInterrupt):
            return state, "quit"
        if not raw:
            continue
        if raw in _WIZARD_QUIT_TOKENS:
            return state, "quit"
        if raw in ("/generate", "/go", "/gen"):
            ok, missing = state.is_ready()
            if ok:
                return state, "generate"
            console.print("[yellow]Can't generate yet — missing:[/] " + ", ".join(missing))
            continue
        if raw.startswith("/"):
            # Allow any other slash command inside refine loop so the user
            # can /model, /effort, /cost, /reset etc. without leaving.
            result = handler.dispatch(raw, state)
            _render(console, result)
            if result.new_state is not None:
                state = result.new_state
            if result.next_action == "exit":
                return state, "quit"
            continue
        # Free text → refinement interpreter; re-render the plan if it landed.
        result = handler.dispatch(raw, state)
        _render(console, result)
        if result.new_state is not None:
            state = result.new_state
            _render(console, handler.dispatch("/plan", state))


def _wizard_paused(state: SessionState, console: Console) -> tuple[SessionState, str]:
    """Universal "user paused" exit. Selections persist; show how to resume."""
    console.print(
        "[yellow]⏸  Wizard paused.[/] Selections preserved — "
        "use slash commands or [bold]/new[/] to resume where you left off."
    )
    return state, "quit"


# Each wizard step is a (label, field, display, picker, format_set) tuple.
# Driving the wizard from a table beats five copy-pasted 7-line blocks
# (one per field) — adding a sixth wizard step is now one row, not 8 lines
# of boilerplate. Each picker takes ``(console, state, handler)`` so it can
# look up dependent fields like ``state.language`` for the framework picker.


@dataclass(frozen=True)
class _WizardStep:
    label: str
    """Capitalised step name used in headings + the ``set`` checkmark line."""

    field: str
    """:class:`SessionState` / :class:`StatePatch` attribute name."""

    display: Callable[[SessionState], str]
    """Render the current value for the keep/change gate prompt."""

    picker: Callable[[Console, SessionState, CommandHandler], Any]
    """Run the actual user-facing pick — questionary select or text input."""

    format_set: Callable[[Any], str]
    """How to render the picked value in the ``✓ recipe: <foo>`` confirmation."""


def _name_default(state: SessionState) -> str:
    """Default project name: previous pick > recipe slug > empty."""
    return state.project_name or (state.recipe.slug if state.recipe else "")


_WIZARD_STEPS: tuple[_WizardStep, ...] = (
    _WizardStep(
        label="Recipe",
        field="recipe",
        display=lambda s: s.recipe.slug if s.recipe else "",
        picker=lambda c, s, h: _select_recipe(c, h.recipes),
        format_set=lambda v: str(v.slug),
    ),
    _WizardStep(
        label="Language",
        field="language",
        display=lambda s: s.language or "",
        picker=lambda c, s, h: _select_language(),
        format_set=str,
    ),
    _WizardStep(
        label="Framework",
        field="framework",
        display=lambda s: s.framework or "",
        picker=lambda c, s, h: _select_framework(s.language or "python"),
        format_set=str,
    ),
    _WizardStep(
        label="Name",
        field="project_name",
        display=lambda s: s.project_name or "",
        picker=lambda c, s, h: _input_name(default=_name_default(s)),
        format_set=str,
    ),
    _WizardStep(
        label="Destination",
        field="dest",
        display=lambda s: str(s.dest) if s.dest else "",
        picker=lambda c, s, h: _input_dest(s.project_name or "demo", s.dest),
        format_set=str,
    ),
)


def _run_new_wizard(
    session: PromptSession[str],
    console: Console,
    handler: CommandHandler,
    state: SessionState,
) -> tuple[SessionState, str]:
    """Guided wizard: arrow-key selections per step, with pause-and-resume.

    Each step gates on whether the field is already set: ``keep``,
    ``change``, or ``pause``. New fields show a one-pick selection (or
    text input for name/dest). Picking the pause option at any step
    returns to the main REPL with whatever's set so far — the user can
    keep adjusting via slash commands and run ``/new`` again to resume.
    """
    console.print(
        "[dim]Use ↑/↓ + Enter to select. Pick "
        "[bold]pause wizard[/bold] at any step to resume later via [bold]/new[/].[/dim]"
    )

    for step in _WIZARD_STEPS:

        def picker(step: _WizardStep = step, state: SessionState = state) -> Any:  # noqa: B023
            """Bind the loop variables so each iteration's picker sees its own
            step + state snapshot — picker is called immediately within this
            iteration, so binding at definition time is equivalent to binding
            at call time but sidesteps Python's late-binding semantics."""
            return step.picker(console, state, handler)

        value, action = _resolve_field(
            step.label,
            getattr(state, step.field),
            step.display(state),
            picker,
        )
        if action in ("stop", "cancel"):
            return _wizard_paused(state, console)
        if action == "set":
            state = apply_patch(state, StatePatch(**{step.field: value}))
            console.print(f"[green]✓[/] {step.label.lower()}: [bold]{step.format_set(value)}[/]")

    console.print(
        "\n[bold #FF6347]Selections complete.[/] Reviewing the plan with cost estimate…\n"
    )
    return _refine_loop(session, console, handler, state)


def run_shell(
    cfg: Config,
    deployments: ResolvedSource,
    blueprints: ResolvedSource,
    *,
    console: Console | None = None,
    prompt_factory: type[PromptSession[str]] = PromptSession,
) -> int:
    """Run the interactive REPL loop until the user exits.

    ``prompt_factory`` lets tests inject a stub PromptSession that yields
    a scripted sequence of lines instead of reading from a TTY. Returns
    the exit code (always 0 in normal operation; non-zero only if recipe
    discovery blows up at session open).

    The loop honors:

    - ``next_action="exit"`` from any cmd_* → break out cleanly
    - EOFError (Ctrl-D) → break out cleanly
    - KeyboardInterrupt (Ctrl-C) → clear input, stay in the loop
    - ``next_action="generate"`` → call ``run_generation``, then back to prompt
    """
    console = console or Console()
    if deployments.path is None:
        console.print("[red]Cannot start shell:[/] deployments source unavailable.")
        return 1

    try:
        recipes = discover_recipes(deployments.path)
    except DiscoveryError as exc:
        console.print(f"[red]Cannot start shell:[/] {exc}")
        return 1

    handler = CommandHandler(recipes=recipes)
    state = SessionState(cfg=cfg, deployments=deployments, blueprints=blueprints)

    history_file = cfg.cache_dir / _HISTORY_FILENAME
    history_file.parent.mkdir(parents=True, exist_ok=True)

    session: PromptSession[str] = prompt_factory(
        message=_PROMPT,
        history=FileHistory(str(history_file)),
        completer=ScaffoldCompleter(
            command_names=handler.commands,
            recipe_slugs=[r.slug for r in recipes],
        ),
        complete_while_typing=True,
        key_bindings=_build_key_bindings(),
    )

    _print_banner(console, deployments, blueprints)

    while True:
        try:
            with patch_stdout():
                line = session.prompt()
        except (EOFError, KeyboardInterrupt):
            console.print("[dim]bye.[/]")
            return 0
        result = handler.dispatch(line, state)
        _render(console, result)
        if result.new_state is not None:
            state = result.new_state
        if result.next_action == "exit":
            return 0
        if result.next_action == "wizard":
            state, terminal = _run_new_wizard(session, console, handler, state)
            if terminal == "generate":
                _run_generation_and_render(state, console)
            continue
        if result.next_action == "generate":
            _run_generation_and_render(state, console)

    # Unreachable in practice; keeps mypy happy if the loop is ever bounded.
    return 0
