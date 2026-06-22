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

from dataclasses import dataclass, replace
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
from agent_scaffold.branding import ACCENT, MUTED, PANEL_BORDER_STYLE
from agent_scaffold.branding import print_banner as render_banner
from agent_scaffold.capabilities import CapabilityKind, load_capabilities
from agent_scaffold.cli_shared import prompt_to_raise_context_cap
from agent_scaffold.config import Config
from agent_scaffold.context import ContextBudgetError, assemble
from agent_scaffold.discovery import (
    DiscoveryError,
    Recipe,
    discover_recipes,
    infer_complexity,
)
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
from agent_scaffold.repl._capabilities import resolve_stack_for_session
from agent_scaffold.repl.commands import CommandHandler, CommandResult
from agent_scaffold.repl.render import render_file_diffs, render_patch_delta
from agent_scaffold.repl.session import SessionState, StatePatch, apply_patch
from agent_scaffold.sources import ResolvedSource
from agent_scaffold.topology import resolve as resolve_topology
from agent_scaffold.writer import FileDiff, WriteMode

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from prompt_toolkit.key_binding.key_processor import KeyPressEvent

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


def _accept_completion_or_submit(buf: Any) -> None:
    """Enter behavior for the multiline prompt.

    If the completion menu has a highlighted item, accept it (so Enter finishes
    a ``/command`` the user is tab-cycling through); otherwise submit the buffer.
    Keeps single-line ``Enter = run`` intact while ``multiline=True`` is on.
    """
    state = buf.complete_state
    if state is not None and state.current_completion is not None:
        buf.apply_completion(state.current_completion)
    else:
        buf.validate_and_handle()


def _build_key_bindings() -> KeyBindings:
    """Key bindings for the REPL prompt.

    - **Ctrl-D** exits cleanly (raises EOFError, caught by the loop).
    - **Ctrl-L** clears the screen.
    - **Enter** submits the input (or accepts a highlighted completion).
    - **Alt+Enter** (Esc then Enter) inserts a newline, so a prompt can grow to
      multiple lines without submitting — the "adjustable" input box.

    Ctrl-C is left to prompt_toolkit's default (clear the current input line)
    rather than killing the shell — that matches the REPL contract the user
    expects from aider / ipython.
    """
    kb = KeyBindings()

    @kb.add(Keys.ControlD)
    def _ctrl_d(event: KeyPressEvent) -> None:
        # Raising EOFError mirrors readline's behavior; the shell catches it.
        raise EOFError

    @kb.add(Keys.ControlL)
    def _ctrl_l(event: KeyPressEvent) -> None:
        event.app.renderer.clear()

    @kb.add(Keys.Enter)
    def _submit(event: KeyPressEvent) -> None:
        _accept_completion_or_submit(event.current_buffer)

    @kb.add(Keys.Escape, Keys.Enter)
    def _newline(event: KeyPressEvent) -> None:
        event.current_buffer.insert_text("\n")

    return kb


_DOCKER_LABELS = {None: "auto", True: "on", False: "off"}


def _render_bottom_toolbar(state: SessionState) -> str:
    """The persistent status line under the prompt (the input box's bottom edge).

    Shows the live selections (recipe / model / docker mode) plus the submit and
    newline keys, so the context and controls are always visible while typing.
    """
    recipe = state.recipe.slug if state.recipe is not None else "no recipe"
    model = state.model or state.cfg.model
    docker = _DOCKER_LABELS[state.use_docker]
    context = f"recipe: {recipe}   model: {model}   docker: {docker}"
    keys = "Enter submit · Alt+Enter newline · /help · Ctrl-D exit"
    return f" {context}   │   {keys} "


def _print_turn_rule(console: Console, state: SessionState) -> None:
    """A dim divider above each prompt so every turn has its own visual space.

    Left-labels the rule with the active recipe when one is selected; otherwise
    a bare dim line. Printed outside ``patch_stdout`` so it scrolls with history.
    """
    if state.recipe is not None:
        console.rule(f"[{MUTED}]{state.recipe.slug}[/]", align="left", style=MUTED)
    else:
        console.rule(style=MUTED)


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
        "  [#FF6347]/context[/]    full context-tier breakdown ([dim]dropped + truncated[/])",
        "  [#FF4500]/generate[/]   confirm + run the pipeline ([dim]alias:[/] [bold]/go[/])",
        "  [#FF4500]/help[/]       list every slash command ([dim]aliases:[/] [bold]/h[/], [bold]/?[/])",
        "  [#DC143C]/exit[/]       leave the shell ([dim]Ctrl-D works too[/])",
        "",
        f"[dim]Deployments:[/] {deployments.label}",
        f"[dim]Blueprints: [/] {blueprints.label}",
        "",
        '[dim]Type any free text to refine the plan ([bold]"swap to sonnet, add postgres"[/]).[/]',
    ]
    render_banner(console, body_lines)


def _build_pipeline_inputs(state: SessionState, console: Console | None = None) -> PipelineInputs:
    """Translate the REPL's SessionState into the pipeline's frozen inputs.

    Mirrors what cmd_new does: assembles context (which the REPL
    intentionally re-does on each generate so refinements between runs
    pick up cleanly), infers topology, threads through all the override
    fields. Raises :class:`PipelineError` early if assemble blows the
    context budget so the shell can render the failure inline.

    ``console`` is optional so existing tests don't need to thread a
    Console through; when ``None``, a context-budget overflow re-raises
    as ``PipelineError`` without prompting the user (the prompt path is
    only reachable through real ``_run_generation_and_render`` calls).
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

    # Bind once so mypy carries the narrowed types into the inner closure.
    recipe = state.recipe
    language = state.language
    framework = state.framework
    cfg = state.cfg
    # Resolve the recipe's capabilities + REPL overrides into a single
    # stack and reuse it for both context assembly (so load_list ``when:
    # capabilities contains '<id>'`` predicates see overrides) and the
    # PipelineInputs that drive generation, manifest, and template copy.
    resolved_stack = resolve_stack_for_session(state)

    from agent_scaffold.catalog import load_catalog_for_config

    top_catalog = load_catalog_for_config(cfg)

    def _do_assemble(active_cfg: Config) -> Any:
        return assemble(
            recipe,
            language,
            framework,
            deployments_path,
            blueprints_path=state.blueprints.path,
            max_context_tokens=active_cfg.max_context_tokens,
            max_link_depth=active_cfg.max_link_depth,
            max_tokens_per_doc=active_cfg.max_tokens_per_doc,
            resolved_stack=resolved_stack,
            catalog=top_catalog,
        )

    try:
        ctx = _do_assemble(cfg)
    except ContextBudgetError as exc:
        bumped = prompt_to_raise_context_cap(console, exc) if console is not None else None
        if bumped is None:
            raise PipelineError(str(exc), phase="context") from exc
        new_cap, new_per_doc = bumped
        cfg = cfg.model_copy(
            update={"max_context_tokens": new_cap, "max_tokens_per_doc": new_per_doc}
        )
        try:
            ctx = _do_assemble(cfg)
        except ContextBudgetError as exc2:
            raise PipelineError(str(exc2), phase="context") from exc2

    topology, roles = resolve_topology(recipe, ctx.body)
    # Selection-vs-default precedence: explicit overrides on state win,
    # otherwise fall back to the Config values load_config produced.
    if state.model:
        cfg = cfg.model_copy(update={"model": state.model})
    if state.max_tokens is not None:
        cfg = cfg.model_copy(update={"max_tokens": state.max_tokens})
    if state.thinking_budget is not None:
        cfg = cfg.model_copy(update={"thinking_budget": state.thinking_budget})

    # Destination already has files: instead of dead-ending on abort, let the
    # user pick what to do (overwrite / skip / diff) — mirrors `agent-scaffold
    # new`. `/write-mode <mode>` set up front skips this prompt.
    write_mode = state.write_mode
    if (
        write_mode is WriteMode.abort
        and console is not None
        and state.dest is not None
        and state.dest.exists()
        and any(state.dest.iterdir())
    ):
        from agent_scaffold.cli_interactive import _select_write_mode

        write_mode = _select_write_mode()

    # Wire the diff-preview gate when the user has opted into WriteMode.diff
    # and we have a console to render through. The closure binds the console
    # so the writer can call it without knowing anything about the REPL.
    pre_write_confirm: Callable[[list[FileDiff]], bool] | None = None
    if write_mode is WriteMode.diff and console is not None:
        _console = console

        def _confirm(diffs: list[FileDiff]) -> bool:
            return _confirm_diff_preview(_console, diffs)

        pre_write_confirm = _confirm

    # Canonicalize the Python module name (hyphens → underscores) like
    # ``cmd_new`` does — otherwise the entry-point / module paths become
    # ``src/research-assistant/...``, an invalid Python package the model never
    # emits, and generation fails the required-files contract.
    from agent_scaffold.cli_interactive import _python_module_name

    module_name = _python_module_name(state.project_name, state.language)

    return PipelineInputs(
        cfg=cfg,
        recipe=state.recipe,
        language=state.language,
        framework=state.framework,
        project_name=module_name,
        raw_project_name=state.project_name,
        dest=state.dest,
        deployments=deployments_path,
        ctx=ctx,
        hints=_load_hints_for(state.language),
        topology=topology,
        roles=roles,
        write_mode=write_mode,
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
        # The agent persona: the Haiku-derived role, else the raw description.
        # Falls back to the recipe's default agent_role when the user skipped
        # the describe step.
        agent_role=state.agent_role or state.agent_description or state.recipe.agent_role,
        agent_title=state.agent_title,
        pre_write_confirm=pre_write_confirm,
        resolved_stack=resolved_stack,
    )


def _load_hints_for(language: str) -> dict[str, object]:
    """Load language hints YAML. Thin wrapper around the leaf module so the
    REPL doesn't have to translate ``UnknownLanguageError`` — at this call
    site the language has already been validated by ``cmd_language`` or
    the wizard, so a missing YAML would be a real bug, not user error."""
    return load_language_hints(language)


def _maybe_autosave_draft(state: SessionState) -> None:
    """Best-effort persist of the active selections to a named draft.

    Named by project name / recipe slug so re-saves overwrite the same draft
    (and the 3-draft cap evicts oldest, not this one). Silent + non-fatal — a
    write failure must never interrupt the REPL.
    """
    from agent_scaffold.repl import drafts

    name = drafts.default_draft_name(state)
    if name is None:
        return  # nothing meaningful selected yet
    try:
        drafts.save_draft(state.cfg.cache_dir, drafts.from_state(state, name))
    except OSError:
        pass


def _hint_saved_drafts(console: Console, cache_dir: Path) -> None:
    """One-line, non-blocking nudge on shell open if drafts exist."""
    from agent_scaffold.repl import drafts

    metas = drafts.list_drafts(cache_dir)
    if not metas:
        return
    names = ", ".join(m.name for m in metas)
    console.print(
        f"[dim]💾 {len(metas)} saved draft(s): {names} — /drafts to list, "
        "/draft load <name> to resume.[/]"
    )


def _run_config_single_var(state: SessionState, console: Console, var: str) -> None:
    """Fill one named env var via the secure form — ``/config <VAR>``.

    For connecting a *managed* service (an external ``REDIS_URL``, a custom
    ``LANGCHAIN_PROJECT``, …) over the in-sandbox default: the value is captured
    through the same secure browser form (credentials) / no-echo prompt (config
    knobs) and exported to the session env, so ``up`` forwards it. Overrides the
    sandbox default for this session.
    """
    from agent_scaffold.preflight import EnvRequirement, PreflightReport, fill_missing
    from agent_scaffold.repl.readiness import hint_for

    req = EnvRequirement(name=var, source="manual", required=False, satisfied=False)
    console.print(
        f"[dim]Setting [bold]{var}[/] — overrides any sandbox default for this session.[/]"
    )
    _print_credential_hints(console, [req])
    fill_missing(PreflightReport(requirements=[req]), console, secure=True, hint_for=hint_for)


def _run_config(state: SessionState, console: Console, *, var: str | None = None) -> None:
    """Interactive setup: fill the Anthropic key + missing stack env vars.

    Owned by the shell (not ``cmd_config``) because it does real getpass I/O —
    keeping the command pure/testable. Reuses ``preflight.fill_missing``: the
    key persists to the auth backend, other secrets export to the session env.
    ``var`` (from ``/config <VAR>``) fills just that one var instead.
    """
    from agent_scaffold.preflight import PreflightReport, fill_missing, render_env_panel
    from agent_scaffold.repl.readiness import config_requirements, hint_for, required_gaps

    if var is not None:
        _run_config_single_var(state, console, var)
        return

    reqs = config_requirements(state)
    console.print(render_env_panel(reqs))
    # Non-secret config knobs are already satisfied (defaults), so the only
    # unsatisfied items are real credentials — the required key + optional ones.
    missing_required = [r for r in reqs if r.required and not r.satisfied]
    missing_optional = [r for r in reqs if not r.required and not r.satisfied]

    if not missing_required and not missing_optional:
        console.print("[green]✓ Everything required is configured.[/] Run /generate.")
        return

    # Prompt for the required key (the only thing that blocks the sandbox).
    # secure=True routes the entry through the local browser paste form so the
    # key is never typed in the terminal (getpass fallback when headless).
    if missing_required:
        _print_credential_hints(console, missing_required)
        fill_missing(
            PreflightReport(requirements=list(missing_required)),
            console,
            secure=True,
            hint_for=hint_for,
        )

    # Optional cloud credentials never block — don't pester. List them and offer
    # to set them now, defaulting to "no" so the happy path is one keystroke.
    if missing_optional:
        from rich.prompt import Confirm

        names = ", ".join(r.name for r in missing_optional)
        console.print(f"[dim]Optional — connect later via /config:[/] {names}")
        if Confirm.ask(
            f"Set {len(missing_optional)} optional credential(s) now?",
            default=False,
            console=console,
        ):
            _print_credential_hints(console, missing_optional)
            fill_missing(
                PreflightReport(requirements=list(missing_optional)),
                console,
                secure=True,
                hint_for=hint_for,
            )

    if required_gaps(state):
        console.print(
            "[yellow]Still missing:[/] " + ", ".join(required_gaps(state)) + " — /config again."
        )
    else:
        console.print("[green]✓ Configured.[/] Run /generate.")


def _print_credential_hints(console: Console, reqs: list[Any]) -> None:
    """Show where to obtain each credential being prompted, when we know."""
    from agent_scaffold.repl.readiness import hint_for

    for req in reqs:
        hint = hint_for(req.name)
        if hint:
            console.print(f"  [dim]{req.name} → get one at:[/] {hint}")


def _run_generation_and_render(state: SessionState, console: Console) -> None:
    """Build inputs, run the pipeline, render the trailing summaries.

    Failures (any :class:`PipelineError`) print to the REPL but don't kill
    the loop — the user can fix the underlying issue and retry.
    """
    # Blocking config gate — the single chokepoint every generate path funnels
    # through (/generate, the confirm-then-generate path, and wizard→generate).
    # Refuse to spend tokens until the required credentials resolve; point the
    # user at /config. Selections are untouched, so nothing is lost.
    from agent_scaffold.repl.readiness import required_gaps

    gaps = required_gaps(state)
    if gaps:
        console.print("[yellow]Not configured yet:[/] " + ", ".join(gaps))
        console.print("Run [bold]/config[/] to set them, then /generate again.")
        return

    try:
        inputs = _build_pipeline_inputs(state, console)
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
        use_docker = _resolve_repl_docker(state, console)
        if use_docker:
            console.print(
                "[dim]Docker is available — bringing the stack up in containers "
                "([bold]/docker off[/] for local processes).[/]"
            )
        _autorun_after_repl_generate(state.dest, console, use_docker=use_docker)
    else:
        print_next_steps(
            state.dest, state.language, report.result.smoke_check, report.result.post_install
        )


def _autorun_after_repl_generate(
    project_dir: Path, console: Console, *, use_docker: bool = False
) -> None:
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
    resolved_stack = _resolve_capability_stack_silently(recipe, capabilities=manifest.capabilities)
    rc = _autorun_after_new(
        project_dir=project_dir,
        recipe=recipe,
        resolved_stack=resolved_stack,
        open_browser=True,
        use_docker=use_docker,
    )
    if rc != 0:
        console.print(f"[yellow]autorun finished with exit code {rc}[/]")


def _run_up(state: SessionState, console: Console) -> None:
    """REPL ``/up`` — restart the project's compose stack + show live URLs.

    Per the user's model ("up restarts the container in that project's compose
    stack"): first tear down this project's previous run (reclaiming its ports —
    only this project's containers, never unrelated host processes), then bring
    it up fresh on the canonical ports (8000 / 3000). Reuses ``_down_inline`` +
    ``_autorun_after_repl_generate``.
    """
    if state.dest is None:
        console.print("[yellow]No project — /generate first.[/]")
        return
    from agent_scaffold.cli import _down_inline, _find_docker_compose

    if _find_docker_compose(state.dest) is not None:
        _down_inline(state.dest, volumes=False, yes=True)  # reclaim this project's ports
    use_docker = _resolve_repl_docker(state, console)
    if use_docker:
        console.print(
            "[dim]Docker is available — bringing the stack up in containers "
            "([bold]/docker off[/] for local processes).[/]"
        )
    _autorun_after_repl_generate(state.dest, console, use_docker=use_docker)


def _run_down(state: SessionState, console: Console, *, volumes: bool) -> None:
    """REPL ``/down`` — stop the project's containers (``docker compose down``)."""
    if state.dest is None:
        console.print("[yellow]No project — nothing to tear down.[/]")
        return
    from agent_scaffold.cli import _down_inline

    # volumes=False never prompts; volumes=True confirms inside _down_inline.
    _down_inline(state.dest, volumes=volumes, yes=not volumes)


def _render(console: Console, result: CommandResult) -> None:
    for msg in result.messages:
        console.print(msg)


def _confirm_refinement(console: Console, patch: StatePatch) -> bool:
    """Prompt the user to confirm or skip a destructive refinement patch.

    Test seam — tests monkeypatch this to return True / False without a
    TTY, mirroring the ``_ask_select`` / ``_ask_text`` pattern used by the
    /new wizard. Production uses Rich's :class:`~rich.prompt.Confirm`,
    which integrates with the same Console used for rendering so the
    prompt lands inline with the preview Panel.
    """
    # Imported locally — Rich.prompt pulls in input handling that the
    # module-level test-import of shell.py doesn't otherwise need.
    from rich.prompt import Confirm

    # patch parameter is intentionally unused in the default implementation
    # (the preview Panel is already on screen by the time we ask). Kept in
    # the signature so monkeypatched stubs can inspect it.
    _ = patch
    return Confirm.ask("Apply this refinement?", default=False, console=console)


def _confirm_generation(console: Console) -> bool:
    """Prompt the user to confirm /generate after dirty-since-plan refinements.

    Test seam — same pattern as :func:`_confirm_refinement`. Default
    ``False`` matches the destructive-action convention: the user must
    type ``y`` to ship, ``Enter`` cancels.
    """
    from rich.prompt import Confirm

    return Confirm.ask(
        "Stack has changed since last /plan. Proceed with generation?",
        default=False,
        console=console,
    )


def _confirm_generate_now(console: Console) -> bool:
    """Post-wizard 'Generate now?' — default **YES** (the happy path is to ship).

    Distinct from :func:`_confirm_generation` (the dirty-stack guard, default no):
    here the user just finished picking everything and the config gate has passed,
    so a single Enter generates. Test seam — monkeypatched in wizard tests.
    """
    from rich.prompt import Confirm

    return Confirm.ask("Generate now?", default=True, console=console)


def _resolve_repl_docker(state: SessionState, console: Console) -> bool:
    """Resolve the tri-state ``use_docker`` to a concrete run mode.

    ``None`` (default) = auto: run in containers when Docker is usable. ``True``
    forces it; ``False`` forces local. Mirrors the terminal's
    :func:`agent_scaffold.cli._resolve_use_docker` probe-and-fallback so the
    one-click run prefers the full container stack but degrades to local
    processes when Docker isn't usable (warning only on an explicit ``/docker on``).
    """
    if state.use_docker is False:
        return False
    from agent_scaffold.steps.docker_up import docker_available

    ok, reason = docker_available()
    if not ok:
        if state.use_docker is True:
            console.print(f"[yellow]Docker not available:[/] {reason} — running locally.")
        return False
    return True


def _confirm_diff_preview(console: Console, diffs: list[FileDiff]) -> bool:
    """Render the full set of file diffs and ask once whether to apply.

    Used as the ``pre_write_confirm`` callback when ``state.write_mode``
    is ``WriteMode.diff``. Test seam — same pattern as the other
    ``_confirm_*`` helpers.

    Returns ``True`` when the user accepts (writes proceed; modifications
    become overwrites), ``False`` when they decline (the writer raises
    ``DiffPreviewCancelled`` and the pipeline turns it into a clean cancel).
    """
    from rich.prompt import Confirm

    if not any(d.status == "modified" for d in diffs):
        # Nothing the user has to review. Auto-accept to keep the diff
        # confirm out of the way when the dest only has new files (or is
        # identical to the planned result).
        console.print("[dim]Diff preview: no modified files — proceeding.[/]")
        return True

    for renderable in render_file_diffs(diffs):
        console.print(renderable)
    return Confirm.ask("Apply these changes?", default=False, console=console)


def _resolve_pending_patch(
    console: Console, state: SessionState, patch: StatePatch
) -> SessionState:
    """Confirm a destructive refinement patch and apply or skip it.

    Returns the new state if the user confirms, or ``state`` unchanged if
    they decline. Either way, prints a brief outcome so the loop history
    shows what happened.
    """
    if not _confirm_refinement(console, patch):
        console.print("[yellow]Skipped.[/] State unchanged.")
        return state
    new_state = apply_patch(state, patch)
    console.print("[green]✓[/] applied refinement")
    console.print(render_patch_delta(state, new_state))
    return new_state


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


_TIER_GROUPS: tuple[tuple[str, str], ...] = (
    ("basic", "Basic — single-agent, recipe-default stack"),
    ("mid", "Mid — multi-step / multi-agent, a few capabilities"),
    ("complex", "Complex — production stack: queue + frontend + host"),
)


def _select_recipe(console: Console, recipes: dict[str, Recipe]) -> Any:
    """Arrow-key recipe pick, grouped by complexity tier.

    Returns ``Recipe``, ``_STOP_SENTINEL``, or ``None`` on Ctrl-C. Tier
    derives from :func:`infer_complexity`; each row shows the agent_pattern
    hint when present so users see "what shape of agent" before "what name".
    """
    if not recipes:
        console.print("[yellow]No recipes available; cancelling wizard.[/]")
        return _STOP_SENTINEL
    import questionary

    grouped: dict[str, list[Recipe]] = {tier: [] for tier, _ in _TIER_GROUPS}
    for r in sorted(recipes.values(), key=lambda r: r.slug):
        grouped[infer_complexity(r)].append(r)
    longest_slug = max(len(r.slug) for r in recipes.values())

    choices: list[Any] = []
    for tier, header in _TIER_GROUPS:
        bucket = grouped.get(tier, [])
        if not bucket:
            continue
        choices.append(questionary.Separator(f"── {header} ──"))
        for r in bucket:
            pattern_hint = f"  · {r.agent_pattern}" if r.agent_pattern else ""
            choices.append(
                questionary.Choice(
                    f"{r.slug:<{longest_slug}}  [{r.status}]  {r.title}{pattern_hint}",
                    value=r,
                )
            )
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


def _select_framework(language: str, deployments_root: Path | None) -> Any:
    """Frameworks come from agent-deployments doc frontmatter (post-SR1b).

    The list is filtered by ``language``: each ``docs/frameworks/<name>.md``
    declares its target language in YAML frontmatter and the picker only
    surfaces matches. Falls back to ``["none"]`` when the deployments tree
    predates the frontmatter — typically because an offline / stale
    snapshot is in use.
    """
    import questionary

    from agent_scaffold.framework_versions import available_frameworks_for_language

    frameworks: list[str] = []
    if deployments_root is not None:
        frameworks = available_frameworks_for_language(deployments_root, language)
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


_OBS_CHOICES: tuple[tuple[str, str], ...] = (
    ("langsmith", "langsmith — best for LangChain/LangGraph; SaaS-only"),
    ("langfuse", "langfuse  — MIT, self-hostable, cheaper at volume"),
    ("none", "none      — skip observability for this project"),
)


def _select_observability() -> Any:
    """Observability backend picker. Mandatory; ``none`` is an explicit choice."""
    import questionary

    choices: list[Any] = [questionary.Choice(label, value=value) for value, label in _OBS_CHOICES]
    choices.append(_separator())
    choices.append(_pause_choice())
    return _ask_select("Observability backend?", choices)


def _format_observability_display(state: SessionState) -> str:
    """Render the user's current observability pick for the keep/change gate."""
    if "obs.langfuse" in state.add_capabilities:
        return "langfuse"
    if "obs.langsmith" in state.add_capabilities:
        return "langsmith"
    if {"obs.langsmith", "obs.langfuse"} <= state.remove_capabilities:
        return "none"
    return ""


def _apply_observability_choice(state: SessionState, value: str) -> SessionState:
    """Translate a {langsmith|langfuse|none} pick into the add/remove pair.

    Mirrors ``cmd_observability`` in repl/commands.py so the wizard and the
    slash command produce identical patches.
    """
    all_obs = ["obs.langsmith", "obs.langfuse"]
    if value == "none":
        patch = StatePatch(remove_capabilities=list(all_obs))
    else:
        target = f"obs.{value}"
        patch = StatePatch(
            add_capabilities=[target],
            remove_capabilities=[c for c in all_obs if c != target],
        )
    return apply_patch(state, patch)


# ---------------------------------------------------------------------------
# Stack mode + customize layer walk
# ---------------------------------------------------------------------------


_STACK_MODE_CHOICES: tuple[tuple[str, str], ...] = (
    ("quick", "quick     — use the recipe's defaults"),
    ("customize", "customize — pick memory, observability, eval, and interface yourself"),
)


def _select_stack_mode() -> Any:
    """Pick how much of the stack the wizard should walk."""
    import questionary

    choices: list[Any] = [
        questionary.Choice(label, value=value) for value, label in _STACK_MODE_CHOICES
    ]
    choices.append(_separator())
    choices.append(_pause_choice())
    return _ask_select("Stack mode?", choices)


def _format_stack_mode_display(state: SessionState) -> str:
    return state.stack_mode


def _is_basic_recipe(state: SessionState) -> bool:
    """True when the picked recipe is basic-tier — Stack mode auto-defaults
    to ``quick`` in that case (no prompt)."""
    return state.recipe is not None and infer_complexity(state.recipe) == "basic"


def _apply_stack_mode_quick(state: SessionState, _value: Any = None) -> SessionState:
    return apply_patch(state, StatePatch(stack_mode="quick"))


# Layer groupings the wizard surfaces. Memory merges three storage kinds so
# the user sees "memory layer" as one decision; obs / eval / interface each
# map to a single kind. Order matches the natural reading flow.
_LAYER_GROUPS: tuple[tuple[str, str, tuple[CapabilityKind, ...]], ...] = (
    ("memory", "Memory", ("relational", "cache", "vector_db")),
    ("observability", "Observability", ("obs",)),
    ("eval", "Eval", ("eval",)),
    ("interface", "Interface", ("frontend",)),
)


def _effective_capability_ids(state: SessionState) -> set[str]:
    """Recipe-declared caps ∪ session adds, minus session removes."""
    recipe_ids = set(state.recipe.capabilities) if state.recipe else set()
    return (recipe_ids | set(state.add_capabilities)) - set(state.remove_capabilities)


def _select_layer(
    state: SessionState,
    kinds: tuple[CapabilityKind, ...],
    layer_label: str,
) -> Any:
    """Multi-select picker for one layer.

    Loads the live capability catalog filtered by ``kinds``; checkboxes
    default-checked when the cap is currently effective on ``state``.
    Returns the picked id list, ``_STOP_SENTINEL``, or ``None``.
    """
    import questionary

    deployments_path = state.deployments.path
    if deployments_path is None:
        return []
    catalog = load_capabilities(deployments_path)
    in_layer = sorted((c for c in catalog.values() if c.kind in kinds), key=lambda c: c.id)
    if not in_layer:
        return []
    effective = _effective_capability_ids(state)
    longest = max(len(cap.id) for cap in in_layer)
    choices = [
        questionary.Choice(
            f"{cap.id:<{longest}}  {cap.docs}",
            value=cap.id,
            checked=(cap.id in effective),
        )
        for cap in in_layer
    ]
    picked = questionary.checkbox(
        f"{layer_label} — pick the categories you want",
        choices=choices,
        qmark="›",
    ).ask()
    if picked is None:
        return None
    return list(picked)


def _apply_layer_choice(
    state: SessionState,
    picked: list[str],
    *,
    kinds: tuple[CapabilityKind, ...],
) -> SessionState:
    """Diff the user's pick against the effective set in this layer.

    Anything dropped lands in ``remove_capabilities`` (apply_patch will also
    pull it from ``add_capabilities`` if it came from a prior step).
    Anything added lands in ``add_capabilities``.
    """
    if picked is None:
        return state
    effective = _effective_capability_ids(state)
    in_layer = {c for c in effective if c.split(".", 1)[0] in kinds}
    picked_set = set(picked)
    to_add = sorted(picked_set - in_layer)
    to_remove = sorted(in_layer - picked_set)
    if not to_add and not to_remove:
        return state
    return apply_patch(
        state,
        StatePatch(add_capabilities=to_add or None, remove_capabilities=to_remove or None),
    )


def _make_layer_step(key: str, label: str, kinds: tuple[CapabilityKind, ...]) -> _WizardStep:
    """Build a ``_WizardStep`` for one layer's customize-mode picker."""

    def display(state: SessionState) -> str:
        effective = _effective_capability_ids(state)
        in_layer = sorted(c for c in effective if c.split(".", 1)[0] in kinds)
        return ", ".join(in_layer) if in_layer else "(none)"

    def picker(_console: Console, state: SessionState, _handler: CommandHandler) -> Any:
        return _select_layer(state, kinds, label)

    def apply(state: SessionState, value: Any) -> SessionState:
        return _apply_layer_choice(state, value, kinds=kinds)

    return _WizardStep(
        label=f"Layer · {label}",
        field=f"_layer_{key}",  # virtual; apply handles persistence
        description=f"Pick the {label.lower()} categories the agent should use.",
        examples=tuple(f"{k}.<name>" for k in kinds),
        display=display,
        picker=picker,
        format_set=lambda v: ", ".join(v) if v else "(none)",
        apply=apply,
        enabled_when=lambda s: s.stack_mode == "customize",
    )


def _print_step_header(console: Console, step: _WizardStep) -> None:
    """Render a Rich panel above each wizard prompt with label + description + examples.

    Centralizes the "what am I picking, and why?" framing so users see the
    trade-off before the questionary list. Examples render as dim hints to
    suggest valid shapes without crowding the prompt.
    """
    from rich.panel import Panel

    body_lines = [f"[bold {ACCENT}]{step.label}[/]"]
    if step.description:
        body_lines.append(f"[{MUTED}]{step.description}[/]")
    if step.examples:
        body_lines.append("")
        for ex in step.examples:
            body_lines.append(f"  [{MUTED}]• {ex}[/]")
    console.print(
        Panel(
            "\n".join(body_lines),
            border_style=PANEL_BORDER_STYLE,
            expand=False,
            padding=(0, 1),
        )
    )


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
    plan_result = handler.dispatch("/plan", state)
    _render(console, plan_result)
    if plan_result.new_state is not None:
        state = plan_result.new_state

    # Auto-flow: right after selections, offer to generate — no separate
    # /generate needed. The config gate breaks the flow when required credentials
    # are missing (directs to /config); when it's clear, a single confirm
    # (default yes) ships it. Either way the user lands in the refine loop below.
    from agent_scaffold.repl.readiness import required_gaps

    gaps = required_gaps(state)
    if not gaps:
        if _confirm_generate_now(console):
            return state, "generate"
    else:
        console.print(
            f"[yellow]Before generating, configure:[/] {', '.join(gaps)} — "
            "run [bold]/config[/], then [bold]/generate[/]."
        )

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
            # Dispatch through cmd_generate so the dirty-since-plan gate applies
            # in the wizard too. It renders any plan / cost panels itself when
            # dirty; otherwise it returns next_action="generate".
            go_result = handler.dispatch(raw, state)
            _render(console, go_result)
            if go_result.new_state is not None:
                state = go_result.new_state
            if go_result.next_action == "generate":
                return state, "generate"
            if go_result.next_action == "confirm_generate":
                if _confirm_generation(console):
                    state = replace(state, dirty_since_plan=False)
                    return state, "generate"
                console.print(
                    "[yellow]Generation cancelled.[/] Use /plan to inspect, then /generate again."
                )
            continue
        if raw.startswith("/"):
            # Allow any other slash command inside refine loop so the user
            # can /config, /model, /effort, /cost, /reset etc. without leaving.
            result = handler.dispatch(raw, state)
            _render(console, result)
            if result.new_state is not None:
                state = result.new_state
            if result.next_action == "exit":
                return state, "quit"
            if result.next_action == "config":
                _run_config(state, console, var=result.config_var)
            continue
        # Free text → refinement interpreter; re-render the plan if it landed.
        result = handler.dispatch(raw, state)
        _render(console, result)
        if result.new_state is not None:
            state = result.new_state
            plan_result = handler.dispatch("/plan", state)
            _render(console, plan_result)
            if plan_result.new_state is not None:
                state = plan_result.new_state


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

    description: str = ""
    """One-line subheader rendered below the step label — explains what the
    choice affects so users understand the trade-off, not just the question."""

    examples: tuple[str, ...] = ()
    """Optional dim hints below the description (e.g. valid framework names)."""

    apply: Callable[[SessionState, Any], SessionState] | None = None
    """Optional custom apply function. Defaults to a straight StatePatch on
    ``field``; observability uses this to translate a {langsmith|langfuse|none}
    pick into the add/remove_capabilities pair from Phase 2."""

    enabled_when: Callable[[SessionState], bool] | None = None
    """When set and it returns ``False``, the step is silently skipped —
    used for layer-walk steps that only run under ``stack_mode=customize``."""

    skip_when: Callable[[SessionState], bool] | None = None
    """When set and it returns ``True``, the step auto-applies its default
    via ``apply`` (no prompt) and emits a dim hint instead of the panel.
    Used by Stack mode to force ``quick`` on basic-tier recipes."""

    skip_message: str = ""
    """Dim hint printed when ``skip_when`` triggers — explains the auto-pick."""


def _name_default(state: SessionState) -> str:
    """Default project name: previous pick > recipe slug > empty."""
    return state.project_name or (state.recipe.slug if state.recipe else "")


_WIZARD_STEPS: tuple[_WizardStep, ...] = (
    _WizardStep(
        label="Recipe",
        field="recipe",
        description=(
            "Which agent shape are we building? Each recipe ships a vetted "
            "stack + prompt + eval harness."
        ),
        examples=("docs-rag-qa", "customer-support-triage", "restaurant-rebooking"),
        display=lambda s: s.recipe.slug if s.recipe else "",
        picker=lambda c, s, h: _select_recipe(c, h.recipes),
        format_set=lambda v: str(v.slug),
    ),
    _WizardStep(
        label="Language",
        field="language",
        description=(
            "Python or TypeScript track. Drives the framework list, package "
            "manager, and emitted file layout."
        ),
        examples=("python", "typescript"),
        display=lambda s: s.language or "",
        picker=lambda c, s, h: _select_language(),
        format_set=str,
    ),
    _WizardStep(
        label="Framework",
        field="framework",
        description=(
            "Agent framework that ties the prompt + tools + graph together. "
            "Some recipes only validate against one — others are framework-agnostic."
        ),
        examples=("langgraph", "pydantic_ai", "vercel_ai_sdk", "none"),
        display=lambda s: s.framework or "",
        picker=lambda c, s, h: _select_framework(
            s.language or "python",
            s.deployments.path,
        ),
        format_set=str,
    ),
    _WizardStep(
        label="Stack mode",
        field="stack_mode",
        description=(
            "Use the recipe's defaults, or walk each layer and pick categories "
            "yourself? Basic-tier recipes default to quick automatically."
        ),
        examples=(
            "quick     — recipe defaults; one extra step then on to generation",
            "customize — pick memory / observability / eval / interface",
        ),
        display=_format_stack_mode_display,
        picker=lambda c, s, h: _select_stack_mode(),
        format_set=str,
        skip_when=_is_basic_recipe,
        skip_message="Stack mode: quick (basic recipe)",
        apply=_apply_stack_mode_quick,
    ),
    _WizardStep(
        label="Observability",
        field="_observability_choice",  # virtual field — `apply` handles persistence
        description=(
            "Where should traces, prompts, and eval runs land? You can swap "
            "this later with /observability."
        ),
        examples=(
            "langsmith — best for LangChain/LangGraph; SaaS-only",
            "langfuse  — MIT, self-hostable, cheaper at volume",
            "none      — skip observability for this project",
        ),
        display=_format_observability_display,
        picker=lambda c, s, h: _select_observability(),
        format_set=str,
        apply=_apply_observability_choice,
        # In customize mode the obs layer is part of the layer walk below;
        # the standalone step would double-prompt.
        enabled_when=lambda s: s.stack_mode != "customize",
    ),
    _make_layer_step("memory", "Memory", ("relational", "cache", "vector_db")),
    _make_layer_step("observability", "Observability", ("obs",)),
    _make_layer_step("eval", "Eval", ("eval",)),
    _make_layer_step("interface", "Interface", ("frontend",)),
    _WizardStep(
        label="Name",
        field="project_name",
        description=(
            "Project + package name. Lowercase letters, digits, underscores, "
            "and hyphens; hyphens get folded to underscores for Python modules."
        ),
        examples=("demo", "rebooking-agent", "doc_qa"),
        display=lambda s: s.project_name or "",
        picker=lambda c, s, h: _input_name(default=_name_default(s)),
        format_set=str,
    ),
    _WizardStep(
        label="Destination",
        field="dest",
        description=(
            "Directory the project will be written to. Defaults to "
            "$CWD/<name>; safe to point at an empty dir or a fresh path."
        ),
        examples=(),
        display=lambda s: str(s.dest) if s.dest else "",
        picker=lambda c, s, h: _input_dest(s.project_name or "demo", s.dest),
        format_set=str,
    ),
)


def _run_describe_step(
    console: Console, handler: CommandHandler, state: SessionState
) -> SessionState:
    """First step: free-text "describe your agent" → Haiku suggestion + seeds.

    Captures a sentence or two, asks Haiku to (a) recommend a recipe and (b)
    derive the backend system prompt (``agent_role``) + chat title
    (``agent_title``). The suggested recipe is pre-selected so the Recipe step
    offers it as the keep-default; role/title are stored to seed generation and
    the frontend. Empty input or a Haiku failure is non-fatal — the wizard
    proceeds with no suggestion. Skipped on resume (``agent_description`` set).
    """
    from agent_scaffold.repl.refine import RefinementError, interpret_description

    console.print()
    console.rule(f"[{ACCENT}]Describe your agent[/]", align="left", style=MUTED)
    console.print(
        f"[{MUTED}]In a sentence or two — what should this agent do, and how should it "
        "behave? We'll suggest a recipe and seed the system prompt + chat title.[/]"
    )
    raw = _ask_text("Describe the agent (Enter to skip)", default="")
    if not raw or not raw.strip():
        console.print(f"[{MUTED}]No description — continuing to the recipe picker.[/]")
        # Mark the step done (empty, not None) so resuming via /new doesn't re-ask.
        return apply_patch(state, StatePatch(agent_description=""))

    description = raw.strip()
    try:
        result = interpret_description(description, handler.recipes.values(), state.cfg)
    except RefinementError as exc:
        # Keep the raw description (it still seeds generation) but skip suggestion.
        console.print(f"[yellow]Couldn't interpret that:[/] {exc} — pick a recipe below.")
        return apply_patch(state, StatePatch(agent_description=description))

    suggested = (
        handler.recipes.get(result.suggested_recipe_slug)
        if result.suggested_recipe_slug
        else None
    )
    state = apply_patch(
        state,
        StatePatch(
            agent_description=description,
            agent_role=result.agent_role,
            agent_title=result.agent_title,
            recipe=suggested,
        ),
    )
    if result.agent_title:
        console.print(f"[green]✓[/] agent: [bold]{result.agent_title}[/]")
    if suggested is not None:
        console.print(
            f"[{MUTED}]Suggested recipe from your description: "
            f"[bold]{suggested.slug}[/] — keep or change it next.[/]"
        )
    return state


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

    # Free-text intent capture runs once, before the picker steps. Skipped on
    # resume so re-running /new doesn't re-ask what was already described.
    if state.agent_description is None:
        state = _run_describe_step(console, handler, state)

    for step in _WIZARD_STEPS:
        # Conditional steps (the customize-mode layer walk) silently skip when
        # their predicate says they're irrelevant for the current selections.
        if step.enabled_when is not None and not step.enabled_when(state):
            continue
        # Auto-skip steps (Stack mode on basic recipes) apply their default
        # and emit a dim hint instead of opening the picker.
        if step.skip_when is not None and step.skip_when(state):
            if step.apply is not None:
                state = step.apply(state, None)
            if step.skip_message:
                console.print(f"[{MUTED}]{step.skip_message}[/]")
            continue

        _print_step_header(console, step)

        def picker(step: _WizardStep = step, state: SessionState = state) -> Any:  # noqa: B023
            """Bind the loop variables so each iteration's picker sees its own
            step + state snapshot — picker is called immediately within this
            iteration, so binding at definition time is equivalent to binding
            at call time but sidesteps Python's late-binding semantics."""
            return step.picker(console, state, handler)

        # Virtual fields (whose `field` doesn't exist on SessionState) skip the
        # "is it already set?" gate and always run the picker; `apply` decides
        # how the picked value lands in state.
        current_value: Any
        if hasattr(state, step.field):
            current_value = getattr(state, step.field)
            # ``stack_mode`` has a non-None default ("quick"); honor a prior
            # explicit pick but don't treat the default as "already set".
            if step.field == "stack_mode" and current_value == "quick":
                current_value = None
        else:
            current_value = None
        value, action = _resolve_field(
            step.label,
            current_value,
            step.display(state),
            picker,
        )
        if action in ("stop", "cancel"):
            return _wizard_paused(state, console)
        if action == "set":
            if step.apply is not None:
                state = step.apply(state, value)
            else:
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

    # The bottom toolbar reads live state through a mutable holder updated each
    # loop turn (the callback is fixed at construction, but state is replaced).
    toolbar_ctx: dict[str, SessionState] = {"state": state}

    def _toolbar() -> str:
        return _render_bottom_toolbar(toolbar_ctx["state"])

    session: PromptSession[str] = prompt_factory(
        message=_PROMPT,
        history=FileHistory(str(history_file)),
        completer=ScaffoldCompleter(
            command_names=handler.commands,
            recipe_slugs=[r.slug for r in recipes],
        ),
        complete_while_typing=True,
        key_bindings=_build_key_bindings(),
        multiline=True,
        bottom_toolbar=_toolbar,
    )

    _print_banner(console, deployments, blueprints)
    _hint_saved_drafts(console, cfg.cache_dir)

    while True:
        # Refresh the toolbar's view of state and draw the turn divider so each
        # prompt sits in its own space (rule above, toolbar below).
        toolbar_ctx["state"] = state
        _print_turn_rule(console, state)
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
        if result.pending_patch is not None:
            state = _resolve_pending_patch(console, state, result.pending_patch)
        # Auto-save the active selections so an accidental exit loses nothing.
        # Covers every slash/refine/draft-load path; the wizard saves again below.
        _maybe_autosave_draft(state)
        if result.next_action == "exit":
            return 0
        if result.next_action == "config":
            _run_config(state, console, var=result.config_var)
            continue
        if result.next_action == "up":
            _run_up(state, console)
            continue
        if result.next_action in ("down", "down_volumes"):
            _run_down(state, console, volumes=result.next_action == "down_volumes")
            continue
        if result.next_action == "wizard":
            state, terminal = _run_new_wizard(session, console, handler, state)
            _maybe_autosave_draft(state)  # the wizard mutates state then continues
            if terminal == "generate":
                _run_generation_and_render(state, console)
            continue
        if result.next_action == "confirm_generate":
            if _confirm_generation(console):
                state = replace(state, dirty_since_plan=False)
                _run_generation_and_render(state, console)
            else:
                console.print(
                    "[yellow]Generation cancelled.[/] Use /plan to inspect, then /generate again."
                )
            continue
        if result.next_action == "generate":
            _run_generation_and_render(state, console)

    # Unreachable in practice; keeps mypy happy if the loop is ever bounded.
    return 0
