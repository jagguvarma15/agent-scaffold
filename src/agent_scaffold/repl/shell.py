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
from typing import TYPE_CHECKING, Any, Literal

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
from agent_scaffold.manifest import ManifestNotFoundError, manifest_path, read_manifest
from agent_scaffold.pipeline import (
    PipelineError,
    PipelineInputs,
    print_next_steps,
    run_generation,
)
from agent_scaffold.progress import RichProgressDisplay
from agent_scaffold.repl._capabilities import resolve_stack_for_session
from agent_scaffold.repl._fuzzy import completions
from agent_scaffold.repl.commands import CommandError, CommandHandler, CommandResult
from agent_scaffold.repl.render import render_patch_delta
from agent_scaffold.repl.session import SessionState, StatePatch, apply_patch
from agent_scaffold.sources import ResolvedSource
from agent_scaffold.tiers import active_tier
from agent_scaffold.topology import resolve as resolve_topology
from agent_scaffold.writer import WriteMode

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
            # Fuzzy ranking: exact-prefix hits first, then close typo matches
            # (e.g. "/observ" -> "/observability", "/genrate" -> "/generate").
            for name in completions(prefix, self._commands):
                yield Completion(f"/{name}", start_position=-len(word), display=f"/{name}")
            return
        # Bare slug completion only at start-of-line.
        for slug in completions(word, self._slugs):
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
        "  [#FF4500]/connect[/]    wire a stack option after generation ([dim]docker or cloud[/])",
        "  [#FF4500]/stack[/]      browse every stack option ([dim]then /layer to pick[/])",
        "  [#FF4500]/open[/]       attach an existing generated project ([dim]then /up, /connect[/])",
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
        resolved_stack=resolved_stack,
        # Preset intent behind the expanded capability ids — recorded in the
        # manifest answers so regenerate/update can see it.
        rag_preset=state.rag_preset,
        hosting_overrides=tuple(sorted(state.hosting_overrides.items())),
        # The effective tier (explicit /tier pick, else the recipe's declared
        # tier) — same precedence as cmd_new's --tier, recorded in the
        # manifest and spec artifact.
        tier=active_tier(state.tier, state.recipe.tier),
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
    write failure must never interrupt the REPL. Skipped once the dest holds a
    generated project — /open <dest> resumes it, not a draft (otherwise every
    post-generate turn would resurrect the draft that generation retired).
    """
    from agent_scaffold.repl import drafts

    name = drafts.default_draft_name(state)
    if name is None:
        return  # nothing meaningful selected yet
    if state.dest is not None and manifest_path(state.dest).is_file():
        return
    try:
        drafts.save_draft(state.cfg.cache_dir, drafts.from_state(state, name))
    except OSError:
        pass


def _retire_drafts_for_dest(state: SessionState, console: Console) -> None:
    """Delete drafts whose dest is the just-generated project.

    Generation consumed their selections; /open <dest> is the resume path
    now. Best-effort like autosave — an OSError never interrupts the REPL.
    """
    from agent_scaffold.repl import drafts

    if state.dest is None:
        return
    dest = state.dest.expanduser().resolve()
    retired: list[str] = []
    try:
        for meta in drafts.list_drafts(state.cfg.cache_dir):
            if meta.dest and Path(meta.dest).expanduser().resolve() == dest:
                if drafts.delete_draft(state.cfg.cache_dir, meta.name):
                    retired.append(meta.name)
    except OSError:
        return
    if retired:
        console.print(
            f"[dim]draft {', '.join(retired)} retired — /open {state.dest} resumes this project[/]"
        )


def _hint_saved_drafts(console: Console, cache_dir: Path) -> None:
    """One-line, non-blocking nudge on shell open if drafts exist.

    Drafts whose dest already holds a generated project are skipped — the
    startup /open hint (or `scaffold <dir>`) is the resume path for those.
    """
    from agent_scaffold.repl import drafts

    metas = [
        m
        for m in drafts.list_drafts(cache_dir)
        if not (m.dest and manifest_path(Path(m.dest)).is_file())
    ]
    if not metas:
        return
    names = ", ".join(m.name for m in metas)
    console.print(
        f"[dim]{len(metas)} saved draft(s): {names} — /draft list to list, "
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
        console.print(
            f"[dim]Optional — wire later with /connect (or /config to just store a value):[/] "
            f"{names}"
        )
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
    console.print()
    if report.result is None or state.dest is None or state.language is None:
        return

    _retire_drafts_for_dest(state, console)

    if state.autorun:
        use_docker = _resolve_repl_docker(state, console)
        if use_docker:
            console.print()
            console.print(
                "[dim]Docker is available — bringing the stack up in containers "
                "([bold]/docker off[/] for local processes).[/]"
            )
        _autorun_after_repl_generate(
            state.dest, console, use_docker=use_docker, teardown_stale=True
        )
    else:
        print_next_steps(
            state.dest, state.language, report.result.smoke_check, report.result.post_install
        )


def _autorun_after_repl_generate(
    project_dir: Path,
    console: Console,
    *,
    use_docker: bool = False,
    teardown_stale: bool = False,
) -> None:
    """REPL mirror of ``cmd_new``'s autorun chain.

    The REPL never raises ``typer.Exit`` on autorun failure — it prints the
    exit-code-as-warning and returns control to the prompt so the user can
    retry, inspect, or just keep going.

    ``teardown_stale=True`` only on the post-generate path: a regenerated
    destination may have containers built from the old files still running.
    ``/up`` passes ``False`` — it already tears down before calling here.
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
        teardown_stale=teardown_stale,
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
        console.print("[yellow]No project — /generate first (or /open <path>).[/]")
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


def _run_connect(state: SessionState, console: Console, *, choice: str) -> None:
    """REPL ``/connect`` — run the connect flow inline (the REPL owns the TTY).

    Bare ``/connect`` lists the project's stack options with their delivery
    mode; ``/connect <option>`` runs the same capture / validate / store /
    wire / verify flow as ``agent-scaffold connect <option>``.
    """
    if state.dest is None:
        console.print("[yellow]No project — /generate first (or /open <path>).[/]")
        return
    from agent_scaffold.integrations import find_docker_compose, run_connect
    from agent_scaffold.manifest import ManifestNotFoundError, read_manifest
    from agent_scaffold.orchestrator import reset_step_state
    from agent_scaffold.stack_options import (
        MODE_CLOUD,
        load_stack_options,
        option_by_id,
    )

    dest = Path(state.dest).expanduser().resolve()
    try:
        manifest = read_manifest(dest)
    except ManifestNotFoundError as exc:
        console.print(f"[red]Error:[/] {exc}")
        return
    options = load_stack_options(manifest.capabilities or [])
    if not options:
        console.print("[yellow]No connectable stack options in this project's manifest.[/]")
        return
    if not choice:
        for option in options:
            mode = "cloud hosted" if option.mode == MODE_CLOUD else "docker"
            console.print(f"  {option.id:<12} {mode:<12} {option.title}")
        console.print("[dim]Connect one with /connect <option>.[/]")
        return
    selected = option_by_id(options, choice)
    if selected is None:
        known = ", ".join(o.id for o in options)
        console.print(f"[red]Unknown option {choice!r}.[/] Available: {known}")
        return
    run_connect(
        dest,
        manifest,
        selected,
        find_docker_compose(dest),
        yes=False,
        reset_step_state=reset_step_state,
    )


def _render(console: Console, result: CommandResult) -> None:
    # One blank line between messages: consecutive panels otherwise stack
    # edge-to-edge and read as a single wall. No leading/trailing padding —
    # the per-turn rule already frames the block.
    for i, msg in enumerate(result.messages):
        if i:
            console.print()
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


def _select_framework(
    language: str,
    deployments_root: Path | None,
    recipe: Any = None,
    console: Console | None = None,
) -> Any:
    """Frameworks come from agent-deployments doc frontmatter (post-SR1b).

    The list is filtered by ``language``: each ``docs/frameworks/<name>.md``
    declares its target language in YAML frontmatter and the picker only
    surfaces matches. When ``recipe`` is set, the list is further filtered
    to the frameworks the recipe's declared dependencies support — the
    generated code follows the recipe's blueprints, so an undeclared pick
    would only record a framework the emitted project does not use. Falls
    back to ``["none"]`` when the deployments tree predates the frontmatter.
    """
    import questionary

    from agent_scaffold.framework_versions import (
        available_frameworks_for_language,
        frameworks_supported_by_recipe,
    )

    frameworks: list[str] = []
    if deployments_root is not None:
        frameworks = available_frameworks_for_language(deployments_root, language)
        if recipe is not None and frameworks:
            supported = frameworks_supported_by_recipe(
                deployments_root, recipe.recipe_dependencies, language
            )
            if supported is not None:
                frameworks = [f for f in frameworks if f in supported]
                if console is not None:
                    console.print(
                        f"[dim]{recipe.slug} generates {language} code against: "
                        f"{', '.join(supported)} — other frameworks are hidden "
                        "so the project matches its manifest.[/]"
                    )
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
    ("langsmith", "langsmith     — best for LangChain/LangGraph; cloud-only"),
    ("langfuse", "langfuse      — MIT; run in docker or point at the cloud"),
    ("grafana-stack", "grafana-stack — metrics + traces dashboards; docker"),
    ("none", "none          — skip observability for this project"),
)

_ALL_OBS_CAPS: tuple[str, ...] = ("obs.langsmith", "obs.langfuse", "obs.grafana-stack")

_HOSTING_LABELS: dict[str, str] = {
    "cloud": "cloud  — managed service; wired by credentials, no container",
    "docker": "docker — self-hosted via the generated compose stack",
}


def _hosting_modes_for(state: SessionState, cap_id: str) -> list[str]:
    """Hosting modes the catalog allows for ``cap_id``.

    Prefers the authored ``hosting:`` metadata; falls back to inferring from
    docker-service presence. Empty when the catalog is unavailable or the id
    is unknown — the caller then skips the hosting question entirely.
    """
    try:
        from agent_scaffold.catalog import load_catalog_for_config

        catalog = load_catalog_for_config(state.cfg)
    except Exception:  # noqa: BLE001 — offline/parse trouble degrades to "no question"
        return []
    entry = next((c for c in catalog.capabilities if c.id == cap_id), None)
    if entry is None:
        return []
    if entry.hosting:
        return [m for m in entry.hosting if m in ("cloud", "docker")]
    return ["docker"] if entry.docker_service else ["cloud"]


def _select_observability(state: SessionState) -> Any:
    """Observability backend picker, then a hosting pick when the backend
    supports more than one mode. Returns ``"none"``, ``(backend, mode)``
    (mode ``None`` when there was nothing to choose), or a sentinel."""
    import questionary

    choices: list[Any] = [questionary.Choice(label, value=value) for value, label in _OBS_CHOICES]
    choices.append(_separator())
    choices.append(_pause_choice())
    backend = _ask_select("Observability backend?", choices)
    if backend is None or backend is _STOP_SENTINEL or backend == "none":
        return backend
    modes = _hosting_modes_for(state, f"obs.{backend}")
    if len(modes) <= 1:
        return (backend, modes[0] if modes else None)
    mode_choices: list[Any] = [
        questionary.Choice(_HOSTING_LABELS.get(m, m), value=m) for m in modes
    ]
    mode_choices.append(_separator())
    mode_choices.append(_pause_choice())
    mode = _ask_select(f"Host {backend} where?", mode_choices)
    if mode is None or mode is _STOP_SENTINEL:
        return mode
    return (backend, mode)


def _format_observability_display(state: SessionState) -> str:
    """Render the user's current observability pick for the keep/change gate."""
    for cap in _ALL_OBS_CAPS:
        if cap in state.add_capabilities:
            name = cap.removeprefix("obs.")
            mode = state.hosting_overrides.get(cap)
            return f"{name} ({mode})" if mode else name
    if set(_ALL_OBS_CAPS) <= state.remove_capabilities:
        return "none"
    return ""


def _format_observability_value(value: Any) -> str:
    if isinstance(value, tuple):
        backend, mode = value
        return f"{backend} ({mode})" if mode else str(backend)
    return str(value)


def _apply_observability_choice(state: SessionState, value: Any) -> SessionState:
    """Translate a backend (+ hosting) pick into the add/remove/hosting patch.

    Mirrors ``cmd_observability`` in repl/commands.py so the wizard and the
    slash command produce identical patches. ``value`` is ``"none"`` or a
    ``(backend, mode)`` tuple; a bare backend string is accepted for the
    slash-command mirror.
    """
    if value == "none":
        return apply_patch(state, StatePatch(remove_capabilities=list(_ALL_OBS_CAPS)))
    backend, mode = value if isinstance(value, tuple) else (value, None)
    target = f"obs.{backend}"
    patch = StatePatch(
        add_capabilities=[target],
        remove_capabilities=[c for c in _ALL_OBS_CAPS if c != target],
        hosting_overrides={target: mode} if mode else None,
    )
    return apply_patch(state, patch)


# ---------------------------------------------------------------------------
# Optional-features menu + customize layer walk
# ---------------------------------------------------------------------------


_FEATURE_CHOICES: tuple[tuple[str, str], ...] = (
    ("rag", "RAG           — retrieval over your documents (simple or advanced)"),
    ("observability", "Observability — traces, prompts, and eval runs"),
    ("guardrails", "Guardrails    — input/output safety classification"),
    ("layers", "More layers   — walk every stack layer and pick each one"),
)


def _default_features_for_recipe(recipe: Recipe | None) -> set[str]:
    """Menu entries pre-checked from the recipe's declared stack."""
    if recipe is None:
        return set()
    declared = list(recipe.capabilities)
    defaults: set[str] = set()
    if any(c.startswith("vector_db.") for c in declared):
        defaults.add("rag")
    if any(c.startswith("obs.") for c in declared):
        defaults.add("observability")
    if any(c.startswith("guardrail.") for c in declared):
        defaults.add("guardrails")
    return defaults


def _ask_checkbox(prompt: str, choices: list[Any]) -> Any:
    """Ask a questionary checkbox; ``None`` on Ctrl-C.

    Test seam — tests monkeypatch this alongside ``_ask_select`` /
    ``_ask_text`` so the wizard runs headlessly.
    """
    import questionary

    return questionary.checkbox(prompt, choices=choices, qmark="›").ask()


def _select_optional_features(state: SessionState) -> Any:
    """The mandatory/optional gate: one multi-select over the feature areas."""
    import questionary

    checked = set(state.optional_features) or _default_features_for_recipe(state.recipe)
    choices = [
        questionary.Choice(label, value=key, checked=key in checked)
        for key, label in _FEATURE_CHOICES
    ]
    return _ask_checkbox(
        "Optional features (space toggles, Enter continues; nothing checked = recipe defaults)",
        choices,
    )


_RAG_CHOICES: tuple[tuple[str, str], ...] = (
    ("simple", "simple  — vector store on the existing database + embeddings; single-stage top-k"),
    ("complex", "complex — hybrid search + embeddings + late reranking"),
    ("custom", "custom  — pick the vector and memory capabilities yourself"),
)


def _select_rag_preset() -> Any:
    import questionary

    choices: list[Any] = [questionary.Choice(label, value=value) for value, label in _RAG_CHOICES]
    choices.append(_separator())
    choices.append(_pause_choice())
    return _ask_select("RAG preset?", choices)


def _rag_bundle_presets(state: SessionState) -> Any:
    """Catalog-published bundles; embedded defaults when the catalog is out."""
    from agent_scaffold.bundles import load_bundles

    try:
        from agent_scaffold.catalog import load_catalog_for_config

        catalog = load_catalog_for_config(state.cfg)
    except Exception:  # noqa: BLE001 — embedded defaults keep the preset working offline
        catalog = None
    return load_bundles(catalog)


def _apply_rag_choice(state: SessionState, value: str) -> SessionState:
    """Expand a RAG preset to capability ids; ``custom`` opens the layer walk."""
    from agent_scaffold.bundles import RAG_PRESET_BUNDLES, expand_bundle

    if value == "custom":
        features = list(state.optional_features)
        if "layers" not in features:
            features.append("layers")
        return apply_patch(state, StatePatch(optional_features=features, rag_preset="custom"))
    ids = expand_bundle(RAG_PRESET_BUNDLES[value], _rag_bundle_presets(state))
    return apply_patch(state, StatePatch(add_capabilities=ids or None, rag_preset=value))


_TIER_NONE = "__no_tier__"
"""Picker sentinel for the "(no tier)" choice — never lands on state."""

_TIER_RECIPE_DEFAULT = "__recipe_default__"
"""Picker sentinel for "clear the explicit pick, follow the recipe tier".

Behaviorally the same clear as ``_TIER_NONE``, but shown when the recipe
declares a tier — there is no "no tier" outcome in that case
(``active_tier`` falls back to the recipe), so the old "(no tier)" label
promised an opt-out the pipeline never honored and the confirmation
misreported what generation would do."""


def _select_tier(state: SessionState) -> Any:
    """Single select over the tier ladder, effective-default first.

    The first entry is what Enter keeps: the recipe-declared tier when one
    exists (labeled as the recipe default), else "(no tier)". The remaining
    ladder follows in T0→T4 order so the progression reads top-down.
    """
    import questionary

    from agent_scaffold.repl._capabilities import session_tier_presets
    from agent_scaffold.tiers import KNOWN_TIERS

    presets = session_tier_presets(state)
    ordered = [n for n in KNOWN_TIERS if n in presets]
    ordered += sorted(set(presets) - set(ordered))
    recipe_tier = state.recipe.tier if state.recipe else None

    def _label(name: str, *, marker: str = "") -> str:
        preset = presets[name]
        desc = f": {preset.description}" if preset.description else ""
        return f"{name} — {preset.title}{desc}{marker}"

    choices: list[Any] = []
    default_name = recipe_tier if recipe_tier in presets else None
    if default_name is not None:
        choices.append(
            questionary.Choice(
                _label(default_name, marker="  (recipe default)"), value=default_name
            )
        )
    else:
        choices.append(questionary.Choice("(no tier) — recipe defaults only", value=_TIER_NONE))
    for name in ordered:
        if name == default_name:
            continue
        choices.append(questionary.Choice(_label(name), value=name))
    if default_name is not None:
        choices.append(
            questionary.Choice(
                f"(clear explicit pick) — follow recipe default {default_name}",
                value=_TIER_RECIPE_DEFAULT,
            )
        )
    choices.append(_separator())
    choices.append(_pause_choice())
    return _ask_select("Capability tier?", choices)


def _apply_tier_choice(state: SessionState, value: str | None) -> SessionState:
    """Land the tier pick; both clear sentinels drop an explicit pick."""
    if value is None:
        # skip_when auto-apply: the recipe declares no tier — leave state as is.
        return state
    if value in (_TIER_NONE, _TIER_RECIPE_DEFAULT):
        # Empty string is the patch-level clear sentinel (None means "don't
        # touch"); only needed when an explicit tier was previously set.
        return apply_patch(state, StatePatch(tier="")) if state.tier else state
    return apply_patch(state, StatePatch(tier=value))


def _format_tier_set(value: Any) -> str:
    if value == _TIER_RECIPE_DEFAULT:
        # The recipe's declared tier still applies — say so instead of the
        # old "none (recipe defaults)", which read as an opt-out.
        return "recipe default"
    if value in (None, _TIER_NONE):
        return "none (recipe defaults)"
    return str(value)


# Layer groupings the wizard surfaces. Memory merges the storage kinds so
# the user sees "memory layer" as one decision; infrastructure covers the
# stateful backbones; tools covers the agent-tier API integrations. Order
# matches the natural reading flow. Hosting and auth are deliberately not
# wizard steps (late/rare decisions) but stay pickable via /layer.
_LAYER_GROUPS: tuple[tuple[str, str, tuple[CapabilityKind, ...]], ...] = (
    ("memory", "Memory", ("relational", "cache", "vector_db", "memory_store")),
    ("infrastructure", "Infrastructure", ("queue", "durable")),
    ("tools", "Tools", ("live_data", "mcp", "embedding", "rerank", "sandbox", "guardrail")),
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


def _make_layer_step(
    key: str,
    label: str,
    kinds: tuple[CapabilityKind, ...],
    enabled_when: Callable[[SessionState], bool] | None = None,
) -> _WizardStep:
    """Build a ``_WizardStep`` for one layer's multi-select picker."""

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
        phase="feature",
        description=f"Pick the {label.lower()} categories the agent should use.",
        examples=tuple(f"{k}.<name>" for k in kinds),
        display=display,
        picker=picker,
        format_set=lambda v: ", ".join(v) if v else "(none)",
        apply=apply,
        # The layer walk opens via the features menu ("More layers") or the
        # standalone /customize command — either signal enables it.
        enabled_when=enabled_when
        or (lambda s: "layers" in s.optional_features or s.stack_mode == "customize"),
    )


def _print_step_header(console: Console, step: _WizardStep) -> None:
    """Render a Rich panel above each wizard prompt with label + description + examples.

    Centralizes the "what am I picking, and why?" framing so users see the
    trade-off before the questionary list. Examples render as dim hints to
    suggest valid shapes without crowding the prompt.
    """
    from rich.panel import Panel

    header = f"[bold {ACCENT}]{step.label}[/]"
    if step.phase == "feature":
        header += f"  [{MUTED}](optional)[/]"
    body_lines = [header]
    if step.description:
        body_lines.append(f"[{MUTED}]{step.description}[/]")
    if step.examples:
        body_lines.append("")
        for ex in step.examples:
            body_lines.append(f"  [{MUTED}]• {ex}[/]")
    # Breathing room above each step panel — otherwise it sits directly on
    # the previous step's confirmation line.
    console.print()
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

    phase: Literal["mandatory", "feature"] = "mandatory"
    """Mandatory steps always run; feature steps are gated by the
    optional-features menu (their ``enabled_when`` reads the picked set).
    Display metadata beyond that — the walk logic is unchanged."""

    enabled_when: Callable[[SessionState], bool] | None = None
    """When set and it returns ``False``, the step is silently skipped —
    used for the feature steps gated by the optional-features menu (and the
    layer walk's ``stack_mode == "customize"`` compatibility path)."""

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
            recipe=s.recipe,
            console=c,
        ),
        format_set=str,
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
        picker=lambda c, s, h: _select_observability(s),
        format_set=_format_observability_value,
        apply=_apply_observability_choice,
        phase="feature",
        # Gated by the features menu. In customize mode the obs layer is part
        # of the layer walk below; the standalone step would double-prompt.
        enabled_when=lambda s: (
            "observability" in s.optional_features and s.stack_mode != "customize"
        ),
    ),
    _make_layer_step("memory", "Memory", ("relational", "cache", "vector_db", "memory_store")),
    _make_layer_step("infrastructure", "Infrastructure", ("queue", "durable")),
    _make_layer_step(
        "tools", "Tools", ("live_data", "mcp", "embedding", "rerank", "sandbox", "guardrail")
    ),
    _make_layer_step("observability", "Observability", ("obs",)),
    _make_layer_step("eval", "Eval", ("eval",)),
    _make_layer_step("interface", "Interface", ("frontend",)),
)


# Assembled walk order: the mandatory selections first (recipe, language,
# framework, name, destination), then the optional-features menu, then only
# the feature steps the menu enabled. _WIZARD_STEPS above holds the pool;
# this tuple is what _run_new_wizard iterates.
_MANDATORY_STEPS: tuple[_WizardStep, ...] = (
    *_WIZARD_STEPS[:3],
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
    _WizardStep(
        label="Tier",
        field="tier",
        description=(
            "Capability tier (T0 chat … T4 enterprise) — each tier seeds a "
            "curated capability set the plan panel lists. Enter keeps the "
            "recipe's declared tier."
        ),
        examples=(),
        display=lambda s: s.tier or "",
        picker=lambda c, s, h: _select_tier(s),
        format_set=_format_tier_set,
        apply=_apply_tier_choice,
        # Prompt only when the recipe declares a tier (the ladder is then a
        # real decision with a sensible default). Undeclared recipes keep
        # today's flow untouched — /tier or free text can still set one.
        skip_when=lambda s: (s.recipe is None or s.recipe.tier is None) and s.tier is None,
        skip_message="No tier declared by the recipe — set one anytime with /tier T0..T4.",
    ),
    _WizardStep(
        label="Optional features",
        field="optional_features",
        description=(
            "Pick the feature areas to configure; everything unpicked stays "
            "on the recipe's defaults. Enter with nothing checked goes "
            "straight to the plan."
        ),
        examples=tuple(label for _key, label in _FEATURE_CHOICES),
        display=lambda s: ", ".join(s.optional_features) if s.optional_features else "",
        picker=lambda c, s, h: _select_optional_features(s),
        format_set=lambda v: ", ".join(v) if v else "none",
    ),
)

_FEATURE_STEPS: tuple[_WizardStep, ...] = (
    _WizardStep(
        label="RAG preset",
        field="rag_preset",
        phase="feature",
        description=(
            "How should the agent retrieve documents? Presets expand to "
            "catalog capability bundles; custom opens the layer walk."
        ),
        examples=tuple(label for _key, label in _RAG_CHOICES),
        display=lambda s: s.rag_preset or "",
        picker=lambda c, s, h: _select_rag_preset(),
        format_set=str,
        apply=_apply_rag_choice,
        enabled_when=lambda s: "rag" in s.optional_features,
    ),
    _WIZARD_STEPS[3],  # Observability (gated on the menu)
    _make_layer_step(
        "guardrails",
        "Guardrails",
        ("guardrail",),
        enabled_when=lambda s: "guardrails" in s.optional_features,
    ),
    *_WIZARD_STEPS[4:],  # the layer walk (menu "layers" or /customize)
)

_WALK_STEPS: tuple[_WizardStep, ...] = (*_MANDATORY_STEPS, *_FEATURE_STEPS)


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
        handler.recipes.get(result.suggested_recipe_slug) if result.suggested_recipe_slug else None
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

    for step in _WALK_STEPS:
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
            # Fields with non-None defaults: honor a prior explicit pick but
            # don't treat the default as "already set".
            if step.field == "stack_mode" and current_value == "quick":
                current_value = None
            if step.field == "optional_features" and current_value == []:
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
    open_dir: Path | None = None,
) -> int:
    """Run the interactive REPL loop until the user exits.

    ``prompt_factory`` lets tests inject a stub PromptSession that yields
    a scripted sequence of lines instead of reading from a TTY. Returns
    the exit code (always 0 in normal operation; non-zero only if recipe
    discovery blows up at session open).

    ``open_dir`` attaches the session to an existing generated project at
    startup (same as typing ``/open <dir>``); a directory without a manifest
    prints a warning and falls back to a fresh session.

    The loop honors:

    - ``next_action="exit"`` from any cmd_* → break out cleanly
    - EOFError (Ctrl-D) → break out cleanly
    - KeyboardInterrupt (Ctrl-C) → clear input, stay in the loop
    - ``next_action="generate"`` → call ``run_generation``, then back to prompt
    """
    console = console or Console()
    if deployments.path is None:
        reason = deployments.fallback_reason or "could not fetch agent-deployments from GitHub"
        console.print(f"[red]Cannot start shell:[/] deployments source unavailable ({reason}).")
        console.print(
            "[dim]Retry when online, or point at a local checkout with "
            "--deployments-path /path/to/agent-deployments.[/]"
        )
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

    # Startup attach: `scaffold <dir>` lands directly on an existing generated
    # project. The direct method call (single-element args) keeps paths with
    # spaces intact — dispatch() would whitespace-split them.
    if open_dir is not None:
        try:
            result = handler.cmd_open([str(open_dir)], state)
        except CommandError as exc:
            console.print(f"[yellow]Could not attach {open_dir}:[/] {exc}")
        else:
            _render(console, result)
            if result.new_state is not None:
                state = result.new_state
    elif manifest_path(Path.cwd()).is_file():
        console.print(
            "[dim]Generated project detected in this directory — /open . to attach it.[/]"
        )

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
        if result.next_action == "connect":
            _run_connect(state, console, choice=result.connect_option or "")
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
