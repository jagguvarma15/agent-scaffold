"""Slash-command dispatcher for the ``agent-scaffold scaffold`` REPL.

Each user-typed line passes through :meth:`CommandHandler.dispatch`, which
classifies it as one of:

1. Slash command (``/recipe demo``) → routes to ``cmd_<name>``.
2. Bare recipe slug (``demo``) → shortcut for ``/recipe demo``.
3. Free text → handed off to the LLM refinement interpreter (PR5 will
   wire that in; this PR returns an "I don't understand" hint).

Convention mirrors aider's ``cmd_*`` naming so commands are discoverable
via introspection — :meth:`CommandHandler.commands` lists every public
slash. Each handler is a small pure function: takes ``(args, state)`` and
returns a :class:`CommandResult` (messages + optional new state + next
action). The shell loop owns I/O and pipeline kickoff.

Adding a new slash command is a 3-line job: write ``cmd_<name>``, give it
a docstring (becomes the ``/help`` line), done.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from rich.console import RenderableType
from rich.table import Table
from rich.text import Text

from agent_scaffold.context import ContextBudgetError, assemble
from agent_scaffold.costs import estimate_preflight
from agent_scaffold.discovery import Recipe
from agent_scaffold.effort import EFFORT_PRESETS
from agent_scaffold.language_hints import available_languages
from agent_scaffold.plan import GenerationPlan
from agent_scaffold.repl.refine import RefinementError, interpret_refinement
from agent_scaffold.repl.render import (
    render_cost,
    render_patch_delta,
    render_state_summary,
)
from agent_scaffold.repl.session import SessionState, StatePatch, apply_patch
from agent_scaffold.topology import resolve as resolve_topology

NextAction = Literal["continue", "generate", "exit", "wizard"]


@dataclass(frozen=True)
class CommandResult:
    """What a command produced.

    ``messages`` are rendered by the shell in order. ``new_state`` of None
    means "state unchanged" — the shell carries the existing state forward
    without forcing every cmd_* to thread it through. ``next_action`` lets
    a handler signal the shell to start generation or exit.
    """

    messages: list[RenderableType] = field(default_factory=list)
    new_state: SessionState | None = None
    next_action: NextAction = "continue"


class CommandError(Exception):
    """Raised inside a cmd_* method for a user-facing validation error.

    Caught by :meth:`CommandHandler.dispatch` and turned into a message
    rather than a stack trace. Use for argument-shape errors and other
    "user typed something we can't honor" situations.
    """


class CommandHandler:
    """Dispatches user input to slash-command methods or free-text refinement.

    Constructed once per REPL session with the discovered ``recipes`` so
    the bare-slug shortcut (``demo`` → ``/recipe demo``) is cheap and
    ``/recipe`` (no args) can list available slugs.
    """

    # Methods discovered by name prefix. Order doesn't matter for dispatch,
    # but the order here drives /help.
    _COMMAND_PREFIX = "cmd_"

    def __init__(self, recipes: list[Recipe]) -> None:
        self.recipes: dict[str, Recipe] = {r.slug: r for r in recipes}
        self._commands: dict[str, Callable[[list[str], SessionState], CommandResult]] = {
            name[len(self._COMMAND_PREFIX) :]: getattr(self, name)
            for name in dir(self)
            if name.startswith(self._COMMAND_PREFIX) and callable(getattr(self, name))
        }
        # Map "/exit" → "exit" so user can type either; both /quit and /q
        # resolve to cmd_exit via aliases below.
        # /go is the original verb; /generate reads more naturally as the
        # "final confirm" step at the end of /new. Both route to cmd_go so
        # there's a single source of truth for is_ready validation.
        self._aliases: dict[str, str] = {
            "quit": "exit",
            "q": "exit",
            "h": "help",
            "?": "help",
            "generate": "go",
            "gen": "go",
        }

    # ----- public surface -------------------------------------------------

    @property
    def commands(self) -> list[str]:
        """Slash-command names in declaration order. Used by /help."""
        return list(self._commands.keys())

    def dispatch(self, line: str, state: SessionState) -> CommandResult:
        """Classify ``line`` and route to the right handler.

        Empty input is a no-op (state unchanged, no message). Cancellation
        / EOF are handled in the shell loop, not here.
        """
        stripped = line.strip()
        if not stripped:
            return CommandResult()

        if stripped.startswith("/"):
            return self._dispatch_slash(stripped[1:], state)

        # Bare recipe slug shortcut: "demo" → "/recipe demo". Only treat as
        # a slug if there's no whitespace and it actually matches; otherwise
        # fall through to the free-text path so a recipe slug fragment in a
        # free-text refinement ("I want a demo agent") doesn't trigger it.
        if " " not in stripped and stripped in self.recipes:
            return self.cmd_recipe([stripped], state)

        return self._dispatch_free_text(stripped, state)

    # ----- dispatch internals --------------------------------------------

    def _dispatch_slash(self, body: str, state: SessionState) -> CommandResult:
        parts = body.split()
        name = self._aliases.get(parts[0], parts[0])
        args = parts[1:]
        handler = self._commands.get(name)
        if handler is None:
            return CommandResult(messages=[self._unknown_command_message(name)])
        try:
            return handler(args, state)
        except CommandError as exc:
            return CommandResult(messages=[Text.from_markup(f"[red]✗[/] {exc}")])

    def _dispatch_free_text(self, text: str, state: SessionState) -> CommandResult:
        """Hand free text to the Haiku-backed refinement interpreter.

        On any failure (network, parse, schema) we surface a yellow warning
        and leave state untouched — the user can retry or drop to slash
        commands. On success we apply the patch and render the delta so
        they can see exactly what changed.
        """
        try:
            patch = interpret_refinement(state, text, state.cfg)
        except RefinementError as exc:
            return CommandResult(
                messages=[
                    Text.from_markup(
                        f"[yellow]Couldn't interpret that refinement:[/] {exc}\n"
                        "Try a slash command ([bold]/help[/]) or rephrase."
                    )
                ]
            )
        if patch.is_empty():
            return CommandResult(
                messages=[Text.from_markup("[dim]No changes from that refinement.[/]")]
            )
        new_state = apply_patch(state, patch)
        return CommandResult(
            messages=[
                Text.from_markup("[green]✓[/] applied refinement"),
                render_patch_delta(state, new_state),
            ],
            new_state=new_state,
        )

    def _unknown_command_message(self, name: str) -> Text:
        candidates = list(self._commands) + list(self._aliases)
        close = difflib.get_close_matches(name, candidates, n=1, cutoff=0.6)
        if close:
            return Text.from_markup(
                f"[red]Unknown command[/] [bold]/{name}[/]. " f"Did you mean [bold]/{close[0]}[/]?"
            )
        return Text.from_markup(f"[red]Unknown command[/] [bold]/{name}[/]. Try [bold]/help[/].")

    # ----- slash commands ------------------------------------------------

    def cmd_help(self, args: list[str], state: SessionState) -> CommandResult:  # noqa: ARG002
        """List available commands and their first-line docstrings."""
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column()
        for name in sorted(self._commands):
            doc = (self._commands[name].__doc__ or "").strip().split("\n", 1)[0]
            table.add_row(f"/{name}", doc)
        return CommandResult(messages=[table])

    def cmd_recipe(self, args: list[str], state: SessionState) -> CommandResult:
        """Select the recipe (e.g. /recipe restaurant-rebooking). Bare /recipe lists slugs."""
        if not args:
            return self._list_recipes()
        slug = args[0]
        recipe = self.recipes.get(slug)
        if recipe is None:
            close = difflib.get_close_matches(slug, list(self.recipes), n=1, cutoff=0.5)
            hint = f" Did you mean [bold]{close[0]}[/]?" if close else ""
            raise CommandError(f"unknown recipe [bold]{slug}[/].{hint}")
        return _state_change(state, StatePatch(recipe=recipe), f"recipe → {recipe.slug}")

    def cmd_language(self, args: list[str], state: SessionState) -> CommandResult:
        """Pick target language (python or typescript)."""
        valid = available_languages()
        if not args:
            raise CommandError(f"usage: /language {'|'.join(valid)}")
        lang = args[0].lower()
        if lang not in valid:
            raise CommandError(f"language must be one of {', '.join(valid)}")
        return _state_change(state, StatePatch(language=lang), f"language → {lang}")

    def cmd_framework(self, args: list[str], state: SessionState) -> CommandResult:
        """Pick framework (e.g. /framework langgraph). Free-form; validated downstream."""
        if not args:
            raise CommandError("usage: /framework <name> (e.g. langgraph, pydantic_ai)")
        framework = args[0]
        return _state_change(state, StatePatch(framework=framework), f"framework → {framework}")

    def cmd_name(self, args: list[str], state: SessionState) -> CommandResult:
        """Set project name (auto-derives /dest if /dest hasn't been set yet)."""
        if not args:
            raise CommandError("usage: /name <project-name>")
        name = args[0]
        patch_kwargs: dict[str, Any] = {"project_name": name}
        # Auto-derive dest into cwd/<name> only if the user hasn't set one.
        if state.dest is None:
            patch_kwargs["dest"] = (Path.cwd() / name).resolve()
        return _state_change(state, StatePatch(**patch_kwargs), f"name → {name}")

    def cmd_dest(self, args: list[str], state: SessionState) -> CommandResult:
        """Override the destination directory."""
        if not args:
            raise CommandError("usage: /dest <path>")
        dest = Path(args[0]).expanduser().resolve()
        return _state_change(state, StatePatch(dest=dest), f"dest → {dest}")

    def cmd_model(self, args: list[str], state: SessionState) -> CommandResult:
        """Override the model id (e.g. /model claude-sonnet-4-6)."""
        if not args:
            raise CommandError("usage: /model <model-id>")
        model = args[0]
        return _state_change(state, StatePatch(model=model), f"model → {model}")

    def cmd_effort(self, args: list[str], state: SessionState) -> CommandResult:
        """Apply an effort preset (low|medium|high). Bundles model + max_tokens + thinking + strict."""
        if not args:
            raise CommandError("usage: /effort low|medium|high")
        level = args[0].lower()
        preset = EFFORT_PRESETS.get(level)
        if preset is None:
            raise CommandError(f"effort must be one of {', '.join(EFFORT_PRESETS)}, got {level!r}")
        patch = StatePatch(
            effort=level,
            model=preset.model,
            max_tokens=preset.max_tokens,
            thinking_budget=preset.thinking,
            strict=preset.strict,
        )
        return _state_change(state, patch, f"effort → {level}")

    def cmd_reset(self, args: list[str], state: SessionState) -> CommandResult:  # noqa: ARG002
        """Drop the current draft. Keeps cfg + resolved sources, clears everything else."""
        fresh = SessionState(
            cfg=state.cfg, deployments=state.deployments, blueprints=state.blueprints
        )
        return CommandResult(
            messages=[Text.from_markup("[green]✓[/] session reset")],
            new_state=fresh,
        )

    def cmd_new(self, args: list[str], state: SessionState) -> CommandResult:  # noqa: ARG002
        """Start a guided wizard: recipe → language → framework → name → dest → plan.

        Each step is an arrow-key picker (↑/↓ + Enter) with a
        ``pause wizard`` option that preserves your selections. Re-run
        ``/new`` to resume — the wizard skips fields that already have
        values and offers a keep / change gate for them. After all
        selections land, you can refine with free text or ``/generate``
        to run the pipeline.
        """
        return CommandResult(
            messages=[Text.from_markup("[bold #FF6347]→ Entering new-project wizard…[/]")],
            new_state=state,
            next_action="wizard",
        )

    def cmd_plan(self, args: list[str], state: SessionState) -> CommandResult:  # noqa: ARG002
        """Re-render the generation plan with the current selections."""
        ok, missing = state.is_ready()
        if not ok:
            return CommandResult(
                messages=[
                    Text.from_markup(
                        "[yellow]Plan needs:[/] "
                        + ", ".join(missing)
                        + " — use the matching slash commands."
                    ),
                    render_state_summary(state),
                ]
            )
        plan = _build_plan(state)
        if isinstance(plan, str):
            return CommandResult(messages=[Text.from_markup(f"[red]✗[/] {plan}")])
        return CommandResult(messages=[plan.render()])

    def cmd_cost(self, args: list[str], state: SessionState) -> CommandResult:  # noqa: ARG002
        """Show the pre-flight cost estimate alone (cheaper than full /plan)."""
        model = state.model
        if model is None:
            return CommandResult(
                messages=[
                    Text.from_markup(
                        "[dim]Set a model first ([bold]/model[/] or [bold]/effort[/]).[/]"
                    )
                ]
            )
        # If recipe + language are set, use the real context token count;
        # otherwise show a rough estimate based on the recipe alone.
        input_tokens = _estimate_input_tokens(state)
        max_tokens = state.max_tokens or 32_000
        preflight = estimate_preflight(
            model,
            input_tokens=input_tokens,
            output_range=(min(8_000, max_tokens), max_tokens),
        )
        return CommandResult(messages=[render_cost(preflight)])

    def cmd_go(self, args: list[str], state: SessionState) -> CommandResult:  # noqa: ARG002
        """Confirm + run the generation pipeline."""
        ok, missing = state.is_ready()
        if not ok:
            return CommandResult(
                messages=[
                    Text.from_markup(
                        "[yellow]Can't generate yet — missing:[/] " + ", ".join(missing)
                    )
                ]
            )
        return CommandResult(
            messages=[Text.from_markup("[bold green]→ Generating…[/]")],
            new_state=state,
            next_action="generate",
        )

    def cmd_exit(self, args: list[str], state: SessionState) -> CommandResult:  # noqa: ARG002
        """Leave the REPL (alias: /quit, /q)."""
        return CommandResult(
            messages=[Text.from_markup("[dim]bye.[/]")],
            new_state=state,
            next_action="exit",
        )

    # ----- helpers --------------------------------------------------------

    def _list_recipes(self) -> CommandResult:
        if not self.recipes:
            return CommandResult(
                messages=[Text.from_markup("[dim]No recipes found in deployments.[/]")]
            )
        table = Table.grid(padding=(0, 2))
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(style="dim", no_wrap=True)
        table.add_column()
        for slug, recipe in sorted(self.recipes.items()):
            table.add_row(slug, recipe.status, recipe.title)
        return CommandResult(messages=[table])


# ---------------------------------------------------------------------------
# Module-level helpers (kept out of the class so they're easy to unit-test)
# ---------------------------------------------------------------------------


def _state_change(state: SessionState, patch: StatePatch, summary: str) -> CommandResult:
    """Apply ``patch`` and return a result containing a ✓ line + the delta."""
    new_state = apply_patch(state, patch)
    return CommandResult(
        messages=[
            Text.from_markup(f"[green]✓[/] {summary}"),
            render_patch_delta(state, new_state),
        ],
        new_state=new_state,
    )


def _build_plan(state: SessionState) -> GenerationPlan | str:
    """Assemble context + build a GenerationPlan from the current state.

    Returns the plan on success, or an error string for the caller to
    render. Kept as a free function so cmd_plan stays a thin wrapper.
    """
    assert state.recipe is not None  # is_ready() guarantees this
    assert state.language is not None
    assert state.framework is not None
    assert state.project_name is not None
    assert state.dest is not None
    deployments_path = state.deployments.path
    if deployments_path is None:
        return "deployments source unavailable; rerun the shell with --deployments-path"
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
        return f"context budget error: {exc}"

    topology, roles = resolve_topology(state.recipe, ctx.body)

    model = state.model or state.cfg.model
    max_tokens = state.max_tokens or state.cfg.max_tokens
    preflight = estimate_preflight(
        model,
        input_tokens=ctx.token_estimate,
        output_range=(min(8_000, max_tokens), max_tokens),
    )

    return GenerationPlan(
        recipe_slug=state.recipe.slug,
        recipe_status=state.recipe.status,
        language=state.language,
        framework=state.framework,
        project_name=state.project_name,
        dest=state.dest,
        topology=topology,
        roles=roles,
        model=model,
        max_tokens=max_tokens,
        thinking_budget=state.thinking_budget or state.cfg.thinking_budget,
        required_files=state.recipe.required_files,
        context_summary=ctx.summary,
        write_mode=state.write_mode,
        warnings=[],
        strict=state.strict,
        service_readiness=[],
        preflight_cost=preflight,
    )


# Rough constants for the /cost shortcut when full context isn't assembled.
_DEFAULT_INPUT_TOKENS_GUESS = 10_000
_CHARS_PER_TOKEN = 4


def _estimate_input_tokens(state: SessionState) -> int:
    """Best-effort input-token count for /cost without running full assemble.

    Used only when /cost runs before the user has picked recipe + language.
    The full /plan computes the real number.
    """
    if state.recipe is None or state.language is None or state.framework is None:
        return _DEFAULT_INPUT_TOKENS_GUESS
    deployments_path = state.deployments.path
    if deployments_path is None:
        return _DEFAULT_INPUT_TOKENS_GUESS
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
    except ContextBudgetError:
        return _DEFAULT_INPUT_TOKENS_GUESS
    return ctx.token_estimate
