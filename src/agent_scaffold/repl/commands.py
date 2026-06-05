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
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from rich.console import RenderableType
from rich.table import Table
from rich.text import Text

from agent_scaffold.capabilities import CapabilityKind, load_capabilities
from agent_scaffold.cli_shared import console as _shared_console
from agent_scaffold.cli_shared import prompt_to_raise_context_cap
from agent_scaffold.context import AssembledContext, ContextBudgetError, assemble
from agent_scaffold.costs import estimate_preflight
from agent_scaffold.discovery import Recipe
from agent_scaffold.effort import EFFORT_PRESETS
from agent_scaffold.language_hints import available_languages
from agent_scaffold.plan import GenerationPlan
from agent_scaffold.repl.refine import REFINEMENT_KEYS, RefinementError, interpret_refinement
from agent_scaffold.repl.render import _DESTRUCTIVE_KEYS as _DESTRUCTIVE_PATCH_KEYS
from agent_scaffold.repl.render import (
    render_cost,
    render_patch_delta,
    render_patch_preview,
    render_state_summary,
)
from agent_scaffold.repl.session import SessionState, StatePatch, apply_patch
from agent_scaffold.topology import resolve as resolve_topology

NextAction = Literal["continue", "generate", "confirm_generate", "exit", "wizard"]


def _patch_is_destructive(patch: StatePatch) -> bool:
    """True iff applying ``patch`` would overwrite a scalar (model, recipe,
    framework, language) or remove existing items (steps, roles, caps).

    Used by :meth:`CommandHandler._dispatch_free_text` to decide whether to
    hand the patch back to the shell loop for confirmation. The destructive
    set is intentionally narrow — purely additive patches (notes,
    add_dependencies, add_steps, add_capabilities) apply silently.
    """
    return any(getattr(patch, key) is not None for key in _DESTRUCTIVE_PATCH_KEYS)


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
    pending_patch: StatePatch | None = None
    """Set by the free-text refinement path when the parsed patch touches a
    destructive key (model / framework / language / recipe / remove_*). The
    shell loop confirms with the user before calling :func:`apply_patch`.
    Other code paths (slash commands, additive refinements) leave this as
    ``None`` and apply directly via ``new_state``."""


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
            # /cost was folded into /plan (cost block is now part of the
            # plan output). Keep the slash for muscle memory — it dispatches
            # to cmd_plan transparently.
            "cost": "plan",
            # `cmd_write_mode` is discovered as `write_mode`; users type
            # `/write-mode` (hyphen reads better at the prompt).
            "write-mode": "write_mode",
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
        preview = render_patch_preview(patch)
        # Destructive patches (overwrite recipe/model/framework/language, or
        # drop steps/roles/capabilities) get a confirmation step in the
        # shell loop before they're applied. Additive patches (notes,
        # add_dependencies, add_steps, add_capabilities) apply inline.
        if _patch_is_destructive(patch):
            return CommandResult(
                messages=[preview],
                pending_patch=patch,
            )
        new_state = apply_patch(state, patch)
        return CommandResult(
            messages=[
                preview,
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
        """List available commands. ``/help refine`` lists free-text refinement keys."""
        # /help refine — render the REFINEMENT_KEYS registry from refine.py
        # so users know what plain-English requests Haiku can interpret.
        if args and args[0].lower() in {"refine", "refinement"}:
            return self._cmd_help_refine()

        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column()
        for name in sorted(self._commands):
            doc = (self._commands[name].__doc__ or "").strip().split("\n", 1)[0]
            table.add_row(f"/{name}", doc)
        return CommandResult(
            messages=[
                table,
                Text.from_markup(
                    "[dim]Free-text refinements like "
                    '[bold]"swap to sonnet, add postgres"[/]'
                    " are also accepted. "
                    "Run [bold]/help refine[/] for the full key list.[/]"
                ),
            ]
        )

    def _cmd_help_refine(self) -> CommandResult:
        """Render the REFINEMENT_KEYS registry as a two-column table.

        Single source of truth lives in :mod:`agent_scaffold.repl.refine`;
        the Haiku system prompt enumerates the same keys, and a test
        (``test_refinement_keys_constant_matches_system_prompt``) keeps
        the two in lockstep.
        """
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column()
        for key, description in REFINEMENT_KEYS.items():
            table.add_row(key, description)
        return CommandResult(
            messages=[
                Text.from_markup(
                    "[bold]Free-text refinement keys[/] "
                    "[dim](Haiku interprets your request into one of these)[/]"
                ),
                table,
            ]
        )

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
        result = _state_change(state, StatePatch(recipe=recipe), f"recipe → {recipe.slug}")
        readiness = _build_service_readiness_line(recipe)
        if readiness is not None:
            result = CommandResult(
                messages=[*result.messages, readiness],
                new_state=result.new_state,
                next_action=result.next_action,
                pending_patch=result.pending_patch,
            )
        return result

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

    def cmd_customize(self, args: list[str], state: SessionState) -> CommandResult:
        """Set stack mode (on|off|toggle). ``on`` enables per-layer customize walk."""
        current = state.stack_mode
        arg = args[0].lower() if args else "toggle"
        target: Literal["quick", "customize"]
        if arg in ("on", "customize"):
            target = "customize"
        elif arg in ("off", "quick"):
            target = "quick"
        elif arg in ("toggle", ""):
            target = "quick" if current == "customize" else "customize"
        else:
            raise CommandError("usage: /customize [on|off|toggle]")
        return _state_change(
            state,
            StatePatch(stack_mode=target),
            f"stack mode → {target}",
        )

    def cmd_layer(self, args: list[str], state: SessionState) -> CommandResult:
        """Inspect or set one layer's capabilities (/layer memory cache.redis vector_db.qdrant).

        With no args: list layers + the current pick for each.
        With one arg (layer name): print available categories within that layer.
        With layer + ids: replace the layer with exactly those ids.
        """
        layer_kinds = _LAYER_GROUPS_BY_KEY
        if not args:
            return _state_change(state, StatePatch(), _format_all_layers(state))
        layer_key = args[0].lower()
        if layer_key not in layer_kinds:
            available = ", ".join(sorted(layer_kinds))
            raise CommandError(f"unknown layer {layer_key!r}; pick one of {available}")
        kinds = layer_kinds[layer_key]
        deployments_path = state.deployments.path
        if deployments_path is None:
            raise CommandError("deployments source is unresolved; cannot inspect layers")
        catalog = load_capabilities(deployments_path)
        candidates = sorted(c.id for c in catalog.values() if c.kind in kinds)
        if len(args) == 1:
            current = _layer_effective_ids(state, kinds)
            cur = ", ".join(current) if current else "(none)"
            opts = ", ".join(candidates) if candidates else "(no catalog entries)"
            return _state_change(
                state,
                StatePatch(),
                f"layer {layer_key}: current = {cur}; available = {opts}",
            )
        # Replace mode: args[1:] is the new id set.
        picked = [a.strip() for a in args[1:] if a.strip()]
        invalid = [p for p in picked if p not in catalog]
        if invalid:
            raise CommandError(
                f"unknown capability id(s) for layer {layer_key}: {', '.join(invalid)}"
            )
        out_of_layer = [p for p in picked if catalog[p].kind not in kinds]
        if out_of_layer:
            raise CommandError(
                f"capabilities {out_of_layer} are not in layer {layer_key!r} "
                f"(layer covers kinds {list(kinds)})"
            )
        effective_in_layer = _layer_effective_ids(state, kinds)
        to_add = [p for p in picked if p not in effective_in_layer]
        to_remove = [c for c in effective_in_layer if c not in picked]
        return _state_change(
            state,
            StatePatch(
                add_capabilities=to_add or None,
                remove_capabilities=to_remove or None,
            ),
            f"layer {layer_key} → {', '.join(picked) if picked else '(none)'}",
        )

    def cmd_observability(self, args: list[str], state: SessionState) -> CommandResult:
        """Pick observability backend (langsmith | langfuse | none).

        Layers an ``add_capabilities`` / ``remove_capabilities`` patch on top of
        the recipe's declared capability set so the swap survives without
        forking the recipe markdown.
        """
        if not args:
            raise CommandError("usage: /observability langsmith | langfuse | none")
        choice = args[0].lower()
        valid = {"langsmith", "langfuse", "none"}
        if choice not in valid:
            raise CommandError(f"observability must be one of {sorted(valid)}, got {choice!r}")
        all_obs_caps = ["obs.langsmith", "obs.langfuse"]
        if choice == "none":
            patch = StatePatch(remove_capabilities=list(all_obs_caps))
        else:
            target = f"obs.{choice}"
            patch = StatePatch(
                add_capabilities=[target],
                remove_capabilities=[c for c in all_obs_caps if c != target],
            )
        return _state_change(state, patch, f"observability → {choice}")

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
        """Re-render the generation plan + cost with the current selections."""
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
        try:
            plan = _build_plan(state)
        except ContextBudgetError as exc:
            bumped = prompt_to_raise_context_cap(_shared_console, exc)
            if bumped is None:
                return CommandResult(
                    messages=[Text.from_markup(f"[red]✗[/] context budget error: {exc}")]
                )
            new_cap, new_per_doc = bumped
            new_cfg = state.cfg.model_copy(
                update={"max_context_tokens": new_cap, "max_tokens_per_doc": new_per_doc}
            )
            new_state = replace(state, cfg=new_cfg, dirty_since_plan=False)
            try:
                plan = _build_plan(new_state)
            except ContextBudgetError as exc2:
                return CommandResult(
                    messages=[Text.from_markup(f"[red]✗[/] context budget error: {exc2}")]
                )
            if isinstance(plan, str):
                return CommandResult(messages=[Text.from_markup(f"[red]✗[/] {plan}")])
            return CommandResult(
                messages=[
                    Text.from_markup(
                        f"[green]✓[/] Context cap raised to {new_cap:,}; per-doc to "
                        f"{new_per_doc:,}. Persisted for this session."
                    ),
                    plan.render(),
                ],
                new_state=new_state,
            )
        if isinstance(plan, str):
            return CommandResult(messages=[Text.from_markup(f"[red]✗[/] {plan}")])
        # /plan folds in the cost estimate so users don't have to run /cost
        # separately. The cost block is appended after the plan panel; if no
        # model is set, the cost helper returns a dim hint.
        cleared_state = replace(state, dirty_since_plan=False) if state.dirty_since_plan else None
        return CommandResult(
            messages=[plan.render(), _build_cost_renderable(state)],
            new_state=cleared_state,
        )

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
        # Refinements applied since the last /plan render mean the user
        # hasn't seen the resolved stack that's about to ship. Show it and
        # gate on confirmation. The shell handles the actual prompt so the
        # confirm-seam stays consistent with _confirm_refinement.
        if state.dirty_since_plan:
            plan_messages: list[RenderableType] = []
            try:
                plan = _build_plan(state)
            except ContextBudgetError as exc:
                return CommandResult(
                    messages=[Text.from_markup(f"[red]✗[/] context budget error: {exc}")]
                )
            if isinstance(plan, str):
                return CommandResult(messages=[Text.from_markup(f"[red]✗[/] {plan}")])
            plan_messages.append(plan.render())
            plan_messages.append(_build_cost_renderable(state))
            return CommandResult(
                messages=plan_messages,
                new_state=state,
                next_action="confirm_generate",
            )
        return CommandResult(
            messages=[Text.from_markup("[bold green]→ Generating…[/]")],
            new_state=state,
            next_action="generate",
        )

    def cmd_autorun(self, args: list[str], state: SessionState) -> CommandResult:
        """Toggle whether ``/go`` chains into ``up`` + welcome panel + browser open.

        Usage: ``/autorun on`` | ``/autorun off`` | ``/autorun`` (toggles).
        Default: on. With autorun off, ``/go`` stops after generation +
        ``print_next_steps`` so you can inspect the generated project before
        running ``up`` by hand.
        """
        from dataclasses import replace

        if not args:
            new_value = not state.autorun
        else:
            token = args[0].strip().lower()
            if token in {"on", "true", "yes", "1"}:
                new_value = True
            elif token in {"off", "false", "no", "0"}:
                new_value = False
            else:
                raise CommandError("usage: /autorun [on|off]")
        new_state = replace(state, autorun=new_value)
        status = "[green]on[/]" if new_value else "[yellow]off[/]"
        return CommandResult(
            messages=[Text.from_markup(f"autorun {status}")],
            new_state=new_state,
        )

    def cmd_write_mode(self, args: list[str], state: SessionState) -> CommandResult:
        """Show or set how /go handles existing files in dest.

        Usage:
            /write-mode                  (show current)
            /write-mode abort|skip|diff|overwrite

        - ``abort``     — refuse to write into a non-empty dest (default).
        - ``skip``      — only create new files; leave existing files alone.
        - ``diff``      — preview a unified diff per changed file and confirm
                          before any writes land. The diff path is owned by
                          the shell loop so the preview lands in the same
                          Console as the rest of the REPL output.
        - ``overwrite`` — always replace existing files. No confirm.
        """
        from agent_scaffold.writer import WriteMode

        if not args:
            current = state.write_mode.value
            return CommandResult(
                messages=[
                    Text.from_markup(f"write mode: [bold]{current}[/]"),
                    Text.from_markup(
                        "[dim]options: abort | skip | diff | overwrite[/]"
                    ),
                ]
            )
        token = args[0].strip().lower()
        try:
            mode = WriteMode(token)
        except ValueError as exc:
            raise CommandError(
                f"unknown mode {token!r}; options: abort | skip | diff | overwrite"
            ) from exc
        new_state = replace(state, write_mode=mode)
        return CommandResult(
            messages=[Text.from_markup(f"write mode → [bold]{mode.value}[/]")],
            new_state=new_state,
        )

    def cmd_exit(self, args: list[str], state: SessionState) -> CommandResult:  # noqa: ARG002
        """Leave the REPL (alias: /quit, /q)."""
        return CommandResult(
            messages=[Text.from_markup("[dim]bye.[/]")],
            new_state=state,
            next_action="exit",
        )

    # ----- lifecycle commands (deploy / down / status / logs) ------------

    def cmd_deploy(self, args: list[str], state: SessionState) -> CommandResult:  # noqa: ARG002
        """Print the deploy command for a host target (always dry-run in the REPL).

        Use: ``/deploy vercel`` / ``/deploy fly`` / ``/deploy railway``.
        The REPL never invokes a real cloud deploy — for that, exit and run
        ``agent-scaffold deploy --target <target> --no-dry-run --yes``.
        """
        if not args:
            raise CommandError("usage: /deploy <vercel|fly|railway>")
        target = args[0].lower()
        if not state.dest:
            raise CommandError("set a project dest first (/dest <path>)")
        from agent_scaffold.deploy import get_plugin

        try:
            plugin = get_plugin(target)
        except KeyError as exc:
            raise CommandError(
                f"unknown deploy target {target!r}; supported: vercel, railway, fly"
            ) from exc
        result = plugin.deploy(Path(state.dest), dry_run=True, yes=False)
        lines: list[RenderableType] = [
            Text.from_markup(f"[bold]{result.target}[/]: {result.summary}"),
        ]
        if result.cmd_run:
            lines.append(Text.from_markup(f"[cyan]$[/] {' '.join(result.cmd_run)}"))
        if result.dashboard_url:
            lines.append(Text.from_markup(f"[dim]dashboard:[/] {result.dashboard_url}"))
        return CommandResult(messages=lines)

    def cmd_eval(self, args: list[str], state: SessionState) -> CommandResult:  # noqa: ARG002
        """Show the eval command for the current project (the REPL never runs it).

        Use: ``/eval`` (defaults to ``--target promptfoo`` against ``state.dest``).
        Evals can take minutes; exit the REPL and run
        ``agent-scaffold eval --cwd <dest>`` to actually run them. ``--json`` and
        ``--update-baseline`` work as on the CLI.
        """
        if not state.dest:
            raise CommandError("set a project dest first (/dest <path>)")
        cmd = f"agent-scaffold eval --cwd {state.dest}"
        return CommandResult(
            messages=[
                Text.from_markup(f"[cyan]$[/] {cmd}  [dim](exit the REPL to run this)[/]"),
                Text.from_markup("[dim]flags:[/] --target promptfoo  --json  --update-baseline"),
            ]
        )

    def cmd_down(self, args: list[str], state: SessionState) -> CommandResult:
        """Show the docker compose down command (the REPL never runs it).

        Use: ``/down`` for plain teardown, ``/down -v`` to also drop volumes.
        Exit the REPL and run ``agent-scaffold down [-v]`` to actually
        tear down the local stack.
        """
        del state
        flags = " -v" if args and args[0] in ("-v", "--volumes") else ""
        return CommandResult(
            messages=[
                Text.from_markup(
                    f"[cyan]$[/] agent-scaffold down{flags} " "[dim](exit the REPL to run this)[/]"
                )
            ]
        )

    def cmd_status(self, args: list[str], state: SessionState) -> CommandResult:  # noqa: ARG002
        """Show the status command for the current project.

        The REPL doesn't run probes itself (they can take seconds and would
        block the prompt); use ``agent-scaffold status`` outside the REPL.
        """
        if not state.dest:
            raise CommandError("set a project dest first (/dest <path>)")
        return CommandResult(
            messages=[
                Text.from_markup(
                    f"[cyan]$[/] agent-scaffold status --cwd {state.dest} "
                    "[dim](exit the REPL to run this)[/]"
                )
            ]
        )

    def cmd_logs(self, args: list[str], state: SessionState) -> CommandResult:
        """Show the logs command for a docker-compose service.

        Use: ``/logs <service>``. The REPL doesn't tail logs itself — exit
        and run ``agent-scaffold logs <service>`` to stream.
        """
        if not args:
            raise CommandError("usage: /logs <service>")
        if not state.dest:
            raise CommandError("set a project dest first (/dest <path>)")
        service = args[0]
        return CommandResult(
            messages=[
                Text.from_markup(
                    f"[cyan]$[/] agent-scaffold logs {service} --cwd {state.dest} "
                    "[dim](exit the REPL to run this)[/]"
                )
            ]
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


# ---------------------------------------------------------------------------
# REPL-scoped assemble() cache
#
# /plan and /cost both call assemble() with identical args drawn from the
# same SessionState; without caching, a /plan immediately followed by /cost
# walks the blueprint tree and re-reads every linked doc twice. Cache key
# captures every assemble() input — recipe, language, framework, paths,
# budgets — so any state change that could affect the output also bypasses
# the cache. Cap at 8 entries to bound memory while still covering a user
# toggling between a handful of recipes in a session.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _AssembleKey:
    recipe_slug: str
    recipe_path: str
    language: str
    framework: str
    deployments_path: str
    blueprints_path: str | None
    max_context_tokens: int
    max_link_depth: int
    max_tokens_per_doc: int


_ASSEMBLE_CACHE_MAX = 8
_assemble_cache: OrderedDict[_AssembleKey, AssembledContext] = OrderedDict()


def _assemble_for_state(state: SessionState) -> AssembledContext:
    """Cached :func:`assemble` for the current REPL state.

    Caller must have already verified ``state.recipe`` / ``state.language`` /
    ``state.framework`` / ``state.deployments.path`` are non-None — this
    helper just wraps the call with a small LRU keyed on every input that
    could change the assembled output.
    """
    deployments_path = state.deployments.path
    assert deployments_path is not None
    assert state.recipe is not None
    assert state.language is not None
    assert state.framework is not None

    key = _AssembleKey(
        recipe_slug=state.recipe.slug,
        recipe_path=str(state.recipe.path),
        language=state.language,
        framework=state.framework,
        deployments_path=str(deployments_path),
        blueprints_path=str(state.blueprints.path) if state.blueprints.path else None,
        max_context_tokens=state.cfg.max_context_tokens,
        max_link_depth=state.cfg.max_link_depth,
        max_tokens_per_doc=state.cfg.max_tokens_per_doc,
    )
    cached = _assemble_cache.get(key)
    if cached is not None:
        _assemble_cache.move_to_end(key)
        return cached
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
    _assemble_cache[key] = ctx
    if len(_assemble_cache) > _ASSEMBLE_CACHE_MAX:
        _assemble_cache.popitem(last=False)
    return ctx


def _clear_assemble_cache() -> None:
    """Test seam — clears the per-state assemble cache between test runs."""
    _assemble_cache.clear()


# Customize-mode layer groupings — mirrors ``_LAYER_GROUPS`` in repl/shell.py
# so the slash command and the wizard step produce identical patches.
_LAYER_GROUPS_BY_KEY: dict[str, tuple[CapabilityKind, ...]] = {
    "memory": ("relational", "cache", "vector_db"),
    "observability": ("obs",),
    "obs": ("obs",),
    "eval": ("eval",),
    "interface": ("frontend",),
    "frontend": ("frontend",),
}


def _layer_effective_ids(state: SessionState, kinds: tuple[CapabilityKind, ...]) -> list[str]:
    """Recipe-declared caps ∪ session adds, minus session removes, filtered to kinds."""
    recipe_ids = set(state.recipe.capabilities) if state.recipe else set()
    effective = (recipe_ids | set(state.add_capabilities)) - set(state.remove_capabilities)
    return sorted(c for c in effective if c.split(".", 1)[0] in kinds)


def _format_all_layers(state: SessionState) -> str:
    """Compact one-line-per-layer summary for ``/layer`` with no args."""
    rows: list[str] = []
    for key in ("memory", "observability", "eval", "interface"):
        kinds = _LAYER_GROUPS_BY_KEY[key]
        ids = _layer_effective_ids(state, kinds)
        rows.append(f"  {key:<14}{', '.join(ids) if ids else '(none)'}")
    return "layers:\n" + "\n".join(rows)


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


# 1-second timeout: short enough to keep `/recipe <slug>` snappy even when
# every probe fails by socket timeout. The CLI plan path uses 5s; the REPL
# trades a little resolution for a responsive slash command.
_RECIPE_PROBE_TIMEOUT_S = 1.0


def _build_service_readiness_line(recipe: Recipe) -> RenderableType | None:
    """Probe the recipe's external services and render the one-liner.

    Returns ``None`` when the recipe has no ``external_services`` so the
    caller skips appending an empty line. Probe failures are caught here
    and rendered as a dim warning instead of bubbling — the brief's
    "non-blocking — generation still works" criterion.
    """
    from agent_scaffold.probes import probe_external_services
    from agent_scaffold.repl.render import render_service_readiness_oneline

    if not recipe.external_services:
        return None
    try:
        results = probe_external_services(
            recipe.external_services, timeout=_RECIPE_PROBE_TIMEOUT_S
        )
    except Exception as exc:  # noqa: BLE001 - readiness is non-blocking
        return Text.from_markup(f"[dim]Services: probe runner failed: {exc}[/]")
    return render_service_readiness_oneline(results)


def _build_plan(state: SessionState) -> GenerationPlan | str:
    """Assemble context + build a GenerationPlan from the current state.

    Returns the plan on success, or an error string for "soft" failures the
    caller should render verbatim (e.g. missing deployments source).
    Propagates :class:`ContextBudgetError` so ``cmd_plan`` can offer the
    user a cap bump and retry — the bump needs to mutate session state, so
    it has to be handled at the command-method layer, not buried here.
    """
    assert state.recipe is not None  # is_ready() guarantees this
    assert state.language is not None
    assert state.framework is not None
    assert state.project_name is not None
    assert state.dest is not None
    deployments_path = state.deployments.path
    if deployments_path is None:
        return "deployments source unavailable; rerun the shell with --deployments-path"
    ctx = _assemble_for_state(state)

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
    if state.deployments.path is None:
        return _DEFAULT_INPUT_TOKENS_GUESS
    try:
        ctx = _assemble_for_state(state)
    except ContextBudgetError:
        return _DEFAULT_INPUT_TOKENS_GUESS
    return ctx.token_estimate


def _build_cost_renderable(state: SessionState) -> Text:
    """Build the cost-estimate renderable used by /plan.

    Centralizes the "model missing → nudge" + "cost unknown → dim hint" UX
    in one place so the plan panel's appended cost block formats consistently
    regardless of state readiness.
    """
    model = state.model
    if model is None:
        return Text.from_markup(
            "[dim]Est. cost unavailable — set a model first "
            "([bold]/model[/] or [bold]/effort[/]).[/]"
        )
    input_tokens = _estimate_input_tokens(state)
    max_tokens = state.max_tokens or 32_000
    preflight = estimate_preflight(
        model,
        input_tokens=input_tokens,
        output_range=(min(8_000, max_tokens), max_tokens),
    )
    return render_cost(preflight)
