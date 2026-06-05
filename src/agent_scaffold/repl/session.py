"""Session state for the ``agent-scaffold scaffold`` REPL.

The REPL is a persistent shell: users make selections incrementally via
slash commands (``/recipe foo``, ``/language python``, …), refine with
free text (``swap to sonnet, add postgres``), inspect with ``/plan``, and
generate with ``/go``. :class:`SessionState` is the source of truth that
flows between every command, and :class:`StatePatch` is the typed delta
applied by both slash commands and the LLM-interpreted refinement layer.

Design choices:

- ``SessionState`` is mutable but updates go through :func:`apply_patch`,
  which returns a new state via ``dataclasses.replace``. That preserves
  the "before" state if a command wants to render a delta.
- ``StatePatch`` is frozen — patches are values, not records, so the same
  patch can be replayed or compared. Every field is ``None`` by default,
  meaning "don't touch this attribute on the state". This matches the
  semantics of a JSON merge patch ([RFC 7396]) and is what the LLM
  refinement interpreter produces.
- ``refinement_notes`` and ``extra_dependencies`` *accumulate* across
  patches rather than overwrite — a sequence of refinements ("use sonnet"
  then "add postgres") composes correctly.

This module is intentionally side-effect-free: no Rich rendering, no
network I/O, no Config / Recipe construction. Those live next door
in :mod:`agent_scaffold.repl.render` and :mod:`agent_scaffold.repl.commands`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar, Literal

from agent_scaffold.config import Config
from agent_scaffold.discovery import Recipe
from agent_scaffold.sources import ResolvedSource
from agent_scaffold.writer import WriteMode

StackMode = Literal["quick", "customize"]
"""``quick`` = use recipe defaults (today's wizard flow). ``customize`` =
walk each layer (memory, obs, eval, interface) and let the user pick
categories. Basic-tier recipes auto-default to ``quick``."""


@dataclass
class SessionState:
    """The mutable selections a user has made so far in the REPL session.

    ``cfg``, ``deployments``, ``blueprints`` are immutable session-scope
    inputs resolved when the shell opens — they don't change inside the
    REPL. Everything else is user-mutable via slash commands or free-text
    refinements.

    ``None`` for a field means "user hasn't picked yet"; :meth:`is_ready`
    converts those into a missing-field list so ``/go`` can refuse with a
    helpful message rather than crash deep in the pipeline.
    """

    # Session-scope inputs (set once at shell open, never patched).
    cfg: Config
    deployments: ResolvedSource
    blueprints: ResolvedSource

    # Required selections — must all be set before ``/go``.
    recipe: Recipe | None = None
    language: str | None = None
    framework: str | None = None
    project_name: str | None = None
    dest: Path | None = None

    # Optional overrides — fall back to config / effort presets when None.
    model: str | None = None
    effort: str | None = None
    max_tokens: int | None = None
    thinking_budget: int | None = None
    strict: bool = False
    write_mode: WriteMode = WriteMode.abort

    # After /go completes, chain into the same up + welcome panel + browser-open
    # flow as ``agent-scaffold new``'s autorun. Default on for interactive REPL
    # users — toggle with /autorun off if you want the staged "generate, then
    # eyeball, then up by hand" loop instead.
    autorun: bool = True

    # Stack mode: "quick" reuses the recipe's declared capability set as-is;
    # "customize" surfaces a layer-walk so the user picks memory / obs / eval /
    # interface categories explicitly. Defaults to "quick"; the wizard auto-
    # downshifts to "quick" for basic-tier recipes without prompting.
    stack_mode: StackMode = "quick"

    # Accumulators populated by free-text refinements + slash commands.
    extra_dependencies: dict[str, dict[str, str]] = field(default_factory=dict)
    """Language -> {package: version}. Merged into the recipe's
    pinned_dependencies before generation."""

    extra_steps: list[str] = field(default_factory=list)
    removed_steps: set[str] = field(default_factory=set)
    removed_roles: set[str] = field(default_factory=set)

    # Capability overrides: layered on top of the recipe's frontmatter
    # ``capabilities:`` list at resolve time. ``add_capabilities`` extends the
    # union; ``remove_capabilities`` subtracts before resolving. Lets users
    # swap obs.langsmith → obs.langfuse without forking the recipe.
    add_capabilities: list[str] = field(default_factory=list)
    remove_capabilities: set[str] = field(default_factory=set)

    refinement_notes: list[str] = field(default_factory=list)
    """Free-text guidance the LLM couldn't translate into a typed patch.
    Appended verbatim to the generation prompt as additional user
    instructions."""

    dirty_since_plan: bool = False
    """True iff a stack-mutating patch has been applied since the user last
    saw the plan panel. ``/plan`` clears it after a successful render;
    ``/go`` re-renders the plan and asks the user to confirm before
    generation while it's set, so refinements never ship silently."""

    # ----- queries -------------------------------------------------------

    # ClassVar so dataclass doesn't treat this as an instance field.
    REQUIRED_FIELDS: ClassVar[tuple[str, ...]] = (
        "recipe",
        "language",
        "framework",
        "project_name",
        "dest",
    )

    def is_ready(self) -> tuple[bool, list[str]]:
        """Returns ``(ok, missing_fields)``. ``/go`` requires ``ok=True``."""
        missing = [name for name in self.REQUIRED_FIELDS if getattr(self, name) is None]
        return (not missing, missing)


@dataclass(frozen=True)
class StatePatch:
    """A typed delta over :class:`SessionState`.

    ``None`` for a scalar field means "don't touch it"; an explicit value
    overwrites. For the accumulators (``add_dependencies``, ``remove_steps``,
    ``remove_roles``, ``notes``), the patch *adds* to what's already there
    rather than replacing — a sequence of patches composes.

    Produced by both the slash-command dispatcher (PR4) and the LLM
    refinement interpreter (PR5). Apply with :func:`apply_patch`.
    """

    # Scalar overrides.
    recipe: Recipe | None = None
    language: str | None = None
    framework: str | None = None
    project_name: str | None = None
    dest: Path | None = None
    model: str | None = None
    effort: str | None = None
    max_tokens: int | None = None
    thinking_budget: int | None = None
    strict: bool | None = None
    write_mode: WriteMode | None = None
    stack_mode: StackMode | None = None
    """When set, switches the session's stack mode. ``customize`` enables the
    layer-walk steps; ``quick`` falls back to recipe defaults."""

    # Accumulators (merged, not overwritten).
    add_dependencies: dict[str, dict[str, str]] | None = None
    """Language -> {package: version}; merged into state.extra_dependencies."""

    add_steps: list[str] | None = None
    remove_steps: list[str] | None = None
    remove_roles: list[str] | None = None
    add_capabilities: list[str] | None = None
    """Capability ids to layer onto the recipe's declared set."""
    remove_capabilities: list[str] | None = None
    """Capability ids to drop from the recipe's declared set."""
    notes: str | None = None
    """Free-text guidance; appended to state.refinement_notes."""

    def is_empty(self) -> bool:
        """True iff the patch is a no-op (every field is ``None``)."""
        for name in self.__dataclass_fields__:
            if getattr(self, name) is not None:
                return False
        return True


def apply_patch(state: SessionState, patch: StatePatch) -> SessionState:
    """Return a new :class:`SessionState` with ``patch`` applied.

    Scalars overwrite when the patch sets them; accumulators merge:

    - ``add_dependencies`` deep-merges into ``extra_dependencies`` so two
      "add postgres" / "add redis" patches don't clobber each other.
    - ``add_steps`` / ``remove_steps`` extend / union into the existing
      sets so the order of refinements doesn't matter.
    - ``notes`` appends as a new line to ``refinement_notes``.

    The original ``state`` is left unchanged — callers that need to render
    a delta keep the "before" snapshot around.
    """
    # Start from a shallow copy so we can mutate the collections without
    # touching the caller's state.
    new_extra_deps = {lang: dict(pkgs) for lang, pkgs in state.extra_dependencies.items()}
    new_extra_steps = list(state.extra_steps)
    new_removed_steps = set(state.removed_steps)
    new_removed_roles = set(state.removed_roles)
    new_add_capabilities = list(state.add_capabilities)
    new_remove_capabilities = set(state.remove_capabilities)
    new_notes = list(state.refinement_notes)

    if patch.add_dependencies:
        for lang, pkgs in patch.add_dependencies.items():
            new_extra_deps.setdefault(lang, {}).update(pkgs)
    if patch.add_steps:
        new_extra_steps.extend(s for s in patch.add_steps if s not in new_extra_steps)
        # If the user re-adds a step they previously removed, honor the add.
        new_removed_steps.difference_update(patch.add_steps)
    if patch.remove_steps:
        new_removed_steps.update(patch.remove_steps)
        # Removing supersedes any earlier add.
        new_extra_steps = [s for s in new_extra_steps if s not in patch.remove_steps]
    if patch.remove_roles:
        new_removed_roles.update(patch.remove_roles)
    if patch.add_capabilities:
        new_add_capabilities.extend(
            c for c in patch.add_capabilities if c not in new_add_capabilities
        )
        # Re-adding a previously removed capability honors the add.
        new_remove_capabilities.difference_update(patch.add_capabilities)
    if patch.remove_capabilities:
        new_remove_capabilities.update(patch.remove_capabilities)
        # Removing supersedes any earlier add.
        new_add_capabilities = [
            c for c in new_add_capabilities if c not in patch.remove_capabilities
        ]
    if patch.notes:
        new_notes.append(patch.notes)

    # dataclasses.replace handles the scalar overwrites — only the fields
    # explicitly set on the patch are passed through. typed as Any so the
    # **expansion accepts the heterogeneous field types.
    scalar_updates: dict[str, Any] = {}
    for name in (
        "recipe",
        "language",
        "framework",
        "project_name",
        "dest",
        "model",
        "effort",
        "max_tokens",
        "thinking_budget",
        "strict",
        "write_mode",
        "stack_mode",
    ):
        value = getattr(patch, name)
        if value is not None:
            scalar_updates[name] = value

    # A patch dirties the plan whenever it touches a field the plan panel
    # actually renders — recipe, language, framework, model, stack mode, or
    # any of the capability / step accumulators. Note-only or autorun-only
    # patches don't dirty: the plan doesn't show them and re-rendering would
    # be noise.
    dirties_plan = any(
        getattr(patch, name) is not None
        for name in (
            "recipe",
            "language",
            "framework",
            "model",
            "stack_mode",
            "add_capabilities",
            "remove_capabilities",
            "add_steps",
            "remove_steps",
        )
    )
    new_dirty = state.dirty_since_plan or dirties_plan

    return replace(
        state,
        extra_dependencies=new_extra_deps,
        extra_steps=new_extra_steps,
        removed_steps=new_removed_steps,
        removed_roles=new_removed_roles,
        add_capabilities=new_add_capabilities,
        remove_capabilities=new_remove_capabilities,
        refinement_notes=new_notes,
        dirty_since_plan=new_dirty,
        **scalar_updates,
    )
