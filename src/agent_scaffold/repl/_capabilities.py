"""REPL-side capability stack resolution.

The CLI's ``cmd_new`` resolves the recipe's declared capabilities against
the deployments catalog once and threads the resulting ``ResolvedStack``
through the pipeline so the generator, manifest, and capability-template
copier all see the same set. The REPL had no equivalent: every
``/observability langfuse``, ``/layer add eval.deepeval``, or free-text
``add cache.redis`` patch updated ``SessionState.add_capabilities`` /
``remove_capabilities`` but never made it back into the resolved stack
that ``PipelineInputs`` or ``context.assemble`` consume.

:func:`resolve_stack_for_session` closes that gap. It re-resolves on every
call so a single ``SessionState`` snapshot always produces the stack that
matches the current overrides. Both ``shell._build_pipeline_inputs`` and
``commands._assemble_for_state`` route through it so the same effective
capability set drives generation and plan rendering.

Living in its own module keeps the call import-safe from both
``commands.py`` and ``shell.py`` without a circular dependency.
"""

from __future__ import annotations

from agent_scaffold.capabilities import (
    ResolvedStack,
    load_capabilities,
)
from agent_scaffold.capabilities import (
    resolve as resolve_capabilities,
)
from agent_scaffold.repl.session import SessionState


def resolve_stack_for_session(state: SessionState) -> ResolvedStack | None:
    """Resolve the recipe's capabilities with REPL overrides applied.

    Returns ``None`` when there's nothing to resolve — no recipe picked
    yet, the deployments source is unavailable, or the recipe declares no
    capabilities and the session hasn't added any. Otherwise builds the
    catalog once and returns the resolved stack; an empty stack
    (no resolvable ids) also degrades to ``None`` so downstream consumers
    that key on "did anything resolve" don't have to special-case it.

    Unknown override ids surface from the slash-command path (``/layer``
    validates against the catalog before patching); this helper trusts
    the patch and falls through to :func:`agent_scaffold.capabilities.resolve`,
    which records unrecognized ids in ``ResolvedStack.unresolved`` rather
    than raising.
    """
    if state.recipe is None or state.deployments.path is None:
        return None
    # No early-return on an empty recipe: every agent ships the default frontend
    # (when the catalog has it), so even a bare recipe resolves to a UI capability.
    catalog = load_capabilities(state.deployments.path)
    stack = resolve_capabilities(
        state.recipe,
        catalog,
        add_capabilities=list(state.add_capabilities),
        remove_capabilities=set(state.remove_capabilities),
        default_frontend=True,
    )
    return stack if stack.capabilities else None


__all__ = ["resolve_stack_for_session"]
