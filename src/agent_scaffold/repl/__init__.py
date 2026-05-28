"""Interactive REPL for ``agent-scaffold scaffold``.

The REPL is a persistent shell — you open it once, drive selections with
slash commands, refine the plan with free text, confirm cost, and generate.
Multiple projects can be scaffolded in one session.

Module layout:

- :mod:`agent_scaffold.repl.session` — state model (``SessionState``, ``StatePatch``).
- :mod:`agent_scaffold.repl.commands` — slash-command dispatcher (``CommandHandler``).
- :mod:`agent_scaffold.repl.refine` — Haiku-interpreted free-text refinements.
- :mod:`agent_scaffold.repl.render` — Rich panels for the in-shell output.
"""

from __future__ import annotations

from agent_scaffold.repl.commands import (
    CommandError,
    CommandHandler,
    CommandResult,
    NextAction,
)
from agent_scaffold.repl.refine import RefinementError, interpret_refinement
from agent_scaffold.repl.session import SessionState, StatePatch, apply_patch

__all__ = [
    "CommandError",
    "CommandHandler",
    "CommandResult",
    "NextAction",
    "RefinementError",
    "SessionState",
    "StatePatch",
    "apply_patch",
    "interpret_refinement",
]
