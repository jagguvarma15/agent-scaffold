"""Interactive REPL for ``agent-scaffold scaffold``.

The REPL is a persistent shell — you open it once, drive selections with
slash commands, refine the plan with free text, confirm cost, and generate.
Multiple projects can be scaffolded in one session. See
:mod:`agent_scaffold.repl.session` for the state model and
:mod:`agent_scaffold.repl.commands` (forthcoming) for the dispatcher.
"""

from __future__ import annotations

from agent_scaffold.repl.session import SessionState, StatePatch, apply_patch

__all__ = ["SessionState", "StatePatch", "apply_patch"]
