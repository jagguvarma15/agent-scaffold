"""Interactive REPL for ``agent-scaffold scaffold``.

The REPL is a persistent shell — you open it once, drive selections with
slash commands, refine the plan with free text, confirm cost, and generate.
Multiple projects can be scaffolded in one session.

Module layout:

- :mod:`agent_scaffold.repl.session` — state model (``SessionState``, ``StatePatch``).
- :mod:`agent_scaffold.repl.commands` — slash-command dispatcher (``CommandHandler``).
- :mod:`agent_scaffold.repl.refine` — Haiku-interpreted free-text refinements.
- :mod:`agent_scaffold.repl.render` — Rich panels for the in-shell output.
- :mod:`agent_scaffold.repl.shell` — ``PromptSession`` loop + ``/new`` wizard.

Callers import directly from the submodule that owns the symbol they
want — there are no package-level re-exports. Tests do the same.
"""

from __future__ import annotations
