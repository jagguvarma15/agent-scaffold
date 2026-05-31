"""Eval-runner provider plugins.

Each module under this package exposes a ``run(project_dir, baseline_total)``
function that returns an :class:`EvalResult`. The CLI dispatcher (``cmd_eval``)
picks the plugin by name (default: ``promptfoo``), runs it, then renders a
regression-table or JSON output.

Plugin shape is intentionally thin — eval runners typically already have
their own CLI (``npx promptfoo``, ``deepeval test``); plugins just wrap the
invocation + JSON parsing + baseline comparison.
"""

from __future__ import annotations

from typing import Any, cast

from agent_scaffold.eval._common import (
    REGRESSION_NOISE_FLOOR,
    EvalCase,
    EvalResult,
    EvalTarget,
)

__all__ = [
    "EVAL_PLUGINS",
    "REGRESSION_NOISE_FLOOR",
    "EvalCase",
    "EvalResult",
    "EvalTarget",
    "get_plugin",
]


def _import_plugins() -> dict[str, Any]:
    """Lazy plugin registry — avoids importing provider modules at package load."""
    from agent_scaffold.eval import promptfoo

    return {"promptfoo": promptfoo}


EVAL_PLUGINS: dict[str, Any] | None = None


def get_plugin(target: str) -> EvalTarget:
    """Return the eval plugin module for ``target``. Raises ``KeyError`` if unknown."""
    global EVAL_PLUGINS
    if EVAL_PLUGINS is None:
        EVAL_PLUGINS = _import_plugins()
    return cast(EvalTarget, EVAL_PLUGINS[target])
