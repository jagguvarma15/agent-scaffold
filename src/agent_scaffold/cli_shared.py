"""Shared CLI singletons used across ``cli.py`` and the ``cli_*`` sub-modules.

Lives in its own module so the extracted command groups (``cli_auth``,
``cli_doctor``, ``cli_secrets``) can import ``console`` without creating
an import cycle through ``cli.py``. The Console is constructed lazily —
import-only — so test fixtures that monkeypatch this attribute see a
single object every site shares.

This is intentionally tiny. New shared helpers belong here only when at
least two command modules need them and they would otherwise force the
new modules to import from ``cli.py``.
"""

from __future__ import annotations

from rich.console import Console

from agent_scaffold.context import ContextBudgetError
from agent_scaffold.effort import EFFORT_PRESETS

console = Console()


def prompt_to_raise_context_cap(
    console: Console,
    exc: ContextBudgetError,
    *,
    non_interactive: bool = False,
) -> tuple[int, int] | None:
    """Offer the user a one-shot bump from medium's 60k cap to high's 100k.

    Default cap is medium (60k). When essentials overflow but would still fit
    under high (100k), prompt y/N. On yes, return ``(new_max_context_tokens,
    new_max_tokens_per_doc)`` — the caller updates its ``Config`` and retries
    ``assemble``. Model + thinking budget stay on the user's current effort:
    only the context-related fields move.

    Returns ``None`` (caller re-raises / bails) when:
    - essentials exceed even high's 100k cap (no preset would fit)
    - ``non_interactive`` is set (CI must opt in explicitly via
      ``--max-context-tokens``)
    - the user declines
    """
    high = EFFORT_PRESETS["high"]
    if exc.essentials_tokens > high.max_context_tokens:
        console.print(
            f"[red]Context budget error:[/] essentials ~{exc.essentials_tokens:,} "
            f"tokens exceed even high effort's {high.max_context_tokens:,} cap. "
            "Trim the recipe's Composes section, or pass a larger "
            "[bold]--max-context-tokens[/] explicitly."
        )
        return None
    if non_interactive:
        console.print(
            f"[red]Context budget error:[/] essentials ~{exc.essentials_tokens:,} "
            f"tokens > {exc.current_cap:,} cap. Non-interactive mode — pass "
            f"[bold]--max-context-tokens {high.max_context_tokens}[/] "
            "(or [bold]--effort high[/]) to retry."
        )
        return None
    console.print(
        f"[yellow]Context budget exceeded:[/] essentials ~{exc.essentials_tokens:,} "
        f"tokens > {exc.current_cap:,} cap."
    )
    answer = (
        console.input(
            f"[bold]Bump cap to {high.max_context_tokens:,} "
            "(high preset's limit, same model) and continue? [Y/n] [/]"
        )
        .strip()
        .lower()
    )
    if answer in ("", "y", "yes"):
        return high.max_context_tokens, high.max_tokens_per_doc
    return None
