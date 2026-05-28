"""Rich rendering helpers for the REPL.

Pure presentation: each function takes data and returns a Rich renderable
(``Panel``, ``Text``, or ``Table``). No I/O, no Console — the shell loop
decides where to print. Splitting them out from ``commands.py`` keeps
business logic and visual choices reviewable in isolation, and lets
snapshot tests pin the user-facing output without spinning up the loop.
"""

from __future__ import annotations

from rich.panel import Panel
from rich.text import Text

from agent_scaffold.costs import PreflightCost
from agent_scaffold.repl.session import SessionState

# Compact one-letter labels make the per-field rows align visually even
# when many are still ``-`` (the placeholder for "not picked yet").
_FIELD_LABELS: tuple[tuple[str, str, str], ...] = (
    ("Recipe", "recipe", "slug"),
    ("Language", "language", ""),
    ("Framework", "framework", ""),
    ("Name", "project_name", ""),
    ("Dest", "dest", ""),
    ("Model", "model", ""),
    ("Effort", "effort", ""),
    ("Write mode", "write_mode", "value"),
)

_UNSET = "[dim]–[/]"


def _format_value(state: SessionState, attr: str, sub_attr: str) -> str:
    value = getattr(state, attr)
    if value is None:
        return _UNSET
    if sub_attr:
        value = getattr(value, sub_attr, value)
    return str(value)


def render_state_summary(state: SessionState) -> Panel:
    """Compact "what's selected so far" panel, shown after each command."""
    rows: list[str] = []
    for label, attr, sub_attr in _FIELD_LABELS:
        rows.append(f"[bold]{label:<11}[/] {_format_value(state, attr, sub_attr)}")
    if state.extra_dependencies:
        added = sum(len(pkgs) for pkgs in state.extra_dependencies.values())
        rows.append(f"[bold]Extra deps[/] +{added} package(s)")
    if state.extra_steps or state.removed_steps:
        rows.append(f"[bold]Steps[/]      +{len(state.extra_steps)} / -{len(state.removed_steps)}")
    if state.removed_roles:
        rows.append(f"[bold]Roles[/]      -{len(state.removed_roles)} removed")
    if state.refinement_notes:
        rows.append(f"[bold]Notes[/]      {len(state.refinement_notes)} refinement(s)")
    return Panel(
        "\n".join(rows),
        title="Session",
        expand=False,
        border_style="#FF8C00",
    )


# Map (attr_name -> human label) for delta rendering. Skips session-scope
# inputs (cfg, deployments, blueprints) that never change.
_DELTA_LABELS: dict[str, str] = {
    "recipe": "recipe",
    "language": "language",
    "framework": "framework",
    "project_name": "name",
    "dest": "dest",
    "model": "model",
    "effort": "effort",
    "max_tokens": "max_tokens",
    "thinking_budget": "thinking",
    "strict": "strict",
    "write_mode": "write_mode",
}


def _label(value: object) -> str:
    if value is None:
        return "–"
    slug = getattr(value, "slug", None)
    if slug is not None:
        return str(slug)
    val = getattr(value, "value", None)
    if val is not None:
        return str(val)
    return str(value)


def render_patch_delta(before: SessionState, after: SessionState) -> Text:
    """Render the diff between two states as ``Δ field: old → new`` lines.

    Used after a free-text refinement or slash-command edit so the user can
    confirm exactly what changed before re-rendering the full plan.
    """
    lines: list[str] = []
    for attr, label in _DELTA_LABELS.items():
        before_v = getattr(before, attr)
        after_v = getattr(after, attr)
        if before_v != after_v:
            lines.append(f"[#FFA500]Δ[/] {label}: {_label(before_v)} → {_label(after_v)}")

    # Accumulators get summarized as counts — a per-package diff would be
    # noisy after a long refinement.
    if before.extra_dependencies != after.extra_dependencies:
        before_n = sum(len(p) for p in before.extra_dependencies.values())
        after_n = sum(len(p) for p in after.extra_dependencies.values())
        if after_n != before_n:
            lines.append(f"[#FFA500]Δ[/] extra deps: {before_n} → {after_n}")
    if before.removed_steps != after.removed_steps:
        added = after.removed_steps - before.removed_steps
        if added:
            lines.append(f"[#FFA500]Δ[/] steps: -{', '.join(sorted(added))}")
    if before.extra_steps != after.extra_steps:
        added_steps = [s for s in after.extra_steps if s not in before.extra_steps]
        if added_steps:
            lines.append(f"[#FFA500]Δ[/] steps: +{', '.join(added_steps)}")
    if before.removed_roles != after.removed_roles:
        added = after.removed_roles - before.removed_roles
        if added:
            lines.append(f"[#FFA500]Δ[/] roles: -{', '.join(sorted(added))}")
    if len(after.refinement_notes) > len(before.refinement_notes):
        new_notes = after.refinement_notes[len(before.refinement_notes) :]
        for note in new_notes:
            # Truncate very long notes so the delta block stays readable.
            display = note if len(note) <= 80 else note[:77] + "…"
            lines.append(f"[#FFA500]Δ[/] note: [dim]{display}[/]")

    if not lines:
        return Text.from_markup("[dim]No changes.[/]")
    return Text.from_markup("\n".join(lines))


def render_cost(preflight: PreflightCost | None) -> Text:
    """One-line cost rendering for the ``/cost`` command.

    Returns a dim line when the model is unknown so the REPL still shows
    *something* (rather than an empty Text) — the user is reminded that
    pricing data is opt-in.
    """
    if preflight is None:
        return Text.from_markup("[dim]Est. cost unavailable — model not in pricing table.[/]")
    return Text.from_markup(f"[bold]Est. cost[/] {preflight.format()}")
