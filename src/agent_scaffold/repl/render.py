"""Rich rendering helpers for the REPL.

Pure presentation: each function takes data and returns a Rich renderable
(``Panel``, ``Text``, or ``Table``). No I/O, no Console — the shell loop
decides where to print. Splitting them out from ``commands.py`` keeps
business logic and visual choices reviewable in isolation, and lets
snapshot tests pin the user-facing output without spinning up the loop.
"""

from __future__ import annotations

from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from agent_scaffold.costs import PreflightCost
from agent_scaffold.doctor import CheckResult, CheckStatus
from agent_scaffold.repl.session import SessionState, StatePatch
from agent_scaffold.writer import FileDiff

_MAX_DIFF_LINES_PER_FILE = 80
"""Cap on unified-diff lines rendered per file before we truncate. Above
this we render the first ``_MAX_DIFF_LINES_PER_FILE`` lines and a
``...N more lines`` tail. Stops a single noisy file from drowning the rest
of the preview."""

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


# Keys whose patch values overwrite or remove existing state — the user
# should see what was interpreted before we apply, and confirm.
_DESTRUCTIVE_KEYS: frozenset[str] = frozenset(
    {
        "recipe",
        "language",
        "framework",
        "model",
        "remove_steps",
        "remove_roles",
        "remove_capabilities",
    }
)


def _patch_field_label(value: object) -> str:
    """Short, readable rendering for any patch field value."""
    if isinstance(value, dict):
        # add_dependencies: {lang: {pkg: ver}} → "python: 2 pkg"
        parts: list[str] = []
        for lang, pkgs in value.items():
            if isinstance(pkgs, dict):
                parts.append(f"{lang}: {len(pkgs)} pkg")
            else:
                parts.append(f"{lang}={pkgs}")
        return ", ".join(parts) if parts else "{}"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    slug = getattr(value, "slug", None)
    if slug is not None:
        return str(slug)
    val = getattr(value, "value", None)
    if val is not None:
        return str(val)
    return str(value)


def render_patch_preview(patch: StatePatch) -> Panel:
    """Render the parsed refinement patch BEFORE it's applied.

    Shows every non-None field so users can audit what Haiku interpreted —
    destructive keys (model, framework, language, recipe, remove_*) render
    red so the eye catches them; additive keys render green.

    Used by :func:`agent_scaffold.repl.commands.CommandHandler._dispatch_free_text`
    to give users a chance to abort a misinterpreted refinement.
    """
    rows: list[str] = []
    for name in patch.__dataclass_fields__:
        value = getattr(patch, name)
        if value is None:
            continue
        color = "red" if name in _DESTRUCTIVE_KEYS else "green"
        rows.append(f"  [{color}]{name}[/]  [dim]→[/]  {_patch_field_label(value)}")
    body = "\n".join(rows) if rows else "[dim](empty patch)[/]"
    return Panel(
        body,
        title="Interpreted refinement",
        expand=False,
        border_style="#FF8C00",
    )


def render_file_diffs(diffs: list[FileDiff]) -> list[Panel | Text]:
    """Render per-file unified diffs for the REPL's ``/write-mode diff`` preview.

    Returns a list of renderables: one summary :class:`Text`, then one
    :class:`Panel` per modified file (capped at
    ``_MAX_DIFF_LINES_PER_FILE`` lines with a ``...N more lines`` tail).
    ``new`` and ``unchanged`` entries roll up into the summary counts so
    the user sees them without being buried under blank panels.
    """
    new = [d for d in diffs if d.status == "new"]
    modified = [d for d in diffs if d.status == "modified"]
    unchanged = [d for d in diffs if d.status == "unchanged"]
    summary = Text.from_markup(
        f"[bold]Diff preview:[/] "
        f"[green]{len(new)} new[/], "
        f"[yellow]{len(modified)} modified[/], "
        f"[dim]{len(unchanged)} unchanged[/]"
    )
    panels: list[Panel | Text] = [summary]
    for diff in modified:
        body_lines = diff.diff_text.splitlines()
        if len(body_lines) > _MAX_DIFF_LINES_PER_FILE:
            extra = len(body_lines) - _MAX_DIFF_LINES_PER_FILE
            body_lines = body_lines[:_MAX_DIFF_LINES_PER_FILE] + [
                f"...{extra} more lines"
            ]
        body = "\n".join(body_lines)
        panels.append(
            Panel(
                Syntax(body, "diff", theme="ansi_dark", line_numbers=False, word_wrap=False),
                title=diff.path,
                border_style="yellow",
                expand=False,
            )
        )
    return panels


def render_service_readiness_oneline(results: list[CheckResult]) -> Text | None:
    """One-line readiness summary for `/recipe <slug>` selection.

    Returns ``None`` when ``results`` is empty so the caller can omit the
    line entirely for recipes without ``external_services``. Format::

        Services: ok postgres (12ms)  fail qdrant (connect refused)  skip langfuse (manual)

    Status labels (plain text, no emojis to match the repo's style):

    - ``ok``   — probe succeeded (CheckStatus.OK).
    - ``warn`` — probe ran but flagged a warning (CheckStatus.WARN).
    - ``fail`` — probe failed (CheckStatus.FAIL).
    - ``skip`` — no probe configured, unknown probe, or the user disabled
      probing (CheckStatus.SKIP).
    """
    if not results:
        return None

    style_for: dict[CheckStatus, str] = {
        CheckStatus.OK: "green",
        CheckStatus.WARN: "yellow",
        CheckStatus.FAIL: "red",
        CheckStatus.SKIP: "dim",
    }

    parts: list[str] = ["[bold]Services:[/]"]
    for r in results:
        label = r.status.value
        color = style_for[r.status]
        # Service id lives at the front of CheckResult.title (formatted as
        # "{id}: ..." by every probe). Strip the prefix for a tighter line.
        name = r.title.split(":", 1)[0]
        suffix = ""
        if r.status == CheckStatus.OK and r.detail:
            # Probes record latency in detail when available.
            suffix = f" [dim]({r.detail})[/]"
        elif r.status in (CheckStatus.FAIL, CheckStatus.WARN) and r.detail:
            suffix = f" [dim]({_truncate(r.detail, 40)})[/]"
        parts.append(f"[{color}]{label}[/] {name}{suffix}")
    return Text.from_markup("  ".join(parts))


def _truncate(text: str, limit: int) -> str:
    """Inline truncator used by the readiness one-liner."""
    text = text.strip().replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_cost(preflight: PreflightCost | None) -> Text:
    """One-line cost rendering for the ``/cost`` command.

    Returns a dim line when the model is unknown so the REPL still shows
    *something* (rather than an empty Text) — the user is reminded that
    pricing data is opt-in.
    """
    if preflight is None:
        return Text.from_markup("[dim]Est. cost unavailable — model not in pricing table.[/]")
    return Text.from_markup(f"[bold]Est. cost[/] {preflight.format()}")
