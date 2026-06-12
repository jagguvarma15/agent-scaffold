"""Consolidated post-generation report panel.

Replaces the pre-Phase-3 layout of three separate panels (``Run summary``,
``Phase summary``, plus a free-floating ``Generated N files`` line) with a
single Rich ``Panel`` that groups everything by intent:

  1. **Selections** — what the user chose (recipe / language / framework /
     observability backend).
  2. **Generation** — tokens / cost / wall time / cache hit ratio.
  3. **Files** — how many landed where (written / overwritten / skipped),
     with a sample of the first few paths.
  4. **Phases** — compact per-phase wall times.
  5. **Notes** — warnings + errors from the run, if any.

The renderer is pure (takes a :class:`GenerationReport` dataclass and
returns a Rich renderable) so tests can snapshot it without spinning up a
real console.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agent_scaffold.branding import ACCENT, ACCENT_DIM, MUTED, OK, PANEL_BORDER_STYLE, WARN
from agent_scaffold.costs import estimate as estimate_cost

if TYPE_CHECKING:
    from agent_scaffold.capabilities import ResolvedStack


_TOP_FILES_LIMIT = 6


@dataclass(frozen=True)
class GenerationReport:
    """Data for one consolidated post-generation panel.

    Frozen so tests can construct fixed instances and snapshot the render.
    All fields except ``recipe_slug`` and ``model`` have sensible empty
    defaults — callers fill what they have; missing sections silently drop.
    """

    # Selections
    recipe_slug: str
    language: str = ""
    framework: str = ""
    observability: str = ""  # "langsmith" | "langfuse" | "none" | "" (recipe default)
    # Generation
    model: str = ""
    wall_seconds: float = 0.0
    cached: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    # Files
    files_written: int = 0
    files_overwritten: int = 0
    files_skipped: int = 0
    top_files: list[str] = field(default_factory=list)
    # Validation-repair rounds that ran (0 = passed first time). Token/cost
    # numbers above already include the repair calls.
    repair_rounds: int = 0
    # Phases + diagnostics
    phase_durations: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    # Where the persistent run.log / events.jsonl for this run live ("" = none).
    run_log_dir: str = ""

    def render(self) -> Panel:
        """Build the consolidated Rich panel."""
        sections: list[RenderableType] = []
        sel = _render_selections(self)
        if sel is not None:
            sections.append(sel)
        gen = _render_generation(self)
        if gen is not None:
            sections.append(gen)
        files = _render_files(self)
        if files is not None:
            sections.append(files)
        phases = _render_phases(self)
        if phases is not None:
            sections.append(phases)
        notes = _render_notes(self)
        if notes is not None:
            sections.append(notes)
        if self.run_log_dir:
            sections.append(Text.from_markup(f"\n[dim]Run log: {self.run_log_dir}[/]"))
        return Panel(
            Group(*sections),
            title=f"[bold {ACCENT}]Generation report[/]",
            border_style=PANEL_BORDER_STYLE,
            expand=False,
            padding=(0, 1),
        )


def print_generation_report(report: GenerationReport, console: Console | None = None) -> None:
    """Print the consolidated post-generation panel.

    The Console defaults to a freshly-constructed one when not passed, so
    tests + non-CLI callers can use it without importing ``cli_shared``.
    """
    if console is None:
        from agent_scaffold.cli_shared import console as default_console

        console = default_console
    console.print(report.render())


def derive_observability(stack: ResolvedStack | None) -> str:
    """Render the chosen observability backend as 'langsmith' | 'langfuse' | 'none'.

    Looks at the resolved stack rather than ``SessionState`` so the report
    surface reflects what *actually* got into the project, not what the
    user typed. Returns ``""`` when nothing observability-related is on
    the stack at all (e.g. a recipe that doesn't ship obs).
    """
    if stack is None:
        return ""
    ids = stack.ids()
    if "obs.langfuse" in ids:
        return "langfuse"
    if "obs.langsmith" in ids:
        return "langsmith"
    if "obs.grafana-stack" in ids:
        return "grafana-stack"
    return ""


# ---------------------------------------------------------------------------
# Section renderers (private)
# ---------------------------------------------------------------------------


def _section_header(label: str) -> Text:
    return Text.from_markup(f"\n[bold {ACCENT_DIM}]{label}[/]")


def _render_selections(report: GenerationReport) -> RenderableType | None:
    """One-row-per-selection mini-table; skipped entirely if nothing's set."""
    rows = [
        ("Recipe", report.recipe_slug),
        ("Language", report.language),
        ("Framework", report.framework),
        ("Observability", report.observability or "[dim](recipe default)[/]"),
    ]
    visible = [(k, v) for k, v in rows if v]
    if not visible:
        return None
    table = Table.grid(padding=(0, 2))
    table.add_column(style=MUTED, justify="right")
    table.add_column()
    for label, value in visible:
        table.add_row(label, value)
    return Group(_section_header("Selections"), table)


def _render_generation(report: GenerationReport) -> RenderableType | None:
    """Tokens / cost / wall time. Skipped when no usage was recorded."""
    if not report.model or (report.input_tokens == 0 and report.output_tokens == 0):
        return None
    mins, secs = divmod(int(report.wall_seconds), 60)
    wall_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
    cache_total = report.cache_read_tokens + report.cache_creation_tokens
    cache_str = ""
    if cache_total:
        denom = max(1, report.input_tokens + cache_total)
        pct = int(100 * report.cache_read_tokens / denom)
        cache_str = f"  [dim]cache hit {pct}%[/]"
    lines = [
        f"[{MUTED}]Model:[/] {report.model}" + ("  [dim][cached][/]" if report.cached else ""),
        f"[{MUTED}]Tokens:[/] {report.input_tokens:,} in / {report.output_tokens:,} out"
        + cache_str,
        f"[{MUTED}]Wall:[/] {wall_str}",
    ]
    if report.repair_rounds:
        lines.append(
            f"[{MUTED}]Repair:[/] {report.repair_rounds} round(s) "
            "[dim](tokens/cost above include the repair calls)[/]"
        )
    cost = estimate_cost(
        report.model,
        input_tokens=report.input_tokens,
        output_tokens=report.output_tokens,
        cache_read_tokens=report.cache_read_tokens,
        cache_write_tokens=report.cache_creation_tokens,
    )
    if cost is not None:
        lines.append(
            f"[{MUTED}]Cost:[/] [bold {OK}]${cost.total:.2f}[/] "
            f"[dim](in ${cost.input_uncached:.2f} / out ${cost.output:.2f}"
            f" / cache r ${cost.cache_read:.2f} w ${cost.cache_write:.2f})[/]"
        )
    return Group(_section_header("Generation"), Text.from_markup("\n".join(lines)))


def _render_files(report: GenerationReport) -> RenderableType | None:
    total = report.files_written + report.files_overwritten + report.files_skipped
    if total == 0 and not report.top_files:
        return None
    summary = (
        f"[bold]{report.files_written}[/] new"
        + (
            f" · [bold]{report.files_overwritten}[/] overwritten"
            if report.files_overwritten
            else ""
        )
        + (f" · [{MUTED}]{report.files_skipped} skipped[/]" if report.files_skipped else "")
    )
    lines = [summary]
    sample = report.top_files[:_TOP_FILES_LIMIT]
    if sample:
        for path in sample:
            lines.append(f"  [{MUTED}]·[/] {path}")
        remaining = max(0, len(report.top_files) - _TOP_FILES_LIMIT)
        if remaining:
            lines.append(f"  [{MUTED}]…and {remaining} more[/]")
    return Group(_section_header("Files"), Text.from_markup("\n".join(lines)))


def _render_phases(report: GenerationReport) -> RenderableType | None:
    if not report.phase_durations:
        return None
    table = Table.grid(padding=(0, 2))
    table.add_column(style=MUTED, justify="right")
    table.add_column()
    for name, secs in report.phase_durations.items():
        mins, s = divmod(int(secs), 60)
        label = f"{mins}m {s:02d}s" if mins else f"{secs:.1f}s"
        table.add_row(name, label)
    return Group(_section_header("Phases"), table)


def _render_notes(report: GenerationReport) -> RenderableType | None:
    if not report.warnings and not report.errors:
        return None
    lines: list[str] = []
    if report.warnings:
        lines.append(f"[bold {WARN}]Warnings[/]")
        for w in report.warnings:
            lines.append(f"  ⚠ {w}")
    if report.errors:
        if lines:
            lines.append("")
        lines.append("[bold red]Errors[/]")
        for e in report.errors:
            lines.append(f"  ✗ {e}")
    return Group(_section_header("Notes"), Text.from_markup("\n".join(lines)))


__all__ = ["GenerationReport", "derive_observability", "print_generation_report"]
