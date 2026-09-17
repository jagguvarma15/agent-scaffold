"""Interactive plan-before-build: surface generation intent before paying for the LLM call."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from agent_scaffold.context import ContextSummary
from agent_scaffold.costs import PreflightCost
from agent_scaffold.doctor import CheckResult, CheckStatus
from agent_scaffold.theme import (
    BORDER_INFO,
    col,
    confirm_kwargs,
    empty,
    fail_glyph,
    info_title,
    off_glyph,
    ok_glyph,
    warn_glyph,
)
from agent_scaffold.theme import row as theme_row
from agent_scaffold.topology import Role, Topology
from agent_scaffold.writer import WriteMode

_SERVICE_ICONS: dict[CheckStatus, str] = {
    CheckStatus.OK: ok_glyph(),
    CheckStatus.WARN: warn_glyph(),
    CheckStatus.FAIL: fail_glyph(),
    CheckStatus.SKIP: off_glyph(),
}

# An unknown status renders as the neutral empty marker, not a bare "?"
# that reads like a rendering bug.
_SERVICE_ICON_FALLBACK = empty()


def _row(label: str, value: str) -> str:
    """One aligned ``label value`` plan row (value column at width 13)."""
    return theme_row(label, value, width=13)


# Recipe frontmatter may carry ONE flat required_files list for every
# language, so a TypeScript run can preview app/main.py-style paths; the
# Files heading then says the list is illustrative rather than letting the
# preview silently lie. Vestigial when the recipe declares
# required_files_by_language — the exact per-language list can't mismatch.
_OTHER_LANG_EXTS: dict[str, tuple[str, ...]] = {
    "python": (".ts", ".tsx", ".js", ".jsx"),
    "typescript": (".py",),
}


class GenerationPlan(BaseModel):
    # CheckResult is a frozen dataclass, not a Pydantic type; allow it through.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    recipe_slug: str
    recipe_status: str
    language: str
    framework: str
    project_name: str
    dest: Path
    topology: Topology
    roles: list[Role] = Field(default_factory=list)
    model: str
    max_tokens: int
    thinking_budget: int | None = None
    required_files: list[str] = Field(default_factory=list)
    context_summary: ContextSummary | None = None
    write_mode: WriteMode = WriteMode.abort
    warnings: list[str] = Field(default_factory=list)
    strict: bool = False
    service_readiness: list[CheckResult] = Field(default_factory=list)
    preflight_cost: PreflightCost | None = None
    stack: list[str] = Field(default_factory=list)
    """Resolved capability ids annotated with their delivery mode
    (``(docker)`` / ``(cloud hosted - connect <option> after generation)``)."""

    def render(self) -> Panel:
        rows: list[str] = [
            _row("Recipe", f"{escape(self.recipe_slug)} ({escape(self.recipe_status)})"),
            _row("Language", escape(self.language)),
            _row("Framework", escape(self.framework)),
            _row(
                "Topology",
                self.topology.value + (f" — {len(self.roles)} role(s)" if self.roles else ""),
            ),
        ]
        for role in self.roles:
            model_for_role = role.model_hint or self.model
            rows.append(f"  • {col(role.name, 14)} {escape(model_for_role)}")
        rows.append(_row("Output", escape(str(self.dest))))
        if self.stack:
            rows.append("[bold]Stack[/]")
            for entry in self.stack:
                rows.append(f"  • {escape(entry)}")
        if self.context_summary is not None:
            rows.append(
                _row(
                    "Context",
                    f"{sum(t.docs for t in self.context_summary.tiers)} docs, "
                    f"~{self.context_summary.total_tokens:,} tokens "
                    f"(cap {self.context_summary.cap:,})",
                )
            )
            non_empty = [t for t in self.context_summary.tiers if t.docs > 0]
            if non_empty:
                label_width = max(len(t.label) for t in non_empty)
                for tier in non_empty:
                    rows.append(
                        f"  [dim]{escape(tier.label.ljust(label_width))}[/]  "
                        f"{tier.docs:>2} docs, {tier.tokens:>7,} tk"
                    )
        rows.append(
            _row(
                "Model",
                f"{escape(self.model)}, max {self.max_tokens:,} out"
                + (f", thinking {self.thinking_budget:,}" if self.thinking_budget else "")
                + (", strict prompt" if self.strict else ""),
            )
        )
        if self.required_files:
            visible = escape(", ".join(self.required_files[:6]))
            more = (
                f", … (+{len(self.required_files) - 6} more)"
                if len(self.required_files) > 6
                else ""
            )
            other_exts = _OTHER_LANG_EXTS.get(self.language, ())
            mismatched = any(f.endswith(other_exts) for f in self.required_files if other_exts)
            if mismatched:
                heading = (
                    f"[bold]Files[/] [dim](recipe manifest — actual paths follow "
                    f"{escape(self.language)})[/]"
                )
                rows.append(f"{heading} {visible}{more}")
            else:
                rows.append(_row("Files", f"{visible}{more}"))
        if self.service_readiness:
            rows.append("[bold]Service readiness[/]")
            for r in self.service_readiness:
                icon = _SERVICE_ICONS.get(r.status, _SERVICE_ICON_FALLBACK)
                name = r.id.removeprefix("service.")
                rows.append(f"  {icon} {col(name, 14)} {escape(r.title)}")
                if r.detail:
                    rows.append(f"      [dim]{escape(r.detail)}[/]")
                if r.status in (CheckStatus.FAIL, CheckStatus.WARN) and r.fix_hint:
                    rows.append(f"      [dim]→[/] {escape(r.fix_hint)}")
        if self.preflight_cost is not None:
            rows.append(_row("Est. cost", self.preflight_cost.format()))
        if self.warnings:
            rows.append("[yellow]Warnings[/]")
            for warning in self.warnings:
                rows.append(f"  • {escape(warning)}")
        return Panel(
            "\n".join(rows),
            title=info_title("Generation plan"),
            expand=False,
            border_style=BORDER_INFO,
        )


def confirm(plan: GenerationPlan, console: Console) -> bool:
    """Render the plan and prompt Y/n. Returns ``True`` if the user accepted."""
    console.print(plan.render())
    try:
        import questionary

        answer = questionary.confirm(
            "Proceed with this plan?", default=True, **confirm_kwargs()
        ).ask()
    except KeyboardInterrupt:
        return False
    return bool(answer)
