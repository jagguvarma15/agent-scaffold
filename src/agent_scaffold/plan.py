"""Interactive plan-before-build: surface generation intent before paying for the LLM call."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from rich.panel import Panel

from agent_scaffold.context import ContextSummary
from agent_scaffold.costs import PreflightCost
from agent_scaffold.doctor import CheckResult, CheckStatus
from agent_scaffold.topology import Role, Topology
from agent_scaffold.writer import WriteMode

_SERVICE_ICONS: dict[CheckStatus, str] = {
    CheckStatus.OK: "[green]✓[/]",
    CheckStatus.WARN: "[yellow]⚠[/]",
    CheckStatus.FAIL: "[red]✗[/]",
    CheckStatus.SKIP: "[dim cyan]⏭[/]",
}

# Recipe frontmatter carries ONE required_files list regardless of the picked
# language, so a TypeScript run can preview app/main.py-style paths. When any
# listed path wears the other language's extension, the Files heading says the
# list is illustrative rather than letting the preview silently lie.
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
            f"[bold]Recipe[/]       {self.recipe_slug} ({self.recipe_status})",
            f"[bold]Language[/]     {self.language}",
            f"[bold]Framework[/]    {self.framework}",
            f"[bold]Topology[/]     {self.topology.value}"
            + (f" — {len(self.roles)} role(s)" if self.roles else ""),
        ]
        for role in self.roles:
            model_for_role = role.model_hint or self.model
            rows.append(f"  • {role.name:<14} {model_for_role}")
        rows.append(f"[bold]Output[/]       {self.dest}")
        if self.stack:
            rows.append("[bold]Stack[/]")
            for entry in self.stack:
                rows.append(f"  • {entry}")
        if self.context_summary is not None:
            rows.append(
                f"[bold]Context[/]      {sum(t.docs for t in self.context_summary.tiers)} docs, "
                f"~{self.context_summary.total_tokens:,} tokens "
                f"(cap {self.context_summary.cap:,})"
            )
            non_empty = [t for t in self.context_summary.tiers if t.docs > 0]
            if non_empty:
                label_width = max(len(t.label) for t in non_empty)
                for tier in non_empty:
                    rows.append(
                        f"  [dim]{tier.label.ljust(label_width)}[/]  "
                        f"{tier.docs:>2} docs, {tier.tokens:>7,} tk"
                    )
        rows.append(
            f"[bold]Model[/]        {self.model}, max {self.max_tokens:,} out"
            + (f", thinking {self.thinking_budget:,}" if self.thinking_budget else "")
            + (", strict prompt" if self.strict else "")
        )
        if self.required_files:
            visible = ", ".join(self.required_files[:6])
            more = (
                f", … (+{len(self.required_files) - 6} more)"
                if len(self.required_files) > 6
                else ""
            )
            other_exts = _OTHER_LANG_EXTS.get(self.language, ())
            mismatched = any(f.endswith(other_exts) for f in self.required_files if other_exts)
            heading = (
                f"[bold]Files[/] [dim](recipe manifest — actual paths follow {self.language})[/]"
                if mismatched
                else "[bold]Files[/]       "
            )
            rows.append(f"{heading} {visible}{more}")
        if self.service_readiness:
            rows.append("[bold]Service readiness[/]")
            for r in self.service_readiness:
                icon = _SERVICE_ICONS.get(r.status, "?")
                name = r.id.removeprefix("service.")
                line = f"  {icon} {name:<14} {r.title}"
                rows.append(line)
                if r.detail:
                    rows.append(f"      [dim]{r.detail}[/]")
                if r.status in (CheckStatus.FAIL, CheckStatus.WARN) and r.fix_hint:
                    rows.append(f"      [dim]→[/] {r.fix_hint}")
        if self.preflight_cost is not None:
            rows.append(f"[bold]Est. cost[/]    {self.preflight_cost.format()}")
        if self.warnings:
            rows.append("[yellow]Warnings[/]")
            for warning in self.warnings:
                rows.append(f"  • {warning}")
        return Panel("\n".join(rows), title="Generation plan", expand=False)


def confirm(plan: GenerationPlan, console: Console) -> bool:
    """Render the plan and prompt Y/n. Returns ``True`` if the user accepted."""
    console.print(plan.render())
    try:
        import questionary

        answer = questionary.confirm("Proceed with this plan?", default=True).ask()
    except KeyboardInterrupt:
        return False
    return bool(answer)
