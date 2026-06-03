"""Interactive plan-before-build: surface generation intent before paying for the LLM call."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from rich.panel import Panel

from agent_scaffold.capabilities import LAYER_ORDER, ResolvedStack
from agent_scaffold.context import ContextSummary
from agent_scaffold.costs import PreflightCost
from agent_scaffold.doctor import CheckResult, CheckStatus
from agent_scaffold.topology import Role, Topology
from agent_scaffold.writer import WriteMode

# Tier label → colour. Mirrors report.py so the pre-gen plan and post-gen
# report panels share the same visual language for complexity.
_TIER_COLORS: dict[str, str] = {"basic": "green", "mid": "#FFA500", "complex": "yellow"}

_SERVICE_ICONS: dict[CheckStatus, str] = {
    CheckStatus.OK: "[green]✓[/]",
    CheckStatus.WARN: "[yellow]⚠[/]",
    CheckStatus.FAIL: "[red]✗[/]",
    CheckStatus.SKIP: "[dim cyan]⏭[/]",
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
    tier: str = ""
    """Complexity tier: ``basic`` / ``mid`` / ``complex``. Surfaced as a
    coloured row in the plan panel so users see the agent shape before
    paying for generation."""
    resolved_stack: ResolvedStack | None = None
    """The capability stack the orchestrator will provision. Renders as the
    Stack section, grouped by ``LAYER_ORDER``."""

    def render(self) -> Panel:
        rows: list[str] = [
            f"[bold]Recipe[/]       {self.recipe_slug} ({self.recipe_status})",
        ]
        if self.tier:
            color = _TIER_COLORS.get(self.tier, "white")
            rows.append(f"[bold]Tier[/]         [bold {color}]{self.tier}[/]")
        rows.extend(
            [
                f"[bold]Language[/]     {self.language}",
                f"[bold]Framework[/]    {self.framework}",
                f"[bold]Topology[/]     {self.topology.value}"
                + (f" — {len(self.roles)} role(s)" if self.roles else ""),
            ]
        )
        for role in self.roles:
            model_for_role = role.model_hint or self.model
            rows.append(f"  • {role.name:<14} {model_for_role}")
        rows.extend(_render_stack_rows(self.resolved_stack))
        rows.append(f"[bold]Output[/]       {self.dest}")
        if self.context_summary is not None:
            rows.append(
                f"[bold]Context[/]      {sum(t.docs for t in self.context_summary.tiers)} docs, "
                f"~{self.context_summary.total_tokens:,} tokens "
                f"(cap {self.context_summary.cap:,})"
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
            rows.append(f"[bold]Files[/]        {visible}{more}")
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
