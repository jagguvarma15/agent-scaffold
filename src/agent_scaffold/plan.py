"""Interactive plan-before-build: surface generation intent before paying for the LLM call."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel

from agent_scaffold.context import ContextSummary
from agent_scaffold.topology import Role, Topology
from agent_scaffold.writer import WriteMode


class GenerationPlan(BaseModel):
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
