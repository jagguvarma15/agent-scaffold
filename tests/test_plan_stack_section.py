"""Tests for the Phase-2 follow-up: Tier row + Stack section in the plan panel."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

from rich.console import Console

from agent_scaffold.capabilities import Capability, ResolvedStack
from agent_scaffold.context import ContextSummary
from agent_scaffold.plan import GenerationPlan, _render_stack_rows
from agent_scaffold.topology import Topology


def _cap(cap_id: str, kind: str) -> Capability:
    return Capability(id=cap_id, kind=kind, path=Path(f"/tmp/{cap_id}.md"))  # type: ignore[arg-type]


def _plan(
    *,
    tier: str = "",
    resolved_stack: ResolvedStack | None = None,
) -> GenerationPlan:
    return GenerationPlan(
        recipe_slug="demo",
        recipe_status="blueprint",
        language="python",
        framework="langgraph",
        project_name="demo",
        dest=Path("/tmp/demo"),
        topology=Topology.SINGLE,
        model="claude-opus-4-7",
        max_tokens=8000,
        tier=tier,
        resolved_stack=resolved_stack,
        context_summary=ContextSummary(
            total_tokens=1000, cap=10000, tiers=[], dropped=[], truncated=[]
        ),
    )


def _render_text(plan: GenerationPlan) -> str:
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=120)
    console.print(plan.render())
    return buf.getvalue()


def test_render_stack_rows_empty_stack_returns_no_rows() -> None:
    assert _render_stack_rows(None) == []
    assert _render_stack_rows(ResolvedStack()) == []


def test_render_stack_rows_groups_by_layer_order() -> None:
    stack = ResolvedStack(
        capabilities=[
            _cap("frontend.nextjs-chat", "frontend"),
            _cap("relational.postgres", "relational"),
            _cap("obs.langfuse", "obs"),
            _cap("tools.filesystem", "tools"),
        ]
    )
    rows = _render_stack_rows(stack)
    # Header + 4 layer rows.
    assert rows[0] == "[bold]Stack[/]"
    # Order follows LAYER_ORDER: relational → tools → obs → frontend.
    assert "relational" in rows[1]
    assert "tools" in rows[2]
    assert "obs" in rows[3]
    assert "frontend" in rows[4]


def test_render_panel_includes_tier_row_when_set() -> None:
    text = _render_text(_plan(tier="complex"))
    assert "Tier" in text
    assert "complex" in text


def test_render_panel_omits_tier_row_when_blank() -> None:
    text = _render_text(_plan())
    # The Tier label should not appear when tier is empty (default).
    assert "Tier" not in text


def test_render_panel_includes_stack_section_with_layered_capabilities() -> None:
    stack = ResolvedStack(
        capabilities=[
            _cap("relational.postgres", "relational"),
            _cap("vector_db.qdrant", "vector_db"),
            _cap("obs.langsmith", "obs"),
        ]
    )
    text = _render_text(_plan(resolved_stack=stack))
    assert "Stack" in text
    assert "relational" in text
    assert "relational.postgres" in text
    assert "vector_db.qdrant" in text
    assert "obs.langsmith" in text


def test_render_panel_omits_stack_section_when_resolved_stack_is_empty() -> None:
    text = _render_text(_plan(resolved_stack=ResolvedStack()))
    # No "Stack" header line; existing sections (Recipe etc.) still render.
    assert "Stack" not in text
    assert "Recipe" in text
