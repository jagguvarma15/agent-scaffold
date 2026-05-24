"""Tests for agent_scaffold.plan."""

from __future__ import annotations

import io
from pathlib import Path

from rich.console import Console

from agent_scaffold.context import ContextSummary, TierStats
from agent_scaffold.plan import GenerationPlan
from agent_scaffold.topology import Role, Topology
from agent_scaffold.writer import WriteMode


def _plan(**overrides) -> GenerationPlan:  # type: ignore[no-untyped-def]
    base = dict(
        recipe_slug="restaurant-rebooking",
        recipe_status="blueprint",
        language="python",
        framework="langgraph",
        project_name="rebooking",
        dest=Path("/tmp/rebooking"),
        topology=Topology.MULTI,
        roles=[
            Role(name="intake", model_hint="sonnet"),
            Role(name="notifier", model_hint="haiku"),
        ],
        model="claude-opus-4-7",
        max_tokens=64000,
        thinking_budget=16000,
        required_files=["pyproject.toml", "Dockerfile", "tests/test_intake.py"],
        context_summary=ContextSummary(
            total_tokens=78_200,
            cap=80_000,
            tiers=[TierStats(tier=1, label="Recipe", docs=1, tokens=2_000)],
            dropped=[],
            truncated=[],
        ),
        write_mode=WriteMode.abort,
        warnings=["Context is 97% of cap"],
        strict=True,
    )
    base.update(overrides)
    return GenerationPlan(**base)  # type: ignore[arg-type]


def _render(plan: GenerationPlan) -> str:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=120)
    console.print(plan.render())
    return buf.getvalue()


def test_plan_render_shows_recipe_topology_and_roles() -> None:
    out = _render(_plan())
    assert "restaurant-rebooking" in out
    assert "multi-agent-flat" in out
    assert "intake" in out
    assert "notifier" in out
    assert "sonnet" in out
    assert "claude-opus-4-7" in out


def test_plan_render_shows_warnings() -> None:
    out = _render(_plan())
    assert "Warnings" in out
    assert "97%" in out


def test_plan_render_truncates_long_required_files_list() -> None:
    plan = _plan(required_files=[f"src/file{i}.py" for i in range(20)])
    out = _render(plan)
    assert "+14 more" in out


def test_plan_render_single_topology_omits_role_lines() -> None:
    plan = _plan(topology=Topology.SINGLE, roles=[], required_files=[])
    out = _render(plan)
    # SINGLE topology with no roles → no role bullets and no "n role(s)" suffix.
    assert "role(s)" not in out
    assert "single" in out


def test_plan_render_omits_thinking_when_disabled() -> None:
    plan = _plan(thinking_budget=None)
    out = _render(plan)
    assert "thinking" not in out
