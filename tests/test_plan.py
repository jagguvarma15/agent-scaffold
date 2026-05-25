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


# ---------------------------------------------------------------------------
# Service readiness section (Q3)
# ---------------------------------------------------------------------------


def test_plan_render_includes_service_readiness_when_set() -> None:
    from agent_scaffold.doctor import CheckResult, CheckStatus

    readiness = [
        CheckResult(
            id="service.redis",
            category="Recipe services",
            status=CheckStatus.OK,
            title="redis: PING ok (localhost:6379)",
        ),
        CheckResult(
            id="service.postgres",
            category="Recipe services",
            status=CheckStatus.FAIL,
            title="postgres: connection failed",
            detail="ConnectionRefusedError",
            fix_hint="docker compose up -d postgres",
        ),
    ]
    out = _render(_plan(service_readiness=readiness))
    assert "Service readiness" in out
    assert "redis" in out
    assert "postgres" in out
    # FAIL status surfaces the fix hint.
    assert "docker compose up" in out


def test_plan_render_skips_service_section_when_empty() -> None:
    out = _render(_plan(service_readiness=[]))
    assert "Service readiness" not in out


def test_plan_concurrent_probes_run_within_two_timeouts() -> None:
    """Probes must run in a thread pool — wall time ≤ 2× per-probe timeout."""
    import time

    from agent_scaffold import probes
    from agent_scaffold.cli import _probe_services_for_plan
    from agent_scaffold.discovery import ExternalService

    services = [ExternalService(id=f"svc-{i}", probe="redis_ping") for i in range(4)]

    def slow_probe(svc, *, timeout=5.0, skip=False):  # type: ignore[no-untyped-def]
        from agent_scaffold.doctor import CheckResult, CheckStatus

        time.sleep(0.5)
        return CheckResult(
            id=f"service.{svc.id}",
            category="Recipe services",
            status=CheckStatus.OK,
            title=f"{svc.id}: slow ok",
        )

    original = probes.run_probe
    probes.run_probe = slow_probe  # type: ignore[assignment]
    try:
        start = time.perf_counter()
        results = _probe_services_for_plan(services, probe_services=True, timeout=1.0)
        elapsed = time.perf_counter() - start
    finally:
        probes.run_probe = original  # type: ignore[assignment]
    assert len(results) == 4
    # Serial would be ~2.0s; with pool max_workers=4 we expect well under 1.5s.
    assert elapsed < 1.5, f"probes ran serially: {elapsed:.2f}s"
