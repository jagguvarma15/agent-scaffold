"""Tests for ``.agent/spec.md`` — the resolved spec the generated project
realizes (deterministic, version-controllable).
"""

from __future__ import annotations

from pathlib import Path

from agent_scaffold.capabilities import Capability, ResolvedStack
from agent_scaffold.contract import GeneratedFile, GenerationResult
from agent_scaffold.discovery import ExternalService, Recipe
from agent_scaffold.spec_artifact import render_spec, spec_path, write_spec_artifact


def _result(name: str = "demo") -> GenerationResult:
    return GenerationResult(
        project_name=name,
        language="python",
        files=[GeneratedFile(path="app/main.py", content="x = 1\n")],
        smoke_check="echo ok",
    )


def _recipe(
    *,
    agent_pattern: str | None = None,
    topology: str | None = None,
    status: str = "unknown",
    agent_role: str | None = None,
    required_files: list[str] | None = None,
    external_services: list[ExternalService] | None = None,
) -> Recipe:
    return Recipe(
        slug="support-triage",
        title="Support Triage",
        path=Path("r.md"),
        status=status,
        agent_pattern=agent_pattern,
        topology=topology,
        agent_role=agent_role,
        required_files=required_files or [],
        external_services=external_services or [],
    )


def test_render_spec_includes_recipe_pattern_topology_target() -> None:
    recipe = _recipe(
        agent_pattern="routing", topology="multi-agent-hierarchical", status="validated"
    )
    text = render_spec(
        recipe=recipe,
        language="python",
        framework="langgraph",
        model="claude-x",
        result=_result(),
        resolved_stack=None,
    )
    assert "# Agent spec — demo" in text
    assert "`support-triage`" in text
    assert "validated" in text
    assert "routing" in text
    assert "multi-agent-hierarchical" in text
    assert "python / langgraph" in text


def test_render_spec_lists_capabilities_role_files_and_env() -> None:
    recipe = _recipe(
        agent_role="You are a careful support agent.",
        required_files=["app/main.py", "README.md"],
        external_services=[ExternalService(id="anthropic", env_vars=["ANTHROPIC_API_KEY"])],
    )
    stack = ResolvedStack(
        capabilities=[
            Capability(
                id="vector_db.qdrant", kind="vector_db", path=Path("q.md"), env_vars=["QDRANT_URL"]
            ),
        ]
    )
    text = render_spec(
        recipe=recipe,
        language="python",
        framework="none",
        model="m",
        result=_result(),
        resolved_stack=stack,
    )
    assert "`vector_db.qdrant`" in text
    assert "You are a careful support agent." in text
    assert "`app/main.py`" in text
    assert "ANTHROPIC_API_KEY" in text  # from the external service
    assert "QDRANT_URL" in text  # from the capability stack


def test_render_spec_is_deterministic() -> None:
    # No timestamp / randomness — same inputs must produce byte-identical output
    # (so a committed spec.md doesn't churn on regeneration).
    recipe = _recipe(agent_pattern="react")
    result = _result()
    first = render_spec(
        recipe=recipe,
        language="python",
        framework="none",
        model="m",
        result=result,
        resolved_stack=None,
    )
    second = render_spec(
        recipe=recipe,
        language="python",
        framework="none",
        model="m",
        result=result,
        resolved_stack=None,
    )
    assert first == second


def test_render_spec_handles_empty_stack() -> None:
    text = render_spec(
        recipe=_recipe(),
        language="python",
        framework="none",
        model="m",
        result=_result(),
        resolved_stack=None,
    )
    assert "## Capabilities" in text
    assert "(none)" in text


def test_write_spec_artifact_writes_under_dot_agent(tmp_path: Path) -> None:
    path = write_spec_artifact(
        tmp_path,
        recipe=_recipe(),
        language="python",
        framework="none",
        model="m",
        result=_result(),
        resolved_stack=None,
    )
    assert path == spec_path(tmp_path)
    assert path.parent.name == ".agent"
    assert path.is_file()
    assert "# Agent spec — demo" in path.read_text(encoding="utf-8")
