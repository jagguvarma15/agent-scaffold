"""Tests for ``Recipe`` parsing of the additive 2026-SOTA frontmatter fields:
``mcp_servers``, ``skills``, ``guardrails``, ``sandbox``, ``durable_workflow``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_scaffold.discovery import (
    MCPServerSpec,
    Recipe,
    SkillSpec,
    discover_recipes,
)


def test_advanced_fields_parsed(
    mock_deployments_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    recipes = {r.slug: r for r in discover_recipes(mock_deployments_path)}
    recipe = recipes["with-advanced-fields"]

    # Two well-formed MCP servers survive; the malformed-transport and empty-id
    # entries drop with warnings.
    assert [s.id for s in recipe.mcp_servers] == ["tavily", "postgres"]
    assert recipe.mcp_servers[0] == MCPServerSpec(
        id="tavily",
        capability="mcp.tavily",
        transport="streamable_http",
        env={"TAVILY_API_KEY": "required"},
    )
    assert recipe.mcp_servers[1].transport == "stdio"

    # Two well-formed skills survive; the no-path entry drops.
    assert [s.id for s in recipe.skills] == ["web-search-loop", "citation-formatting"]
    assert recipe.skills[0] == SkillSpec(
        id="web-search-loop",
        path="skills/web-search-loop/SKILL.md",
        triggers=["research", "look up", "investigate"],
    )
    assert recipe.skills[1].triggers == []  # missing triggers field → empty default

    # guardrails: BAD_FORMAT dropped, duplicate deduped.
    assert recipe.guardrails == ["guardrail.llama-guard"]

    # sandbox + durable_workflow are scalar capability id strings.
    assert recipe.sandbox == "sandbox.e2b"
    assert recipe.durable_workflow == "durable.temporal"

    err = capsys.readouterr().err
    # The malformed entries should surface in warnings.
    assert "transport must be" in err
    assert "missing/empty 'id'" in err
    assert "missing/empty 'path'" in err
    assert "must match" in err  # the BAD_FORMAT guardrail entry
    assert "declared twice" in err


def test_advanced_fields_default_empty_when_absent(mock_deployments_path: Path) -> None:
    recipes = {r.slug: r for r in discover_recipes(mock_deployments_path)}
    # Pre-existing fixture recipe has none of the new fields — must default
    # safely.
    triage = recipes["customer-support-triage"]
    assert triage.mcp_servers == []
    assert triage.skills == []
    assert triage.guardrails == []
    assert triage.sandbox is None
    assert triage.durable_workflow is None


def test_recipe_model_defaults() -> None:
    # Direct construction with no advanced-field args must succeed.
    r = Recipe(slug="x", title="X", path=Path("/tmp/x.md"))
    assert r.mcp_servers == []
    assert r.skills == []
    assert r.guardrails == []
    assert r.sandbox is None
    assert r.durable_workflow is None


def test_sandbox_rejects_malformed_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A recipe with sandbox: 'not_dotted' must warn and fall back to None."""
    recipes_dir = tmp_path / "docs" / "recipes"
    recipes_dir.mkdir(parents=True)
    (recipes_dir / "bad-sandbox.md").write_text(
        "---\nstatus: blueprint\nlanguages: [python]\nsandbox: not_dotted\n---\n\n# Bad Sandbox\n",
        encoding="utf-8",
    )

    recipes = discover_recipes(tmp_path)
    assert recipes[0].sandbox is None
    err = capsys.readouterr().err
    assert "sandbox" in err and "must match" in err


def test_capability_id_list_accepts_string_or_list(tmp_path: Path) -> None:
    """guardrails: <single-id> as a bare string should coerce to a single-entry list."""
    recipes_dir = tmp_path / "docs" / "recipes"
    recipes_dir.mkdir(parents=True)
    (recipes_dir / "string-guardrail.md").write_text(
        "---\nstatus: blueprint\nlanguages: [python]\nguardrails: guardrail.llama-guard\n---\n\n# String Guardrail\n",
        encoding="utf-8",
    )

    recipes = discover_recipes(tmp_path)
    assert recipes[0].guardrails == ["guardrail.llama-guard"]
