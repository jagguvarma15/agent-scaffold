"""Tests for the capability-aware prompt rendering in ``generator``."""

from __future__ import annotations

from pathlib import Path

from agent_scaffold.context import AssembledContext
from agent_scaffold.generator import (
    GenerationRequest,
    _render_capabilities_block,
    _render_user_message,
)


def _request(capabilities_brief: list[dict[str, object]] | None = None) -> GenerationRequest:
    ctx = AssembledContext(
        recipe_path=Path("/fake/recipe.md"),
        referenced_paths=[],
        body="# recipe body",
        token_estimate=10,
        summary=None,
    )
    return GenerationRequest(
        project_name="demo",
        target_language="python",
        framework="langgraph",
        assembled_context=ctx,
        language_hints={"manifest": "pyproject.toml", "entry_point": "{project_name}/main.py"},
        capabilities_brief=capabilities_brief or [],
    )


def test_capabilities_block_empty_when_no_caps() -> None:
    assert _render_capabilities_block(_request()) == ""


def test_capabilities_block_lists_env_and_docker() -> None:
    req = _request(
        [
            {
                "id": "vector_db.qdrant",
                "kind": "vector_db",
                "env_vars": ["QDRANT_URL", "QDRANT_API_KEY"],
                "docker_service": "qdrant",
                "emit_globs": [],
            },
            {
                "id": "frontend.nextjs-chat",
                "kind": "frontend",
                "env_vars": ["NEXT_PUBLIC_AGENT_URL"],
                "docker_service": None,
                "emit_globs": ["frontend/**"],
            },
        ]
    )
    block = _render_capabilities_block(req)
    assert "vector_db.qdrant" in block
    assert "`QDRANT_URL`" in block and "`QDRANT_API_KEY`" in block
    assert "`qdrant`" in block
    assert "frontend.nextjs-chat" in block
    assert "frontend/**" in block
    assert "do NOT re-emit" in block


def test_user_message_renders_capabilities_block_inline() -> None:
    req = _request(
        [
            {
                "id": "cache.redis",
                "kind": "cache",
                "env_vars": ["REDIS_URL"],
                "docker_service": "redis",
                "emit_globs": [],
            }
        ]
    )
    context, tail = _render_user_message(req)
    assert "# Resolved capabilities" in tail
    assert "cache.redis" in tail
    # No leftover placeholder.
    assert "{capabilities_block}" not in context
    assert "{capabilities_block}" not in tail


def test_user_message_template_omits_block_when_empty() -> None:
    context, tail = _render_user_message(_request())
    assert "# Resolved capabilities" not in (context + tail)
    assert "{capabilities_block}" not in (context + tail)


# ---------------------------------------------------------------------------
# MCP servers block
# ---------------------------------------------------------------------------


def _mcp_request(brief: list[dict[str, object]] | None = None) -> GenerationRequest:
    req = _request()
    return req.model_copy(update={"mcp_servers_brief": brief or []})


def test_mcp_block_empty_when_no_servers() -> None:
    from agent_scaffold.generator import _render_mcp_block

    assert _render_mcp_block(_mcp_request()) == ""


def test_mcp_block_lists_servers_and_instructions() -> None:
    from agent_scaffold.generator import _render_mcp_block

    block = _render_mcp_block(
        _mcp_request(
            [
                {
                    "id": "arrowhead",
                    "capability": "mcp.arrowhead",
                    "transport": "streamable_http",
                    "endpoint": "http://127.0.0.1:8004/mcp",
                    "env_vars": ["ARROWHEAD_API_KEY"],
                }
            ]
        )
    )
    assert "# MCP servers" in block
    assert "mcp.arrowhead" in block
    assert "http://127.0.0.1:8004/mcp" in block
    assert "`ARROWHEAD_API_KEY`" in block
    assert "do NOT" in block
    assert "mcp.json" in block


def test_user_message_renders_mcp_block_in_the_tail_only() -> None:
    req = _mcp_request(
        [
            {
                "id": "tavily",
                "capability": "mcp.tavily",
                "transport": "streamable_http",
                "endpoint": "https://mcp.tavily.example/mcp/",
                "env_vars": ["TAVILY_API_KEY"],
            }
        ]
    )
    context, tail = _render_user_message(req)
    assert "# MCP servers" in tail
    assert "# MCP servers" not in context
    assert "{mcp_block}" not in (context + tail)


def test_user_message_omits_mcp_block_when_empty() -> None:
    context, tail = _render_user_message(_mcp_request())
    assert "# MCP servers" not in (context + tail)
    assert "{mcp_block}" not in (context + tail)
