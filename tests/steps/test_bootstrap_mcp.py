"""Tests for the ``bootstrap_mcp`` step."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

from agent_scaffold.discovery import MCPServerSpec, Recipe
from agent_scaffold.manifest import Manifest
from agent_scaffold.orchestrator import StepContext, StepEvent, StepStatus
from agent_scaffold.steps.bootstrap_mcp import (
    MCP_REGISTRY_FILENAME,
    BootstrapMcpStep,
    build_registry,
)


def _stack(*caps: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(capabilities=list(caps), unresolved=[])


def _tavily_cap() -> SimpleNamespace:
    return SimpleNamespace(
        id="mcp.tavily",
        endpoint="https://mcp.tavily.example/mcp/",
        env_vars=["TAVILY_API_KEY"],
    )


def _tavily_server() -> MCPServerSpec:
    return MCPServerSpec(
        id="tavily",
        capability="mcp.tavily",
        transport="streamable_http",
        env={"TAVILY_API_KEY": "required", "TAVILY_REGION": "hint"},
    )


def test_detect_skips_without_mcp_servers(
    ctx_factory: Callable[..., StepContext],
    recipe_factory: Callable[..., Recipe],
    patch_load_recipe: Callable[[Recipe | None], None],
) -> None:
    patch_load_recipe(recipe_factory())
    outcome = BootstrapMcpStep().detect(ctx_factory())
    assert outcome.status is StepStatus.SKIPPED
    assert "no mcp_servers" in outcome.reason


def test_apply_writes_the_streamable_http_entry(
    ctx_factory: Callable[..., StepContext],
    recipe_factory: Callable[..., Recipe],
    patch_load_recipe: Callable[[Recipe | None], None],
    tmp_path: Path,
) -> None:
    patch_load_recipe(recipe_factory(mcp_servers=[_tavily_server()]))
    ctx = ctx_factory(
        resolved_stack=_stack(_tavily_cap()),
        runtime_env={"TAVILY_API_KEY": "sekret-value"},
    )
    result = BootstrapMcpStep().apply(ctx)
    assert result.status is StepStatus.DONE

    written = json.loads((tmp_path / MCP_REGISTRY_FILENAME).read_text(encoding="utf-8"))
    entry = written["mcpServers"]["tavily"]
    assert written["version"] == 1
    assert entry["capability"] == "mcp.tavily"
    assert entry["transport"] == "streamable_http"
    assert entry["url"] == "https://mcp.tavily.example/mcp/"
    assert entry["headers"] == {"Authorization": "Bearer ${TAVILY_API_KEY}"}
    assert entry["env"] == {
        "TAVILY_API_KEY": "${TAVILY_API_KEY}",
        "TAVILY_REGION": "${TAVILY_REGION}",
    }
    assert entry["required_env"] == ["TAVILY_API_KEY"]
    assert entry["optional_env"] == ["TAVILY_REGION"]
    # Placeholders only: the runtime secret value must never reach the file.
    raw = (tmp_path / MCP_REGISTRY_FILENAME).read_text(encoding="utf-8")
    assert "sekret-value" not in raw


def test_stdio_entry_carries_a_launcher_slot(
    ctx_factory: Callable[..., StepContext],
    recipe_factory: Callable[..., Recipe],
    patch_load_recipe: Callable[[Recipe | None], None],
    tmp_path: Path,
) -> None:
    server = MCPServerSpec(id="local", capability="mcp.local", transport="stdio")
    patch_load_recipe(recipe_factory(mcp_servers=[server]))
    BootstrapMcpStep().apply(ctx_factory(resolved_stack=_stack()))
    entry = json.loads((tmp_path / MCP_REGISTRY_FILENAME).read_text(encoding="utf-8"))[
        "mcpServers"
    ]["local"]
    assert entry["command"] is None
    assert entry["args"] == []
    assert "url" not in entry


def test_apply_then_detect_is_idempotent(
    ctx_factory: Callable[..., StepContext],
    recipe_factory: Callable[..., Recipe],
    patch_load_recipe: Callable[[Recipe | None], None],
) -> None:
    patch_load_recipe(recipe_factory(mcp_servers=[_tavily_server()]))
    ctx = ctx_factory(resolved_stack=_stack(_tavily_cap()))
    step = BootstrapMcpStep()
    assert step.detect(ctx).status is StepStatus.PENDING
    assert step.apply(ctx).status is StepStatus.DONE
    assert step.detect(ctx).status is StepStatus.DONE
    again = step.apply(ctx)
    assert again.status is StepStatus.DONE
    assert "already current" in again.detail


def test_a_stale_registry_is_rewritten(
    ctx_factory: Callable[..., StepContext],
    recipe_factory: Callable[..., Recipe],
    patch_load_recipe: Callable[[Recipe | None], None],
    tmp_path: Path,
) -> None:
    patch_load_recipe(recipe_factory(mcp_servers=[_tavily_server()]))
    ctx = ctx_factory(resolved_stack=_stack(_tavily_cap()))
    step = BootstrapMcpStep()
    step.apply(ctx)
    target = tmp_path / MCP_REGISTRY_FILENAME
    target.write_text('{"version": 1, "mcpServers": {}}\n', encoding="utf-8")
    assert step.detect(ctx).status is StepStatus.PENDING
    step.apply(ctx)
    written = json.loads(target.read_text(encoding="utf-8"))
    assert "tavily" in written["mcpServers"]


def test_unresolved_capability_writes_a_null_url_and_warns(
    ctx_factory: Callable[..., StepContext],
    recipe_factory: Callable[..., Recipe],
    patch_load_recipe: Callable[[Recipe | None], None],
    event_log: list[StepEvent],
    tmp_path: Path,
) -> None:
    patch_load_recipe(recipe_factory(mcp_servers=[_tavily_server()]))
    result = BootstrapMcpStep().apply(ctx_factory(resolved_stack=_stack()))
    assert result.status is StepStatus.DONE
    entry = json.loads((tmp_path / MCP_REGISTRY_FILENAME).read_text(encoding="utf-8"))[
        "mcpServers"
    ]["tavily"]
    assert entry["url"] is None
    lines = [event.line for event in event_log if hasattr(event, "line")]
    assert any("not in the resolved stack" in line for line in lines)


def test_fingerprint_tracks_the_registry(
    ctx_factory: Callable[..., StepContext],
    recipe_factory: Callable[..., Recipe],
    patch_load_recipe: Callable[[Recipe | None], None],
) -> None:
    step = BootstrapMcpStep()
    patch_load_recipe(recipe_factory(mcp_servers=[_tavily_server()]))
    ctx = ctx_factory(resolved_stack=_stack(_tavily_cap()))
    first = step.fingerprint(ctx)
    assert first == step.fingerprint(ctx)
    patch_load_recipe(recipe_factory(mcp_servers=[]))
    assert step.fingerprint(ctx) != first


def test_build_registry_is_deterministic() -> None:
    servers = [_tavily_server()]
    stack = _stack(_tavily_cap())
    assert json.dumps(build_registry(servers, stack), sort_keys=True) == json.dumps(
        build_registry(servers, stack), sort_keys=True
    )


def test_bootstrap_mcp_registered_and_ordered() -> None:
    from agent_scaffold.steps import ALL_STEP_CLASSES, BootstrapMcpStep, default_steps_for

    assert BootstrapMcpStep in ALL_STEP_CLASSES
    manifest = Manifest(
        recipe="test-recipe",
        language="python",
        framework="none",
        model="claude-test",
        generated_at="2026-05-24T00:00:00+00:00",
    )
    ids = [s.id for s in default_steps_for(manifest, None)]
    assert "bootstrap_mcp" in ids
    assert ids.index("wire_credentials") < ids.index("bootstrap_mcp")
    assert ids.index("bootstrap_mcp") < ids.index("launch_backend")
