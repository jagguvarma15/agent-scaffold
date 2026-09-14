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


def _arrowhead_cap() -> SimpleNamespace:
    return SimpleNamespace(
        id="mcp.arrowhead",
        kind="mcp",
        transport="streamable_http",
        endpoint="http://127.0.0.1:8004/mcp",
        env_vars=[],
        docker=SimpleNamespace(service="arrowhead", ports=["127.0.0.1:8004:8000"]),
    )


def test_detect_skips_without_mcp_servers(
    ctx_factory: Callable[..., StepContext],
    recipe_factory: Callable[..., Recipe],
    patch_load_recipe: Callable[[Recipe | None], None],
) -> None:
    patch_load_recipe(recipe_factory())
    outcome = BootstrapMcpStep().detect(ctx_factory())
    assert outcome.status is StepStatus.SKIPPED
    assert "no MCP servers" in outcome.reason


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


def test_opted_in_capability_synthesizes_a_registry_entry(
    ctx_factory: Callable[..., StepContext],
    recipe_factory: Callable[..., Recipe],
    patch_load_recipe: Callable[[Recipe | None], None],
    tmp_path: Path,
) -> None:
    """A bundle/wizard-added mcp capability gets its registry with no recipe binding."""
    patch_load_recipe(recipe_factory())  # recipe declares no mcp_servers
    ctx = ctx_factory(resolved_stack=_stack(_arrowhead_cap()))
    step = BootstrapMcpStep()
    assert step.detect(ctx).status is StepStatus.PENDING
    assert step.apply(ctx).status is StepStatus.DONE
    entry = json.loads((tmp_path / MCP_REGISTRY_FILENAME).read_text(encoding="utf-8"))[
        "mcpServers"
    ]["arrowhead"]
    assert entry["capability"] == "mcp.arrowhead"
    assert entry["transport"] == "streamable_http"
    assert entry["url"] == "http://127.0.0.1:8004/mcp"
    assert entry["containerUrl"] == "http://arrowhead:8000/mcp"
    assert entry["required_env"] == []


def test_unresolvable_recipe_still_writes_synthesized_entries(
    ctx_factory: Callable[..., StepContext],
    patch_load_recipe: Callable[[Recipe | None], None],
    tmp_path: Path,
) -> None:
    """The stack from the manifest carries the bindings even without a recipe."""
    patch_load_recipe(None)
    ctx = ctx_factory(resolved_stack=_stack(_arrowhead_cap()))
    result = BootstrapMcpStep().apply(ctx)
    assert result.status is StepStatus.DONE
    written = json.loads((tmp_path / MCP_REGISTRY_FILENAME).read_text(encoding="utf-8"))
    assert "arrowhead" in written["mcpServers"]


def test_no_recipe_and_no_mcp_capability_skips(
    ctx_factory: Callable[..., StepContext],
    patch_load_recipe: Callable[[Recipe | None], None],
) -> None:
    patch_load_recipe(None)
    outcome = BootstrapMcpStep().detect(ctx_factory(resolved_stack=_stack(_tavily_cap())))
    # _tavily_cap carries no kind, so nothing synthesizes: still a skip.
    assert outcome.status is StepStatus.SKIPPED


def test_fingerprint_tracks_synthesized_bindings(
    ctx_factory: Callable[..., StepContext],
    recipe_factory: Callable[..., Recipe],
    patch_load_recipe: Callable[[Recipe | None], None],
) -> None:
    step = BootstrapMcpStep()
    patch_load_recipe(recipe_factory())
    with_mcp = step.fingerprint(ctx_factory(resolved_stack=_stack(_arrowhead_cap())))
    without = step.fingerprint(ctx_factory(resolved_stack=_stack()))
    assert with_mcp != without


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
    # The registry file must exist before docker compose starts: the app
    # service bind-mounts ./mcp.json, and Docker materialises a missing
    # bind-mount source as a directory the step then cannot write.
    assert ids.index("bootstrap_mcp") < ids.index("docker_up")
    assert ids.index("bootstrap_mcp") < ids.index("launch_backend")


def test_container_url_derived_from_the_docker_fragment() -> None:
    from agent_scaffold.steps.bootstrap_mcp import container_url

    cap = SimpleNamespace(
        id="mcp.arrowhead",
        endpoint="http://127.0.0.1:8004/mcp",
        env_vars=[],
        docker=SimpleNamespace(service="arrowhead", ports=["127.0.0.1:8004:8000"]),
    )
    assert container_url(cap, cap.endpoint) == "http://arrowhead:8000/mcp"
    # Hosted servers (no docker fragment) have no in-network address.
    hosted = SimpleNamespace(
        id="mcp.tavily", endpoint="https://mcp.tavily.example/mcp/", docker=None
    )
    assert container_url(hosted, hosted.endpoint) is None
    assert container_url(None, "") is None


def test_registry_entry_carries_the_container_url() -> None:
    cap = SimpleNamespace(
        id="mcp.arrowhead",
        endpoint="http://127.0.0.1:8004/mcp",
        env_vars=[],
        docker=SimpleNamespace(service="arrowhead", ports=["127.0.0.1:8004:8000"]),
    )
    server = MCPServerSpec(id="arrowhead", capability="mcp.arrowhead", transport="streamable_http")
    entry = build_registry([server], _stack(cap))["mcpServers"]["arrowhead"]
    assert entry["url"] == "http://127.0.0.1:8004/mcp"
    assert entry["containerUrl"] == "http://arrowhead:8000/mcp"
    # A hosted entry omits the key entirely.
    hosted_entry = build_registry([_tavily_server()], _stack(_tavily_cap()))["mcpServers"]["tavily"]
    assert "containerUrl" not in hosted_entry


def test_apply_reclaims_an_empty_docker_created_directory(
    ctx_factory: Callable[..., StepContext],
    recipe_factory: Callable[..., Recipe],
    patch_load_recipe: Callable[[Recipe | None], None],
    tmp_path: Path,
) -> None:
    """A docker_up that ran against a missing bind-mount source leaves an
    empty mcp.json directory behind; apply removes it and writes the file."""
    (tmp_path / MCP_REGISTRY_FILENAME).mkdir()
    patch_load_recipe(recipe_factory(mcp_servers=[_tavily_server()]))
    ctx = ctx_factory(resolved_stack=_stack(_tavily_cap()))
    result = BootstrapMcpStep().apply(ctx)
    assert result.status is StepStatus.DONE
    target = tmp_path / MCP_REGISTRY_FILENAME
    assert target.is_file()
    assert "tavily" in json.loads(target.read_text(encoding="utf-8"))["mcpServers"]


def test_apply_fails_gracefully_on_a_non_empty_directory(
    ctx_factory: Callable[..., StepContext],
    recipe_factory: Callable[..., Recipe],
    patch_load_recipe: Callable[[Recipe | None], None],
    tmp_path: Path,
) -> None:
    target = tmp_path / MCP_REGISTRY_FILENAME
    target.mkdir()
    (target / "keep.txt").write_text("user data", encoding="utf-8")
    patch_load_recipe(recipe_factory(mcp_servers=[_tavily_server()]))
    ctx = ctx_factory(resolved_stack=_stack(_tavily_cap()))
    result = BootstrapMcpStep().apply(ctx)
    assert result.status is StepStatus.FAILED
    assert result.error is not None and "non-empty directory" in result.error
    assert (target / "keep.txt").read_text(encoding="utf-8") == "user data"


def test_detect_reports_reclaim_for_directory_artifact(
    ctx_factory: Callable[..., StepContext],
    recipe_factory: Callable[..., Recipe],
    patch_load_recipe: Callable[[Recipe | None], None],
    tmp_path: Path,
) -> None:
    """A docker-created mcp.json directory surfaces in detect, not just apply."""
    (tmp_path / MCP_REGISTRY_FILENAME).mkdir()
    patch_load_recipe(recipe_factory(mcp_servers=[_tavily_server()]))
    ctx = ctx_factory(resolved_stack=_stack(_tavily_cap()))
    outcome = BootstrapMcpStep().detect(ctx)
    assert outcome.status is StepStatus.PENDING
    assert "reclaim" in outcome.reason
