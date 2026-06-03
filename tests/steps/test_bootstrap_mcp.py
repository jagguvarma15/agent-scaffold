"""Tests for the ``bootstrap_mcp`` step."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from agent_scaffold.capabilities import Capability, MCPServerFragment, ResolvedStack
from agent_scaffold.orchestrator import StepContext, StepStatus
from agent_scaffold.steps.bootstrap_mcp import BootstrapMcpStep


def _cap(cap_id: str, *, mcp: MCPServerFragment | None = None) -> Capability:
    return Capability(id=cap_id, kind="tools", path=Path(f"/tmp/{cap_id}.md"), mcp=mcp)


def test_detect_skipped_without_any_tools_capability(
    ctx_factory: Callable[..., StepContext],
) -> None:
    ctx = ctx_factory(resolved_stack=ResolvedStack())
    result = BootstrapMcpStep().detect(ctx)
    assert result.status is StepStatus.SKIPPED


def test_detect_skipped_when_tools_cap_lacks_mcp_fragment(
    ctx_factory: Callable[..., StepContext],
) -> None:
    # tools.* capability without mcp: in its frontmatter — `mcp_servers()` filters it out.
    stack = ResolvedStack(capabilities=[_cap("tools.no-mcp")])
    ctx = ctx_factory(resolved_stack=stack)
    result = BootstrapMcpStep().detect(ctx)
    assert result.status is StepStatus.SKIPPED


def test_apply_writes_mcp_json_in_anthropic_schema(
    ctx_factory: Callable[..., StepContext], tmp_path: Path
) -> None:
    fs = _cap(
        "tools.filesystem",
        mcp=MCPServerFragment(
            name="filesystem",
            transport="stdio",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "."],
        ),
    )
    web = _cap(
        "tools.web-search",
        mcp=MCPServerFragment(
            name="web-search",
            transport="stdio",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-brave-search"],
            env={"BRAVE_API_KEY": "${BRAVE_API_KEY}"},
        ),
    )
    ctx = ctx_factory(resolved_stack=ResolvedStack(capabilities=[fs, web]), project_dir=tmp_path)
    result = BootstrapMcpStep().apply(ctx)
    assert result.status is StepStatus.DONE
    target = tmp_path / ".mcp.json"
    assert target.is_file()
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert "mcpServers" in payload
    servers = payload["mcpServers"]
    assert set(servers) == {"filesystem", "web-search"}
    # stdio entries get command + args (no "transport"/"type" key per Anthropic schema).
    assert servers["filesystem"]["command"] == "npx"
    assert servers["filesystem"]["args"] == [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        ".",
    ]
    # env placeholder makes it through verbatim.
    assert servers["web-search"]["env"] == {"BRAVE_API_KEY": "${BRAVE_API_KEY}"}


def test_apply_writes_http_transport_with_type_field(
    ctx_factory: Callable[..., StepContext], tmp_path: Path
) -> None:
    remote = _cap(
        "tools.remote",
        mcp=MCPServerFragment(name="remote", transport="http", url="https://mcp.example.com"),
    )
    ctx = ctx_factory(resolved_stack=ResolvedStack(capabilities=[remote]), project_dir=tmp_path)
    BootstrapMcpStep().apply(ctx)
    payload = json.loads((tmp_path / ".mcp.json").read_text(encoding="utf-8"))
    assert payload["mcpServers"]["remote"]["type"] == "http"
    assert payload["mcpServers"]["remote"]["url"] == "https://mcp.example.com"


def test_detect_returns_done_when_existing_file_matches(
    ctx_factory: Callable[..., StepContext], tmp_path: Path
) -> None:
    fs = _cap(
        "tools.filesystem",
        mcp=MCPServerFragment(name="filesystem", command="npx", args=["-y", "fs"]),
    )
    ctx = ctx_factory(resolved_stack=ResolvedStack(capabilities=[fs]), project_dir=tmp_path)
    # First apply writes the file.
    BootstrapMcpStep().apply(ctx)
    # Re-detecting should report DONE — no rewrite needed.
    result = BootstrapMcpStep().detect(ctx)
    assert result.status is StepStatus.DONE


def test_apply_overwrites_stale_file(
    ctx_factory: Callable[..., StepContext], tmp_path: Path
) -> None:
    # An older .mcp.json from a previous run with a different server set.
    (tmp_path / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"stale": {"command": "old"}}}), encoding="utf-8"
    )
    fs = _cap(
        "tools.filesystem",
        mcp=MCPServerFragment(name="filesystem", command="npx"),
    )
    ctx = ctx_factory(resolved_stack=ResolvedStack(capabilities=[fs]), project_dir=tmp_path)
    BootstrapMcpStep().apply(ctx)
    payload = json.loads((tmp_path / ".mcp.json").read_text(encoding="utf-8"))
    assert set(payload["mcpServers"]) == {"filesystem"}


def test_fingerprint_stable_across_calls(
    ctx_factory: Callable[..., StepContext],
) -> None:
    fs = _cap(
        "tools.filesystem",
        mcp=MCPServerFragment(name="filesystem", command="npx", args=["a", "b"]),
    )
    ctx = ctx_factory(resolved_stack=ResolvedStack(capabilities=[fs]))
    step = BootstrapMcpStep()
    assert step.fingerprint(ctx) == step.fingerprint(ctx)


def test_fingerprint_changes_when_server_changes(
    ctx_factory: Callable[..., StepContext], tmp_path: Path
) -> None:
    a = _cap("tools.x", mcp=MCPServerFragment(name="x", command="alpha"))
    b = _cap("tools.x", mcp=MCPServerFragment(name="x", command="beta"))
    ctx_a = ctx_factory(resolved_stack=ResolvedStack(capabilities=[a]), project_dir=tmp_path)
    ctx_b = ctx_factory(resolved_stack=ResolvedStack(capabilities=[b]), project_dir=tmp_path)
    assert BootstrapMcpStep().fingerprint(ctx_a) != BootstrapMcpStep().fingerprint(ctx_b)
