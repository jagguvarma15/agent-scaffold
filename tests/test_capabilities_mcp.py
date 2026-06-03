"""Tests for the ``tools`` capability kind + ``MCPServerFragment`` parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_scaffold.capabilities import (
    LAYER_ORDER,
    Capability,
    MCPServerFragment,
    ResolvedStack,
    _KNOWN_KINDS,
    _coerce_mcp,
    _parse_capability_file,
    load_capabilities,
)


def _write_tools_capability(
    deployments_root: Path,
    *,
    name: str,
    frontmatter_body: str,
) -> Path:
    """Drop a tools/<name>.md under ``docs/capabilities/`` and return the path."""
    target = deployments_root / "docs" / "capabilities" / "tools" / f"{name}.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        f"---\n{frontmatter_body}\n---\n\n# tools.{name}\n\nbody\n",
        encoding="utf-8",
    )
    return target


def test_tools_added_to_known_kinds_and_layer_order() -> None:
    assert "tools" in _KNOWN_KINDS
    assert "tools" in LAYER_ORDER
    # Tools slots between vector_db (storage / retrieval) and obs (signal).
    order = list(LAYER_ORDER)
    assert order.index("vector_db") < order.index("tools") < order.index("obs")


def test_coerce_mcp_stdio_happy_path() -> None:
    fragment = _coerce_mcp(
        {
            "name": "web-search",
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-brave-search"],
            "env": {"BRAVE_API_KEY": "${BRAVE_API_KEY}"},
        },
        capability_id="tools.web-search",
    )
    assert fragment is not None
    assert fragment.name == "web-search"
    assert fragment.transport == "stdio"
    assert fragment.command == "npx"
    assert fragment.args == ["-y", "@modelcontextprotocol/server-brave-search"]
    assert fragment.env == {"BRAVE_API_KEY": "${BRAVE_API_KEY}"}


def test_coerce_mcp_http_happy_path() -> None:
    fragment = _coerce_mcp(
        {"name": "remote", "transport": "http", "url": "https://mcp.example.com"},
        capability_id="tools.remote",
    )
    assert fragment is not None
    assert fragment.transport == "http"
    assert fragment.url == "https://mcp.example.com"


def test_coerce_mcp_stdio_requires_command(capsys: pytest.CaptureFixture[str]) -> None:
    fragment = _coerce_mcp(
        {"name": "broken", "transport": "stdio"}, capability_id="tools.broken"
    )
    assert fragment is None
    assert "mcp.command required" in capsys.readouterr().err


def test_coerce_mcp_http_requires_url(capsys: pytest.CaptureFixture[str]) -> None:
    fragment = _coerce_mcp(
        {"name": "broken", "transport": "http"}, capability_id="tools.broken"
    )
    assert fragment is None
    assert "mcp.url required" in capsys.readouterr().err


def test_coerce_mcp_unknown_transport_warns(capsys: pytest.CaptureFixture[str]) -> None:
    fragment = _coerce_mcp(
        {"name": "x", "transport": "carrier-pigeon", "command": "noop"},
        capability_id="tools.x",
    )
    assert fragment is None
    assert "must be 'stdio' or 'http'" in capsys.readouterr().err


def test_coerce_mcp_missing_input_is_none() -> None:
    assert _coerce_mcp(None, capability_id="tools.x") is None


def test_load_capabilities_round_trips_tools_kind(tmp_path: Path) -> None:
    _write_tools_capability(
        tmp_path,
        name="filesystem",
        frontmatter_body=(
            "id: tools.filesystem\n"
            "kind: tools\n"
            "provides: [filesystem]\n"
            "mcp:\n"
            "  name: filesystem\n"
            "  transport: stdio\n"
            "  command: npx\n"
            "  args: ['-y', '@modelcontextprotocol/server-filesystem', '.']\n"
        ),
    )
    catalog = load_capabilities(tmp_path)
    assert "tools.filesystem" in catalog
    cap = catalog["tools.filesystem"]
    assert cap.kind == "tools"
    assert cap.mcp is not None
    assert cap.mcp.name == "filesystem"
    assert cap.mcp.command == "npx"


def test_resolved_stack_mcp_servers_filters_tools_only() -> None:
    fs = Capability(
        id="tools.filesystem",
        kind="tools",
        path=Path("/tmp/fs.md"),
        mcp=MCPServerFragment(name="filesystem", command="npx"),
    )
    redis = Capability(id="cache.redis", kind="cache", path=Path("/tmp/redis.md"))
    stack = ResolvedStack(capabilities=[redis, fs])
    servers = stack.mcp_servers()
    assert len(servers) == 1
    assert servers[0].name == "filesystem"


def test_parse_capability_file_rejects_tools_without_mcp_fragment_implicit_kind_check(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # An explicit "kind: tools" without an mcp fragment still loads — mcp is
    # optional at the model level. The bootstrap step is what skips it. This
    # test pins that behavior so the loader doesn't silently drop authors'
    # in-flight work.
    target = _write_tools_capability(
        tmp_path,
        name="no-mcp",
        frontmatter_body="id: tools.no-mcp\nkind: tools\n",
    )
    capability = _parse_capability_file(target, root=tmp_path / "docs" / "capabilities")
    assert capability is not None
    assert capability.mcp is None
    # No warnings about the missing fragment — just the resolved-empty state.
    assert "mcp" not in capsys.readouterr().err.lower()
