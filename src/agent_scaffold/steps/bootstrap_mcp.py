"""``bootstrap_mcp`` step: emit ``.mcp.json`` for the generated project.

Collects every ``tools.*`` capability's ``mcp:`` fragment from the resolved
stack and writes them as a single ``.mcp.json`` at the project root using
the Anthropic ``mcpServers`` schema:

.. code-block:: json

    {
      "mcpServers": {
        "web-search": {
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-brave-search"],
          "env": {"BRAVE_API_KEY": "${BRAVE_API_KEY}"}
        }
      }
    }

The format matches what Claude Code, Claude Desktop, Cursor, and other
MCP clients read from a project-scope config. Env values are written
verbatim — recipe authors use ``${VAR}`` placeholders so actual secrets
stay in ``.env.local`` and the manifest stays diffable.

Skips cleanly when the recipe declares no ``tools.*`` capability — no
empty ``.mcp.json`` for projects that don't use MCP.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_scaffold.capabilities import MCPServerFragment, ResolvedStack
from agent_scaffold.orchestrator import (
    DetectionResult,
    StepContext,
    StepLog,
    StepResult,
    StepStatus,
    compute_fingerprint,
)

_MCP_JSON = ".mcp.json"


@dataclass
class BootstrapMcpStep:
    """Materialise the resolved stack's MCP servers into ``.mcp.json``."""

    id: str = "bootstrap_mcp"
    description: str = "Write project-scope .mcp.json from tools.* capabilities"
    depends_on: tuple[str, ...] = ("wire_credentials",)
    troubleshoot: dict[str, str] = field(
        default_factory=lambda: {
            "missing": (
                "no tools.* capabilities on the recipe — add one (e.g. "
                "tools.filesystem) via /layer tools <id> or recipe frontmatter"
            ),
            "invalid": (
                "an MCP fragment failed validation — check the capability's "
                "frontmatter for required command/url"
            ),
        }
    )

    def detect(self, ctx: StepContext) -> DetectionResult:
        servers = self._mcp_servers(ctx)
        if not servers:
            return DetectionResult(
                StepStatus.SKIPPED,
                reason="no tools.* capability declared; .mcp.json not needed",
            )
        target = ctx.project_dir / _MCP_JSON
        if target.is_file() and _matches_desired(target, servers):
            return DetectionResult(StepStatus.DONE, reason=".mcp.json already current")
        return DetectionResult(
            StepStatus.PENDING,
            reason=f"emit .mcp.json with {len(servers)} server(s)",
        )

    def apply(self, ctx: StepContext) -> StepResult:
        servers = self._mcp_servers(ctx)
        if not servers:
            return StepResult(StepStatus.SKIPPED, detail="no tools.* capabilities")
        target = ctx.project_dir / _MCP_JSON
        body = _render_mcp_json(servers)
        target.write_text(body, encoding="utf-8")
        ctx.emit(
            StepLog(
                step_id=self.id,
                line=f"wrote {_MCP_JSON} with {len(servers)} server(s): "
                + ", ".join(s.name for s in servers),
            )
        )
        return StepResult(
            StepStatus.DONE,
            detail=f"wrote {len(servers)} MCP server(s) to {_MCP_JSON}",
        )

    def fingerprint(self, ctx: StepContext) -> str:
        return compute_fingerprint({"servers": _server_signature(self._mcp_servers(ctx))})

    def _mcp_servers(self, ctx: StepContext) -> list[MCPServerFragment]:
        stack = ctx.resolved_stack
        if not isinstance(stack, ResolvedStack):
            return []
        return stack.mcp_servers()


def _server_signature(servers: list[MCPServerFragment]) -> list[dict[str, Any]]:
    """Stable serialisation for fingerprinting — sorted keys, no path refs."""
    out: list[dict[str, Any]] = []
    for s in servers:
        out.append(
            {
                "name": s.name,
                "transport": s.transport,
                "command": s.command,
                "args": list(s.args),
                "url": s.url,
                "env": dict(sorted(s.env.items())),
            }
        )
    return out


def _render_mcp_json(servers: list[MCPServerFragment]) -> str:
    """Render servers as the Anthropic ``mcpServers`` schema."""
    payload: dict[str, dict[str, dict[str, Any]]] = {"mcpServers": {}}
    for s in servers:
        entry: dict[str, Any] = {}
        if s.transport == "stdio":
            if s.command:
                entry["command"] = s.command
            if s.args:
                entry["args"] = list(s.args)
        else:
            entry["type"] = "http"
            if s.url:
                entry["url"] = s.url
        if s.env:
            entry["env"] = dict(s.env)
        payload["mcpServers"][s.name] = entry
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _matches_desired(path: Path, servers: list[MCPServerFragment]) -> bool:
    """True when ``path`` already contains exactly the payload we'd write."""
    try:
        existing = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return existing == _render_mcp_json(servers)


__all__ = ["BootstrapMcpStep"]
