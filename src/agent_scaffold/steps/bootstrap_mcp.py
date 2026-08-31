"""``bootstrap_mcp`` step: write the MCP server registry the recipe declares.

For each ``mcp_servers`` entry on the recipe, one entry lands in ``mcp.json``
at the generated project's root: transport, the bound capability, the endpoint
URL (streamable HTTP) or a launcher slot (stdio), and the env var NAMES the
server needs. Values are always ``${VAR}`` placeholders — a secret value never
reaches the file; the generated agent expands them from its process env at
boot, after ``wire_credentials`` has stored the real values.

The registry is derived, step-owned state (the generation prompt tells the
model not to emit it), so unlike ``emit_deploy_configs`` a stale or hand-edited
``mcp.json`` is rewritten rather than preserved: wrong env names or a dropped
server would break the agent silently.

stdio entries carry ``command: null`` until the deployments capability schema
grows launcher fields; frameworks that spawn stdio servers fill the command in
their generated code for now.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from agent_scaffold.discovery import (
    DiscoveryError,
    MCPServerSpec,
    Recipe,
    discover_recipes,
)
from agent_scaffold.orchestrator import (
    DetectionResult,
    StepContext,
    StepLog,
    StepResult,
    StepStatus,
    compute_fingerprint,
)

MCP_REGISTRY_FILENAME = "mcp.json"

_AUTH_SUFFIXES = ("_API_KEY", "_TOKEN")
_REQUIRED_SENTINEL = "required"


@dataclass
class BootstrapMcpStep:
    """Write the framework-agnostic MCP server registry (``mcp.json``)."""

    id: str = "bootstrap_mcp"
    description: str = "Write the MCP server registry (mcp.json)"
    depends_on: tuple[str, ...] = ()
    troubleshoot: dict[str, str] = field(
        default_factory=lambda: {
            "unresolved": (
                "the recipe binds an mcp capability the catalog does not "
                "contain — update the deployments source or fix the id"
            ),
        }
    )

    # ---- detection ----------------------------------------------------

    def detect(self, ctx: StepContext) -> DetectionResult:
        recipe = _load_recipe(ctx)
        if recipe is None:
            return DetectionResult(StepStatus.SKIPPED, reason="recipe not resolvable")
        if not recipe.mcp_servers:
            return DetectionResult(StepStatus.SKIPPED, reason="recipe declares no mcp_servers")
        desired = build_registry(recipe.mcp_servers, ctx.resolved_stack)
        target = ctx.project_dir / MCP_REGISTRY_FILENAME
        if target.is_file():
            try:
                on_disk = json.loads(target.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                on_disk = None
            if on_disk == desired:
                return DetectionResult(StepStatus.DONE, reason="mcp.json is current")
        return DetectionResult(
            StepStatus.PENDING,
            reason=f"write {len(desired['mcpServers'])} MCP server entr(y/ies)",
        )

    # ---- apply --------------------------------------------------------

    def apply(self, ctx: StepContext) -> StepResult:
        recipe = _load_recipe(ctx)
        if recipe is None:
            return StepResult(StepStatus.SKIPPED, detail="recipe not resolvable")
        if not recipe.mcp_servers:
            return StepResult(StepStatus.SKIPPED, detail="recipe declares no mcp_servers")
        capabilities = _capabilities_by_id(ctx.resolved_stack)
        for server in recipe.mcp_servers:
            capability = capabilities.get(server.capability)
            if capability is None:
                ctx.emit(
                    StepLog(
                        step_id=self.id,
                        line=(
                            f"mcp: capability {server.capability!r} for server "
                            f"{server.id!r} is not in the resolved stack; entry "
                            "written without an endpoint"
                        ),
                    )
                )
            elif (
                server.transport == "streamable_http"
                and not (getattr(capability, "endpoint", None) or "").strip()
            ):
                ctx.emit(
                    StepLog(
                        step_id=self.id,
                        line=(
                            f"mcp: capability {server.capability!r} declares no "
                            f"endpoint; server {server.id!r} written with url: null"
                        ),
                    )
                )
        desired = build_registry(recipe.mcp_servers, ctx.resolved_stack)
        target = ctx.project_dir / MCP_REGISTRY_FILENAME
        rendered = json.dumps(desired, indent=2, sort_keys=True) + "\n"
        if target.is_file() and target.read_text(encoding="utf-8") == rendered:
            return StepResult(StepStatus.DONE, detail="mcp.json already current")
        target.write_text(rendered, encoding="utf-8")
        ctx.emit(
            StepLog(
                step_id=self.id,
                line=f"mcp: wrote {len(desired['mcpServers'])} server entr(y/ies) to mcp.json",
            )
        )
        return StepResult(
            StepStatus.DONE,
            detail=f"wrote {MCP_REGISTRY_FILENAME} ({len(desired['mcpServers'])} server(s))",
        )

    # ---- fingerprint --------------------------------------------------

    def fingerprint(self, ctx: StepContext) -> str:
        recipe = _load_recipe(ctx)
        servers = recipe.mcp_servers if recipe else []
        return compute_fingerprint({"registry": build_registry(servers, ctx.resolved_stack)})


# ---- registry construction --------------------------------------------


def build_registry(servers: list[MCPServerSpec], stack: Any) -> dict[str, Any]:
    """The desired ``mcp.json`` content for ``servers`` against ``stack``.

    Pure and deterministic: the same recipe + resolved stack always renders
    byte-identical JSON, so ``detect()``/``fingerprint()`` can compare it.
    """
    capabilities = _capabilities_by_id(stack)
    entries = {
        server.id: _registry_entry(server, capabilities.get(server.capability))
        for server in servers
    }
    return {"version": 1, "mcpServers": entries}


def _capabilities_by_id(stack: Any) -> dict[str, Any]:
    if stack is None:
        return {}
    return {capability.id: capability for capability in stack.capabilities}


def _registry_entry(server: MCPServerSpec, capability: Any) -> dict[str, Any]:
    required = sorted(
        var
        for var, sentinel in server.env.items()
        if str(sentinel).strip().lower() == _REQUIRED_SENTINEL
    )
    entry: dict[str, Any] = {
        "capability": server.capability,
        "transport": server.transport,
        "env": {var: f"${{{var}}}" for var in sorted(server.env)},
        "required_env": required,
        "optional_env": sorted(var for var in server.env if var not in set(required)),
    }
    if server.transport == "streamable_http":
        endpoint = (getattr(capability, "endpoint", None) or "").strip() if capability else ""
        entry["url"] = endpoint or None
        container = container_url(capability, endpoint)
        if container is not None:
            entry["containerUrl"] = container
        auth_var = _auth_header_var(server, capability)
        entry["headers"] = (
            {"Authorization": f"Bearer ${{{auth_var}}}"} if auth_var is not None else {}
        )
    else:
        entry["command"] = None
        entry["args"] = []
    return entry


def container_url(capability: Any, endpoint: str) -> str | None:
    """The in-network address of a self-hosted server, for containerized backends.

    A capability whose docker fragment publishes a port is reachable from a
    sibling compose service at ``service:container-port``, not at the host
    loopback address its ``endpoint`` names. Derived from the fragment's
    service name, the first port mapping's container side, and the endpoint's
    scheme and path; None for hosted servers (no docker fragment) or when the
    endpoint is unset.
    """
    docker = getattr(capability, "docker", None) if capability else None
    if docker is None or not endpoint:
        return None
    service = getattr(docker, "service", None)
    ports = getattr(docker, "ports", None) or []
    if not service or not ports:
        return None
    container_port = str(ports[0]).rsplit(":", 1)[-1].split("/")[0]
    parsed = urlsplit(endpoint)
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{parsed.scheme or 'http'}://{service}:{container_port}{parsed.path or '/'}{query}"


def _auth_header_var(server: MCPServerSpec, capability: Any) -> str | None:
    """The env var whose value rides as a bearer header, if any.

    First match wins across the server's own env hints (declaration order)
    then the capability's declared env vars: a name ending ``_API_KEY`` or
    ``_TOKEN`` is taken to be the credential.
    """
    candidates = list(server.env)
    for var in getattr(capability, "env_vars", None) or []:
        if var not in server.env:
            candidates.append(var)
    for var in candidates:
        if var.upper().endswith(_AUTH_SUFFIXES):
            return var
    return None


def _load_recipe(ctx: StepContext) -> Recipe | None:
    from agent_scaffold.config import load_config
    from agent_scaffold.sources import SourceFetchError, resolve_deployments

    try:
        cfg = load_config()
    except Exception:  # noqa: BLE001 — config issues must not crash detect()
        return None
    try:
        dep = resolve_deployments(
            override=cfg.deployments_path,
            mode=cfg.deployments_source,
            cache_dir=cfg.cache_dir,
        )
    except SourceFetchError:
        return None
    if dep.path is None:
        return None
    try:
        recipes = discover_recipes(dep.path)
    except DiscoveryError:
        return None
    return next((r for r in recipes if r.slug == ctx.manifest.recipe), None)


__all__ = ["BootstrapMcpStep", "MCP_REGISTRY_FILENAME", "build_registry", "container_url"]
