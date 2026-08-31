---
status: blueprint
languages: [python]
capabilities: [cache.redis]
mcp_servers:
  - id: tavily
    capability: mcp.tavily
    transport: streamable_http
    env: { TAVILY_API_KEY: required }
---

# Recipe With MCP Server

Test recipe binding a single well-formed MCP server so the resolution seed,
the registry step, and the prompt wiring can be exercised end to end against
the mock catalog.
