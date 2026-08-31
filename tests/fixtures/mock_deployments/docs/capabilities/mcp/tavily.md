---
id: mcp.tavily
kind: mcp
provides: [web_search]
env_vars: [TAVILY_API_KEY]
transport: streamable_http
endpoint: https://mcp.tavily.example/mcp/
probe: tavily_mcp_ping
docs: |
  Tavily search exposed over the Model Context Protocol.
---

# Capability: mcp.tavily

Test fixture body. Register the server with your framework's MCP client
support and call its search tools.
