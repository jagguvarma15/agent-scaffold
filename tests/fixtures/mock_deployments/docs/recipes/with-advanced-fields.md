---
status: blueprint
languages: [python]
mcp_servers:
  - id: tavily
    capability: mcp.tavily
    transport: streamable_http
    env: { TAVILY_API_KEY: required }
  - id: postgres
    capability: mcp.postgres
    transport: stdio
    env: { POSTGRES_URL: required }
  - id: malformed-transport
    capability: mcp.broken
    transport: rocket
  - id: ""
    capability: mcp.empty
skills:
  - id: web-search-loop
    path: skills/web-search-loop/SKILL.md
    triggers: [research, "look up", investigate]
  - id: citation-formatting
    path: skills/citation-formatting/SKILL.md
  - {id: no-path}
guardrails:
  - guardrail.llama-guard
  - BAD_FORMAT
  - guardrail.llama-guard
sandbox: sandbox.e2b
durable_workflow: durable.temporal
---

# Advanced Fields Recipe

Test recipe declaring `mcp_servers`, `skills`, `guardrails`, `sandbox`, and
`durable_workflow` frontmatter fields. Mixes well-formed and malformed entries
so the coercers' warn-and-drop behavior is exercised.
