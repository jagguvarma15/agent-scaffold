---
status: blueprint
languages: [python]
external_services:
  - id: anthropic
    required: true
    env_vars: [ANTHROPIC_API_KEY]
    probe: anthropic_list_models
    explain: anthropic
  - id: redis
    required: true
    env_vars: [REDIS_URL]
    default_local: redis://localhost:6379
    docker_service: redis
    probe: redis_ping
    explain: redis
  - id: langfuse
    required: false
    env_vars: [LANGFUSE_HOST]
    probe: langfuse_health
    explain: langfuse
  - id: unknown-service
    probe: probe_that_does_not_exist
  - id: no-probe-svc
    env_vars: [SOMETHING_URL]
  - not a mapping
  - {}
---

# With External Services

Test recipe declaring a mix of external services so discovery + probes can be exercised end-to-end.
