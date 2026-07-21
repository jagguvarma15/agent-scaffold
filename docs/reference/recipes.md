# Recipe format

Recipes are markdown files with optional YAML frontmatter, discovered from the deployments repo's `docs/recipes/` directory.

```yaml
---
status: blueprint
languages: [python, typescript]
required_files:
  - Dockerfile
  - docker-compose.yml
  - .github/workflows/ci.yml
---
```

- `status` — free-form label shown in the recipe picker (e.g. `validated`, `blueprint`).
- `languages` — supported target languages; intersected with the available language hints.
- `required_files` — additional paths that the generated project MUST contain. These are enforced by the contract validator on top of the built-in four (manifest, entry point, `README.md`, `.env.example`). Paths follow the same safety rules as generated files (relative, no `..`, no leading `/`); unsafe entries are warned about and dropped during discovery.

## recipe_dependencies (optional)

Per-language extra dependencies the recipe needs. Merged into `pinned_dependencies` from the language hints before being shown to the LLM. Use when a recipe references infrastructure clients (Redis, Postgres drivers), observability (structlog, langfuse), or framework adjuncts not in the default language profile.

```yaml
---
recipe_dependencies:
  python:
    redis: ">=5.0.0"
    structlog: ">=24.1.0"
  typescript:
    ioredis: "^5.4.0"
    pino: "^9.0.0"
---
```

Recipe-declared versions win over language-default versions on conflict. Malformed entries (non-mapping shape) are warned about and ignored during discovery.

## external_services (optional)

The infrastructure the recipe depends on. `agent-scaffold doctor --recipe <slug>` probes each entry; `agent-scaffold new --plan` renders a per-service readiness row before the LLM call.

```yaml
---
external_services:
  - id: anthropic
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
---
```

Per-entry fields:

| Field | Default | Meaning |
|-------|---------|---------|
| `id` | — | Short stable slug (`anthropic`, `redis`, `postgres`, ...). Required. |
| `required` | `true` | Whether the service must be present for the recipe to work. |
| `env_vars` | `[]` | Env vars that may carry the connection URL / credentials, in priority order. |
| `default_local` | none | Used when no `env_vars` entry is set. |
| `docker_service` | none | Name of the matching service in a bundled `docker-compose.yml` (consumed by the `up` orchestrator). |
| `probe` | none | Registered probe name. See the table below. |
| `migrations` | none | Migration tool (`alembic`, `prisma`, ...). |
| `explain` | none | Slug under `docs/getting-started/<slug>.md` for `--explain`. |
| `mock_available` | `false` | A fallback mock adapter exists if the real service is unreachable. |

Bundled probes:

| `probe` value | What it does | Address from |
|---------------|--------------|--------------|
| `anthropic_list_models` | `models.list(limit=1)` via the resolved key | `auth` resolution (env → keyring → file) |
| `redis_ping` | Raw-socket Redis `PING`/`PONG` | first env var, else `default_local` |
| `postgres_select_one` | `psycopg.connect(...).cursor().execute("SELECT 1")` (TCP-only fallback if `psycopg` not installed) | first env var, else `default_local` |
| `langfuse_health` | `GET {host}/api/public/health` | first env var, else `default_local` |
| `kafka_metadata` | TCP connect + `kafka-python` metadata (TCP-only fallback if not installed) | first env var, else `default_local` |

Unknown probe names log a warning and produce a `SKIP` at runtime instead of crashing the audit.
