# agent-scaffold

`agent-scaffold` generates runnable AI agent projects from markdown specs. It ships with bundled knowledge from [agent-deployments](https://github.com/jagguvarma15/agent-deployments) — pick a recipe, target language, and framework, and the CLI assembles the relevant docs, asks Claude to emit a complete project, validates the response, and writes the files atomically into your destination of choice.

## The three-repo ecosystem

This repo is one of three that work together as a single pipeline:

```
agent-blueprints     →     agent-deployments    →     agent-scaffold
(architecture)             (specs)                    (generator)
"how to think"             "what to build"            "build it for me"
patterns + tradeoffs       9 production-shaped        reads spec, asks LLM,
framework-agnostic         markdown blueprints        writes runnable project
```

- **[agent-blueprints](https://github.com/jagguvarma15/agent-blueprints)** — framework-agnostic patterns, tradeoffs, and design guidance. Start here if you want to design before you build.
- **[agent-deployments](https://github.com/jagguvarma15/agent-deployments)** — opinionated, production-shaped markdown specs for nine concrete agents (Python + TypeScript tracks).
- **[agent-scaffold](https://github.com/jagguvarma15/agent-scaffold)** *(this repo)* — a CLI that consumes a deployment spec, asks Claude to emit a complete project, and writes the files atomically to disk.

## Install

The package is published on PyPI as **`agent-scaffold-cli`** (the CLI command itself is still `agent-scaffold`).

```bash
pipx install agent-scaffold-cli
# or
uv tool install agent-scaffold-cli
# or, for one-off use:
uvx --from agent-scaffold-cli agent-scaffold --help
```

### Local development

```bash
git clone https://github.com/jagguvarma15/agent-scaffold
cd agent-scaffold
uv sync
```

## Quickstart

```bash
export ANTHROPIC_API_KEY=sk-ant-...
agent-scaffold new
```

The bundled recipes work out of the box. To use a custom agent-deployments checkout instead:

```bash
export AGENT_SCAFFOLD_DEPLOYMENTS_PATH=/path/to/agent-deployments
agent-scaffold new
```

The interactive `new` flow walks you through:

1. The path to your `agent-deployments` repo (default from config).
2. A recipe from `docs/recipes/*.md`.
3. A target language (Python or TypeScript).
4. A framework (e.g. `pydantic_ai`, `langgraph`, `vercel_ai_sdk`, or `none`).
5. A project name and destination directory.

You'll see a context summary, a generation step, a static validation pass, and a "next steps" footer with the smoke-check command.

## Configuration

| Source | Variable / key | Purpose |
| --- | --- | --- |
| Env | `ANTHROPIC_API_KEY` | Required. The Anthropic API key used by the generator. |
| Env | `AGENT_SCAFFOLD_DEPLOYMENTS_PATH` | Default path to your `agent-deployments` checkout. |
| Env | `AGENT_SCAFFOLD_MODEL` | Override the model (default `claude-opus-4-7`). |
| Env | `AGENT_SCAFFOLD_THINKING_BUDGET` | Extended-thinking token budget. Omit to disable. |
| Env | `AGENT_SCAFFOLD_EFFORT` | Default effort preset (`low` / `medium` / `high`). |
| Env | `AGENT_SCAFFOLD_CACHE_DIR` | Override the cache root (default `~/.cache/agent-scaffold`). |
| Env | `AGENT_SCAFFOLD_CONFIG_PATH` | Override the TOML fallback location. |
| TOML | `~/.config/agent-scaffold/config.toml` | Fallback for `deployments_path`, `model`, and `thinking_budget`. |

Run `uv run agent-scaffold config` to print the resolved configuration (the API key is masked).

A typical config file:

```toml
deployments_path = "/Users/me/code/agent-deployments"
model = "claude-opus-4-7"
```

## Generation effort

`--effort` picks a preset bundle of model + token budget + extended-thinking budget + prompt strictness:

| Effort | Model | max_tokens | Thinking | Strict prompt |
|--------|-------|------------|----------|----------------|
| low    | Haiku 4.5  | 16,000 | off    | no  |
| medium | Sonnet 4.6 | 32,000 | 8,000  | no  |
| high   | Opus 4.7   | 64,000 | 16,000 | yes |

Explicit `--model`, `--max-tokens`, `--thinking`, and `--strict` override preset values. Precedence: preset → explicit flag → env / TOML.

Strict mode (`--strict` or `--effort high`) loads `system_strict.md`, which instructs the LLM to emit Docker / docker-compose / GitHub Actions / structured-logging / three-tier tests when the spec references those components.

## Pointing at your own `agent-deployments`

Either set `AGENT_SCAFFOLD_DEPLOYMENTS_PATH`, write `deployments_path` to the TOML config, or pass `--deployments-path` to `agent-scaffold new`. The directory must contain `docs/recipes/*.md` files; cross-cutting / framework / pattern / stack docs are picked up automatically based on the recipe's references.

## Recipe frontmatter

Recipes are markdown files with optional YAML frontmatter:

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

### `recipe_dependencies` (optional)

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

### `external_services` (optional)

The infrastructure the recipe depends on. `agent-scaffold doctor --recipe <slug>` probes each entry; `agent-scaffold new --plan` renders a per-service ✓/✗ readiness row before the LLM call.

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
| `docker_service` | none | Name of the matching service in a bundled `docker-compose.yml` (consumed by the upcoming `up` orchestrator). |
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

## Adding a new target language

Drop a YAML file into [`src/agent_scaffold/languages/`](src/agent_scaffold/languages/) modeled after [python.yaml](src/agent_scaffold/languages/python.yaml) or [typescript.yaml](src/agent_scaffold/languages/typescript.yaml). Required keys:

- `language`, `package_manager`, `project_layout`, `entry_point`, `manifest`
- `required_tools` (formatter / type_checker / test)
- `pinned_dependencies`, `framework_dependencies`
- `forbidden`, `smoke_check`

The CLI reads them on demand; no code changes needed unless you also want a language-specific static-validation tier (see [`src/agent_scaffold/validator.py`](src/agent_scaffold/validator.py)).

## Troubleshooting

### Contract parse failures

If Claude returns malformed JSON, agent-scaffold:

1. Saves the raw response to `~/.cache/agent-scaffold/failures/<timestamp>.json`.
2. Prints a warning and asks Claude to repair the response.
3. If the repair still fails, saves that raw response too and aborts with file pointers.

You can re-run `agent-scaffold new` with `AGENT_SCAFFOLD_CACHE_DIR` set to inspect failures elsewhere.

### `--write-mode` choices

| Mode | Behavior |
| --- | --- |
| `abort` (default) | Refuse to write into a non-empty destination. |
| `skip` | Keep existing files, write only new ones. |
| `diff` | Show a unified diff per file and prompt before overwriting. |
| `overwrite` | Replace everything. |

All writes stage to a sibling temp directory and `os.replace` into place, so a failure mid-generation leaves the destination untouched.

### Re-running validation

`agent-scaffold validate /path/to/generated --tier static|build|smoke` reruns one of the post-generation tiers without re-invoking the LLM.

## CLI commands

| Command | Purpose |
| --- | --- |
| `agent-scaffold new` | Interactive project generator. |
| `agent-scaffold up [project_dir]` | Provision a generated project: install deps, start docker services, prompt for missing API keys. `--plan` to preview, `--yes` for CI, `--resume / --retry / --skip / --force / --only` for re-runs. |
| `agent-scaffold regenerate <project> <file>` | Re-prompt the model for a single file in an existing project. |
| `agent-scaffold validate <project> --tier ...` | Re-run a post-generation validation tier. |
| `agent-scaffold doctor` | Read-only audit of local tools (`python`, `uv`, `docker`, `ruff`). `--recipe <slug>` adds Authentication + per-`external_services` rows. `--no-probes` skips network probes. `--timeout N` (1–30s) caps each probe. `--json` for machine-readable output. `--explain <topic>` opens the matching getting-started doc. |
| `agent-scaffold auth login` | Capture an Anthropic key (browser or paste), validate it via `models.list()`, and store it. |
| `agent-scaffold auth status` | Show the active credential backend, stored credentials (masked), and the resolution order. `--json` for machine-readable output. |
| `agent-scaffold auth logout` | Remove a stored credential from every backend it lives in (`--all` to wipe everything). |
| `agent-scaffold auth setup-token <name>` | Store a long-lived CI token in the mode-0600 file backend (`--stdin` for piped input). |
| `agent-scaffold config` | Print the resolved configuration. |

## Step orchestrator

Provisioning verbs (`up`, `update`) plug into a state-tracked step framework: each step has a `detect()` (read-only) and an `apply()` (idempotent), with progress recorded in `<project>/.scaffold/state.json`. From that one design, the flag set `--only / --skip / --force / --retry / --resume` and `--plan`-before-build all fall out naturally.

See [`docs/design/orchestrator.md`](docs/design/orchestrator.md) for the contract, state-file shape, decision table, and the anti-patterns to avoid when authoring new steps.

## Credentials

`agent-scaffold` resolves the Anthropic API key in this order:

1. `ANTHROPIC_API_KEY` environment variable
2. `python-keyring` (macOS Keychain / Windows Credential Manager / Linux Secret Service / KDE Wallet)
3. INI file at `$XDG_CONFIG_HOME/agent-scaffold/credentials` (mode `0600`)

The plaintext keyring backend is **refused**: if `keyring.get_keyring()` reports `PlaintextKeyring` (or any non-OS-native backend), `auth login` falls back to the mode-0600 file backend with a warning. Pass `--use-file` or `--use-env` to override the default.

```bash
agent-scaffold auth login              # browser flow
agent-scaffold auth login --no-browser # paste flow (headless / SSH)
agent-scaffold auth status             # show what's stored where
agent-scaffold auth logout --all       # nuke every stored credential
echo "$TOKEN" | agent-scaffold auth setup-token ci-prod --stdin
```

## License

MIT (see [LICENSE](LICENSE)).
