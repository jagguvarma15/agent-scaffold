# agent-scaffold

`agent-scaffold` generates runnable AI agent projects from markdown specs. It fetches the [agent-deployments](https://github.com/jagguvarma15/agent-deployments) catalog at runtime (one URL hardcoded, overridable via `--catalog-url`) and falls back to an embedded snapshot when offline — pick a recipe, target language, and framework, and the CLI assembles the relevant docs, asks Claude to emit a complete project, validates the response, and writes the files atomically into your destination of choice.

## The three-repo ecosystem

This repo is one of three that work together as a single pipeline:

```
agent-blueprints     →     agent-deployments    →     agent-scaffold
(architecture)             (specs)                    (generator)
"how to think"             "what to build"            "build it for me"
patterns + tradeoffs       11 production-shaped       reads spec, asks LLM,
framework-agnostic         markdown blueprints        writes runnable project
```

- **[agent-blueprints](https://github.com/jagguvarma15/agent-blueprints)** — framework-agnostic patterns, tradeoffs, and design guidance. Start here if you want to design before you build.
- **[agent-deployments](https://github.com/jagguvarma15/agent-deployments)** — opinionated, production-shaped markdown specs for eleven concrete agents (Python + TypeScript tracks).
- **[agent-scaffold](https://github.com/jagguvarma15/agent-scaffold)** *(this repo)* — a CLI that consumes a deployment spec, asks Claude to emit a complete project, and writes the files atomically to disk.

## Install

The package is published on PyPI as **`agent-scaffold-cli`** and installs two equivalent binaries: `agent-scaffold` (long form) and `scaffold` (short, `claude`-style). Bare `scaffold` (no subcommand) drops you straight into the interactive REPL; everything else (`scaffold new`, `scaffold doctor`, `scaffold --help`, …) mirrors the `agent-scaffold` subcommands.

**One-line install (recommended).** Installs the CLI, adds it to your PATH, and offers to store your Anthropic key — the way `claude`'s installer works:

```bash
curl -fsSL https://raw.githubusercontent.com/jagguvarma15/agent-scaffold/main/install.sh | sh
```

**Or install manually.** A plain `pip install` can't put the binaries on your PATH (wheels run no code at install time), so use `pipx`/`uv tool` and run their one-time PATH step:

```bash
pipx install agent-scaffold-cli && pipx ensurepath
# or
uv tool install agent-scaffold-cli && uv tool update-shell
# or, for one-off use (no install, no PATH change):
uvx --from agent-scaffold-cli scaffold --help
```

Either way, restart your shell afterward, then store your Anthropic key once with `scaffold auth login` (the one-line installer prompts for it during setup). `scaffold` won't start without a key.

### Local development

```bash
git clone https://github.com/jagguvarma15/agent-scaffold
cd agent-scaffold
uv sync
make install-dev   # exposes `scaffold` + `agent-scaffold` on PATH (editable)
```

## Quickstart

```bash
export ANTHROPIC_API_KEY=sk-ant-...
agent-scaffold scaffold   # interactive shell — recommended
# or, one-shot:
agent-scaffold new
```

### One command to a running stack

By default, `agent-scaffold new` (interactive) chains into the full lifecycle: generation → `up` (install deps, start docker, run migrations, seed data, launch the frontend dev server) → welcome panel → open the frontend in your browser. The screencast looks like:

```
$ agent-scaffold new
  ... (generation)
  ✓ Files written: 46
  ✓ Validation passed (static)

  ─── Provisioning ─────────────────────────────────────
  → install_deps         ✓ done
  → docker_up            ✓ 5 services up + healthy
  → wire_credentials     ✓ all keys resolved
  → migrations           ✓ alembic upgrade head
  → seed                 ✓ 50 restaurants, 80 reservations
  → emit_deploy_configs  ✓ vercel.json written
  → launch_frontend      ✓ http://localhost:3000

  ╭── Ready — local URLs ──────────────────╮
  │ Frontend: http://localhost:3000        │
  │ Backend:  http://localhost:8000        │
  │ Grafana:  http://localhost:3002        │
  │ ...                                    │
  ╰────────────────────────────────────────╯
  Opening http://localhost:3000 in your browser…
```

Escape hatches when you want the staged-by-hand flow instead:

- `--no-autorun` — generate only, then print the legacy "Next steps" hints.
- `--no-open-browser` — autorun completes but doesn't launch a browser tab.
- `--non-interactive` (the CI shape) — autorun is implicitly off so generation-only CI scripts keep their one-shot behavior.

In the REPL, the same default applies: `/generate` runs the full chain. Toggle with `/autorun off` (per session) for the staged flow.

### Interactive shell

`agent-scaffold scaffold` opens a persistent REPL. Make selections with
slash commands, refine the plan with free text, see the cost estimate,
generate when you're ready, then start the next project — all without
re-launching the CLI.

```
scaffold › /recipe restaurant-rebooking
scaffold › /language python
scaffold › /framework langgraph
scaffold › /name demo
scaffold › /plan
  [renders the generation plan with token + cost estimate]
scaffold › swap to sonnet and skip the smoke test
  ✓ applied refinement
  Δ model: claude-opus-4-7 → claude-sonnet-4-6
  Δ steps: -smoke_test
scaffold › /generate
  [runs the generation pipeline]
scaffold › /exit
```

Type `/help` inside the shell for the full command list. Free-text
input ("use Sonnet, add Redis") is interpreted by a tiny Haiku call
(~$0.002) into a typed patch over the plan. Run `/help refine` for the
full list of accepted refinement keys.

Command and slug completion is fuzzy — `/observ`<Tab> reaches
`/observability` and a mistyped `/genrate` still suggests `/generate`.
Unknown commands and capability ids get a "did you mean" hint, and a
partial `/stack <query>` or `/recipe <query>` filters to matching rows
(`/stack qdr` narrows to `vector_db.qdrant`).

The `/new` wizard walks the mandatory picks first (recipe, language,
framework, name, destination), then one optional-features menu: RAG,
Observability, Guardrails, More layers. Only the features you check get
their own step — Enter with nothing checked goes straight to the plan.
The RAG step offers `simple` (single-stage retrieval on pgvector plus
embeddings) or `complex` (hybrid search plus reranking), expanded from
the catalog's published bundles; `custom` opens the full layer walk. The
observability step asks where the backend runs when it supports both
modes (`/observability langfuse cloud` mirrors it) — cloud keeps the
capability but drops its compose service and wires the endpoint by
credentials. The same presets work non-interactively:
`agent-scaffold new --bundle rag-simple --obs-hosting langfuse=cloud`.

While composing, `/stack` browses the entire capability catalog grouped
by layer — delivery (docker, cloud hosted, or docker with a cloud
override), cost tier, and provisioning time per option, with your current
picks marked. `/stack <id>` shows a detail card (description, env vars,
connect handle); `/layer <layer> <ids...>` applies picks across memory,
infrastructure, tools, observability, eval, interface, hosting, and auth.

After generation the shell stays useful: `/up` brings the stack up,
`/status` checks readiness, `/connect <option>` wires a cloud hosted
integration (LangSmith, managed Redis/Postgres), and `/down` tears the
stack back down.

**Resume work.** Selections autosave to a named draft as you go (at most
3 are kept; `/drafts` lists them, `/draft load <name>` resumes one). Once
a project generates, its draft is retired — from then on `/open <dir>`
(alias `/load`), or launching with `scaffold <dir>`, attaches the shell
to the generated project so `/up`, `/connect`, and `/status` work on it.
Loading a draft whose destination was already generated attaches to the
project instead of rehydrating the stale selections.

**Startup sync.** Each shell launch checks GitHub for newer
deployments/blueprints content before the banner (the banner label reads
"up to date" or "updated"); pass `--no-sync` to skip the check and start
from the cache.

By default, the CLI auto-fetches the latest `main` commit from
[`agent-deployments`](https://github.com/jagguvarma15/agent-deployments) and
[`agent-blueprints`](https://github.com/jagguvarma15/agent-blueprints),
caches each by commit SHA under `~/.cache/agent-scaffold/`, and rewrites
blueprint URLs in deployments docs so the LLM actually reads the
canonical pattern content. Offline fallback chain: cached catalog at
`~/.cache/agent-scaffold/catalog/` → embedded `_embedded_catalog.json`
shipped in the wheel. Blueprint links are silently skipped when the
blueprints repo can't be fetched.

To use a local checkout instead (typical for repo development):

```bash
export AGENT_SCAFFOLD_DEPLOYMENTS_PATH=/path/to/agent-deployments
export AGENT_SCAFFOLD_BLUEPRINTS_PATH=/path/to/agent-blueprints
agent-scaffold new
# or per-invocation:
agent-scaffold new --deployments-path . --blueprints-path ../agent-blueprints
```

The interactive `new` flow walks you through:

1. A recipe from `docs/recipes/*.md`.
2. A target language (Python or TypeScript).
3. A framework (e.g. `pydantic_ai`, `langgraph`, `vercel_ai_sdk`, or `none`).
4. A project name and destination directory.

You'll see the resolved source labels, a context summary, a generation step, a static validation pass, and a "next steps" footer with the smoke-check command.

## Configuration

| Source | Variable / key | Purpose |
| --- | --- | --- |
| Env | `ANTHROPIC_API_KEY` | Required. The Anthropic API key used by the generator. |
| Env | `AGENT_SCAFFOLD_DEPLOYMENTS_PATH` | Local-checkout override for `agent-deployments` (defaults to auto-fetch from GitHub). |
| Env | `AGENT_SCAFFOLD_BLUEPRINTS_PATH` | Local-checkout override for `agent-blueprints` (defaults to auto-fetch from GitHub). |
| Env | `AGENT_SCAFFOLD_DEPLOYMENTS_SOURCE` | `auto` only (default). `bundled` mode was removed in v0.3 — the catalog + on-disk fetch cache replaces it. |
| Env | `AGENT_SCAFFOLD_BLUEPRINTS_SOURCE` | `auto` (default) or `skip` (no fetch; drop blueprint URLs from context). |
| Env | `AGENT_SCAFFOLD_CATALOG_URL` | Override the catalog URL. Default: `raw.githubusercontent.com/jagguvarma15/agent-deployments/main/catalog.yaml`. |
| Env | `AGENT_SCAFFOLD_MODEL` | Override the model (default `claude-opus-4-7`). |
| Env | `AGENT_SCAFFOLD_THINKING_BUDGET` | Extended-thinking token budget. Omit to disable. |
| Env | `AGENT_SCAFFOLD_EFFORT` | Default effort preset (`low` / `medium` / `high`). |
| Env | `AGENT_SCAFFOLD_CACHE_DIR` | Override the cache root (default `~/.cache/agent-scaffold`). |
| Env | `AGENT_SCAFFOLD_CACHE_TTL` | Prompt-cache TTL for the stable prefix: `5m` (default, cheaper writes) or `1h` (keeps the prefix warm across repeated regenerations within the hour). |
| Env | `AGENT_SCAFFOLD_CONFIG_PATH` | Override the TOML fallback location. |
| TOML | `~/.config/agent-scaffold/config.toml` | Fallback for `deployments_path`, `model`, and `thinking_budget`. |

Run `agent-scaffold config` (or `scaffold config`) to print the resolved configuration (the API key is masked).

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

## Where docs come from

The CLI resolves two sources before assembling the LLM context:

1. **agent-deployments** — recipes + cross-cutting / framework / pattern / stack docs.
2. **agent-blueprints** — canonical pattern overviews referenced by deployments docs.

Resolution order for each repo (highest priority first):

1. `--deployments-path` / `--blueprints-path` flag on `agent-scaffold new`.
2. `AGENT_SCAFFOLD_DEPLOYMENTS_PATH` / `AGENT_SCAFFOLD_BLUEPRINTS_PATH` env var.
3. `deployments_path` / `blueprints_path` in `~/.config/agent-scaffold/config.toml`.
4. **Auto-fetch from GitHub** (default) — pulls the latest `main` commit, caches by SHA under `~/.cache/agent-scaffold/{deployments,blueprints}/<sha>/`. Uses ETag-conditional GET so unchanged refs don't consume rate-limit quota.
5. Offline fallback — catalog falls through cached → embedded JSON (frozen at wheel-build time). Blueprints is skipped with a warning (blueprint URLs in deployments docs drop out of context).

Override the auto-fetch behavior per-invocation:

```bash
# Skip network for blueprints (deployments still fetches; the cache or
# embedded catalog serves offline runs after the first fetch).
agent-scaffold new --blueprints-source skip

# Use my local fork of deployments, auto-fetch blueprints.
agent-scaffold new --deployments-path ~/code/my-deployments
```

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
| `agent-scaffold up [project_dir]` | Provision a generated project: install deps, start docker services, prompt for missing API keys, run alembic migrations, seed dev data, run smoke tests, launch the frontend dev server in the background, and (opt-in) commit/push + open `$EDITOR`. Ends with a welcome panel listing every live local URL. `--plan` to preview, `--yes` for CI, `--resume / --retry / --skip / --force / --only` for re-runs, `--yes --confirm-commit-push` to fully automate the opt-in commit step. |
| `agent-scaffold update [project_dir]` | Re-run the recipe and 3-way-merge template changes against your edits. Copier-style: snapshots the generated tree on `new`, uses it as the merge base, writes `<<<<<<< user / ======= / >>>>>>> template` markers on conflicts. `--dry-run` previews the plan, `--continue` finalises after manual resolution. |
| `agent-scaffold down --cwd <project>` | Stop the local stack: kills the frontend dev server (SIGTERM the process group), then `docker compose down`. `-v` also removes named volumes (destroys local Postgres / Qdrant / Redis state — requires confirmation). |
| `agent-scaffold logs <service> --cwd <project>` | Tail container logs. The reserved name `frontend` tails the dev server's log file at `.scaffold/frontend.log` instead of going through docker. `-f/--no-follow`, `--tail N`. |
| `agent-scaffold eval --cwd <project>` | Run the project's eval suite via the matching `eval.*` capability's plugin (default: Promptfoo via `npx`). Exits 1 if the total score drops below the baseline (stored in `manifest.answers["eval_baseline"]` by the `bootstrap_evals` step during `up`). `--update-baseline` persists the new total and exits 0. `--json` for machine-readable output. Recipes without an `eval.*` capability exit 0 with a friendly note. |
| `agent-scaffold regenerate <project> <file>` | Re-prompt the model for a single file in an existing project. |
| `agent-scaffold validate <project> --tier ...` | Re-run a post-generation validation tier. |
| `agent-scaffold doctor` | Read-only audit of local tools (`python`, `uv`, `docker`, `ruff`). `--recipe <slug>` adds Authentication + per-`external_services` rows. `--no-probes` skips network probes. `--timeout N` (1–30s) caps each probe. `--json` for machine-readable output. `--explain <topic>` opens the matching getting-started doc. |
| `agent-scaffold auth login` | Capture an Anthropic key (browser or paste), validate it via `models.list()`, and store it. |
| `agent-scaffold auth status` | Show the active credential backend, stored credentials (masked), and the resolution order. `--json` for machine-readable output. |
| `agent-scaffold auth logout` | Remove a stored credential from every backend it lives in (`--all` to wipe everything). |
| `agent-scaffold auth setup-token <name>` | Store a long-lived CI token in the mode-0600 file backend (`--stdin` for piped input). |
| `agent-scaffold secrets list` | Inventory every credential the CLI knows about, masked. `--json` for machine output. |
| `agent-scaffold secrets purge` | Survey + wipe every stored credential (keyring + file + `./.env.local`). `--yes` for CI; `--keep-env-local` to preserve project secrets. |
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

# Cross-backend revocation (keyring + file + ./.env.local in one go)
agent-scaffold secrets list
agent-scaffold secrets purge --yes
```

## Security model

The CLI follows a nine-point hardening checklist for secret handling:
**no secrets in argv, `getpass` instead of `input`, `SecretStr` typing,
`shell=False`, mode-0600 credential files, plaintext-keyring refusal,
output redaction, enforced `.gitignore`, and first-class revocation via
`secrets purge`**. Each rule is locked in by an audit test under
`tests/security/` so regressions block CI.

See [`docs/design/security.md`](docs/design/security.md) for the full
rationale and per-rule references.

## License

MIT (see [LICENSE](LICENSE)).
