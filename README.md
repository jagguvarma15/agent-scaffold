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
| `agent-scaffold regenerate <project> <file>` | Re-prompt the model for a single file in an existing project. |
| `agent-scaffold validate <project> --tier ...` | Re-run a post-generation validation tier. |
| `agent-scaffold doctor` | Read-only audit of local tools (`python`, `uv`, `docker`, `ruff`). Supports `--json` for machine-readable output and `--explain <topic>` to open a getting-started doc. Reserved flags `--recipe` / `--no-probes` will become active in later Track B briefs. |
| `agent-scaffold config` | Print the resolved configuration. |

## License

MIT (see [LICENSE](LICENSE)).
