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
| Env | `AGENT_SCAFFOLD_MAX_TOKENS` | Override the API `max_tokens` (default `32000`). |
| Env | `AGENT_SCAFFOLD_CACHE_DIR` | Override the cache root (default `~/.cache/agent-scaffold`). |
| Env | `AGENT_SCAFFOLD_CONFIG_PATH` | Override the TOML fallback location. |
| Flag | `--model` | Per-run model override (interactive picker if omitted). |
| TOML | `~/.config/agent-scaffold/config.toml` | Fallback for `deployments_path`, `model`, and `max_tokens`. |

Run `uv run agent-scaffold config` to print the resolved configuration (the API key is masked).

A typical config file:

```toml
deployments_path = "/Users/me/code/agent-deployments"
model = "claude-opus-4-7"
```

## Pointing at your own `agent-deployments`

Either set `AGENT_SCAFFOLD_DEPLOYMENTS_PATH`, write `deployments_path` to the TOML config, or pass `--deployments-path` to `agent-scaffold new`. The directory must contain `docs/recipes/*.md` files; cross-cutting / framework / pattern / stack docs are picked up automatically based on the recipe's references.


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



## License

MIT (see [LICENSE](LICENSE)).
