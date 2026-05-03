# agent-forge

`agent-forge` generates runnable AI agent projects from markdown specs. Point it at a clone of [agent-deployments](https://github.com/jagguvarma15/agent-deployments), pick a recipe, target language, and framework, and the CLI assembles the relevant docs, asks Claude to emit a complete project, validates the response, and writes the files atomically into your destination of choice.

## Install

```bash
# Once published to PyPI:
uv tool install agent-forge
```

For local development:

```bash
git clone https://github.com/your-org/agent-forge
cd agent-forge
uv sync
```

## Quickstart

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export AGENT_FORGE_DEPLOYMENTS_PATH=/path/to/agent-deployments
uv run agent-forge new
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
| Env | `AGENT_FORGE_DEPLOYMENTS_PATH` | Default path to your `agent-deployments` checkout. |
| Env | `AGENT_FORGE_MODEL` | Override the model (default `claude-opus-4-5`). |
| Env | `AGENT_FORGE_CACHE_DIR` | Override the cache root (default `~/.cache/agent-forge`). |
| Env | `AGENT_FORGE_CONFIG_PATH` | Override the TOML fallback location. |
| TOML | `~/.config/agent-forge/config.toml` | Fallback for `deployments_path` and `model`. |

Run `uv run agent-forge config` to print the resolved configuration (the API key is masked).

A typical config file:

```toml
deployments_path = "/Users/me/code/agent-deployments"
model = "claude-opus-4-5"
```

## Pointing at your own `agent-deployments`

Either set `AGENT_FORGE_DEPLOYMENTS_PATH`, write `deployments_path` to the TOML config, or pass `--deployments-path` to `agent-forge new`. The directory must contain `docs/recipes/*.md` files; cross-cutting / framework / pattern / stack docs are picked up automatically based on the recipe's references.

## Adding a new target language

Drop a YAML file into [`src/agent_forge/languages/`](src/agent_forge/languages/) modeled after [python.yaml](src/agent_forge/languages/python.yaml) or [typescript.yaml](src/agent_forge/languages/typescript.yaml). Required keys:

- `language`, `package_manager`, `project_layout`, `entry_point`, `manifest`
- `required_tools` (formatter / type_checker / test)
- `pinned_dependencies`, `framework_dependencies`
- `forbidden`, `smoke_check`

The CLI reads them on demand; no code changes needed unless you also want a language-specific static-validation tier (see [`src/agent_forge/validator.py`](src/agent_forge/validator.py)).

## Troubleshooting

### Contract parse failures

If Claude returns malformed JSON, agent-forge:

1. Saves the raw response to `~/.cache/agent-forge/failures/<timestamp>.json`.
2. Prints a warning and asks Claude to repair the response.
3. If the repair still fails, saves that raw response too and aborts with file pointers.

You can re-run `agent-forge new` with `AGENT_FORGE_CACHE_DIR` set to inspect failures elsewhere.

### `--write-mode` choices

| Mode | Behavior |
| --- | --- |
| `abort` (default) | Refuse to write into a non-empty destination. |
| `skip` | Keep existing files, write only new ones. |
| `diff` | Show a unified diff per file and prompt before overwriting. |
| `overwrite` | Replace everything. |

All writes stage to a sibling temp directory and `os.replace` into place, so a failure mid-generation leaves the destination untouched.

### Re-running validation

`agent-forge validate /path/to/generated --tier static|build|smoke` reruns one of the post-generation tiers without re-invoking the LLM.

## Project layout

```
agent-forge/
|- pyproject.toml
|- src/agent_forge/
|  |- cli.py            # Typer commands (new / config / validate)
|  |- config.py         # Config model + env/TOML loader
|  |- discovery.py      # Recipe discovery in docs/recipes/
|  |- context.py        # Assemble the LLM context bundle
|  |- contract.py       # Parse + validate the LLM JSON contract
|  |- generator.py      # Anthropic client wrapper (with retries)
|  |- writer.py         # Atomic file writer with diff/skip/overwrite
|  |- validator.py      # Static / build / smoke validation tiers
|  |- prompts/          # system.md, user_template.md, repair.md
|  `- languages/        # python.yaml, typescript.yaml, ...
`- tests/
   |- fixtures/         # mock_deployments + canned LLM responses
   `- test_*.py
```

## License

MIT (see [LICENSE](LICENSE)).
