# Configuration

| Source | Variable / key | Purpose |
| --- | --- | --- |
| Env | `ANTHROPIC_API_KEY` | Required. The Anthropic API key used by the generator. |
| Env | `AGENT_SCAFFOLD_DEPLOYMENTS_PATH` | Local-checkout override for `agent-deployments` (defaults to auto-fetch from GitHub). |
| Env | `AGENT_SCAFFOLD_BLUEPRINTS_PATH` | Local-checkout override for `agent-blueprints` (defaults to auto-fetch from GitHub). |
| Env | `AGENT_SCAFFOLD_DEPLOYMENTS_SOURCE` | `auto` only (default). `bundled` mode was removed in v0.3 — the catalog + on-disk fetch cache replaces it. |
| Env | `AGENT_SCAFFOLD_BLUEPRINTS_SOURCE` | `auto` (default) or `skip` (no fetch; drop blueprint URLs from context). |
| Env | `AGENT_SCAFFOLD_CATALOG_URL` | Override the catalog URL. Default: `raw.githubusercontent.com/jagguvarma15/agent-deployments/main/catalog.yaml`. |
| Env | `AGENT_SCAFFOLD_MODEL` | Override the model. The default comes from `models.DEFAULT_MODEL` (currently `claude-opus-4-8`). |
| Env | `AGENT_SCAFFOLD_REPAIR_MODEL` | Model for the validation-repair call only (default `claude-sonnet-5`); set it equal to your model to keep repairs on the session model. |
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
model = "claude-opus-4-8"
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
