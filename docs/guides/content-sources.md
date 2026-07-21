# Content sources

The CLI resolves two sources before assembling the LLM context:

1. **agent-deployments** — recipes + cross-cutting / framework / pattern / stack docs.
2. **agent-blueprints** — canonical pattern overviews referenced by deployments docs.

## Resolution order

For each repo, highest priority first:

1. `--deployments-path` / `--blueprints-path` flag on `agent-scaffold new`.
2. `AGENT_SCAFFOLD_DEPLOYMENTS_PATH` / `AGENT_SCAFFOLD_BLUEPRINTS_PATH` env var.
3. `deployments_path` / `blueprints_path` in `~/.config/agent-scaffold/config.toml`.
4. **Auto-fetch from GitHub** (default) — pulls the latest `main` commit, caches by SHA under `~/.cache/agent-scaffold/{deployments,blueprints}/<sha>/`. The branch-head check probes with `git ls-remote` first (no REST rate limit, uses your logged-in git credentials) and falls back to the GitHub API with ETag-conditional GET.
5. Offline fallback — catalog falls through cached → embedded JSON (frozen at wheel-build time). Blueprints is skipped with a warning (blueprint URLs in deployments docs drop out of context).

## Auto-fetch and the cache

By default, the CLI auto-fetches the latest `main` commit from [agent-deployments](https://github.com/jagguvarma15/agent-deployments) and [agent-blueprints](https://github.com/jagguvarma15/agent-blueprints), caches each by commit SHA under `~/.cache/agent-scaffold/`, and rewrites blueprint URLs in deployments docs so the LLM actually reads the canonical pattern content. When GitHub can't be reached, the shell banner says it's serving cached trees (with their date) and `/sync` retries in-session.

Override the auto-fetch behavior per-invocation:

```bash
# Skip network for blueprints (deployments still fetches; the cache or
# embedded catalog serves offline runs after the first fetch).
agent-scaffold new --blueprints-source skip

# Use my local fork of deployments, auto-fetch blueprints.
agent-scaffold new --deployments-path ~/code/my-deployments
```

## Local checkouts

To use a local checkout instead (typical for repo development):

```bash
export AGENT_SCAFFOLD_DEPLOYMENTS_PATH=/path/to/agent-deployments
export AGENT_SCAFFOLD_BLUEPRINTS_PATH=/path/to/agent-blueprints
agent-scaffold new
# or per-invocation:
agent-scaffold new --deployments-path . --blueprints-path ../agent-blueprints
```
