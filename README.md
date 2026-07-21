# agent-scaffold

`agent-scaffold` generates runnable AI agent projects from markdown specs. It fetches the [agent-deployments](https://github.com/jagguvarma15/agent-deployments) catalog at runtime (one URL hardcoded, overridable via `--catalog-url`) and falls back to an embedded snapshot when offline — pick a recipe, target language, and framework, and the CLI assembles the relevant docs, asks Claude to emit a complete project, validates the response, and writes the files atomically into your destination of choice.

## The three-repo ecosystem

`agent-blueprints` and `agent-deployments` are the two arms that feed this CLI — the engine that composes a selection into a real, running agent:

```
agent-blueprints ──────────┐
(design core)              │
kernel · patterns · IR     ├──>  agent-scaffold  ──>  running agent
                           │     (composition        (code + server,
agent-deployments ─────────┘      engine)             live)
(verified options)
adapters · recipes · catalog
```

- **[agent-blueprints](https://github.com/jagguvarma15/agent-blueprints)** — the design core: the kernel, framework-agnostic patterns at five levels, and the spec/IR a selection compiles to. *What the agent is and how it's shaped.*
- **[agent-deployments](https://github.com/jagguvarma15/agent-deployments)** — the verified options: a port-typed registry of vetted adapters and production-shaped recipes, indexed by `catalog.yaml`. *Which concrete options realize each port.*
- **[agent-scaffold](https://github.com/jagguvarma15/agent-scaffold)** *(this repo)* — the composition engine: validates a selection, binds each port to a deployments option, and emits a complete, running project.

## Install

The package is published on PyPI as **`agent-scaffold-cli`** and installs two equivalent binaries: `agent-scaffold` (long form) and `scaffold` (short, `claude`-style). Bare `scaffold` (no subcommand) drops you straight into the interactive REPL.

**One-line install (recommended).** Installs the CLI, adds it to your PATH, and offers to store your Anthropic key:

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

## Quickstart

```bash
export ANTHROPIC_API_KEY=sk-ant-...
agent-scaffold scaffold   # interactive shell — recommended
# or, one-shot:
agent-scaffold new
```

By default the interactive flow chains generation into a running stack: install deps, start docker, run migrations, seed data, launch the frontend, open the browser. See the [quickstart guide](https://jagguvarma15.github.io/agent-scaffold/getting-started/quickstart/) for the full lifecycle and its escape hatches.

## Documentation

The full manual lives at **[jagguvarma15.github.io/agent-scaffold](https://jagguvarma15.github.io/agent-scaffold/)**:

- [Installation](https://jagguvarma15.github.io/agent-scaffold/getting-started/installation/) and [quickstart](https://jagguvarma15.github.io/agent-scaffold/getting-started/quickstart/)
- [Interactive shell](https://jagguvarma15.github.io/agent-scaffold/getting-started/interactive-shell/) — slash commands, free-text refinement, the `/new` wizard
- [Configuration](https://jagguvarma15.github.io/agent-scaffold/guides/configuration/) — env vars, TOML fallback, effort presets
- [Project lifecycle](https://jagguvarma15.github.io/agent-scaffold/guides/project-lifecycle/) — `up`, `update`, `down`, `logs`, `eval`
- [CLI reference](https://jagguvarma15.github.io/agent-scaffold/reference/cli/) and [REPL reference](https://jagguvarma15.github.io/agent-scaffold/reference/repl/)
- [Recipe format](https://jagguvarma15.github.io/agent-scaffold/reference/recipes/) — frontmatter, dependencies, external services
- [Credentials](https://jagguvarma15.github.io/agent-scaffold/guides/credentials/) and the [security model](https://jagguvarma15.github.io/agent-scaffold/design/security/)
- [Troubleshooting](https://jagguvarma15.github.io/agent-scaffold/guides/troubleshooting/)

Contributor docs stay in the repo: [CONTRIBUTING.md](https://github.com/jagguvarma15/agent-scaffold/blob/main/CONTRIBUTING.md), [SECURITY.md](https://github.com/jagguvarma15/agent-scaffold/blob/main/SECURITY.md), [CHANGELOG.md](https://github.com/jagguvarma15/agent-scaffold/blob/main/CHANGELOG.md).

### Local development

```bash
git clone https://github.com/jagguvarma15/agent-scaffold
cd agent-scaffold
uv sync
make install-dev   # exposes `scaffold` + `agent-scaffold` on PATH (editable)
```

## License

MIT (see [LICENSE](https://github.com/jagguvarma15/agent-scaffold/blob/main/LICENSE)).
