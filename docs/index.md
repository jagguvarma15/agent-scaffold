# agent-scaffold

Generate runnable AI agent projects from markdown specs. Pick a recipe, a target language, and a framework — the CLI assembles the relevant docs, asks Claude to emit a complete project, validates the response, and writes the files atomically into your destination of choice.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/jagguvarma15/agent-scaffold/main/install.sh | sh
```

The one-liner installs the CLI, puts it on your PATH, and offers to store your Anthropic key. Prefer pipx or uv, or want a one-off run without installing? See [Installation](getting-started/installation.md).

## What you get

**Spec to running project.** A recipe is a production-shaped markdown spec, not a code template. The generator reads it together with framework guides and architecture patterns, asks Claude for the complete project, validates the response against a contract (required files, path safety, static lint, build, smoke check), and stages every file to a temp directory before atomically moving it into place — a failed generation leaves your destination untouched.

**One command to a running stack.** By default `new` chains straight into provisioning: install dependencies, start docker services, wire credentials, run migrations, seed dev data, launch the frontend, open the browser. The [interactive shell](getting-started/interactive-shell.md) keeps the session alive afterwards for `/up`, `/status`, `/connect`, and `/down`.

**Secrets handled properly.** A nine-rule hardening model governs every credential the CLI touches: keyring-first storage with a mode-0600 file fallback, no secrets in argv, output redaction, enforced `.gitignore`, and first-class revocation — each rule locked in by an audit test. See the [security model](design/security.md).

## The three-repo ecosystem

```
agent-blueprints     →     agent-deployments    →     agent-scaffold
(architecture)             (specs)                    (generator)
"how to think"             "what to build"            "build it for me"
```

- [agent-blueprints](https://github.com/jagguvarma15/agent-blueprints) — framework-agnostic patterns, tradeoffs, and design guidance.
- [agent-deployments](https://github.com/jagguvarma15/agent-deployments) — opinionated markdown specs for eleven concrete agents (Python and TypeScript tracks).
- [agent-scaffold](https://github.com/jagguvarma15/agent-scaffold) — this CLI, which consumes a deployment spec and writes a runnable project.

## Next steps

- [Quickstart](getting-started/quickstart.md) — from API key to a running agent.
- [Interactive shell](getting-started/interactive-shell.md) — the recommended way to work.
- [CLI reference](reference/cli.md) and [REPL reference](reference/repl.md) — every command.
