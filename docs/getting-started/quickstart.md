# Quickstart

```bash
export ANTHROPIC_API_KEY=sk-ant-...
agent-scaffold scaffold   # interactive shell — recommended
# or, one-shot:
agent-scaffold new
```

(Skip the `export` if you stored the key with `scaffold auth login` — the CLI resolves it from the keyring automatically.)

The interactive `new` flow walks you through:

1. A recipe from the deployments catalog.
2. A target language (Python or TypeScript).
3. A framework (e.g. `pydantic_ai`, `langgraph`, `vercel_ai_sdk`, or `none`).
4. A project name and destination directory.

You'll see the resolved source labels, a context summary, a generation step, a static validation pass, and a "next steps" footer with the smoke-check command.

## One command to a running stack

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

- `--no-autorun` — generate only, then print the "Next steps" hints.
- `--no-open-browser` — autorun completes but doesn't launch a browser tab.
- `--non-interactive` (the CI shape) — autorun is implicitly off so generation-only CI scripts keep their one-shot behavior.

In the REPL, the same default applies: `/generate` runs the full chain. Toggle with `/autorun off` (per session) for the staged flow.

## Next steps

- [Interactive shell](interactive-shell.md) — slash commands, free-text refinement, the `/new` wizard, and drafts.
- [Project lifecycle](../guides/project-lifecycle.md) — `up`, `update`, `down`, `logs`, `eval`, and re-run flags.
- [Configuration](../guides/configuration.md) — every env var, the TOML fallback, and effort presets.
