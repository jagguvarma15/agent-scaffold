# CLI reference

Both binaries expose the same commands: `agent-scaffold <command>` and `scaffold <command>`. Bare `scaffold` with no subcommand opens the interactive shell.

## Start here

| Command | Purpose |
| --- | --- |
| `agent-scaffold scaffold [project_dir]` | Open the interactive shell (the recommended way to work). With a path, attaches to an existing generated project. See the [REPL reference](repl.md). |

## Generate

| Command | Purpose |
| --- | --- |
| `agent-scaffold new` | Interactive project generator. By default chains into the full lifecycle (generation → `up` → browser); `--no-autorun`, `--no-open-browser`, `--non-interactive` opt out. Supports `--bundle`, `--obs-hosting`, `--effort`, `--model`, `--write-mode`, and source overrides. |
| `agent-scaffold regenerate <project> <file>` | Re-prompt the model for a single file in an existing project. |
| `agent-scaffold validate <project> --tier static\|build\|smoke` | Re-run a post-generation validation tier without re-invoking the LLM. |
| `agent-scaffold lint-content` | Lint a resolved agent-deployments source against the content-drift rules. |

## Run and deploy

| Command | Purpose |
| --- | --- |
| `agent-scaffold up [project_dir]` | Provision a generated project: install deps, start docker services, prompt for missing API keys, run migrations, seed dev data, run smoke tests, launch the frontend dev server, and (opt-in) commit/push. `--plan` to preview, `--yes` for CI, `--resume / --retry / --skip / --force / --only` for re-runs. |
| `agent-scaffold update [project_dir]` | Re-run the recipe and 3-way-merge template changes against your edits. `--dry-run` previews, `--continue` finalises after manual conflict resolution. |
| `agent-scaffold deploy --target <t>` | Push the project to a cloud provider declared by a `host.*` capability. |
| `agent-scaffold down --cwd <project>` | Stop the local stack: kill the frontend dev server, then `docker compose down`. `-v` also removes named volumes (destroys local state; asks for confirmation). Never touches cloud. |
| `agent-scaffold eval --cwd <project>` | Run the project's eval suite via the matching `eval.*` capability's plugin. Exits 1 if the score drops below the stored baseline; `--update-baseline` persists a new one. |
| `agent-scaffold logs <service> --cwd <project>` | Tail container logs. The reserved name `frontend` tails the dev server's log file at `.scaffold/frontend.log`. `-f/--no-follow`, `--tail N`. |

## Setup

| Command | Purpose |
| --- | --- |
| `agent-scaffold connect [integration]` | Connect a stack option: internal docker or cloud hosted. |
| `agent-scaffold status` | Probe every capability the recipe declared and print a health table. |
| `agent-scaffold config` | Print the resolved configuration (the API key is masked). |
| `agent-scaffold doctor` | Read-only audit of local tools. `--recipe <slug>` adds auth + per-service rows, `--no-probes` skips network probes, `--timeout N` caps each probe, `--json` for machine output, `--explain <topic>` opens the matching doc. Never mutates. |

## auth — manage Anthropic credentials

| Command | Purpose |
| --- | --- |
| `agent-scaffold auth login` | Capture an Anthropic key (browser or `--no-browser` paste flow), validate it, and store it keyring-first. |
| `agent-scaffold auth status` | Show the active credential backend, stored credentials (masked), and the resolution order. `--json` for machine output. |
| `agent-scaffold auth logout` | Remove a stored credential from every backend it lives in (`--all` to wipe everything). |
| `agent-scaffold auth setup-token <name>` | Store a long-lived CI token in the mode-0600 file backend (`--stdin` for piped input). |

## secrets — survey, store, and purge

| Command | Purpose |
| --- | --- |
| `agent-scaffold secrets list` | Inventory every credential the CLI knows about, masked. `--json` for machine output. |
| `agent-scaffold secrets set <NAME>` | Store one project secret in the encrypted vault (prompted, never echoed). |
| `agent-scaffold secrets unset <NAME>` | Remove one project secret from the encrypted vault. |
| `agent-scaffold secrets purge` | Survey + wipe every stored credential (keyring + file + `./.env.local`). `--yes` for CI; `--keep-env-local` preserves project secrets. |
