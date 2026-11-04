# Project lifecycle

A generated project isn't done at generation — the CLI carries it through provisioning, iteration, and teardown. Every verb below works from the terminal (`agent-scaffold <verb>`) and most have REPL twins (`/up`, `/down`, `/status`, `/connect`) that operate on the session's attached project.

## up — provision everything

`agent-scaffold up [project_dir]` provisions a generated project: install deps, start docker services, prompt for missing API keys, run migrations, seed dev data, run smoke tests, launch the frontend dev server in the background, and (opt-in) commit/push + open `$EDITOR`. Ends with a welcome panel listing every live local URL.

- `--plan` — preview the step list without running anything.
- `--yes` — non-interactive (CI); add `--confirm-commit-push` to fully automate the opt-in commit step.
- `--resume` / `--retry` / `--skip <step>` / `--force <step>` / `--only <step>` — re-run control, driven by the recorded per-step state.

## update — re-run the recipe, keep your edits

`agent-scaffold update [project_dir]` re-runs the recipe and 3-way-merges template changes against your edits. Copier-style: it snapshots the generated tree on `new`, uses it as the merge base, and writes `<<<<<<< user / ======= / >>>>>>> template` markers on conflicts. `--dry-run` previews the plan, `--continue` finalises after manual resolution.

## down — tear down local, never cloud

`agent-scaffold down --cwd <project>` stops the local stack. It stops the frontend and backend dev servers (SIGTERM to the process group, then SIGKILL if they have not exited after a few seconds), then runs `docker compose down --remove-orphans`. It then checks every port the project uses and reports any that are still held, with the owner and the commands to free it, and exits 1. A project with no `docker-compose.yml` stops its servers and finishes there. Leaving the REPL while the attached project is still running asks whether to stop it first. `-v` also removes named volumes — that destroys local Postgres / Qdrant / Redis state, so it asks for confirmation.

## logs, eval, status, connect

- `agent-scaffold logs <service> --cwd <project>` — tail container logs. The reserved name `frontend` tails the dev server's log file at `.scaffold/frontend.log` instead of going through docker. `-f/--no-follow`, `--tail N`.
- `agent-scaffold eval --cwd <project>` — run the project's eval suite via the matching `eval.*` capability's plugin (default: Promptfoo via `npx`). Exits 1 if the total score drops below the stored baseline; `--update-baseline` persists a new one. Recipes without an `eval.*` capability exit 0 with a friendly note.
- `agent-scaffold status` — probe every capability the recipe declared and print a health table.
- `agent-scaffold connect [integration]` — wire a stack option to internal docker or a cloud host (LangSmith, managed Redis/Postgres).

## The step orchestrator

Provisioning verbs (`up`, `update`) plug into a state-tracked step framework: each step has a `detect()` (read-only) and an `apply()` (idempotent), with progress recorded in `<project>/.scaffold/state.json`. From that one design, the flag set `--only / --skip / --force / --retry / --resume` and `--plan`-before-build all fall out naturally.

See the [step orchestrator design](../design/orchestrator.md) for the contract, state-file shape, decision table, and the anti-patterns to avoid when authoring new steps.
