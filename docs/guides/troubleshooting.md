# Troubleshooting

## Contract parse failures

If Claude returns malformed JSON, agent-scaffold:

1. Saves the raw response to `~/.cache/agent-scaffold/failures/<timestamp>.json`.
2. Prints a warning and asks Claude to repair the response.
3. If the repair still fails, saves that raw response too and aborts with file pointers.

You can re-run `agent-scaffold new` with `AGENT_SCAFFOLD_CACHE_DIR` set to inspect failures elsewhere.

## --write-mode choices

| Mode | Behavior |
| --- | --- |
| `abort` (default) | Refuse to write into a non-empty destination. |
| `skip` | Keep existing files, write only new ones. |
| `diff` | Show a unified diff per file and prompt before overwriting. |
| `overwrite` | Replace everything. |

All writes stage to a sibling temp directory and `os.replace` into place, so a failure mid-generation leaves the destination untouched.

## Re-running validation

`agent-scaffold validate /path/to/generated --tier static|build|smoke` reruns one of the post-generation tiers without re-invoking the LLM.

## Host port already in use

`agent-scaffold up` (and the autorun after `new`) checks every host port the run is about to bind - the docker compose ports, plus the local backend (usually 8000) and frontend (usually 3000) - before anything binds them.

**By default** an interactive run shows who holds each port and asks before every stop or kill. A non-interactive run (`--yes`, `--autorun-yes`) fails fast on compose ports with the conflict table and the commands to run yourself; it never kills anything. `--yes` is deliberately not consent to stop things.

**`--free-ports`** is the consent. It stops what holds the ports with no prompt, in this order:

1. this project's own previous stack, with one `docker compose down`;
2. any other container holding a port, with `docker stop <name>`;
3. any other host process, with SIGTERM, then SIGKILL after a short grace period.

Each port is re-checked afterwards, and a table lists exactly what was stopped, what was left alone, and why. If a port is still held, the run stops before provisioning and prints the manual commands.

**What it will never touch**, even with the flag: Docker Desktop and other container-runtime processes (on macOS the process holding a docker-published port is Docker Desktop's own proxy, so killing it would take Docker down), `agent-scaffold` itself and its parent processes (your shell, the REPL), pid 1 and system daemons, processes owned by another user, and anything running as root. When it cannot tell what a process is, it refuses. Those ports are reported as "left alone" with the reason, and you stop them yourself. A project's own running backend or frontend is never treated as a foreign conflict when `up` is re-run.

Ways to turn it on:

| Where | How |
| --- | --- |
| One run | `agent-scaffold up --free-ports` or `agent-scaffold new --free-ports` (`--no-free-ports` forces it off) |
| REPL session | `/free_ports on` (alias `/free-ports`); the toolbar shows `free-ports: on` while armed |
| Every run | `AGENT_SCAFFOLD_FREE_PORTS=1`, or `free_ports = true` in `~/.config/agent-scaffold/config.toml` |

Notes: ports are freed before the plan runs, so if a later step such as `install_deps` fails, something was stopped for a server that never started. A process that is restarted by a supervisor (`uvicorn --reload`, `restart: always`) can come back; the run reports the port as still in use rather than looping. On Windows, owner lookup is unavailable, so every holder is reported and you stop it yourself.

## Environment audit

`agent-scaffold doctor` is a read-only audit of local tools (`python`, `uv`, `docker`, `ruff`). `--recipe <slug>` adds authentication and per-service readiness rows for everything the recipe declares; `--explain <topic>` opens the matching getting-started doc from the deployments repo. It never mutates anything, so it's safe to run first when something looks wrong.
