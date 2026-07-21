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

## Environment audit

`agent-scaffold doctor` is a read-only audit of local tools (`python`, `uv`, `docker`, `ruff`). `--recipe <slug>` adds authentication and per-service readiness rows for everything the recipe declares; `--explain <topic>` opens the matching getting-started doc from the deployments repo. It never mutates anything, so it's safe to run first when something looks wrong.
