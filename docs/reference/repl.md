# REPL reference

`agent-scaffold scaffold` opens the shell. Input is either a slash command, a bare recipe slug (`demo` means `/recipe demo`), or free text interpreted as a plan refinement. Completion is fuzzy; unknown commands get a "did you mean" hint.

## Session

| Command | Purpose |
| --- | --- |
| `/help` | List available commands. `/help refine` lists free-text refinement keys. |
| `/config` | Set up credentials: the Anthropic key + any env vars the stack needs. |
| `/reset` | Drop the current draft. Keeps config + resolved sources, clears everything else. |
| `/draft` | List, save, resume, or delete a named selection draft (at most 3 kept). |
| `/exit` | Leave the REPL. |

## Compose the plan

| Command | Purpose |
| --- | --- |
| `/new` | Guided wizard: recipe → language → framework → name → dest → optional features → plan. |
| `/recipe <slug>` | Select the recipe. Bare `/recipe` lists slugs; a partial query filters. |
| `/language <lang>` | Pick the target language (`python` or `typescript`). |
| `/framework <name>` | Pick the framework. Validated against the recipe's declared dependencies. |
| `/tier [T0..T4]` | Show or set the capability tier (`/tier clear` resets). |
| `/layer <layer> <ids...>` | Inspect or set one layer's capabilities (e.g. `/layer memory cache.redis vector_db.qdrant`). |
| `/observability <backend> [cloud\|docker]` | Pick the observability backend and where it runs. |
| `/stack [<layer>\|<id>]` | Browse every stack option in the catalog, grouped by layer; `/stack <id>` shows a detail card. |
| `/name <project>` | Set the project name (auto-derives `/dest` if not set). |
| `/dest <path>` | Override the destination directory. |
| `/model <id>` | Override the model id (e.g. `/model claude-sonnet-5`). |
| `/effort low\|medium\|high` | Apply an effort preset (model + max_tokens + thinking + strict). |

## Inspect

| Command | Purpose |
| --- | --- |
| `/plan` | Render the generation plan + cost with the current selections. |
| `/context` | Show the full context-tier breakdown plus dropped / truncated lists. |
| `/status` | Readiness check: Anthropic key, Docker, and the selected stack's env vars. |
| `/sync` | Re-sync deployments + blueprints from GitHub and reload recipes. |

## Generate and run

| Command | Purpose |
| --- | --- |
| `/generate` | Confirm + run the generation pipeline (the final step of `/new`). |
| `/autorun on\|off` | Whether `/generate` chains into `up` + welcome panel + browser open. |
| `/docker on\|off\|auto` | Run mode for `/up` and autorun: containers, local, or auto. Never affects generation. |
| `/write_mode <mode>` | How `/generate` handles existing files in dest (`abort`, `skip`, `diff`, `overwrite`). |
| `/open <path>` | Attach the session to an existing generated project. |
| `/up` | Bring the generated project's stack up (docker sandbox / local servers). |
| `/down` | Tear down the local stack: stop the servers + `docker compose down`. |
| `/connect [<option>]` | Connect a stack option (docker or cloud hosted) — runs in the REPL. |

## Aliases

| Alias | Resolves to |
| --- | --- |
| `/quit`, `/q` | `/exit` |
| `/h`, `/?` | `/help` |
| `/go`, `/gen` | `/generate` |
| `/write-mode` | `/write_mode` |
| `/load` | `/open` |
| `/drafts` | `/draft` |

## Deprecated

These still dispatch for one release with a migration hint, then disappear: `/deploy` (use `agent-scaffold deploy --target <t>`), `/eval` (use `agent-scaffold eval --cwd <dest>`), `/logs` (use `agent-scaffold logs <service>`), `/cost` (cost is part of `/plan`).

## Free-text refinement keys

Anything that isn't a slash command is interpreted by a small Haiku call into a typed patch over the session. The accepted keys:

| Key | Meaning |
| --- | --- |
| `model` | Override model (e.g. `claude-sonnet-5`, `claude-haiku-4-5`, `claude-opus-4-8`). |
| `effort` | Preset bundle: `low` / `medium` / `high` (model + tokens + thinking + strict). |
| `framework` | Framework name (e.g. `langgraph`, `pydantic_ai`, `vercel_ai_sdk`). |
| `language` | Target language: `python` / `typescript`. |
| `strict` | Toggle the strict generation prompt. |
| `max_tokens` | Anthropic max_tokens cap for this run. |
| `thinking_budget` | Extended-thinking token budget (null disables). |
| `stack_mode` | Capability stack mode: `quick` / `customize`. |
| `tier` | Capability tier T0..T4; `none` clears. |
| `rag_preset` | RAG preset: `simple` / `complex` (expands to catalog capability bundles). |
| `add_dependencies` | Extra pins to inject: `{language: {package: version}}`. |
| `add_steps` | Extra post-write steps to run (e.g. `[docker_up, seed]`). |
| `remove_steps` | Post-write steps to skip (e.g. `[smoke_test]`). |
| `remove_roles` | Multi-agent roles to drop. |
| `add_capabilities` | Capability ids to enable (e.g. `[obs.langfuse]`). |
| `remove_capabilities` | Capability ids to drop. |
| `hosting_overrides` | Hosting per capability: `{capability id: cloud \| docker}`. |
| `notes` | Free-form guidance appended verbatim to the LLM prompt. |
