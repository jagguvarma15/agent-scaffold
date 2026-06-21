# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- **Every agent ships a frontend by default.** When a recipe declares no `frontend.*` capability of its own, capability resolution now auto-includes the default `frontend.minimal-chat` (`resolve(..., default_frontend=True)` on the generation paths) — so every generated project gets a frontend + backend in the docker sandbox, no per-recipe authoring. A recipe that already declares a frontend keeps it; and the auto-include is **inert-safe** — if the catalog doesn't have `frontend.minimal-chat` yet, nothing is added and no "unresolved" warning fires.
- **Frontend can run as a container in the sandbox.** A frontend capability that sets `serve_in_container: true` (it ships a `frontend/Dockerfile`) now gets a built `frontend` service in the generated `docker-compose.yml` — a new deterministic `normalize_frontend_service` pass (mirror of the backend one) adds `build: ./frontend`, the UI port, the backend URL wired to the **host-mapped** backend port (the browser runs on the host, so it reaches the backend at `localhost`, not the in-network name), and `depends_on` the backend. `LaunchFrontendStep` gains `served_by_docker`, skipping the local `pnpm dev` when the frontend is the container. So one `docker compose up` brings up frontend + backend as containers. The pass stays inert until a capability opts in (`serve_in_container`), so it never references a missing build — a frontend without a Dockerfile still runs locally.
- **One-click run: the wizard flows straight into generate + Docker.** After `/new` collects your selections it now auto-offers to generate — a single confirm (default yes) ships it, no separate `/generate` needed. The config gate still applies: if a required credential is missing the flow breaks with a "configure X via `/config`" directive instead of auto-running (and `/config` works inline in the wizard's refine loop). Post-generate, the stack defaults to **running in Docker** when Docker is usable — `use_docker` is now tri-state (`None` = auto, `/docker on|off` forces it) and resolves to containers-when-available, falling back to local processes otherwise — so generation flows into the whole multi-container stack coming up with one keystroke.
- **Named selection drafts in the REPL.** Your recipe/language/framework/capability picks now persist to `~/.cache/agent-scaffold/drafts/<name>.json` (atomic, schema-versioned — the `orchestrator.state.json` pattern), auto-saved on every selection change so an accidental exit loses nothing. `/drafts` lists them (most recent first, with relative timestamps); `/draft save [name]`, `/draft load <name>` (re-resolves the recipe from its slug against the current deployments, degrading gracefully if it was removed), and `/draft delete <name>` manage them. At most **3** drafts are kept — a new save beyond the cap evicts the oldest. Drafts hold selections only, never secrets. On shell open a one-line hint surfaces any saved drafts to resume.
- **REPL `/config`, a real `/status`, and a blocking generate-gate.** The `scaffold` shell is now the complete setup surface: `/config` walks the credentials the current selections need — the Anthropic key (stored to the keyring/file backend) plus any external service/tool secret — and prompts for the missing ones without echoing; docker-provided infra (postgres/redis) is never asked for since `up` wires it. `/status` is now a fast, local readiness check (Anthropic key, Docker availability, the selected stack's env vars) instead of a "run this outside the REPL" stub. And `/generate` gains a blocking pre-check: if a required credential is missing it refuses to spend tokens, names the gap, and points at `/config` — selections are kept, so nothing is lost. All three route through one `repl/readiness.py` module so they agree on what "configured" means.
- **Grouped `--help` + a REPL-first banner.** The terminal `--help` now groups commands into "Start here / Setup / Generate / Run & deploy" panels (Typer `rich_help_panel`) instead of one flat list, and the bare-command banner leads with `scaffold` as the one entry point — the `agent-scaffold <verb>` commands are framed as the scripting/CI layer.

- **Tiered prompt-cache breakpoints.** Recipes with a structured `load_list` now produce cache-tier context segments: stable docs (blueprints patterns, frameworks, stack, project-layout) form a **hot** block cached with Anthropic's 1-hour TTL; the recipe body, capabilities, and remaining docs form a **warm** block on the default 5-minute TTL; the per-run project tail stays uncached. With the system block (1h in tiered mode — the API requires 1h entries before 5m ones) that's 3 of the 4 allowed breakpoints. Hot-tier docs survive warm-tier churn between runs, so re-generating in the same hour re-reads the expensive pattern/framework corpus at 0.1× price. Blocks under the minimum cacheable size collapse into their neighbor; recipes without a `load_list` keep the existing single cached context block. `load_list[].cache_tier` is honored when authored, with the same path-based defaults the deployments catalog generator publishes.
- **Alias-scan demotion.** When a recipe declares a `load_list`, the alias and cross-cutting prose scans are skipped — the author's curated declaration wins over heuristics (a stray "redis" in a design-rationale paragraph no longer pulls in the whole stack doc). Explicit Composes links and the transitive walk still apply; recipes without a `load_list` keep today's heuristic behavior unchanged.
- **Post-`up` smoke repair (one round).** When the `smoke_test` step fails during `up`/autorun, interactive runs are offered a single model-driven repair round: the failure output and implicated files go through the same repair engine as the post-write loop, the patch is written atomically, and the smoke step re-runs. Hard-capped at one round, so worst-case LLM calls per golden-path run stay at generate + 2 validation repairs + 1 smoke repair. Never offered in `--yes`/non-interactive runs.

### Fixed

- **`/config` prompts only real credentials, with hints.** Non-secret config knobs a capability declares (e.g. `LANGCHAIN_TRACING_V2`, `LANGCHAIN_PROJECT`, `LANGCHAIN_ENDPOINT` — flags/names/endpoints with sensible defaults) are now shown ✓ "config" and never prompted; only actual credentials (by a `KEY`/`TOKEN`/`SECRET`/`PASSWORD` name heuristic) are asked for — the required Anthropic key, then optional cloud credentials behind a default-no confirm. Each credential prompt shows where to get it (e.g. `LANGCHAIN_API_KEY → smith.langchain.com → Settings → API Keys`). No more being asked to "fill" four LangSmith vars when only one is a secret.
- **Only the Anthropic key blocks generation — the rest is "connect later".** The REPL gate previously demanded *every* env var a capability declared, so selecting `obs.langsmith` made all four `LANGCHAIN_*` vars required even though tracing is optional and only `LANGCHAIN_API_KEY` is a credential. Now the agent key is the sole hard requirement; docker-provided infra (postgres/redis) is shown ✓ "in sandbox" (the containers supply it — no longer a confusing unconfigured ✓), and external/cloud credentials are shown ○ optional. `/config` prompts for the key and lists optional cloud credentials behind a default-no "set now?" — no pestering. The minimal docker sandbox runs with just the key.

- **Generated backend boots with the resolved Anthropic key.** The per-run runtime env now injects `ANTHROPIC_API_KEY` from wherever the CLI itself resolves it — keyring or mode-0600 credentials file (e.g. set via `scaffold auth login`, as the installer does) — when it isn't already supplied by the shell env, `.env.local`, or the project vault. A generated agent that constructs its Anthropic client at startup no longer crashes with `Could not resolve authentication method`. Shell / `.env.local` / vault still take precedence.
- **`launch_backend` reports the real startup failure.** A backend that exits during boot is now detected immediately and reported as a crash with its exit code and log tail (with an API-key suggested fix when the tail shows the auth error), instead of waiting out the full readiness window and misattributing the failure to "backing services not running".
- **Docker mode runs the stack with zero env setup.** A deterministic post-generation pass (`contract.normalize_app_service`) now guarantees the backend container can boot: the app service gains `ANTHROPIC_API_KEY` (+ any capability secret var) via the no-value `${VAR:-}` passthrough Compose fills from the env it runs in (the resolved `runtime_env`), so the key reaches the container with no plaintext file; a dangling `env_file: .env` (a hard `docker compose up` error — that file is never generated) is rewritten to `{path, required: false}`. In-network connection strings (`DATABASE_URL`, `REDIS_URL`) and other services' config (`POSTGRES_*`) are left untouched. The generation prompts were tightened to emit this directly; the pass is the safety net.
- **Docker-mode backend crash is surfaced, not hidden.** The generated app service has no compose healthcheck, so `docker compose up --wait` reported "healthy" even when the backend crash-looped. `docker_up` now confirms the app container didn't exit on boot after `--wait` and fails with its `docker compose logs` tail (and the API-key suggested fix) — symmetry with the local-mode crash detection.
- **Frontend dev server reaches the backend automatically.** In Docker mode the backend runs containerized; `launch_frontend` now defaults the frontend capability's backend-URL var (`NEXT_PUBLIC_AGENT_URL` for nextjs-chat, `AGENT_URL` for streamlit) to `http://localhost:<backend-port>`, so the local `pnpm dev` server talks to it with no manual config. A user-set value still wins.
- **REPL canonicalizes the Python module name.** The `scaffold` REPL now passes the underscored module name (hyphens → underscores) to the pipeline like `agent-scaffold new` does, so a project named `research-assistant` no longer makes the contract demand the invalid Python path `src/research-assistant/main.py` and fail every generation. `raw_project_name` keeps the original.
- **Recipe-declared entry layout wins over the language default.** A recipe whose `required_files` declare their own application entry (e.g. an `app/` layout) no longer also has to satisfy the generic, model-invisible `src/<pkg>/main.py` entry-point requirement — the two were double-required and conflicting. The language default still applies to recipes that declare no entry of their own.
- **`launch_backend` detects a top-level `app/` layout.** Local-mode backend auto-start previously scanned only `src/<pkg>/`, so an `app/`-layout recipe (a real `app/__init__.py` package with `app/main.py`) skipped instead of launching. It now also resolves a top-level `app/` / `api/` / `backend/` / `server/` package and runs it as `uvicorn app.main:app` (or `python -m app.main`). A project carrying both layouts keeps prior behaviour — the `src/` package wins.
- **Required-files validation reports every gap at once.** `validate_required_files` previously raised on the *first* missing file, so the single repair round only ever learned about one — a recipe missing several files (e.g. an `app/` layout's `app/main.py`, `app/agent/researcher.py`, `app/tools/web_search.py`) could never be satisfied: repair added one, then generation failed on the next. It now collects **all** missing required files into one error, so the repair round adds them together. The error reads `missing required file(s): a, b, c` (manifest/entry-point roles still annotated).
- **The model is told recipe-required paths are exact.** The generation prompt's recipe-required block is now emphatic: emit each required file at its EXACT path, don't relocate (e.g. into `src/`) or rename, and — when the recipe's layout differs from the language's idiomatic one — a thin re-export shim at the required path is acceptable. This stops the model from silently substituting its preferred `src/<pkg>/` layout for a recipe's `app/` layout and failing the contract.

### Changed

- New cache block boundaries invalidate previously warm Anthropic prompt caches once (one-time extra cache-write cost on the first run after upgrading). Hot-tier writes cost 2× base input (vs 1.25× for 5m) and pay for themselves on the second generation inside the hour.

- **Live validation output.** The validator now streams each subprocess line as a `bash_line` progress event instead of buffering everything until exit — a multi-minute `uv sync` or `tsc` run shows live output instead of a frozen spinner. The Rich panel renders the last 3 lines (redacted) under the active command, mirroring the provisioning step display; the plain non-TTY display prints each line; per-run logs record every line. Full combined output is still captured (chronologically interleaved now) for the repair loop. Validation tiers also gain the streaming runner's process-group timeout kill. On POSIX, smoke checks run via `/bin/sh -c` with `shell=False`; `shell=True` survives only in the buffered Windows fallback.
- `stream_subprocess` gains a `line_callback` parameter for callers outside the step framework.

- **`.scaffold/run-summary.md` in every generated project.** A durable, human-readable record that travels with the project: recipe + status, language/framework/model, deployments snapshot SHA, file count, validation outcome (including repair rounds), env var names with set/missing status (names only — values never appear), start instructions, and the run-log path. Each `agent-scaffold up` refreshes a Provisioning section with the latest step summary. The welcome panel and the "Next steps" footer both point at it.
- Welcome panel gains "Run summary" and "Run log" pointer rows.

### Changed

- **The plan-confirm panel now defaults ON for every interactive `new`** (previously only at `--effort high`) — one Y/n gate showing context size, cost estimate, and service readiness before any tokens are spent. `--no-plan` opts out; non-interactive runs are unaffected.

- **Project-scoped encrypted secrets vault.** Service credentials for generated projects (QDRANT_URL, LANGFUSE_SECRET_KEY, …) now live encrypted in the OS-native keyring, namespaced per project (`project:<name>-<pathhash>:<VAR>`); the plaintext-keyring refusal and mode-0600 file fallback from the Anthropic-key flow apply unchanged. A **names-only index** (never values) in the credentials file powers listing and presence checks without keyring consent prompts. `wire_credentials` stores to the vault first, `.env.local` only as a last-resort fallback; the manifest records `secrets_namespace`.
- **Runtime env injection.** `up` resolves one environment per run — shell env > vault > `.env.local` (vault read in a single batch) — and every step subprocess (docker compose, uv, alembic, seed, smoke, pnpm/frontend) receives it via `env=`. Docker `${VAR}` interpolation works without any plaintext file.
- **"LLMs can never read secrets" guarantee, enforced by tests.** Secret values are architecturally confined to subprocess environments; every outbound path now redacts: the generation progress panel (operation hints/summaries/errors), repair-prompt validation output, REPL free-text refinements and serialized state sent to Haiku, run logs, and events. `tests/security/test_secrets_never_reach_llm.py` plants credentials in env/.env.local, runs the full golden path (generate + repair + refine), and asserts no planted fragment appears in any recorded LLM payload, run artifact, or console output.
- `agent-scaffold secrets set/unset --project <dir>` manage individual vault entries (getpass, masked echo); `secrets list` shows per-project vaults (names + backend only); `secrets purge` clears vault namespaces along with the other backends (list `--json` schema_version bumped to 2 with a `projects` array).
- Tests now run against an in-memory keyring + throwaway credentials file by default (autouse fixture) — no test can touch the developer's real keychain.

- **Bounded validate→repair loop.** When post-generation validation fails, the pipeline now feeds the failing command's output (redacted, capped) plus the implicated files' current bodies back to the model for targeted fixes — up to 2 rounds. The model returns only changed files (`{"files": [{path, content}]}`); patches pass the same path-safety rules as generation plus a structural constraint (existing files, or new files inside directories the project already populates — no new directory trees). Patched files are written atomically, re-formatted, re-validated, and folded into the manifest/snapshot. Repair prompts carry the recipe body only (not the full assembled context) so repair calls stay cheap.
- **Validation now runs static + build tiers** on the golden path (`ruff check` + `uv sync` for Python, `tsc --noEmit` + `pnpm install` for TypeScript) instead of static only — a project that doesn't resolve its dependencies isn't "running in one go". `--skip-validation` still bypasses everything.
- **Unrecovered validation failures now exit non-zero.** After the repair rounds are exhausted, the run raises a pipeline error with the failing tier's output excerpt and recovery hints. The project and its manifest stay on disk so `validate` / `regenerate` / `update` work for manual recovery.
- **Run-cumulative token accounting.** The generation report now sums usage across every API call in the run (generate + JSON repair + validation-repair rounds) instead of reporting only the last call, and shows a `Repair: N round(s)` line when the loop fired.
- New prompt `prompts/validation_repair.md` (registered in the prompts signature, so response caches bust correctly), `generator.repair_validation()`, `contract.parse_file_patch()`, `validator.tier_command()`.

- **Pre-flight gate before the LLM call.** `agent-scaffold new` now checks, *before spending tokens*, that the env vars the run will eventually need are resolvable and that the recipe's external services respond. The required set is the union of three sources: the recipe's `external_services[].env_vars`, the catalog's auto-derived `env_contract` (entries with a `default` count as satisfied), and the resolved capability stack's `env_vars`. Missing values can be filled at the gate (getpass, no echo): `ANTHROPIC_API_KEY` persists to the auth backend immediately; project secrets are exported for the run and written to the project's `.env.local` (mode 0600) right after the write phase. The gate is warn-only — generation never blocks. Service probes now run for **every** interactive `new` (previously only with the plan panel at `--effort high`), and the plan panel reuses the gate's results instead of probing twice; failures against docker-managed services render as "not running — `up` starts it via docker compose" instead of an alarming ✗. Non-interactive runs print missing names (never values) to stderr and skip probing.
- New module `agent_scaffold.preflight` (`run_preflight`, `collect_env_requirements`, `persist_filled`, panel renderers) and `agent_scaffold.envfile` (`.env.local` read/write + presence helpers, extracted from the `wire_credentials` step so both stages share one definition of "present").
- `RecipeEntry.env_contract` parsing in the catalog model (`EnvContractEntry`: `name` / `source_capability` / `default`).

- **Persistent per-run logs.** Every `agent-scaffold new` run writes artifacts under `~/.cache/agent-scaffold/runs/<run_id>/`: `run.log` (human-readable, one timestamped line per event) and `events.jsonl` (machine-readable event stream covering generation, file writes, validation, and — when autorun fires — every provisioning step). Both sinks are secret-redacted before anything touches disk. Run directories are pruned to the 20 most recent. The generation report panel ends with the run-log path, and pipeline failures print `Full log: …` so the evidence survives the scrollback.
- **Plain progress output for non-TTY runs.** When stdout isn't an interactive terminal (CI, pipes), generation progress degrades from the Rich Live panel to flat, grep-able one-line-per-event output on stderr — matching the existing behavior of the provisioning step display.
- `GenerationDisplay` protocol in `progress.py`; `pipeline.run_generation` now accepts any conforming display (Rich, plain, null, or the run-log tee).

- **Catalog-driven loader.** New `agent_scaffold.catalog` module fetches the deployments catalog from a single hardcoded URL (`DEFAULT_CATALOG_URL`), validates it via Pydantic, caches it under `~/.cache/agent-scaffold/catalog/`, and falls back to an embedded JSON shipped in the wheel. The alias / cross-cutting / framework-gating maps and the blueprint URL pattern all come from the catalog — scaffold has zero hardcoded knowledge of the catalog's content. Override with `--catalog-url` or `$AGENT_SCAFFOLD_CATALOG_URL`. See `MANIFEST_SCHEMA.md` in agent-deployments for the catalog schema.
- New env: `AGENT_SCAFFOLD_CATALOG_URL`.
- New CLI flag: `--catalog-url` on `agent-scaffold new`.
- `scripts/embed_catalog.py` refreshes the embedded fallback JSON from the live catalog. Run before `uv build` (the `publish.yml` workflow does this automatically; `scripts/build_hooks.py` also wires this into local `uv build` via a hatch hook).
- **`launch_frontend` orchestrator step.** Spawns the frontend dev server as a detached background process after `install_deps`. Writes `<project>/.scaffold/frontend.pid` so subsequent `up` / `down` / `logs` runs find and manage the same process. SKIPs when the recipe ships no `frontend/package.json`.
- **Welcome panel after `agent-scaffold up`.** Lists every live local URL — frontend, backend, Grafana, Langfuse, Qdrant, Tempo, eval command — derived from the resolved capability stack, plus `agent-scaffold down` as the stop hint.
- `agent-scaffold logs frontend` tails the dev server's `.scaffold/frontend.log` via `tail -f`, with a pure-Python fallback when `tail` isn't on PATH.
- `default_port` field on `languages/python.yaml` (8000) and `languages/typescript.yaml` (3000), read by the welcome panel.
- `_open_browser_safe()` helper, swallows headless/CI failures.
- **`agent-scaffold new` autorun (default on for interactive runs).** After generation succeeds, chains into `up` + welcome panel + browser open. `--no-autorun` keeps the staged-by-hand flow; `--non-interactive` (CI shape) implicitly disables autorun so existing CI scripts don't suddenly start spinning up docker. `--no-open-browser` runs `up` without launching a browser.
- `/autorun on|off` slash command in the REPL — toggles the same chain after `/go`. `SessionState` gained an `autorun: bool = True` field.
- `_run_up_inline()` helper factored out of `cmd_up` so `cmd_new` and the REPL can share the orchestrator-run + welcome-panel code path without duplicating it.
- **`agent-scaffold eval` verb.** Runs the project's eval suite via the matching `eval.*` capability's plugin (default: Promptfoo via `npx`). Exits 1 on regression vs the stored baseline (per-case delta threshold ±0.01 to ignore sampling noise). `--update-baseline` persists the new total; `--json` for machine-readable output; recipes without `eval.*` exit 0 with a friendly note.
- **`bootstrap_evals` orchestrator step.** Runs the eval suite once during `up` (after `smoke_test`, before `emit_deploy_configs`) and stores `result.total` in `manifest.answers["eval_baseline"]`. SKIPs cleanly when no `eval.*` capability is declared or `npx` isn't on PATH.
- **Eval plugin system at `agent_scaffold.eval/`.** Lazy registry, `EvalResult` + `EvalCase` dataclasses, regression-noise floor constant. Ships one plugin (`promptfoo`) that parses Promptfoo's JSON output (both `results.results[]` and flatter top-level shapes), clamps runaway LLM-judged scores into `[0, 1]`, and computes the delta against an optional baseline.
- `update_manifest_answer(project_dir, key, value)` helper in `manifest.py` for round-tripping a single answers entry.
- `/eval` slash command in the REPL — prints the `agent-scaffold eval --cwd <dest>` command line (REPL never runs the eval itself; can take minutes).
- Welcome panel now includes the eval baseline on the Eval row when it's set.
- `eval = []` extra in `pyproject.toml` (placeholder — Promptfoo is Node-based, no pip dep needed).

### Fixed

- Catalog loading no longer fails against catalogs published by generator 1.3+, which list `stack` / `cross_cutting_docs` / `pattern_docs` entries as `{path, tags, when_to_load}` mappings instead of bare path strings. Both shapes now parse; mapping entries normalize to their `path`.

### Changed

- The wheel ships ~320 KB (was ~1.2 MB). The 916 KB `_bundled_deployments/` snapshot has been removed in favor of the ~55 KB embedded catalog JSON.
- `assemble()` now requires `catalog: Catalog` as a keyword argument. Callers obtain it via `catalog.load_catalog_for_config(cfg)`.
- The CI / publish workflows no longer call `scripts/sync_deployments.sh`. Wheel artifacts now go through `scripts/embed_catalog.py` at publish time, and local `uv build` runs the same refresh via a hatch build hook (`scripts/build_hooks.py`).
- `agent-scaffold down` now stops the frontend dev server (SIGTERM the process group, remove the PID file, reset the step state) before tearing down docker compose.
- The legacy "Next steps" panel printed by `cmd_new` is suppressed when autorun fires (the welcome panel covers it). Still printed when `--no-autorun` or `--non-interactive`.

### Removed (breaking)

- `--deployments-source=bundled` flag value. The mode raises `typer.BadParameter` with a migration hint pointing at the catalog flow. The catalog + on-disk fetch cache replaces the bundled snapshot's offline-first-run role.
- `src/agent_scaffold/_bundled_deployments/` (entire tree, 916 KB).
- `scripts/sync_deployments.sh`.
- `ALIAS_TABLE`, `CROSS_CUTTING`, `FRAMEWORK_LANGUAGE`, `FRAMEWORK_DOC_TO_ID`, `_BLUEPRINT_URL_RE` constants from `context.py`. The catalog provides equivalent data via `catalog.aliases`, `catalog.cross_cutting`, `catalog.frameworks`, and `catalog.build_secondary_url_re()`.

### Migration

| If you were doing... | Do this instead |
|---|---|
| `agent-scaffold new --deployments-source bundled` | Drop the flag; the catalog + cache covers offline runs after the first fetch. For air-gapped CI, pre-warm `~/.cache/agent-scaffold/` or pin `--catalog-url file://...`. |
| `from agent_scaffold.context import ALIAS_TABLE` | Load a `Catalog` via `catalog.load_catalog_for_config(cfg)`, then read `catalog.aliases`. |
| `from agent_scaffold.context import _BLUEPRINT_URL_RE` | Use `catalog.build_secondary_url_re(catalog)`. |
| `from agent_scaffold._bundled_deployments import bundled_docs_path` | No replacement. Bundle is gone. The runtime tarball cache at `~/.cache/agent-scaffold/deployments/<sha>/` covers offline docs once first-fetched. |

## 0.2.255 — 2026-05-28

The first minor since `0.1.1`. Headline change: the new **`agent-scaffold scaffold` REPL** — a persistent shell with a guided wizard, slash commands, LLM-interpreted refinements, pre-flight cost estimates, and a tab-completing prompt — replaces the one-shot prompt-for-each-input flow that `agent-scaffold new` used to drive. `new` still works for scripted / non-interactive runs. Also: deployments + blueprints now auto-fetch from GitHub (cached by SHA, ETag-conditional), and the CLI itself has been pulled apart into smaller focused modules (`cli_auth`, `cli_doctor`, `cli_secrets`, plus shared leafs `effort` + `language_hints` + `_scaffold_dir`).

### Highlights

- **Interactive REPL: `agent-scaffold scaffold`.** Persistent shell with the orange→red figlet banner, a `/new` wizard, free-text refinements ("swap to sonnet, add postgres, skip the smoke test"), pre-flight cost estimates, and tab-completion. Stays open until `/exit` so multiple projects can be scaffolded in one session.
- **Auto-fetched deployments + blueprints.** `agent-scaffold new` no longer prompts for a path. The CLI pulls the latest `main` commit from `jagguvarma15/agent-deployments` and `jagguvarma15/agent-blueprints`, caches each by SHA under `~/.cache/agent-scaffold/`, and uses ETag-conditional GETs so unchanged refs don't consume GitHub rate-limit quota. Falls back to the bundled deployments copy when offline.
- **Live bug fix:** REPL free-text refinements actually reach the generator now. Earlier preview builds collected them into `SessionState` and rendered them in the delta panel but never threaded them into `PipelineInputs` — so prose like `"add postgres, swap to sonnet"` was a silent no-op.

### Added

- **`agent-scaffold scaffold` — interactive REPL.** Persistent shell with the orange→red figlet banner, slash commands (`/help`, `/recipe`, `/language`, `/framework`, `/name`, `/dest`, `/model`, `/effort`, `/plan`, `/cost`, `/reset`, `/go`, `/exit`), tab-completion for commands + recipe slugs, history at `~/.cache/agent-scaffold/repl_history`, Ctrl-D / Ctrl-L key bindings.
- **`/new` guided wizard with arrow-key selection.** Steps through recipe → language → framework → name → dest via questionary picks; each step has a `pause wizard` option that preserves selections so a follow-up `/new` resumes from the first unset field with a keep/change gate.
- **`/generate` (alias `/gen`)** — user-facing verb for the final confirm step; both route through the existing `cmd_go` validator.
- **`agent_scaffold.branding`** — shared figlet+gradient logo used by both the top-level `agent-scaffold` banner and the REPL welcome screen so they stay visually consistent.
- **`prompt-toolkit>=3.0,<4`** pinned as an explicit dependency (previously transitive via questionary; the shell drives `PromptSession`, `FileHistory`, and `patch_stdout` directly).
- **REPL slash-command dispatcher.** `agent_scaffold.repl.commands.CommandHandler` introspects its own `cmd_*` methods to register slash commands with bare-recipe-slug shortcuts and fuzzy "did you mean" matching on typos.
- **LLM-interpreted free-text refinements in the REPL.** Anything that isn't a slash command or a bare recipe slug goes through a Haiku-only `interpret_refinement` call that turns prose like `"swap to sonnet and skip the smoke test"` into a typed `StatePatch`. ~$0.002/call. Network failures, malformed JSON, and schema-mismatched values surface a yellow warning and leave state intact rather than crashing the loop.
- **Pre-flight cost estimate in the plan panel.** The plan-confirm panel now shows `Est. cost: $X (input $Y, output ~$Z ±$W)` before any LLM call, so you see what a run will cost before paying. Input cost is exact (from the assembled context size); output is a range bracketed by the configured `--max-tokens`.
- **Auto-fetch deployments + blueprints from GitHub.** `agent-scaffold new` no longer prompts for the deployments path. By default the CLI pulls the latest `main` commit from both `jagguvarma15/agent-deployments` and `jagguvarma15/agent-blueprints`, caches each by SHA under `~/.cache/agent-scaffold/`, and uses ETag-conditional GETs so unchanged refs don't consume GitHub rate-limit quota. Falls back to the bundled deployments copy when offline.
- **Blueprint URL rewriting in context assembly.** `https://github.com/jagguvarma15/agent-blueprints/(tree|blob|raw)/main/<path>` links in deployments docs now resolve to local files in the fetched blueprints tree, so the LLM actually reads the canonical pattern content the deployments docs point to (subject to the existing context budget).
- New flags on `agent-scaffold new`: `--blueprints-path`, `--deployments-source [auto|bundled]`, `--blueprints-source [auto|skip]`.
- New env vars: `AGENT_SCAFFOLD_BLUEPRINTS_PATH`, `AGENT_SCAFFOLD_DEPLOYMENTS_SOURCE`, `AGENT_SCAFFOLD_BLUEPRINTS_SOURCE`.
- New module `agent_scaffold.sources` with safe-extract tarball handling (rejects `..`, absolute paths, symlink escapes — Python 3.11 floor, no `tarfile.data_filter` dependency).

### Changed

- `Config.deployments_path` is now `Path | None` — empty means "use the resolver default" (auto-fetch). `load_config` no longer raises when no path is set; resolution is deferred to `sources.resolve_*`.
- Removed the "Path to agent-deployments repo:" interactive prompt from `agent-scaffold new`.
- **New leaf modules `agent_scaffold.effort` and `agent_scaffold.language_hints`** carry the shared effort-preset table and language-YAML loader respectively. `cli.py` and `repl/*.py` now import from them instead of each carrying a near-copy.
- **New `topology.resolve(recipe, ctx_body)` helper** returns the `(Topology, list[Role])` pair so `cmd_new`, `cmd_plan`, and `_build_pipeline_inputs` no longer duplicate the explicit-frontmatter / inference / `SINGLE` fallback dance verbatim.
- **Split `cli.py` (2615 LOC → 1883 LOC) into focused sibling modules.** `cli_auth.py` owns the `auth` sub-app + credentials commands; `cli_secrets.py` owns the `secrets list` / `secrets purge` survey + flow; `cli_doctor.py` owns the `doctor` sub-app, the `--explain` lookup, and the auth/service `Check` adapters. A new `cli_shared.py` holds the single `console = Console()` singleton imported by every sibling. `cli.py` keeps the project-generation pipeline (`cmd_new`, `cmd_update`, `cmd_regenerate`, `cmd_scaffold`, `cmd_up`, `cmd_validate`, `cmd_config`) and registers the sub-apps via `app.add_typer(...)`.

### Fixed

- **REPL free-text refinements now reach the generator.** `SessionState.extra_dependencies`, `extra_steps`, `removed_steps`, `removed_roles`, and `refinement_notes` were collected and rendered in the delta panel but never threaded into `PipelineInputs` or the LLM prompt — so prose like `"add postgres, swap to sonnet, skip docker_up"` was a silent no-op. Added the five fields to `PipelineInputs`, `GenerationRequest`, and the `cache_inputs` fingerprint; rendered them as a `# User refinements` block in the user-message tail (per-run, never cached); `_build_pipeline_inputs` now passes them through from `SessionState`.
- **`/effort high` in the REPL now matches `--effort high` on the CLI.** The two surfaces each carried their own `EFFORT_PRESETS` dict; the REPL's was missing `max_context_tokens`/`max_link_depth`/`max_tokens_per_doc`, so the same keyword silently produced different context budgets. Unified into a single `agent_scaffold.effort.EFFORT_PRESETS` mapping typed as `EffortPreset` frozen dataclasses.
- **REPL `/language` validation now picks up new language YAMLs automatically.** Replaced the `_VALID_LANGUAGES = ("python", "typescript")` constant with a call to `agent_scaffold.language_hints.available_languages()`, so dropping in `rust.yaml` is picked up by both `/language` validation and the wizard list without code changes.

### Performance

- **Cached the bundled prompt reads.** `generator._load_prompt` and `prompts_signature` are now wrapped in `functools.lru_cache` — the wheel-bundled prompt files don't change at runtime, so the prior behaviour of re-reading and re-hashing them on every `run_generation` was pure waste.
- **Per-state assemble cache in the REPL.** `repl/commands._assemble_for_state` wraps `context.assemble` with a small LRU keyed on every input that could change the output (recipe + paths + budgets). The `/plan` → `/cost` flow used to walk the blueprint tree twice; it now walks it once per state change.

### Typing

- **`pipeline._run_post_gen_formatter` renamed to `pipeline.run_post_gen_formatter`.** Drops the leading underscore on a function that was both re-exported via `__all__` and imported by `cli.cmd_regenerate` — the underscore was misleading public-vs-private signalling.
- **`pipeline.RunReport.report` is now typed `WriteReport | None`** instead of `Any | None`. The comment claiming the cycle ("typed Any to avoid an import cycle in writer") was incorrect — `writer.py` is a leaf module. `mypy --strict` still passes.

### Internal

- **New `agent_scaffold._scaffold_dir.SCAFFOLD_DIR = ".scaffold"`** centralises the per-project metadata directory name. Updated `manifest.py`, `orchestrator.py`, `template_snapshot.py`, `writer.py`, `cli.py`, and `steps/commit_push.py` to import it instead of spelling the literal six times.
- **REPL wizard step table.** `repl/shell._run_new_wizard` now drives its five steps (recipe / language / framework / name / dest) from a `_WIZARD_STEPS` tuple of `_WizardStep` dataclasses instead of five copy-pasted 7-line blocks. Adding a sixth step is now a single table row.
- **Dropped the `agent_scaffold.repl` package re-exports.** Every existing caller (in-tree and tests) already imports from the symbol's owning submodule; the `__init__.py` re-exports were pure noise.

### Versioning note

Patch number is the count of commits to `main` since `21dcbfa` (the `0.1.1` PyPI-rename commit). Future releases on the `0.2.x` line will keep this scheme until a breaking change moves to `0.3`.

## 0.1.1 — 2026-05-06

### Fixed
- Source `__version__` from installed package metadata so `agent-scaffold --version` matches the published wheel.

### Changed
- Renamed PyPI distribution to `agent-scaffold-cli` (CLI command remains `agent-scaffold`).
- Aligned `.gitignore` and wheel build with the `agent-scaffold` rename.

## 0.1.0 — 2026-05-03

### Added
- Initial public release on PyPI.
- `agent-scaffold new` interactive flow: pick a recipe, language, framework, project name, and destination.
- `agent-scaffold validate --tier {static,build,smoke}` reruns post-generation tiers without re-invoking the LLM.
- `agent-scaffold config` prints the resolved configuration with the API key masked.
- Pipeline: `config → discovery → context → generator → contract → writer → validator`.
- Bundled `agent-deployments` docs shipped inside the wheel; override with `AGENT_SCAFFOLD_DEPLOYMENTS_PATH` or `--deployments-path`.
- Pluggable language targets via YAML (`src/agent_scaffold/languages/`), with `python.yaml` and `typescript.yaml` shipped.
- Atomic writes via temp dir + `os.replace`; `--write-mode` choices: `abort` (default), `skip`, `diff`, `overwrite`.
- Path-safety validation in the contract layer (rejects absolute paths, parent traversal, symlink escapes).
- Failure capture: malformed contracts written to `~/.cache/agent-scaffold/failures/<timestamp>.json` with automatic repair-attempt round-trip.
