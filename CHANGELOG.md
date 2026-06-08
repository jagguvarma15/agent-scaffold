# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

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
