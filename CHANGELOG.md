# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added
- **`agent-scaffold scaffold` — interactive REPL.** Persistent shell with the orange→red figlet banner, slash commands (`/help`, `/recipe`, `/language`, `/framework`, `/name`, `/dest`, `/model`, `/effort`, `/plan`, `/cost`, `/reset`, `/go`, `/exit`), tab-completion for commands + recipe slugs, history at `~/.cache/agent-scaffold/repl_history`, Ctrl-D / Ctrl-L key bindings. Stays open until `/exit` so multiple projects can be scaffolded in one session.
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
- **Split `cli.py` (2615 LOC → 1883 LOC) into focused sibling modules.** `cli_auth.py` owns the `auth` sub-app + credentials commands; `cli_secrets.py` owns the `secrets list` / `secrets purge` survey + flow; `cli_doctor.py` owns the `doctor` sub-app, the `--explain` lookup, and the auth/service `Check` adapters. A new `cli_shared.py` holds the single `console = Console()` singleton imported by every sibling. `cli.py` keeps the project-generation pipeline (`cmd_new`, `cmd_update`, `cmd_regenerate`, `cmd_scaffold`, `cmd_up`, `cmd_validate`, `cmd_config`) and registers the sub-apps via `app.add_typer(...)`. Tests that previously monkeypatched `agent_scaffold.cli.X` were updated to target the new owning module.

### Fixed
- **`/effort high` in the REPL now matches `--effort high` on the CLI.** The two surfaces each carried their own `EFFORT_PRESETS` dict; the REPL's was missing `max_context_tokens`/`max_link_depth`/`max_tokens_per_doc`, so the same keyword silently produced different context budgets. Unified into a single `agent_scaffold.effort.EFFORT_PRESETS` mapping typed as `EffortPreset` frozen dataclasses.
- **REPL `/language` validation now picks up new language YAMLs automatically.** Replaced the `_VALID_LANGUAGES = ("python", "typescript")` constant with a call to `agent_scaffold.language_hints.available_languages()`, so dropping in `rust.yaml` is picked up by both `/language` validation and the wizard list without code changes.

### Fixed
- **REPL free-text refinements now reach the generator.** `SessionState.extra_dependencies`, `extra_steps`, `removed_steps`, `removed_roles`, and `refinement_notes` were collected and rendered in the delta panel but never threaded into `PipelineInputs` or the LLM prompt — so prose like `"add postgres, swap to sonnet, skip docker_up"` was a silent no-op. Added the five fields to `PipelineInputs`, `GenerationRequest`, and the `cache_inputs` fingerprint; rendered them as a `# User refinements` block in the user-message tail (per-run, never cached); `_build_pipeline_inputs` now passes them through from `SessionState`.

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
