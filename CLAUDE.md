# agent-scaffold

Generate runnable AI agent projects from markdown specs in an `agent-deployments` repo.

## Commands

```bash
uv run pytest                  # run tests
uv run pytest -m integration   # run integration tests only
uv run ruff check src/ tests/  # lint
uv run ruff format src/ tests/ # format
uv run mypy src/               # type check
uv run agent-scaffold --help      # CLI usage
make install-dev                # expose `scaffold` + `agent-scaffold` on PATH (editable)
```

## Architecture

Pipeline: `config → catalog → sources → discovery → context → pipeline.run_generation → cli`

| Module | Responsibility |
|--------|---------------|
| `config.py` | Load env vars + TOML config, resolve `Config`. Deployments / blueprints paths are optional hints. |
| `catalog.py` | Load + validate the deployments catalog: synced-tree copy first (`catalog.yaml` inside the sources cache, same commit as the docs), then network fetch of `DEFAULT_CATALOG_URL` (explicit URL overrides always fetch). Pydantic models, ETag-cached fetch, embedded JSON fallback. Provides the alias / cross-cutting / framework-gating maps `context.assemble` consumes. |
| `bundles.py` | Named flat capability bundles (rag-simple, rag-complex, guardrails-basic): catalog-published `bundles:` with embedded fallbacks. Expanded into `resolve(add_capabilities=...)` by the wizard's RAG step and `new --bundle`. |
| `repl/_fuzzy.py` | rapidfuzz-backed fuzzy matching for the REPL: `suggest` (did-you-mean), `filter_matches` (/stack, /recipe narrowing), `completions` (tab completion). Degrades to a difflib fallback (with a warning) when rapidfuzz is missing from a stale env instead of crashing the shell; difflib otherwise stays only for 3-way merge and unified diffs. |
| `sources.py` | Resolve where deployments + blueprints come from: GitHub auto-fetch (cached by SHA, ETag-conditional, 300s HEAD TTL). Kw-only `refresh=True` bypasses the TTL; the REPL syncs at startup by default (`--no-sync` opts out). Blueprints supports `skip` mode; deployments fetches or fails. |
| `discovery.py` | Scan `agent-deployments/docs/recipes/` for markdown recipe specs |
| `context.py` | Assemble recipe + linked docs into a single prompt context. Rewrites `github.com/.../agent-blueprints/...` URLs in deployments docs to local files in the fetched blueprints tree. |
| `generator.py` | Call Anthropic API with system/user prompts, retry on transient errors |
| `contract.py` | Parse + validate JSON response (path safety, required files) |
| `writer.py` | Atomic file writing with skip/diff/overwrite modes |
| `validator.py` | Post-generation validation: static lint, build, smoke check |
| `pipeline.py` | Post-plan orchestration (`run_generation`): generate → write → gitignore → verify → format → validate → manifest. Reusable by `cmd_new` and the `scaffold` REPL. Recoverable failures raise `PipelineError`. |
| `repl/` | Interactive shell (`agent-scaffold scaffold`). `session.py` (state + StatePatch), `commands.py` (slash dispatcher incl. the `/stack` catalog browser), `refine.py` (Haiku-interpreted free text), `render.py` (Rich panels), `shell.py` (PromptSession loop; the `/new` wizard walks mandatory steps then an optional-features menu gating RAG/observability/guardrails/layer steps), `drafts.py` (persisted selection drafts, LRU 3, retired after generate), `readiness.py` (config gate), `_capabilities.py` (stack resolution + hosting overrides). |
| `cli.py` | Typer CLI: prompt collection + plan-confirm + delegate to `pipeline.run_generation`. `cmd_scaffold` opens the REPL. |

## Conventions

- Python 3.11+, strict mypy, ruff for lint+format
- Pydantic models for all data structures
- Custom exceptions carry context (e.g., `ContractParseError.raw`, `.reason`)
- Tests use monkeypatching for Anthropic client; fixtures in `tests/fixtures/`
- src layout: `src/agent_scaffold/`
