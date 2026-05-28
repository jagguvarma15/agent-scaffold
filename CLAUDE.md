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
```

## Architecture

Pipeline: `config → sources → discovery → context → generator → contract → writer → validator`

| Module | Responsibility |
|--------|---------------|
| `config.py` | Load env vars + TOML config, resolve `Config`. Deployments / blueprints paths are optional hints. |
| `sources.py` | Resolve where deployments + blueprints come from: GitHub auto-fetch (cached by SHA, ETag-conditional) with bundled / skip fallback. |
| `discovery.py` | Scan `agent-deployments/docs/recipes/` for markdown recipe specs |
| `context.py` | Assemble recipe + linked docs into a single prompt context. Rewrites `github.com/.../agent-blueprints/...` URLs in deployments docs to local files in the fetched blueprints tree. |
| `generator.py` | Call Anthropic API with system/user prompts, retry on transient errors |
| `contract.py` | Parse + validate JSON response (path safety, required files) |
| `writer.py` | Atomic file writing with skip/diff/overwrite modes |
| `validator.py` | Post-generation validation: static lint, build, smoke check |
| `cli.py` | Typer CLI orchestrating the full pipeline |

## Conventions

- Python 3.11+, strict mypy, ruff for lint+format
- Pydantic models for all data structures
- Custom exceptions carry context (e.g., `ContractParseError.raw`, `.reason`)
- Tests use monkeypatching for Anthropic client; fixtures in `tests/fixtures/`
- src layout: `src/agent_scaffold/`
