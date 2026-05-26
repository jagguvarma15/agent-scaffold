# Contributing to agent-scaffold

Thanks for your interest! This guide covers bug reports, feature requests, language-target additions, and code contributions.

## Getting started

```bash
git clone https://github.com/jagguvarma15/agent-scaffold
cd agent-scaffold
uv sync
uv run pytest
```

The full toolchain (matches CI):

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run mypy src/
uv run pytest
uv run pytest -m integration   # requires ANTHROPIC_API_KEY
```

## Project layout

`agent-scaffold` is structured as a pipeline:

```
config → discovery → context → generator → contract → writer → validator
```

Each module owns one stage; please keep contributions scoped to the stage they belong to. See [`CLAUDE.md`](CLAUDE.md) for the responsibility table.

## Types of contributions

### Bug reports

Open an issue with:
- The `agent-scaffold --version` output
- The exact command you ran and what you expected
- The generated output or error (redact API keys)
- If the LLM produced a malformed contract, attach the file from `~/.cache/agent-scaffold/failures/`

### Adding a target language

Drop a YAML file into `src/agent_scaffold/languages/` modeled after [`python.yaml`](src/agent_scaffold/languages/python.yaml) or [`typescript.yaml`](src/agent_scaffold/languages/typescript.yaml). Required keys are listed in the README. Add a fixture under `tests/fixtures/` and a discovery test.

### Pipeline changes

Changes to `contract.py` or `writer.py` are security-sensitive (path validation, atomic writes). Please include:
- Tests covering the new behaviour
- Tests covering adversarial inputs (escaping paths, symlinks, absolute paths)
- A short note in the PR explaining why the new behaviour is safe

### Documentation

Improvements to `README.md`, `CLAUDE.md`, or inline docstrings are always welcome.

## Code style

- Python 3.11+, strict mypy, ruff for lint + format (see `pyproject.toml` for config)
- Pydantic models for all data structures
- Custom exceptions carry context (e.g. `ContractParseError.raw`, `.reason`)
- Tests use monkeypatching for the Anthropic client; fixtures live in `tests/fixtures/`
- src layout under `src/agent_scaffold/`

## Commit messages

Use concise, descriptive messages following the existing style:
- `feat: add support for <language>`
- `fix: reject parent-relative paths in writer`
- `chore: bump version to 0.1.2`
- `docs: clarify --write-mode semantics`

## PR checklist

- [ ] `uv run ruff check src/ tests/` passes
- [ ] `uv run mypy src/` passes
- [ ] `uv run pytest` passes
- [ ] Added or updated tests for the changed behaviour
- [ ] Updated `CHANGELOG.md` under `## Unreleased` if user-visible
- [ ] No secrets, API keys, or `.env` files committed

## Security rules (the nine-point checklist)

If your change touches credential handling — auth, wire_credentials, file
writes near secrets, subprocess output that might echo a key — read
[`docs/design/security.md`](docs/design/security.md) first. The summary:

1. Never accept secrets as positional/flag args (use env vars or stdin/getpass).
2. Use `getpass.getpass()` for interactive paste; never `input()` for secrets.
3. Wrap credentials in `pydantic.SecretStr` immediately after capture.
4. `subprocess.run([...], shell=False)` always; list-form arguments only.
5. Use `agent_scaffold._filesec.secure_write` for any file holding a secret.
6. Never trust a plaintext keyring backend — refuse and surface the error.
7. Run any user-visible string that might contain a credential through
   `agent_scaffold._redact.redact` before logging or persisting.
8. Use `agent_scaffold.writer.ensure_gitignore_defaults` to guarantee the
   secret-safety block lands in every project's `.gitignore`.
9. Anything storable must be revocable via `agent-scaffold secrets purge`.

Each rule has an audit test under `tests/security/`. If your change needs
to bend one of them, add the exception to the corresponding allow-list with
a one-line justification — don't disable the test.

## Questions?

Open a discussion or issue. Happy to help scope a contribution before you invest time building.
