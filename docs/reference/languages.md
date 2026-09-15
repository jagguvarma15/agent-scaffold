# Target languages

Language profiles are YAML files bundled with the CLI. To add a new target language, drop a YAML file into [`src/agent_scaffold/languages/`](https://github.com/jagguvarma15/agent-scaffold/tree/main/src/agent_scaffold/languages) modeled after [python.yaml](https://github.com/jagguvarma15/agent-scaffold/blob/main/src/agent_scaffold/languages/python.yaml) or [typescript.yaml](https://github.com/jagguvarma15/agent-scaffold/blob/main/src/agent_scaffold/languages/typescript.yaml). Required keys:

- `language`, `package_manager`, `project_layout`, `entry_point`, `manifest`
- `required_tools` (formatter / type_checker / test)
- `pinned_dependencies`, `framework_dependencies`
- `forbidden`, `smoke_check`

The CLI reads them on demand; no code changes needed unless you also want a language-specific static-validation tier (see [`validator.py`](https://github.com/jagguvarma15/agent-scaffold/blob/main/src/agent_scaffold/validator.py)).

## Local bring-up per language

`agent-scaffold up` without `--docker` provisions both tracks:

| Step | Python | TypeScript |
|---|---|---|
| `install_deps` | `uv lock` + `uv sync` | the lockfile's manager: `pnpm install --frozen-lockfile`, `npm ci`, or `yarn install --frozen-lockfile` (plain `pnpm install` when no lockfile exists yet) |
| `launch_backend` | `uv run python -m <module>` or `uv run uvicorn <module>:app` | the package.json `dev` / `start` script when present, else `tsx` on the manifest-recorded entry point |
| `smoke_test` | `scripts/smoke.sh`, else `pytest -m smoke` | `scripts/smoke.sh`, else the package.json `smoke` script, else the manifest-recorded `smoke_check` command |

The TypeScript backend binds the profile's default port (3000) unless the project ships a `frontend/` package — the frontend dev server owns 3000, so the backend falls back to 8000. Both tracks export `PORT` to the spawned process.
