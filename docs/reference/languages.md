# Target languages

Language profiles are YAML files bundled with the CLI. To add a new target language, drop a YAML file into [`src/agent_scaffold/languages/`](https://github.com/jagguvarma15/agent-scaffold/tree/main/src/agent_scaffold/languages) modeled after [python.yaml](https://github.com/jagguvarma15/agent-scaffold/blob/main/src/agent_scaffold/languages/python.yaml) or [typescript.yaml](https://github.com/jagguvarma15/agent-scaffold/blob/main/src/agent_scaffold/languages/typescript.yaml). Required keys:

- `language`, `package_manager`, `project_layout`, `entry_point`, `manifest`
- `required_tools` (formatter / type_checker / test)
- `pinned_dependencies`, `framework_dependencies`
- `forbidden`, `smoke_check`

The CLI reads them on demand; no code changes needed unless you also want a language-specific static-validation tier (see [`validator.py`](https://github.com/jagguvarma15/agent-scaffold/blob/main/src/agent_scaffold/validator.py)).
