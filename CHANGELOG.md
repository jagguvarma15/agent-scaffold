# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

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
