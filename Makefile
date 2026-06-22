.PHONY: build test lint typecheck install-dev uninstall-dev

# `uv build` runs the hatch custom hook (scripts/build_hooks.py), which refreshes
# src/agent_scaffold/_embedded_catalog.json from the live catalog before the
# wheel is assembled. No pre-step needed.
build:
	uv build

test:
	uv run pytest --cov=agent_scaffold

lint:
	uv run ruff check src/ tests/
	uv run mypy src/

typecheck:
	uv run mypy src/

# Install both `agent-scaffold` and `scaffold` binaries onto PATH as an editable
# tool, so changes to src/ are picked up without reinstalling.
install-dev:
	uv tool install --force --editable .

uninstall-dev:
	uv tool uninstall agent-scaffold-cli
