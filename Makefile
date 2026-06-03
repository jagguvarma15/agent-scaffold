.PHONY: sync-deployments build test lint typecheck install-dev uninstall-dev

sync-deployments:
	bash scripts/sync_deployments.sh

build: sync-deployments
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
