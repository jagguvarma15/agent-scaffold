.PHONY: sync-deployments build test lint typecheck

sync-deployments:
	bash scripts/sync_deployments.sh

build: sync-deployments
	uv build

test:
	uv run pytest --cov=agent_forge

lint:
	uv run ruff check src/ tests/
	uv run mypy src/

typecheck:
	uv run mypy src/
