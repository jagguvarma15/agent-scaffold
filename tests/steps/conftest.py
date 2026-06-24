"""Fixtures shared across the steps test suite."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from agent_scaffold.discovery import ExternalService, Recipe
from agent_scaffold.manifest import Manifest
from agent_scaffold.orchestrator import (
    OrchestratorState,
    StepContext,
    StepEvent,
)


@pytest.fixture
def manifest_factory() -> Callable[..., Manifest]:
    """Build a Manifest with sensible defaults; override per test as needed."""

    def make(
        *,
        language: str = "python",
        recipe: str = "test-recipe",
        framework: str = "none",
        model: str = "claude-test",
        entry_point: str | None = None,
    ) -> Manifest:
        return Manifest(
            recipe=recipe,
            language=language,
            framework=framework,
            model=model,
            generated_at="2026-05-24T00:00:00+00:00",
            entry_point=entry_point,
        )

    return make


@pytest.fixture
def event_log() -> list[StepEvent]:
    return []


@pytest.fixture
def ctx_factory(
    tmp_path: Path,
    manifest_factory: Callable[..., Manifest],
    event_log: list[StepEvent],
) -> Callable[..., StepContext]:
    def make(
        project_dir: Path | None = None,
        manifest: Manifest | None = None,
        callback: Callable[[StepEvent], None] | None = None,
        resolved_stack: object | None = None,
    ) -> StepContext:
        m = manifest or manifest_factory()
        if callback is None:
            callback = event_log.append
        return StepContext(
            project_dir=project_dir or tmp_path,
            manifest=m,
            state=OrchestratorState(),
            callback=callback,
            timeout=30.0,
            resolved_stack=resolved_stack,
        )

    return make


@pytest.fixture
def recipe_factory() -> Callable[..., Recipe]:
    def make(
        *,
        slug: str = "test-recipe",
        external_services: list[ExternalService] | None = None,
    ) -> Recipe:
        return Recipe(
            slug=slug,
            title="Test Recipe",
            path=Path("/nonexistent/recipe.md"),
            external_services=external_services or [],
        )

    return make


@pytest.fixture
def patch_load_recipe(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Callable[[Recipe | None], None]]:
    """Stub the per-step ``_load_recipe`` helpers so tests don't touch disk."""

    handles: list[tuple[str, str]] = []

    def install(recipe: Recipe | None) -> None:
        for mod_name, attr in (
            ("agent_scaffold.steps.docker_up", "_load_recipe"),
            ("agent_scaffold.steps.wire_credentials", "_load_recipe"),
            ("agent_scaffold.steps.migrations", "_load_recipe"),
            ("agent_scaffold.steps.seed", "_load_recipe"),
        ):
            monkeypatch.setattr(f"{mod_name}.{attr}", lambda _ctx, r=recipe: r)
            handles.append((mod_name, attr))

    yield install
