"""Tests for the .scaffold/run-summary.md writer."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_scaffold.contract import GeneratedFile, GenerationResult
from agent_scaffold.discovery import ExternalService, Recipe
from agent_scaffold.run_summary import (
    append_provisioning_section,
    run_summary_path,
    write_run_summary,
)
from agent_scaffold.validator import ValidationResult, ValidationTier


def _recipe(services: list[ExternalService] | None = None) -> Recipe:
    return Recipe(
        slug="docs-rag-qa",
        title="Recipe: docs-rag-qa",
        status="Blueprint (validated)",
        path=Path("docs/recipes/docs-rag-qa.md"),
        external_services=services or [],
    )


def _result() -> GenerationResult:
    return GenerationResult(
        project_name="demo_agent",
        language="python",
        files=[GeneratedFile(path="src/demo_agent/main.py", content="x")],
        post_install=["uv sync"],
        smoke_check="uv run pytest -m smoke",
        known_limitations=["No retry on rate limits"],
    )


def _write(tmp_path: Path, **overrides: object) -> str:
    kwargs: dict = {
        "recipe": _recipe(),
        "language": "python",
        "framework": "langgraph",
        "model": "claude-opus-4-7",
        "result": _result(),
        "template_sha": "a" * 64,
        "validation_results": [
            ValidationResult(tier=ValidationTier.static, passed=True, output=""),
            ValidationResult(tier=ValidationTier.build, passed=True, output=""),
        ],
        "repair_rounds": 0,
        "resolved_stack": None,
        "run_log_dir": "/cache/runs/20260612T000000Z-abc123",
    }
    kwargs.update(overrides)
    path = write_run_summary(tmp_path, **kwargs)
    assert path == run_summary_path(tmp_path)
    return path.read_text(encoding="utf-8")


def test_summary_contains_all_sections(tmp_path: Path) -> None:
    text = _write(tmp_path)
    assert "# Run summary — demo_agent" in text
    assert "`docs-rag-qa` (Blueprint (validated))" in text
    assert "python / langgraph" in text
    assert "claude-opus-4-7" in text
    assert f"`{'a' * 16}`" in text  # shortened template sha
    assert "- Files: 1" in text
    assert "static ✓, build ✓" in text
    assert "No retry on rate limits" in text
    assert "agent-scaffold up" in text
    assert "uv run pytest -m smoke" in text
    assert "/cache/runs/20260612T000000Z-abc123" in text


def test_validation_line_variants(tmp_path: Path) -> None:
    text = _write(tmp_path, validation_results=[], repair_rounds=0)
    assert "- Validation: skipped" in text

    text = _write(
        tmp_path,
        validation_results=[
            ValidationResult(tier=ValidationTier.static, passed=False, output="boom")
        ],
        repair_rounds=2,
    )
    assert "static ✗ FAILING (after 2 repair rounds)" in text


def test_environment_section_names_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://user:supersecretvalue@localhost:6379")
    monkeypatch.delenv("QDRANT_URL", raising=False)
    recipe = _recipe(
        [
            ExternalService(id="redis", env_vars=["REDIS_URL"], required=True),
            ExternalService(id="qdrant", env_vars=["QDRANT_URL"], required=True),
        ]
    )
    text = _write(tmp_path, recipe=recipe)
    assert "`REDIS_URL` — set (redis)" in text
    assert "`QDRANT_URL` — MISSING (qdrant)" in text
    # The value must never appear — names only, by design.
    assert "supersecretvalue" not in text


def test_provisioning_section_appends_and_replaces(tmp_path: Path) -> None:
    _write(tmp_path)
    append_provisioning_section(tmp_path, {"done": 3, "skipped": 1, "failed": 0})
    text = run_summary_path(tmp_path).read_text(encoding="utf-8")
    assert "## Provisioning" in text
    assert "3 done, 1 skipped" in text

    # Second `up` replaces the section instead of stacking another.
    append_provisioning_section(tmp_path, {"done": 4, "skipped": 0, "failed": 0})
    text = run_summary_path(tmp_path).read_text(encoding="utf-8")
    assert text.count("## Provisioning") == 1
    assert "4 done" in text
    assert "3 done" not in text


def test_provisioning_append_is_noop_without_summary(tmp_path: Path) -> None:
    append_provisioning_section(tmp_path, {"done": 1})  # must not raise or create
    assert not run_summary_path(tmp_path).exists()
