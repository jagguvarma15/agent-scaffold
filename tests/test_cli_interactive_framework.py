"""Recipe-aware framework validation on the flag-driven CLI path.

The REPL wizard filters its questionary choices; this locks the same rule
into ``cli_interactive._select_framework`` for ``agent-scaffold new
--framework <x>``: a framework the recipe's declared dependencies cannot
generate is a hard BadParameter, not a silently recorded label.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import typer

from agent_scaffold.cli_interactive import _select_framework
from agent_scaffold.discovery import Recipe


def _write_framework_docs(root: Path) -> None:
    fw = root / "docs" / "frameworks"
    fw.mkdir(parents=True, exist_ok=True)
    (fw / "pydantic-ai.md").write_text(
        "---\nid: pydantic_ai\nlanguage: python\npackage: pydantic-ai\n"
        'versions:\n  minimum: ">=0.1.0"\n---\n\nBody.\n',
        encoding="utf-8",
    )
    (fw / "crewai.md").write_text(
        "---\nid: crewai\nlanguage: python\npackage: crewai\n"
        'versions:\n  minimum: ">=0.100.0"\n---\n\nBody.\n',
        encoding="utf-8",
    )


def _recipe(tmp_path: Path, deps: dict[str, dict[str, str]] | None = None) -> Recipe:
    md = tmp_path / "demo.md"
    md.write_text("# Demo\n", encoding="utf-8")
    return Recipe(
        slug="demo",
        title="Demo",
        path=md,
        recipe_dependencies=deps or {},
    )


def test_explicit_framework_undeclared_by_recipe_is_bad_parameter(tmp_path: Path) -> None:
    _write_framework_docs(tmp_path)
    recipe = _recipe(tmp_path, {"python": {"pydantic-ai": ">=0.1.0"}})
    with pytest.raises(typer.BadParameter) as exc:
        _select_framework(tmp_path, "python", "crewai", True, recipe=recipe)
    assert "pydantic_ai" in str(exc.value)


def test_explicit_framework_declared_by_recipe_passes_and_normalizes(tmp_path: Path) -> None:
    _write_framework_docs(tmp_path)
    recipe = _recipe(tmp_path, {"python": {"pydantic-ai": ">=0.1.0"}})
    assert (
        _select_framework(tmp_path, "python", "pydantic-ai", True, recipe=recipe) == "pydantic_ai"
    )


def test_none_is_always_allowed(tmp_path: Path) -> None:
    _write_framework_docs(tmp_path)
    recipe = _recipe(tmp_path, {"python": {"pydantic-ai": ">=0.1.0"}})
    assert _select_framework(tmp_path, "python", "none", True, recipe=recipe) == "none"


def test_agnostic_recipe_keeps_language_list(tmp_path: Path) -> None:
    _write_framework_docs(tmp_path)
    recipe = _recipe(tmp_path, {"python": {"redis": ">=5.0.0"}})
    assert _select_framework(tmp_path, "python", "crewai", True, recipe=recipe) == "crewai"
