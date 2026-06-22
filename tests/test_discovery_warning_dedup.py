"""Process-level dedupe of repeated discovery/capability warnings.

Before this fix the same malformed-recipe warning fired ~150 times in one
orchestrator run because every bootstrap step re-calls `discover_recipes`
(and a few re-call `load_capabilities`). The dedupe set inside `_warn`
caps each unique message at one print per process.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_scaffold.capabilities import load_capabilities
from agent_scaffold.discovery import discover_recipes


def _write_recipe_with_bad_external_services(deployments: Path) -> None:
    recipes = deployments / "docs" / "recipes"
    recipes.mkdir(parents=True, exist_ok=True)
    (recipes / "broken.md").write_text(
        "---\n"
        "title: Broken Recipe\n"
        "status: validated\n"
        "external_services:\n"
        "  - {}\n"  # malformed: mapping with no id
        "  - foo: bar\n"  # malformed: mapping missing id
        "---\n\n# Broken Recipe\n",
        encoding="utf-8",
    )


def test_discovery_warning_fires_once_per_process(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    deployments = tmp_path / "deployments"
    _write_recipe_with_bad_external_services(deployments)

    # First call: malformed-entry warnings should print.
    discover_recipes(deployments)
    first_err = capsys.readouterr().err
    assert first_err.count("broken.md: external_services[0]") == 1
    assert first_err.count("broken.md: external_services[1]") == 1

    # Second + third calls: same warnings must NOT re-emit.
    discover_recipes(deployments)
    discover_recipes(deployments)
    repeat_err = capsys.readouterr().err
    assert "external_services" not in repeat_err


def test_capabilities_warning_fires_once_per_process(
    mock_deployments_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The mock fixture ships a malformed.no_frontmatter capability that
    triggers `missing frontmatter`. Three loads → exactly one print."""
    load_capabilities(mock_deployments_path)
    load_capabilities(mock_deployments_path)
    load_capabilities(mock_deployments_path)
    err = capsys.readouterr().err
    assert err.count("missing frontmatter") == 1
