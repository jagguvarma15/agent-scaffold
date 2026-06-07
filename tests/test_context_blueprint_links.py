"""Tests for context.assemble rewriting agent-blueprints URLs to local paths."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pytest
import yaml

from agent_scaffold.catalog import Catalog
from agent_scaffold.context import (
    _rewrite_blueprint_url as _real_rewrite,
)
from agent_scaffold.context import (
    _view_from_catalog,
)
from agent_scaffold.context import (
    assemble as _real_assemble,
)
from agent_scaffold.discovery import Recipe

# Catalog became a required kwarg in vX+1; bind the test fixture once and
# shadow the imported symbols so each test body stays unchanged.
_TEST_CATALOG_PATH = Path(__file__).parent / "fixtures" / "catalog_minimal.yaml"
_TEST_CATALOG: Catalog = Catalog.model_validate(
    yaml.safe_load(_TEST_CATALOG_PATH.read_text(encoding="utf-8"))
)
_TEST_VIEW = _view_from_catalog(_TEST_CATALOG)

assemble = partial(_real_assemble, catalog=_TEST_CATALOG)
_rewrite_blueprint_url = partial(_real_rewrite, view=_TEST_VIEW)


@pytest.fixture
def blueprints_tree(tmp_path: Path) -> Path:
    """A miniature blueprints tree with an event-driven pattern."""
    root = tmp_path / "blueprints"
    pattern_dir = root / "patterns" / "event-driven"
    pattern_dir.mkdir(parents=True)
    (pattern_dir / "overview.md").write_text(
        "# Event-Driven Overview\n\nCanonical event-driven pattern.\n",
        encoding="utf-8",
    )
    (pattern_dir / "design.md").write_text("# Event-Driven Design\n", encoding="utf-8")
    return root


@pytest.fixture
def deployments_tree(tmp_path: Path) -> Path:
    """Tiny deployments tree whose recipe links out to blueprints."""
    root = tmp_path / "deployments"
    (root / "docs" / "recipes").mkdir(parents=True)
    recipe_md = root / "docs" / "recipes" / "demo.md"
    recipe_md.write_text(
        "---\n"
        "status: blueprint\n"
        "languages: [python]\n"
        "---\n"
        "# Demo recipe\n\n"
        "Uses the event-driven pattern. See "
        "[blueprints/event-driven]"
        "(https://github.com/jagguvarma15/agent-blueprints/tree/main/patterns/event-driven).\n",
        encoding="utf-8",
    )
    return root


# ---------------------------------------------------------------------------
# _rewrite_blueprint_url
# ---------------------------------------------------------------------------


def test_rewrite_tree_link_targets_overview(blueprints_tree: Path) -> None:
    url = "https://github.com/jagguvarma15/agent-blueprints/tree/main/patterns/event-driven"
    out = _rewrite_blueprint_url(url, blueprints_tree)
    assert out == blueprints_tree / "patterns" / "event-driven" / "overview.md"


def test_rewrite_blob_link_targets_file(blueprints_tree: Path) -> None:
    url = (
        "https://github.com/jagguvarma15/agent-blueprints/blob/main/patterns/event-driven/design.md"
    )
    out = _rewrite_blueprint_url(url, blueprints_tree)
    assert out == blueprints_tree / "patterns" / "event-driven" / "design.md"


def test_rewrite_strips_trailing_slash(blueprints_tree: Path) -> None:
    url = "https://github.com/jagguvarma15/agent-blueprints/tree/main/patterns/event-driven/"
    out = _rewrite_blueprint_url(url, blueprints_tree)
    assert out == blueprints_tree / "patterns" / "event-driven" / "overview.md"


def test_rewrite_rejects_other_repos(blueprints_tree: Path) -> None:
    url = "https://github.com/someone-else/agent-blueprints/tree/main/patterns/x"
    assert _rewrite_blueprint_url(url, blueprints_tree) is None


def test_rewrite_rejects_non_main_branches(blueprints_tree: Path) -> None:
    url = "https://github.com/jagguvarma15/agent-blueprints/tree/develop/patterns/event-driven"
    assert _rewrite_blueprint_url(url, blueprints_tree) is None


def test_rewrite_returns_none_when_no_blueprints_root() -> None:
    url = "https://github.com/jagguvarma15/agent-blueprints/tree/main/patterns/event-driven"
    assert _rewrite_blueprint_url(url, None) is None


def test_rewrite_returns_none_when_file_missing(tmp_path: Path) -> None:
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    url = "https://github.com/jagguvarma15/agent-blueprints/tree/main/patterns/event-driven"
    assert _rewrite_blueprint_url(url, empty_root) is None


# ---------------------------------------------------------------------------
# assemble — blueprints integration
# ---------------------------------------------------------------------------


def _make_recipe(deployments_root: Path) -> Recipe:
    recipe_path = deployments_root / "docs" / "recipes" / "demo.md"
    return Recipe(
        slug="demo",
        title="Demo recipe",
        path=recipe_path,
        status="blueprint",
        languages=["python"],
    )


def test_assemble_pulls_in_blueprint_content_when_path_provided(
    deployments_tree: Path, blueprints_tree: Path
) -> None:
    recipe = _make_recipe(deployments_tree)
    ctx = assemble(
        recipe,
        "python",
        "langgraph",
        deployments_tree,
        blueprints_path=blueprints_tree,
        max_context_tokens=20_000,
    )
    assert "Canonical event-driven pattern." in ctx.body
    assert any(
        p == blueprints_tree / "patterns" / "event-driven" / "overview.md"
        for p in ctx.referenced_paths
    )
    # Marker uses the "blueprints/" prefix so the LLM can tell where each
    # referenced doc originated.
    assert "blueprints/patterns/event-driven/overview.md" in ctx.body


def test_assemble_skips_blueprint_links_when_path_is_none(deployments_tree: Path) -> None:
    recipe = _make_recipe(deployments_tree)
    ctx = assemble(
        recipe,
        "python",
        "langgraph",
        deployments_tree,
        blueprints_path=None,  # offline / skipped
        max_context_tokens=20_000,
    )
    # No crash, just no blueprints content.
    assert "Canonical event-driven pattern." not in ctx.body
    assert ctx.referenced_paths == []  # only the recipe; the URL was dropped
