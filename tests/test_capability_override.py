"""Tests for user-level capability overrides in resolve() + StatePatch.

The REPL lets the user swap obs.langsmith → obs.langfuse without forking
the recipe markdown. ``add_capabilities`` / ``remove_capabilities`` carry
the swap; ``capabilities.resolve()`` honors them at resolution time.
"""

from __future__ import annotations

from pathlib import Path

from agent_scaffold.capabilities import (
    load_capabilities,
    resolve,
)
from agent_scaffold.config import Config
from agent_scaffold.discovery import Recipe
from agent_scaffold.repl.session import (
    ResolvedSource,
    SessionState,
    StatePatch,
    apply_patch,
)
from agent_scaffold.sources import DEPLOYMENTS_SPEC


def _recipe(slug: str, capabilities: list[str], tmp_path: Path) -> Recipe:
    return Recipe(slug=slug, title="t", path=tmp_path / f"{slug}.md", capabilities=capabilities)


def test_resolve_adds_user_capability(
    mock_deployments_path: Path, tmp_path: Path
) -> None:
    """add_capabilities layers on top; recipe order wins for overlap."""
    catalog = load_capabilities(mock_deployments_path)
    recipe = _recipe("demo", ["cache.redis"], tmp_path)
    stack = resolve(recipe, catalog, add_capabilities=["obs.langfuse"])
    ids = stack.ids()
    assert ids[0] == "cache.redis"
    assert "obs.langfuse" in ids


def test_resolve_removes_recipe_capability(
    mock_deployments_path: Path, tmp_path: Path
) -> None:
    """remove_capabilities drops before resolution (never lands in unresolved)."""
    catalog = load_capabilities(mock_deployments_path)
    recipe = _recipe("demo", ["obs.langsmith", "cache.redis"], tmp_path)
    stack = resolve(
        recipe, catalog, remove_capabilities={"obs.langsmith"}
    )
    assert "obs.langsmith" not in stack.ids()
    assert "obs.langsmith" not in stack.unresolved
    assert "cache.redis" in stack.ids()


def test_resolve_swap_observability_backend(
    mock_deployments_path: Path, tmp_path: Path
) -> None:
    """The real-world case: swap obs.langsmith for obs.langfuse."""
    catalog = load_capabilities(mock_deployments_path)
    recipe = _recipe("demo", ["obs.langsmith"], tmp_path)
    stack = resolve(
        recipe,
        catalog,
        add_capabilities=["obs.langfuse"],
        remove_capabilities={"obs.langsmith"},
    )
    ids = stack.ids()
    assert "obs.langfuse" in ids
    assert "obs.langsmith" not in ids


def test_resolve_dedupes_user_additions(
    mock_deployments_path: Path, tmp_path: Path
) -> None:
    """User-added cap that the recipe already declares is a no-op."""
    catalog = load_capabilities(mock_deployments_path)
    recipe = _recipe("demo", ["obs.langsmith"], tmp_path)
    stack = resolve(recipe, catalog, add_capabilities=["obs.langsmith"])
    assert stack.ids().count("obs.langsmith") == 1


def _empty_state(tmp_path: Path) -> SessionState:
    cfg = Config(
        anthropic_api_key="test-key",
        cache_dir=tmp_path / "cache",
        failures_dir=tmp_path / "cache" / "failures",
    )
    src = ResolvedSource(
        spec=DEPLOYMENTS_SPEC,
        path=tmp_path / "deployments",
        label="test",
        kind="explicit-path",
        commit_sha=None,
    )
    return SessionState(cfg=cfg, deployments=src, blueprints=src)


def test_apply_patch_accumulates_capability_overrides(tmp_path: Path) -> None:
    """Two successive patches compose: first adds langfuse, second drops langsmith."""
    state = _empty_state(tmp_path)
    state = apply_patch(state, StatePatch(add_capabilities=["obs.langfuse"]))
    state = apply_patch(state, StatePatch(remove_capabilities=["obs.langsmith"]))
    assert state.add_capabilities == ["obs.langfuse"]
    assert state.remove_capabilities == {"obs.langsmith"}


def test_apply_patch_remove_supersedes_earlier_add(tmp_path: Path) -> None:
    """If a later patch removes what an earlier patch added, the remove wins."""
    state = _empty_state(tmp_path)
    state = apply_patch(state, StatePatch(add_capabilities=["obs.langfuse"]))
    state = apply_patch(state, StatePatch(remove_capabilities=["obs.langfuse"]))
    assert state.add_capabilities == []
    assert state.remove_capabilities == {"obs.langfuse"}


def test_apply_patch_add_after_remove_clears_removal(tmp_path: Path) -> None:
    """Re-adding a previously removed cap honors the add (drops from removals)."""
    state = _empty_state(tmp_path)
    state = apply_patch(state, StatePatch(remove_capabilities=["obs.langfuse"]))
    state = apply_patch(state, StatePatch(add_capabilities=["obs.langfuse"]))
    assert state.add_capabilities == ["obs.langfuse"]
    assert state.remove_capabilities == set()
