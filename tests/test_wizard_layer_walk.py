"""Tests for the customize-mode layer walk + Stack-mode auto-skip in repl/shell.py.

Covers the new layer-walk helpers added in Phase 2:
- ``_apply_layer_choice`` diffs the user's pick against the effective stack and
  produces the correct add/remove ``StatePatch``.
- ``_make_layer_step`` builds a ``_WizardStep`` whose ``enabled_when`` only
  fires under ``stack_mode == "customize"``.
- ``_is_basic_recipe`` + ``_apply_stack_mode_quick`` form the auto-skip path
  for basic-tier recipes.

These are unit tests that bypass questionary — the orchestration walk is
exercised via the existing wizard integration test (which now naturally
auto-skips Stack mode for the basic mock recipe).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_scaffold.config import Config
from agent_scaffold.discovery import Recipe
from agent_scaffold.repl.session import SessionState
from agent_scaffold.repl.shell import (
    _apply_layer_choice,
    _default_features_for_recipe,
    _effective_capability_ids,
    _make_layer_step,
)
from agent_scaffold.sources import DEPLOYMENTS_SPEC, ResolvedSource


@pytest.fixture
def base_state(tmp_path: Path) -> SessionState:
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


def _recipe(slug: str, *, capabilities: list[str], topology: str | None = None) -> Recipe:
    """Build a Recipe without touching disk."""
    return Recipe(
        slug=slug,
        title=slug,
        path=Path(f"/tmp/{slug}.md"),
        capabilities=capabilities,
        topology=topology,
    )


def test_effective_ids_merges_recipe_with_overrides(base_state: SessionState) -> None:
    recipe = _recipe("x", capabilities=["cache.redis", "obs.langsmith"])
    state = base_state
    state.recipe = recipe
    state.add_capabilities = ["obs.langfuse"]
    state.remove_capabilities = {"obs.langsmith"}
    assert _effective_capability_ids(state) == {"cache.redis", "obs.langfuse"}


def test_apply_layer_choice_adds_new_and_removes_dropped(
    base_state: SessionState,
) -> None:
    recipe = _recipe("x", capabilities=["vector_db.qdrant", "relational.postgres", "obs.langsmith"])
    base_state.recipe = recipe
    new_state = _apply_layer_choice(
        base_state,
        picked=["vector_db.pgvector", "relational.postgres"],
        kinds=("relational", "cache", "vector_db"),
    )
    # qdrant was in-layer and got dropped → goes into remove_capabilities.
    assert "vector_db.qdrant" in new_state.remove_capabilities
    # postgres stays unchanged → not touched.
    assert "relational.postgres" not in new_state.add_capabilities
    assert "relational.postgres" not in new_state.remove_capabilities
    # pgvector is new → adds.
    assert "vector_db.pgvector" in new_state.add_capabilities
    # obs.langsmith is outside the layer → untouched.
    assert "obs.langsmith" not in new_state.remove_capabilities


def test_apply_layer_choice_is_idempotent_when_nothing_changes(
    base_state: SessionState,
) -> None:
    recipe = _recipe("x", capabilities=["cache.redis", "vector_db.qdrant"])
    base_state.recipe = recipe
    new_state = _apply_layer_choice(
        base_state,
        picked=["cache.redis", "vector_db.qdrant"],
        kinds=("relational", "cache", "vector_db"),
    )
    assert new_state.add_capabilities == []
    assert new_state.remove_capabilities == set()


def test_apply_layer_choice_handles_none_pick(base_state: SessionState) -> None:
    # Ctrl-C from questionary returns None; we must not crash and must leave
    # state untouched.
    new_state = _apply_layer_choice(
        base_state,
        picked=None,  # type: ignore[arg-type]
        kinds=("obs",),
    )
    assert new_state is base_state


def test_make_layer_step_gates_on_menu_or_customize(
    base_state: SessionState,
) -> None:
    """The layer walk opens via the features menu ("More layers") or the
    standalone /customize path — either signal enables the steps."""
    step = _make_layer_step("memory", "Memory", ("relational", "cache", "vector_db"))
    assert step.enabled_when is not None
    base_state.stack_mode = "quick"
    base_state.optional_features = []
    assert step.enabled_when(base_state) is False
    base_state.optional_features = ["layers"]
    assert step.enabled_when(base_state) is True
    base_state.optional_features = []
    base_state.stack_mode = "customize"
    assert step.enabled_when(base_state) is True


def test_make_layer_step_honors_custom_gating(base_state: SessionState) -> None:
    """A custom enabled_when (the guardrails feature step) replaces the
    default menu-or-customize gate entirely."""
    step = _make_layer_step(
        "guardrails",
        "Guardrails",
        ("guardrail",),
        enabled_when=lambda s: "guardrails" in s.optional_features,
    )
    assert step.enabled_when is not None
    base_state.stack_mode = "customize"
    base_state.optional_features = []
    assert step.enabled_when(base_state) is False
    base_state.optional_features = ["guardrails"]
    assert step.enabled_when(base_state) is True


def test_default_features_pre_check_from_recipe(base_state: SessionState) -> None:
    """The menu pre-checks the features the recipe's declared stack implies."""
    assert _default_features_for_recipe(None) == set()
    recipe = _recipe("plain", capabilities=["cache.redis"])
    assert _default_features_for_recipe(recipe) == set()
    recipe = _recipe(
        "ragged", capabilities=["vector_db.qdrant", "obs.langfuse", "guardrail.llama-guard"]
    )
    assert _default_features_for_recipe(recipe) == {"rag", "observability", "guardrails"}
