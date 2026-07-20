"""Tests for ``agent_scaffold.repl.session`` — SessionState + StatePatch.

The REPL's data model. apply_patch returns a new state (the original is
left intact so renderers can diff before/after), scalar patches overwrite,
accumulator patches merge so a sequence of refinements composes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_scaffold.config import Config
from agent_scaffold.discovery import Recipe
from agent_scaffold.repl.session import SessionState, StatePatch, apply_patch
from agent_scaffold.sources import DEPLOYMENTS_SPEC, ResolvedSource
from agent_scaffold.writer import WriteMode


@pytest.fixture
def base_state(tmp_path: Path) -> SessionState:
    """A SessionState with only the session-scope inputs set."""
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


@pytest.fixture
def demo_recipe(tmp_path: Path) -> Recipe:
    recipe_md = tmp_path / "demo.md"
    recipe_md.write_text("# Demo\n", encoding="utf-8")
    return Recipe(slug="demo", title="Demo", path=recipe_md)


# ---------------------------------------------------------------------------
# is_ready
# ---------------------------------------------------------------------------


def test_is_ready_false_for_empty_state_and_lists_missing(base_state: SessionState) -> None:
    ok, missing = base_state.is_ready()
    assert ok is False
    assert set(missing) == {"recipe", "language", "framework", "project_name", "dest"}


def test_is_ready_true_when_all_required_set(base_state: SessionState, demo_recipe: Recipe) -> None:
    state = apply_patch(
        base_state,
        StatePatch(
            recipe=demo_recipe,
            language="python",
            framework="langgraph",
            project_name="demo",
            dest=Path("/tmp/demo"),
        ),
    )
    ok, missing = state.is_ready()
    assert ok is True
    assert missing == []


# ---------------------------------------------------------------------------
# apply_patch — immutability + scalar overwrites
# ---------------------------------------------------------------------------


def test_apply_patch_returns_new_state_and_leaves_original_unchanged(
    base_state: SessionState,
) -> None:
    patched = apply_patch(base_state, StatePatch(language="python"))
    assert patched.language == "python"
    assert base_state.language is None
    # Identity check — apply_patch must not return the same object.
    assert patched is not base_state


def test_apply_patch_scalar_overwrite_replaces_previous_value(
    base_state: SessionState,
) -> None:
    s1 = apply_patch(base_state, StatePatch(model="claude-opus-4-7"))
    s2 = apply_patch(s1, StatePatch(model="claude-sonnet-4-6"))
    assert s2.model == "claude-sonnet-4-6"
    assert s1.model == "claude-opus-4-7"  # original snapshot unchanged


def test_apply_patch_none_scalar_does_not_clear_existing_value(
    base_state: SessionState,
) -> None:
    """A patch with model=None means 'don't touch it', not 'clear it'."""
    s1 = apply_patch(base_state, StatePatch(model="claude-opus-4-7"))
    s2 = apply_patch(s1, StatePatch(language="python"))
    assert s2.model == "claude-opus-4-7"


def test_apply_patch_write_mode_overwrites(base_state: SessionState) -> None:
    s = apply_patch(base_state, StatePatch(write_mode=WriteMode.overwrite))
    assert s.write_mode == WriteMode.overwrite


# ---------------------------------------------------------------------------
# apply_patch — accumulators merge across patches
# ---------------------------------------------------------------------------


def test_apply_patch_dependencies_merge_across_patches(base_state: SessionState) -> None:
    s1 = apply_patch(base_state, StatePatch(add_dependencies={"python": {"postgres": ">=14"}}))
    s2 = apply_patch(s1, StatePatch(add_dependencies={"python": {"redis": ">=7"}}))
    assert s2.extra_dependencies == {"python": {"postgres": ">=14", "redis": ">=7"}}


def test_apply_patch_dependencies_across_languages(base_state: SessionState) -> None:
    s = apply_patch(
        base_state,
        StatePatch(add_dependencies={"python": {"a": "1"}, "typescript": {"b": "2"}}),
    )
    assert s.extra_dependencies == {"python": {"a": "1"}, "typescript": {"b": "2"}}


def test_apply_patch_notes_append_rather_than_replace(base_state: SessionState) -> None:
    s1 = apply_patch(base_state, StatePatch(notes="use ECS not GKE"))
    s2 = apply_patch(s1, StatePatch(notes="strict tenant isolation"))
    assert s2.refinement_notes == ["use ECS not GKE", "strict tenant isolation"]


def test_apply_patch_add_steps_dedupes(base_state: SessionState) -> None:
    s1 = apply_patch(base_state, StatePatch(add_steps=["docker_up"]))
    s2 = apply_patch(s1, StatePatch(add_steps=["docker_up", "seed"]))
    assert s2.extra_steps == ["docker_up", "seed"]


def test_apply_patch_remove_step_supersedes_earlier_add(base_state: SessionState) -> None:
    s1 = apply_patch(base_state, StatePatch(add_steps=["smoke_test"]))
    s2 = apply_patch(s1, StatePatch(remove_steps=["smoke_test"]))
    assert s2.extra_steps == []
    assert s2.removed_steps == {"smoke_test"}


def test_apply_patch_add_after_remove_reinstates_step(base_state: SessionState) -> None:
    s1 = apply_patch(base_state, StatePatch(remove_steps=["smoke_test"]))
    s2 = apply_patch(s1, StatePatch(add_steps=["smoke_test"]))
    assert "smoke_test" in s2.extra_steps
    assert "smoke_test" not in s2.removed_steps


def test_apply_patch_remove_roles_accumulates(base_state: SessionState) -> None:
    s1 = apply_patch(base_state, StatePatch(remove_roles=["kafka-consumer"]))
    s2 = apply_patch(s1, StatePatch(remove_roles=["evaluator"]))
    assert s2.removed_roles == {"kafka-consumer", "evaluator"}


# ---------------------------------------------------------------------------
# StatePatch helpers
# ---------------------------------------------------------------------------


def test_statepatch_is_empty_for_default() -> None:
    assert StatePatch().is_empty() is True


def test_statepatch_is_empty_false_when_any_field_set() -> None:
    assert StatePatch(language="python").is_empty() is False
    assert StatePatch(add_dependencies={"python": {"x": "1"}}).is_empty() is False
    assert StatePatch(notes="something").is_empty() is False


def test_apply_patch_empty_is_noop(base_state: SessionState) -> None:
    out = apply_patch(base_state, StatePatch())
    # Field-by-field equality; .is_ready output is identity-free.
    assert out.is_ready() == base_state.is_ready()
    assert out.model is base_state.model
    assert out.extra_dependencies == base_state.extra_dependencies


# ---------------------------------------------------------------------------
# dirty_since_plan flag
# ---------------------------------------------------------------------------


def test_dirty_since_plan_defaults_false(base_state: SessionState) -> None:
    assert base_state.dirty_since_plan is False


@pytest.mark.parametrize(
    "patch_kwargs",
    [
        {"language": "python"},
        {"framework": "langgraph"},
        {"model": "claude-sonnet-4-6"},
        {"stack_mode": "customize"},
        {"add_capabilities": ["obs.langfuse"]},
        {"remove_capabilities": ["obs.langsmith"]},
        {"add_steps": ["docker_up"]},
        {"remove_steps": ["seed"]},
    ],
)
def test_apply_patch_sets_dirty_for_stack_mutating_fields(
    base_state: SessionState, patch_kwargs: dict
) -> None:
    out = apply_patch(base_state, StatePatch(**patch_kwargs))
    assert out.dirty_since_plan is True


def test_apply_patch_sets_dirty_when_recipe_changes(
    base_state: SessionState, demo_recipe: Recipe
) -> None:
    out = apply_patch(base_state, StatePatch(recipe=demo_recipe))
    assert out.dirty_since_plan is True


@pytest.mark.parametrize(
    "patch_kwargs",
    [
        {"notes": "use ECS not GKE"},
        {"project_name": "demo"},
        {"dest": Path("/tmp/demo")},
        {"add_dependencies": {"python": {"redis": ">=7"}}},
        {"effort": "medium"},
        {"max_tokens": 8000},
        {"strict": True},
        {"write_mode": WriteMode.overwrite},
    ],
)
def test_apply_patch_does_not_dirty_for_non_stack_fields(
    base_state: SessionState, patch_kwargs: dict
) -> None:
    out = apply_patch(base_state, StatePatch(**patch_kwargs))
    assert out.dirty_since_plan is False


def test_apply_patch_preserves_dirty_across_clean_patches(
    base_state: SessionState,
) -> None:
    s1 = apply_patch(base_state, StatePatch(model="claude-opus-4-7"))
    assert s1.dirty_since_plan is True
    s2 = apply_patch(s1, StatePatch(notes="bonus context"))
    assert s2.dirty_since_plan is True


def test_apply_patch_empty_does_not_set_dirty(base_state: SessionState) -> None:
    out = apply_patch(base_state, StatePatch())
    assert out.dirty_since_plan is False


# ---------------------------------------------------------------------------
# Tier patches
# ---------------------------------------------------------------------------


def test_tier_patch_sets_and_dirties_plan(base_state: SessionState) -> None:
    state = apply_patch(base_state, StatePatch(tier="T2"))
    assert state.tier == "T2"
    assert state.dirty_since_plan is True


def test_tier_empty_string_clears_to_recipe_fallback(base_state: SessionState) -> None:
    state = apply_patch(base_state, StatePatch(tier="T2"))
    cleared = apply_patch(state, StatePatch(tier=""))
    assert cleared.tier is None


def test_tier_none_leaves_state_untouched(base_state: SessionState) -> None:
    state = apply_patch(base_state, StatePatch(tier="T4"))
    untouched = apply_patch(state, StatePatch(notes="just a note"))
    assert untouched.tier == "T4"
