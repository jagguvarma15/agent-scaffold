"""Tests for ``agent_scaffold.repl.render`` — REPL rendering helpers.

These aren't pixel-snapshot tests (Rich markup spans line breaks and color
codes, which fight diff readability); they assert structural invariants
the human reader cares about: which labels appear, which deltas surface,
and that the unknown-model path doesn't return an empty string.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_scaffold.config import Config
from agent_scaffold.costs import estimate_preflight
from agent_scaffold.discovery import Recipe
from agent_scaffold.repl.render import (
    render_cost,
    render_patch_delta,
    render_state_summary,
)
from agent_scaffold.repl.session import SessionState, StatePatch, apply_patch
from agent_scaffold.sources import DEPLOYMENTS_SPEC, ResolvedSource
from agent_scaffold.writer import WriteMode


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


@pytest.fixture
def demo_recipe(tmp_path: Path) -> Recipe:
    recipe_md = tmp_path / "demo.md"
    recipe_md.write_text("# Demo\n", encoding="utf-8")
    return Recipe(slug="demo", title="Demo", path=recipe_md)


def _panel_text(panel: object) -> str:
    """Pull the raw text content out of a Rich Panel for substring asserts."""
    # Panel.renderable is the inner content; for our case it's a str.
    return str(getattr(panel, "renderable", panel))


# ---------------------------------------------------------------------------
# render_state_summary
# ---------------------------------------------------------------------------


def test_state_summary_shows_dash_placeholders_for_unset_fields(
    base_state: SessionState,
) -> None:
    text = _panel_text(render_state_summary(base_state))
    for label in ("Recipe", "Language", "Framework", "Name", "Dest"):
        assert label in text
    # `–` is the unset marker.
    assert "–" in text


def test_state_summary_shows_recipe_slug(base_state: SessionState, demo_recipe: Recipe) -> None:
    state = apply_patch(base_state, StatePatch(recipe=demo_recipe, language="python"))
    text = _panel_text(render_state_summary(state))
    assert "demo" in text  # the slug
    assert "python" in text


def test_state_summary_shows_extra_deps_count_when_present(
    base_state: SessionState,
) -> None:
    state = apply_patch(
        base_state,
        StatePatch(add_dependencies={"python": {"postgres": ">=14", "redis": ">=7"}}),
    )
    text = _panel_text(render_state_summary(state))
    assert "Extra deps" in text
    assert "+2" in text


def test_state_summary_shows_step_counts_when_modified(base_state: SessionState) -> None:
    state = apply_patch(
        base_state, StatePatch(add_steps=["docker_up"], remove_steps=["smoke_test"])
    )
    text = _panel_text(render_state_summary(state))
    assert "Steps" in text
    assert "+1" in text and "-1" in text


def test_state_summary_shows_notes_count(base_state: SessionState) -> None:
    state = apply_patch(base_state, StatePatch(notes="use ECS not GKE"))
    text = _panel_text(render_state_summary(state))
    assert "Notes" in text
    assert "1 refinement" in text


# ---------------------------------------------------------------------------
# render_patch_delta
# ---------------------------------------------------------------------------


def test_patch_delta_empty_when_states_match(base_state: SessionState) -> None:
    text = render_patch_delta(base_state, base_state)
    assert "No changes" in text.plain


def test_patch_delta_renders_scalar_change(base_state: SessionState) -> None:
    after = apply_patch(base_state, StatePatch(model="claude-sonnet-4-6"))
    text = render_patch_delta(base_state, after).plain
    assert "model:" in text
    assert "claude-sonnet-4-6" in text


def test_patch_delta_uses_recipe_slug_label(base_state: SessionState, demo_recipe: Recipe) -> None:
    after = apply_patch(base_state, StatePatch(recipe=demo_recipe))
    text = render_patch_delta(base_state, after).plain
    assert "recipe:" in text
    assert "demo" in text  # slug, not the repr


def test_patch_delta_summarizes_dep_count_change(base_state: SessionState) -> None:
    after = apply_patch(
        base_state,
        StatePatch(add_dependencies={"python": {"postgres": ">=14"}}),
    )
    text = render_patch_delta(base_state, after).plain
    assert "extra deps:" in text
    assert "0 → 1" in text


def test_patch_delta_shows_added_and_removed_steps(base_state: SessionState) -> None:
    after = apply_patch(
        base_state, StatePatch(add_steps=["docker_up"], remove_steps=["smoke_test"])
    )
    text = render_patch_delta(base_state, after).plain
    assert "steps: +docker_up" in text
    assert "steps: -smoke_test" in text


def test_patch_delta_shows_new_notes_with_truncation(base_state: SessionState) -> None:
    long_note = "x" * 200
    after = apply_patch(base_state, StatePatch(notes=long_note))
    text = render_patch_delta(base_state, after).plain
    assert "note:" in text
    # Truncation marker; the full 200-char note shouldn't be inline.
    assert "…" in text
    assert long_note not in text


def test_patch_delta_handles_write_mode_change(base_state: SessionState) -> None:
    after = apply_patch(base_state, StatePatch(write_mode=WriteMode.overwrite))
    text = render_patch_delta(base_state, after).plain
    assert "write_mode:" in text
    assert "overwrite" in text


# ---------------------------------------------------------------------------
# render_cost
# ---------------------------------------------------------------------------


def test_render_cost_with_known_model_shows_dollar_amount() -> None:
    pre = estimate_preflight("claude-sonnet-4-6", input_tokens=10_000)
    text = render_cost(pre).plain
    assert "Est. cost" in text
    assert "$" in text


def test_render_cost_unknown_model_returns_dim_unavailable_line() -> None:
    text = render_cost(None).plain
    assert "unavailable" in text


def test_state_summary_shows_stack_picks(base_state: SessionState) -> None:
    state = apply_patch(
        base_state,
        StatePatch(add_capabilities=["obs.langsmith"], remove_capabilities=["obs.langfuse"]),
    )
    text = _panel_text(render_state_summary(state))
    assert "Stack" in text
    assert "+obs.langsmith" in text
    assert "-obs.langfuse" in text


def test_patch_delta_shows_capability_changes(base_state: SessionState) -> None:
    after = apply_patch(base_state, StatePatch(add_capabilities=["cache.redis"]))
    text = str(render_patch_delta(base_state, after))
    assert "stack add" in text
    assert "cache.redis" in text
