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


def test_default_features_pre_check_mcp_only_when_the_recipe_binds_it() -> None:
    """MCP pre-checks from an mcp_servers binding or a declared mcp.* cap —
    never for an unbound recipe (explicit opt-in everywhere else)."""
    from agent_scaffold.discovery import MCPServerSpec

    bound = _recipe("bound", capabilities=["cache.redis"]).model_copy(
        update={
            "mcp_servers": [
                MCPServerSpec(
                    id="arrowhead", capability="mcp.arrowhead", transport="streamable_http"
                )
            ]
        }
    )
    assert "mcp" in _default_features_for_recipe(bound)
    declared = _recipe("declared", capabilities=["mcp.tavily"])
    assert "mcp" in _default_features_for_recipe(declared)
    plain = _recipe("plain", capabilities=["cache.redis"])
    assert "mcp" not in _default_features_for_recipe(plain)


def test_effective_ids_include_mcp_server_bindings(base_state: SessionState) -> None:
    """mcp_servers-bound capabilities show as effective (checked in pickers);
    unchecking flows through remove_capabilities like any other id."""
    from agent_scaffold.discovery import MCPServerSpec

    base_state.recipe = _recipe("bound", capabilities=["cache.redis"]).model_copy(
        update={
            "mcp_servers": [
                MCPServerSpec(
                    id="arrowhead", capability="mcp.arrowhead", transport="streamable_http"
                )
            ]
        }
    )
    assert "mcp.arrowhead" in _effective_capability_ids(base_state)
    base_state.remove_capabilities = {"mcp.arrowhead"}
    assert "mcp.arrowhead" not in _effective_capability_ids(base_state)


def test_feature_steps_include_a_gated_mcp_layer_step(base_state: SessionState) -> None:
    """The MCP step fires for the dedicated menu pick AND the full layer walk
    (the walk's Tools step no longer covers mcp, so this step owns the kind
    in every mode) — but never when nothing opted in."""
    from agent_scaffold.repl.shell import _FEATURE_STEPS

    step = next(s for s in _FEATURE_STEPS if s.label == "Layer · MCP servers")
    assert step.enabled_when is not None
    base_state.stack_mode = "quick"
    base_state.optional_features = []
    assert step.enabled_when(base_state) is False
    base_state.optional_features = ["mcp"]
    assert step.enabled_when(base_state) is True
    base_state.optional_features = ["layers"]
    assert step.enabled_when(base_state) is True
    base_state.optional_features = []
    base_state.stack_mode = "customize"
    assert step.enabled_when(base_state) is True


def test_select_layer_rows_are_single_line_and_stable_width(
    base_state: SessionState, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A capability's multi-line docs block renders as one truncated summary
    line, and the id column width comes from the whole wizard catalog, not
    the current layer."""
    from types import SimpleNamespace

    from agent_scaffold.repl import shell as shell_module

    catalog = {
        "cache.redis": SimpleNamespace(
            id="cache.redis",
            kind="cache",
            docs="Session + tool-result cache.\n\nSecond paragraph.\nThird line.\n" + "x" * 400,
        ),
        "live_data.super-long-capability-id": SimpleNamespace(
            id="live_data.super-long-capability-id", kind="live_data", docs="Web search."
        ),
    }
    monkeypatch.setattr(shell_module, "load_capabilities", lambda _p: catalog)
    captured: dict[str, list] = {}

    def fake_checkbox(_prompt: str, choices: list) -> list:
        captured["choices"] = choices
        return []

    monkeypatch.setattr(shell_module, "_ask_checkbox", fake_checkbox)
    assert shell_module._select_layer(base_state, ("cache",), "Memory") == []

    rows = [
        c.title
        for c in captured["choices"]
        if hasattr(c, "title") and isinstance(c.title, str) and c.title.startswith("cache.")
    ]
    assert len(rows) == 1
    row = rows[0]
    assert "\n" not in row
    assert len(row) <= 80
    assert "Session + tool-result cache." in row
    # Column width spans the whole wizard catalog (the long live_data id),
    # not just this layer's entries.
    assert row.startswith("cache.redis" + " " * (len("live_data.super-long-capability-id") - 11))


def test_wizard_tools_step_excludes_mcp_and_guardrail(base_state: SessionState) -> None:
    """The walk's Tools step narrows to WIZARD_TOOLS_KINDS; mcp and guardrail
    belong to their dedicated steps, so one walk never asks a kind twice."""
    from agent_scaffold.repl import shell as shell_module
    from agent_scaffold.repl.layers import WIZARD_TOOLS_KINDS

    assert "mcp" not in WIZARD_TOOLS_KINDS
    assert "guardrail" not in WIZARD_TOOLS_KINDS
    tools_step = next(s for s in shell_module._WIZARD_STEPS if s.label == "Layer · Tools")
    # The step's display reads its kinds closure: an mcp capability on state
    # must not show as part of the Tools layer.
    state = base_state
    state.recipe = _recipe("x", capabilities=["mcp.arrowhead", "live_data.tavily"])
    assert "mcp.arrowhead" not in tools_step.display(state)
    assert "live_data.tavily" in tools_step.display(state)


def test_obs_layer_step_yields_to_the_dedicated_feature_step(
    base_state: SessionState,
) -> None:
    """No menu combination asks observability twice: the layer-walk obs step
    gates off exactly when the dedicated feature step will prompt."""
    from agent_scaffold.repl import shell as shell_module

    obs_layer = next(s for s in shell_module._WIZARD_STEPS if s.label == "Layer · Observability")
    dedicated = next(s for s in shell_module._WALK_STEPS if s.label == "Observability")
    assert obs_layer.enabled_when is not None and dedicated.enabled_when is not None
    for features, mode in (
        ([], "quick"),
        (["observability"], "quick"),
        (["layers"], "quick"),
        (["observability", "layers"], "quick"),
        (["observability"], "customize"),
    ):
        base_state.optional_features = features
        base_state.stack_mode = mode
        prompts = int(bool(obs_layer.enabled_when(base_state))) + int(
            bool(dedicated.enabled_when(base_state))
        )
        expects = 0 if features == [] and mode == "quick" else 1
        assert prompts == expects, (features, mode)
