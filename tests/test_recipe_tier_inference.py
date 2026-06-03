"""Tests for :func:`agent_scaffold.discovery.infer_complexity`.

Three rules in priority order:
1. Explicit ``complexity:`` in frontmatter wins.
2. Any capability kind in ``{queue, frontend, host}`` ⇒ ``complex``.
3. Multi-step topology or more than four capabilities ⇒ ``mid``; else ``basic``.

Tests construct ``Recipe`` instances directly so the helper can be exercised
without spinning up a deployments tree.
"""

from __future__ import annotations

from pathlib import Path

from agent_scaffold.discovery import Recipe, infer_complexity


def _recipe(**fields: object) -> Recipe:
    defaults: dict[str, object] = {"slug": "x", "title": "X", "path": Path("/tmp/x.md")}
    defaults.update(fields)
    return Recipe(**defaults)  # type: ignore[arg-type]


def test_explicit_complexity_overrides_inference() -> None:
    # Recipe ships queue + frontend + host (would infer "complex") but pins "basic".
    recipe = _recipe(
        complexity="basic",
        topology="multi-agent-flat",
        capabilities=["queue.kafka", "frontend.nextjs-chat", "host.vercel"],
    )
    assert infer_complexity(recipe) == "basic"


def test_queue_capability_marks_complex() -> None:
    assert infer_complexity(_recipe(capabilities=["queue.kafka"])) == "complex"


def test_frontend_capability_marks_complex() -> None:
    assert infer_complexity(_recipe(capabilities=["frontend.streamlit"])) == "complex"


def test_host_capability_marks_complex() -> None:
    assert infer_complexity(_recipe(capabilities=["host.fly"])) == "complex"


def test_multi_agent_topology_with_no_complex_kinds_is_mid() -> None:
    recipe = _recipe(
        topology="multi-agent-hierarchical",
        capabilities=["relational.postgres", "cache.redis", "obs.langfuse"],
    )
    assert infer_complexity(recipe) == "mid"


def test_chain_topology_is_mid() -> None:
    assert infer_complexity(_recipe(topology="chain")) == "mid"


def test_parallel_topology_is_mid() -> None:
    assert infer_complexity(_recipe(topology="parallel")) == "mid"


def test_more_than_four_capabilities_is_mid_even_when_topology_single() -> None:
    recipe = _recipe(
        topology="single",
        capabilities=[
            "relational.postgres",
            "cache.redis",
            "vector_db.qdrant",
            "obs.langfuse",
            "eval.promptfoo",
        ],
    )
    assert infer_complexity(recipe) == "mid"


def test_single_topology_with_few_capabilities_is_basic() -> None:
    recipe = _recipe(
        topology="single",
        capabilities=["relational.postgres", "cache.redis", "obs.langfuse"],
    )
    assert infer_complexity(recipe) == "basic"


def test_missing_topology_and_capabilities_defaults_to_basic() -> None:
    assert infer_complexity(_recipe()) == "basic"


def test_unknown_explicit_complexity_falls_through_to_inference(
    mock_deployments_path: Path,
) -> None:
    # Direct construction can't carry an unknown literal — Pydantic rejects it.
    # The frontmatter coercer is what handles unknown strings, so route through
    # the loader. Build a temporary recipe file via monkey-patching is overkill;
    # instead, assert the helper itself ignores out-of-band values when passed
    # via model_construct (bypasses validation).
    recipe = Recipe.model_construct(
        slug="x",
        title="X",
        path=Path("/tmp/x.md"),
        complexity="enterprise",  # type: ignore[arg-type]
        capabilities=["queue.kafka"],
    )
    # "enterprise" is not one of the three valid tiers → falls through to
    # capability-based inference → "complex" because queue.* is declared.
    assert infer_complexity(recipe) == "complex"
