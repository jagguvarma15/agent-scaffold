"""Tests for the Phase-4 Tier row + Layers section in the report panel."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

from rich.console import Console

from agent_scaffold.capabilities import Capability, ResolvedStack
from agent_scaffold.discovery import Recipe
from agent_scaffold.report import (
    GenerationReport,
    derive_layers,
    derive_tier,
    print_generation_report,
)


def _render(report: GenerationReport) -> str:
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=120)
    print_generation_report(report, console)
    return buf.getvalue()


def _cap(cap_id: str, kind: str) -> Capability:
    return Capability(id=cap_id, kind=kind, path=Path(f"/tmp/{cap_id}.md"))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# derive_* helpers
# ---------------------------------------------------------------------------


def test_derive_tier_returns_inferred_value() -> None:
    basic = Recipe(slug="b", title="B", path=Path("/tmp/b.md"))
    assert derive_tier(basic) == "basic"
    complex_recipe = Recipe(
        slug="c", title="C", path=Path("/tmp/c.md"), capabilities=["host.vercel"]
    )
    assert derive_tier(complex_recipe) == "complex"


def test_derive_tier_with_no_recipe_is_empty() -> None:
    assert derive_tier(None) == ""


def test_derive_layers_follows_layer_order() -> None:
    stack = ResolvedStack(
        capabilities=[
            _cap("frontend.nextjs-chat", "frontend"),
            _cap("relational.postgres", "relational"),
            _cap("obs.langfuse", "obs"),
            _cap("tools.filesystem", "tools"),
        ]
    )
    layers = derive_layers(stack)
    # LAYER_ORDER puts relational → tools → obs → frontend in that sequence.
    assert list(layers) == ["relational", "tools", "obs", "frontend"]
    assert layers["tools"] == ["tools.filesystem"]


def test_derive_layers_with_none_stack_is_empty() -> None:
    assert derive_layers(None) == {}


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def test_render_tier_row_appears_with_inferred_label() -> None:
    text = _render(GenerationReport(recipe_slug="demo", tier="mid"))
    assert "Tier" in text
    assert "mid" in text


def test_render_omits_tier_row_when_blank() -> None:
    text = _render(GenerationReport(recipe_slug="demo"))
    assert "Tier" not in text


def test_render_layers_section_lists_each_kind_with_ids() -> None:
    text = _render(
        GenerationReport(
            recipe_slug="demo",
            layers={
                "relational": ["relational.postgres"],
                "tools": ["tools.filesystem", "tools.web-search"],
                "obs": ["obs.langfuse"],
            },
        )
    )
    assert "Layers" in text
    assert "relational.postgres" in text
    assert "tools.filesystem" in text
    assert "tools.web-search" in text
    assert "obs.langfuse" in text


def test_render_layers_section_omitted_when_empty() -> None:
    text = _render(GenerationReport(recipe_slug="demo"))
    assert "Layers" not in text


def test_render_layers_with_empty_kind_skips_that_row() -> None:
    text = _render(
        GenerationReport(recipe_slug="demo", layers={"tools": []}),
    )
    # No layer-row text should appear when the list is empty for the only kind.
    assert "tools" not in text or "Layers" not in text


def test_render_full_panel_keeps_existing_sections_alongside_new_ones() -> None:
    text = _render(
        GenerationReport(
            recipe_slug="docs-rag-qa",
            tier="basic",
            language="python",
            framework="langgraph",
            observability="langfuse",
            layers={
                "relational": ["relational.postgres"],
                "vector_db": ["vector_db.qdrant"],
                "obs": ["obs.langfuse"],
            },
            model="claude-opus-4-7",
            wall_seconds=45.0,
            input_tokens=5000,
            output_tokens=1000,
            files_written=12,
        )
    )
    # Every section should be present and in order. Anchor on unique
    # substrings — "Generation" alone matches the panel title too.
    selections_idx = text.index("Selections")
    layers_idx = text.index("Layers")
    tokens_idx = text.index("Tokens:")
    files_idx = text.index("12 new")
    assert selections_idx < layers_idx < tokens_idx < files_idx
