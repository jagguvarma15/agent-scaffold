"""Tests for the consolidated post-generation report panel."""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from agent_scaffold.capabilities import Capability, ResolvedStack
from agent_scaffold.report import (
    GenerationReport,
    derive_observability,
    print_generation_report,
)


def _render_to_text(report: GenerationReport) -> str:
    """Render the panel into a plain string for substring assertions."""
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, width=120)
    print_generation_report(report, console)
    return buffer.getvalue()


def test_render_includes_all_filled_sections() -> None:
    text = _render_to_text(
        GenerationReport(
            recipe_slug="restaurant-rebooking",
            language="python",
            framework="langgraph",
            observability="langfuse",
            model="claude-opus-4-7",
            wall_seconds=569.0,
            input_tokens=33_206,
            output_tokens=31_533,
            files_written=44,
            top_files=["src/main.py", "README.md"],
            phase_durations={"generate": 568.0, "write": 0.1},
            warnings=["something to note"],
        )
    )
    assert "Generation report" in text
    assert "Selections" in text
    assert "restaurant-rebooking" in text
    assert "langfuse" in text
    assert "Generation" in text
    assert "claude-opus-4-7" in text
    assert "33,206" in text
    assert "Files" in text
    assert "44" in text
    assert "src/main.py" in text
    assert "Phases" in text
    assert "Notes" in text
    assert "something to note" in text


def test_render_elides_empty_sections() -> None:
    """Recipe-only report (no usage, no files, no phases) skips those sections."""
    text = _render_to_text(GenerationReport(recipe_slug="lonely-recipe"))
    assert "Generation report" in text
    assert "lonely-recipe" in text
    # Sections that had no data should not appear at all.
    assert "Generation\n" not in text and "Generation " not in text.split("Selections")[-1]
    assert "Phases" not in text
    assert "Notes" not in text


def test_render_top_files_truncated_with_more_indicator() -> None:
    top = [f"src/file{i}.py" for i in range(12)]
    text = _render_to_text(
        GenerationReport(recipe_slug="x", files_written=12, top_files=top)
    )
    # Limit is 6; the remaining 6 should be summarised.
    assert "src/file0.py" in text
    assert "src/file5.py" in text
    assert "and 6 more" in text


def test_render_includes_cost_when_estimable() -> None:
    text = _render_to_text(
        GenerationReport(
            recipe_slug="x",
            model="claude-sonnet-4-6",
            input_tokens=10_000,
            output_tokens=5_000,
        )
    )
    assert "Cost:" in text
    assert "$" in text


def test_derive_observability_from_stack() -> None:
    tmp = Capability(id="obs.langfuse", kind="obs", path=__file__)  # type: ignore[arg-type]
    stack = ResolvedStack(capabilities=[tmp])
    assert derive_observability(stack) == "langfuse"

    langsmith = Capability(id="obs.langsmith", kind="obs", path=__file__)  # type: ignore[arg-type]
    stack = ResolvedStack(capabilities=[langsmith])
    assert derive_observability(stack) == "langsmith"

    empty = ResolvedStack(capabilities=[])
    assert derive_observability(empty) == ""

    assert derive_observability(None) == ""
