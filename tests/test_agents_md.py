"""Tests for agent_scaffold.agents_md — the AGENTS.md writer."""

from __future__ import annotations

from pathlib import Path

from agent_scaffold.agents_md import render_agents_md, write_agents_md
from agent_scaffold.capabilities import Capability, ResolvedStack
from agent_scaffold.contract import GeneratedFile, GenerationResult
from agent_scaffold.discovery import Recipe

_HINTS = {
    "language": "python",
    "package_manager": "uv",
    "manifest": "pyproject.toml",
    "required_tools": {"formatter": "ruff", "type_checker": "mypy", "test": "pytest"},
}


def _recipe(tmp_path: Path) -> Recipe:
    md = tmp_path / "demo.md"
    md.write_text("# Demo\n", encoding="utf-8")
    return Recipe(slug="demo", title="Demo agent", path=md)


def _result() -> GenerationResult:
    return GenerationResult(
        project_name="demo_agent",
        language="python",
        files=[GeneratedFile(path="README.md", content="# hi\n")],
        smoke_check="uv run python -c 'from demo_agent.main import agent'",
    )


def _stack(tmp_path: Path) -> ResolvedStack:
    cap = Capability(
        id="vector_db.pgvector",
        kind="vector_db",
        path=tmp_path / "cap.md",
        env_vars=["DATABASE_URL"],
        docker={"service": "postgres", "image": "pgvector/pgvector:pg16"},
    )
    return ResolvedStack(capabilities=[cap])


def test_render_carries_commands_services_env_and_boundaries(tmp_path: Path) -> None:
    text = render_agents_md(
        recipe=_recipe(tmp_path),
        language="python",
        framework="langgraph",
        hints=_HINTS,
        result=_result(),
        resolved_stack=_stack(tmp_path),
        agent_role="You answer support tickets.",
    )
    assert text.startswith("# demo_agent")
    assert "Role: You answer support tickets." in text
    # Commands derived from the language hints.
    assert "`uv sync`" in text
    assert "`uv run pytest`" in text
    assert "`uv run python -c 'from demo_agent.main import agent'`" in text
    # Services + environment from the resolved stack.
    assert "- `postgres`" in text
    assert "- `DATABASE_URL`" in text
    # Boundaries are always present.
    assert ".agent/spec.md" in text
    assert ".env.local" in text


def test_render_emits_real_toolchain_commands(tmp_path: Path) -> None:
    # The generic "{tool} check ." template rendered commands that don't
    # exist: `prettier check .` (prettier wants --check) and `tsc .` (tsc
    # rejects a directory argument). Both languages must render invocations
    # their tools actually accept.
    ts_hints = {
        "language": "typescript",
        "package_manager": "pnpm",
        "required_tools": {"formatter": "prettier", "type_checker": "tsc", "test": "vitest"},
    }
    text = render_agents_md(
        recipe=_recipe(tmp_path),
        language="typescript",
        framework="vercel-ai-sdk",
        hints=ts_hints,
        result=_result(),
        resolved_stack=None,
    )
    assert "`pnpm exec vitest run`" in text
    assert "`pnpm exec prettier --check .`" in text
    assert "`pnpm exec tsc --noEmit`" in text
    assert "prettier check" not in text

    py_text = render_agents_md(
        recipe=_recipe(tmp_path),
        language="python",
        framework="langgraph",
        hints=_HINTS,
        result=_result(),
        resolved_stack=None,
    )
    assert "`uv run ruff check .`" in py_text
    assert "`uv run mypy .`" in py_text


def test_render_is_deterministic(tmp_path: Path) -> None:
    kwargs = dict(
        recipe=_recipe(tmp_path),
        language="python",
        framework="langgraph",
        hints=_HINTS,
        result=_result(),
        resolved_stack=None,
    )
    assert render_agents_md(**kwargs) == render_agents_md(**kwargs)


def test_write_creates_when_absent(tmp_path: Path) -> None:
    dest = tmp_path / "proj"
    dest.mkdir()
    path = write_agents_md(
        dest,
        recipe=_recipe(tmp_path),
        language="python",
        framework="langgraph",
        hints=_HINTS,
        result=_result(),
        resolved_stack=None,
    )
    assert path is not None
    assert path.read_text(encoding="utf-8").startswith("# demo_agent")


def test_write_preserves_existing_user_owned_file(tmp_path: Path) -> None:
    """AGENTS.md is user-owned once it exists — regeneration never clobbers."""
    dest = tmp_path / "proj"
    dest.mkdir()
    existing = dest / "AGENTS.md"
    existing.write_text("# my tuned instructions\n", encoding="utf-8")
    path = write_agents_md(
        dest,
        recipe=_recipe(tmp_path),
        language="python",
        framework="langgraph",
        hints=_HINTS,
        result=_result(),
        resolved_stack=None,
    )
    assert path is None
    assert existing.read_text(encoding="utf-8") == "# my tuned instructions\n"
