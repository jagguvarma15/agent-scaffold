"""Tests for ``check_frontend_collisions`` in ``contract``."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from agent_scaffold.capabilities import Capability, EmitFile, ResolvedStack
from agent_scaffold.contract import (
    ContractParseError,
    GeneratedFile,
    GenerationResult,
    check_frontend_collisions,
)


def _frontend_cap(emit_files: list[EmitFile]) -> Capability:
    return Capability(
        id="frontend.nextjs-chat",
        kind="frontend",
        path=Path("/fake/nextjs.md"),
        emit_files=emit_files,
    )


def _result(*paths: str) -> GenerationResult:
    return GenerationResult(
        project_name="demo",
        language="python",
        files=[GeneratedFile(path=p, content="x") for p in paths],
        smoke_check="pytest",
    )


def test_no_op_without_stack() -> None:
    assert check_frontend_collisions(_result("frontend/app/page.tsx"), None) == []


def test_no_op_without_frontend_capability() -> None:
    stack = ResolvedStack(
        capabilities=[Capability(id="obs.langsmith", kind="obs", path=Path("/x.md"))]
    )
    assert check_frontend_collisions(_result("frontend/app/page.tsx"), stack) == []


def test_glob_collision_detected() -> None:
    cap = _frontend_cap([EmitFile(source="templates/nextjs-chat/**", dest="frontend/")])
    stack = ResolvedStack(capabilities=[cap])
    result = _result("frontend/app/page.tsx", "backend/main.py")
    collisions = check_frontend_collisions(result, stack)
    assert len(collisions) == 1
    assert "frontend/app/page.tsx" in collisions[0]


def test_backend_files_not_flagged() -> None:
    cap = _frontend_cap([EmitFile(source="templates/nextjs-chat/**", dest="frontend/")])
    stack = ResolvedStack(capabilities=[cap])
    result = _result("backend/main.py", "README.md", "docker-compose.yml")
    assert check_frontend_collisions(result, stack) == []


def test_single_file_collision_detected() -> None:
    cap = _frontend_cap([EmitFile(source="templates/vercel.json", dest="vercel.json")])
    stack = ResolvedStack(capabilities=[cap])
    collisions = check_frontend_collisions(_result("vercel.json"), stack)
    assert len(collisions) == 1


def test_strict_mode_raises() -> None:
    cap = _frontend_cap([EmitFile(source="templates/nextjs-chat/**", dest="frontend/")])
    stack = ResolvedStack(capabilities=[cap])
    with pytest.raises(ContractParseError) as exc:
        check_frontend_collisions(_result("frontend/app/page.tsx"), stack, strict=True)
    assert "frontend/app/page.tsx" in exc.value.reason


def test_non_strict_logs_but_returns(caplog: pytest.LogCaptureFixture) -> None:
    cap = _frontend_cap([EmitFile(source="templates/nextjs-chat/**", dest="frontend/")])
    stack = ResolvedStack(capabilities=[cap])
    with caplog.at_level(logging.WARNING, logger="agent_scaffold.contract"):
        collisions = check_frontend_collisions(
            _result("frontend/app/page.tsx"), stack, strict=False
        )
    assert collisions
    assert any("frontend collision" in rec.message for rec in caplog.records)


def test_exact_dest_match_for_glob_root() -> None:
    """A file at exactly the glob root prefix (no trailing slash) should also collide."""
    cap = _frontend_cap([EmitFile(source="templates/streamlit/**", dest="frontend/")])
    stack = ResolvedStack(capabilities=[cap])
    # "frontend" with no slash — matches because dest is `frontend` after rstrip.
    collisions = check_frontend_collisions(_result("frontend"), stack)
    assert collisions
