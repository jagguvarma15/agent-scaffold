"""Tests for ``agent_scaffold.steps.bootstrap_langsmith`` (Phase 2)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from agent_scaffold.capabilities import Capability, ResolvedStack
from agent_scaffold.orchestrator import StepContext, StepStatus
from agent_scaffold.steps import bootstrap_langsmith as bls
from agent_scaffold.steps.bootstrap_langsmith import BootstrapLangSmithStep


def _cap(tmp_path: Path) -> Capability:
    return Capability(
        id="obs.langsmith",
        kind="obs",
        path=tmp_path / "ls.md",
        env_vars=["LANGCHAIN_API_KEY"],
    )


def _stack(cap: Capability) -> ResolvedStack:
    return ResolvedStack(capabilities=[cap])


def test_detect_skipped_without_capability(
    ctx_factory: Callable[..., StepContext],
) -> None:
    result = BootstrapLangSmithStep().detect(ctx_factory())
    assert result.status is StepStatus.SKIPPED


def test_detect_skipped_without_api_key(
    ctx_factory: Callable[..., StepContext],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    result = BootstrapLangSmithStep().detect(
        ctx_factory(resolved_stack=_stack(_cap(tmp_path)))
    )
    assert result.status is StepStatus.SKIPPED
    assert "LANGCHAIN_API_KEY" in result.reason


def test_detect_pending_when_ready(
    ctx_factory: Callable[..., StepContext],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGCHAIN_API_KEY", "ls__test")
    result = BootstrapLangSmithStep().detect(
        ctx_factory(resolved_stack=_stack(_cap(tmp_path)))
    )
    assert result.status is StepStatus.PENDING


def test_apply_creates_project_when_missing(
    ctx_factory: Callable[..., StepContext],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGCHAIN_API_KEY", "ls__test")
    created: list[str] = []

    class FakeNotFound(Exception):
        pass

    class FakeClient:
        def __init__(self, **_kw: Any) -> None:
            pass

        def read_project(self, project_name: str) -> None:
            raise FakeNotFound("404 Project not found")

        def create_project(self, project_name: str) -> None:
            created.append(project_name)

    sys = __import__("sys")
    monkeypatch.setitem(sys.modules, "langsmith", type("M", (), {"Client": FakeClient}))

    ctx = ctx_factory(resolved_stack=_stack(_cap(tmp_path)), project_dir=tmp_path)
    result = BootstrapLangSmithStep().apply(ctx)
    assert result.status is StepStatus.DONE
    assert created == ["test-recipe"]  # manifest_factory default
    env_text = (tmp_path / ".env.local").read_text(encoding="utf-8")
    assert "LANGCHAIN_TRACING_V2=true" in env_text
    assert "LANGCHAIN_PROJECT=test-recipe" in env_text
    assert "LANGCHAIN_ENDPOINT=https://api.smith.langchain.com" in env_text


def test_apply_skips_create_when_project_exists(
    ctx_factory: Callable[..., StepContext],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGCHAIN_API_KEY", "ls__test")
    create_calls: list[str] = []

    class FakeClient:
        def __init__(self, **_kw: Any) -> None:
            pass

        def read_project(self, project_name: str) -> None:
            return  # found

        def create_project(self, project_name: str) -> None:
            create_calls.append(project_name)

    sys = __import__("sys")
    monkeypatch.setitem(sys.modules, "langsmith", type("M", (), {"Client": FakeClient}))

    result = BootstrapLangSmithStep().apply(
        ctx_factory(resolved_stack=_stack(_cap(tmp_path)), project_dir=tmp_path)
    )
    assert result.status is StepStatus.DONE
    assert create_calls == []
    assert "exists" in result.detail


def test_apply_skipped_when_sdk_missing(
    ctx_factory: Callable[..., StepContext],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGCHAIN_API_KEY", "ls__test")
    sys = __import__("sys")
    monkeypatch.delitem(sys.modules, "langsmith", raising=False)

    real_import = (
        __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__
    )

    def fake_import(name: str, *args: Any, **kw: Any) -> Any:
        if name == "langsmith":
            raise ImportError("no module named langsmith")
        return real_import(name, *args, **kw)

    monkeypatch.setattr("builtins.__import__", fake_import)
    result = BootstrapLangSmithStep().apply(
        ctx_factory(resolved_stack=_stack(_cap(tmp_path)), project_dir=tmp_path)
    )
    assert result.status is StepStatus.SKIPPED
    assert "SDK not installed" in result.detail


def test_apply_failed_when_create_raises(
    ctx_factory: Callable[..., StepContext],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGCHAIN_API_KEY", "ls__test")

    class FakeClient:
        def __init__(self, **_kw: Any) -> None:
            pass

        def read_project(self, project_name: str) -> None:
            raise Exception("404 Not Found")

        def create_project(self, project_name: str) -> None:
            raise RuntimeError("quota exceeded")

    sys = __import__("sys")
    monkeypatch.setitem(sys.modules, "langsmith", type("M", (), {"Client": FakeClient}))

    result = BootstrapLangSmithStep().apply(
        ctx_factory(resolved_stack=_stack(_cap(tmp_path)), project_dir=tmp_path)
    )
    assert result.status is StepStatus.FAILED
    assert "create_project failed" in (result.error or "")


def test_write_tracing_env_is_idempotent(tmp_path: Path) -> None:
    bls._write_tracing_env(tmp_path, "demo", "https://x")
    first = (tmp_path / ".env.local").read_text(encoding="utf-8")
    added = bls._write_tracing_env(tmp_path, "demo", "https://x")
    second = (tmp_path / ".env.local").read_text(encoding="utf-8")
    assert added == 0
    assert first == second


def test_write_tracing_env_appends_to_existing(tmp_path: Path) -> None:
    (tmp_path / ".env.local").write_text("EXISTING=1\n", encoding="utf-8")
    bls._write_tracing_env(tmp_path, "demo", "https://x")
    body = (tmp_path / ".env.local").read_text(encoding="utf-8")
    assert "EXISTING=1" in body
    assert "LANGCHAIN_TRACING_V2=true" in body
