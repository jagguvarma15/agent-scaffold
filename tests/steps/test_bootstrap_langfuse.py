"""Tests for ``agent_scaffold.steps.bootstrap_langfuse``."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from agent_scaffold.capabilities import Capability, ResolvedStack
from agent_scaffold.orchestrator import StepContext, StepStatus
from agent_scaffold.steps.bootstrap_langfuse import BootstrapLangfuseStep


def _cap(tmp_path: Path) -> Capability:
    return Capability(
        id="obs.langfuse",
        kind="obs",
        path=tmp_path / "langfuse.md",
    )


def _stack(tmp_path: Path) -> ResolvedStack:
    return ResolvedStack(capabilities=[_cap(tmp_path)])


def test_detect_skipped_without_capability(
    ctx_factory: Callable[..., StepContext],
) -> None:
    """No obs.langfuse on the recipe → silent skip; no key probe."""
    result = BootstrapLangfuseStep().detect(ctx_factory())
    assert result.status is StepStatus.SKIPPED
    assert "obs.langfuse" in result.reason


def test_detect_skipped_without_keys(
    ctx_factory: Callable[..., StepContext],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capability declared but no keys → skip with the create-project hint."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    result = BootstrapLangfuseStep().detect(
        ctx_factory(resolved_stack=_stack(tmp_path), project_dir=tmp_path)
    )
    assert result.status is StepStatus.SKIPPED
    assert "LANGFUSE_PUBLIC_KEY" in result.reason


def test_apply_writes_env_vars(
    ctx_factory: Callable[..., StepContext],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3001")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    result = BootstrapLangfuseStep().apply(
        ctx_factory(resolved_stack=_stack(tmp_path), project_dir=tmp_path)
    )
    assert result.status is StepStatus.DONE
    env_local = (tmp_path / ".env.local").read_text(encoding="utf-8")
    assert "LANGFUSE_HOST=http://localhost:3001" in env_local
    assert "LANGFUSE_PUBLIC_KEY=pk-test" in env_local
    assert "LANGFUSE_SECRET_KEY=sk-test" in env_local


def test_apply_idempotent(
    ctx_factory: Callable[..., StepContext],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-running against an already-populated .env.local is a no-op."""
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3001")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    step = BootstrapLangfuseStep()

    first = step.apply(ctx_factory(resolved_stack=_stack(tmp_path), project_dir=tmp_path))
    assert first.status is StepStatus.DONE
    assert "wrote 3" in (first.detail or "")

    second = step.apply(ctx_factory(resolved_stack=_stack(tmp_path), project_dir=tmp_path))
    assert second.status is StepStatus.DONE
    assert "already" in (second.detail or "")


def test_apply_skipped_without_keys(
    ctx_factory: Callable[..., StepContext],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defense-in-depth: apply() also checks for missing keys."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    result = BootstrapLangfuseStep().apply(
        ctx_factory(resolved_stack=_stack(tmp_path), project_dir=tmp_path)
    )
    assert result.status is StepStatus.SKIPPED
    assert not (tmp_path / ".env.local").exists()
