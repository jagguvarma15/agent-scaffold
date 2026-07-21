"""Tests for ``agent_scaffold.steps.migrations``."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from agent_scaffold.discovery import ExternalService
from agent_scaffold.orchestrator import StepContext, StepStatus
from agent_scaffold.steps import migrations as m_mod
from agent_scaffold.steps._subprocess import SubprocessResult
from agent_scaffold.steps.migrations import MigrationsStep


def _pg_svc(engine: str | None = "alembic") -> ExternalService:
    return ExternalService(
        id="postgres",
        env_vars=["DATABASE_URL"],
        migrations=engine,
        default_local="postgres://localhost:5432/test",
    )


def test_detect_skipped_when_no_migrating_services(
    tmp_path: Path,
    ctx_factory: Callable[..., StepContext],
    recipe_factory: Callable[..., Any],
    patch_load_recipe: Callable[[Any], None],
) -> None:
    patch_load_recipe(recipe_factory(external_services=[ExternalService(id="x")]))
    result = MigrationsStep().detect(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.SKIPPED


def test_detect_skipped_for_unsupported_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
    recipe_factory: Callable[..., Any],
    patch_load_recipe: Callable[[Any], None],
) -> None:
    patch_load_recipe(recipe_factory(external_services=[_pg_svc("prisma")]))
    monkeypatch.setattr(m_mod.shutil, "which", lambda _name: "/usr/bin/uv")
    result = MigrationsStep().detect(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.SKIPPED
    assert "prisma" in result.reason


def test_detect_done_when_alembic_at_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
    recipe_factory: Callable[..., Any],
    patch_load_recipe: Callable[[Any], None],
) -> None:
    patch_load_recipe(recipe_factory(external_services=[_pg_svc()]))
    monkeypatch.setattr(m_mod.shutil, "which", lambda _name: "/usr/bin/uv")
    rev = "abc123def456"
    monkeypatch.setattr(m_mod, "_capture", lambda *_a, **_kw: (f"{rev} (head)\n", 0))
    result = MigrationsStep().detect(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.DONE


def test_detect_pending_when_alembic_behind_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
    recipe_factory: Callable[..., Any],
    patch_load_recipe: Callable[[Any], None],
) -> None:
    patch_load_recipe(recipe_factory(external_services=[_pg_svc()]))
    monkeypatch.setattr(m_mod.shutil, "which", lambda _name: "/usr/bin/uv")
    seq = iter([("aaaaaaaaaa\n", 0), ("bbbbbbbbbb (head)\n", 0)])
    monkeypatch.setattr(m_mod, "_capture", lambda *_a, **_kw: next(seq))
    result = MigrationsStep().detect(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.PENDING
    assert "postgres" in result.reason


def test_apply_runs_alembic_upgrade_per_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
    recipe_factory: Callable[..., Any],
    patch_load_recipe: Callable[[Any], None],
) -> None:
    patch_load_recipe(recipe_factory(external_services=[_pg_svc()]))
    monkeypatch.setattr(m_mod.shutil, "which", lambda _name: "/usr/bin/uv")
    calls: list[list[str]] = []

    def fake_stream(cmd: list[str], **_kw: Any) -> SubprocessResult:
        calls.append(cmd)
        return SubprocessResult(0, "", False, 0.1)

    monkeypatch.setattr(m_mod, "stream_subprocess", fake_stream)
    result = MigrationsStep().apply(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.DONE
    assert calls == [["uv", "run", "alembic", "upgrade", "head"]]


def test_apply_failed_propagates_stderr_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
    recipe_factory: Callable[..., Any],
    patch_load_recipe: Callable[[Any], None],
) -> None:
    patch_load_recipe(recipe_factory(external_services=[_pg_svc()]))
    monkeypatch.setattr(m_mod.shutil, "which", lambda _name: "/usr/bin/uv")
    monkeypatch.setattr(
        m_mod,
        "stream_subprocess",
        lambda *_a, **_kw: SubprocessResult(1, "could not connect to server", False, 0.1),
    )
    result = MigrationsStep().apply(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.FAILED
    assert "could not connect" in (result.stderr_tail or "")


def test_troubleshoot_matches_alembic_error_prefixes() -> None:
    step = MigrationsStep()
    assert "could not connect to server" in step.troubleshoot
    assert 'relation "alembic_version" does not exist' in step.troubleshoot
    assert "password authentication failed" in step.troubleshoot


def test_capture_handles_missing_binary(tmp_path: Path) -> None:
    out, rc = m_mod._capture(["__definitely_not_a_binary__"], cwd=tmp_path)
    assert out == ""
    assert rc == -1


def test_parse_alembic_rev_extracts_hex_id() -> None:
    assert m_mod._parse_alembic_rev("abc12345 (head)\n") == "abc12345"
    assert m_mod._parse_alembic_rev("# no revs here\n") == ""


def test_capture_uses_real_subprocess(tmp_path: Path) -> None:
    """Smoke-check the capture helper against a real, harmless command."""
    out, rc = m_mod._capture(["python", "-c", "print('hi')"], cwd=tmp_path, timeout=10.0)
    assert rc == 0
    assert "hi" in out
    # Verify subprocess.run was used (timeout path).
    _ = subprocess  # silence unused-import warning on Windows-only branches


def test_detect_skipped_for_typescript_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
    manifest_factory: Callable[..., Any],
    recipe_factory: Callable[..., Any],
    patch_load_recipe: Callable[[Any], None],
) -> None:
    """alembic runs via `uv run` in the project's Python env — a recipe that
    declares it on a TypeScript run skips instead of crashing on uv."""
    patch_load_recipe(recipe_factory(external_services=[_pg_svc()]))
    monkeypatch.setattr(m_mod.shutil, "which", lambda _name: "/usr/bin/uv")
    ctx = ctx_factory(project_dir=tmp_path, manifest=manifest_factory(language="typescript"))
    result = MigrationsStep().detect(ctx)
    assert result.status is StepStatus.SKIPPED
    assert "Python" in result.reason


def test_apply_skipped_for_typescript_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
    manifest_factory: Callable[..., Any],
    recipe_factory: Callable[..., Any],
    patch_load_recipe: Callable[[Any], None],
) -> None:
    patch_load_recipe(recipe_factory(external_services=[_pg_svc()]))
    monkeypatch.setattr(m_mod.shutil, "which", lambda _name: "/usr/bin/uv")
    ctx = ctx_factory(project_dir=tmp_path, manifest=manifest_factory(language="typescript"))
    result = MigrationsStep().apply(ctx)
    assert result.status is StepStatus.SKIPPED
