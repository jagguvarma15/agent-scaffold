"""Tests for ``agent_scaffold.steps.install_deps``."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from agent_scaffold.orchestrator import StepContext, StepStatus
from agent_scaffold.steps import install_deps as id_mod
from agent_scaffold.steps._subprocess import SubprocessResult
from agent_scaffold.steps.install_deps import InstallDepsStep


def _seed_project(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\nversion='0'\n", encoding="utf-8")


def test_detect_skips_non_python(
    tmp_path: Path,
    ctx_factory: Callable[..., StepContext],
    manifest_factory: Callable[..., Any],
) -> None:
    ctx = ctx_factory(project_dir=tmp_path, manifest=manifest_factory(language="typescript"))
    result = InstallDepsStep().detect(ctx)
    assert result.status is StepStatus.SKIPPED


def test_detect_skips_when_no_pyproject(
    tmp_path: Path, ctx_factory: Callable[..., StepContext]
) -> None:
    result = InstallDepsStep().detect(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.SKIPPED


def test_detect_pending_when_no_lock(
    tmp_path: Path, ctx_factory: Callable[..., StepContext]
) -> None:
    _seed_project(tmp_path)
    result = InstallDepsStep().detect(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.PENDING
    assert "uv.lock" in result.reason


def test_detect_pending_when_no_venv(
    tmp_path: Path, ctx_factory: Callable[..., StepContext]
) -> None:
    _seed_project(tmp_path)
    (tmp_path / "uv.lock").write_text("# lock\n", encoding="utf-8")
    result = InstallDepsStep().detect(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.PENDING


def test_detect_pending_when_lock_newer_than_venv(
    tmp_path: Path, ctx_factory: Callable[..., StepContext]
) -> None:
    _seed_project(tmp_path)
    (tmp_path / "uv.lock").write_text("# v1\n", encoding="utf-8")
    venv = tmp_path / ".venv"
    venv.mkdir()
    older = time.time() - 60
    os.utime(venv, (older, older))
    result = InstallDepsStep().detect(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.PENDING
    assert "newer" in result.reason


def test_detect_done_when_venv_fresh(
    tmp_path: Path, ctx_factory: Callable[..., StepContext]
) -> None:
    _seed_project(tmp_path)
    lock = tmp_path / "uv.lock"
    lock.write_text("# v1\n", encoding="utf-8")
    venv = tmp_path / ".venv"
    venv.mkdir()
    # ensure venv mtime > lock mtime
    older = time.time() - 60
    os.utime(lock, (older, older))
    result = InstallDepsStep().detect(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.DONE


def test_apply_returns_failed_when_uv_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
) -> None:
    _seed_project(tmp_path)
    monkeypatch.setattr(id_mod.shutil, "which", lambda _name: None)
    result = InstallDepsStep().apply(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.FAILED
    assert "uv" in (result.error or "")


def test_apply_runs_uv_lock_then_sync_when_no_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
) -> None:
    _seed_project(tmp_path)
    monkeypatch.setattr(id_mod.shutil, "which", lambda _name: "/usr/bin/uv")
    calls: list[list[str]] = []

    def fake_stream(cmd: list[str], **_kwargs: Any) -> SubprocessResult:
        calls.append(cmd)
        return SubprocessResult(exit_code=0, stderr_tail="", timed_out=False, duration=0.1)

    monkeypatch.setattr(id_mod, "stream_subprocess", fake_stream)
    result = InstallDepsStep().apply(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.DONE
    assert calls == [["uv", "lock"], ["uv", "sync"]]


def test_apply_skips_lock_when_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
) -> None:
    _seed_project(tmp_path)
    (tmp_path / "uv.lock").write_text("# v1\n", encoding="utf-8")
    monkeypatch.setattr(id_mod.shutil, "which", lambda _name: "/usr/bin/uv")
    calls: list[list[str]] = []

    def fake_stream(cmd: list[str], **_kwargs: Any) -> SubprocessResult:
        calls.append(cmd)
        return SubprocessResult(exit_code=0, stderr_tail="", timed_out=False, duration=0.1)

    monkeypatch.setattr(id_mod, "stream_subprocess", fake_stream)
    InstallDepsStep().apply(ctx_factory(project_dir=tmp_path))
    assert calls == [["uv", "sync"]]


def test_apply_failed_on_nonzero_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
) -> None:
    _seed_project(tmp_path)
    (tmp_path / "uv.lock").write_text("# v1\n", encoding="utf-8")
    monkeypatch.setattr(id_mod.shutil, "which", lambda _name: "/usr/bin/uv")

    def fake_stream(_cmd: list[str], **_kwargs: Any) -> SubprocessResult:
        return SubprocessResult(
            exit_code=1, stderr_tail="No solution found", timed_out=False, duration=0.1
        )

    monkeypatch.setattr(id_mod, "stream_subprocess", fake_stream)
    result = InstallDepsStep().apply(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.FAILED
    assert "exit 1" in (result.error or "")
    assert "No solution found" in (result.stderr_tail or "")


def test_apply_failed_on_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
) -> None:
    _seed_project(tmp_path)
    (tmp_path / "uv.lock").write_text("# v1\n", encoding="utf-8")
    monkeypatch.setattr(id_mod.shutil, "which", lambda _name: "/usr/bin/uv")

    def fake_stream(_cmd: list[str], **_kwargs: Any) -> SubprocessResult:
        return SubprocessResult(exit_code=-1, stderr_tail="", timed_out=True, duration=600.0)

    monkeypatch.setattr(id_mod, "stream_subprocess", fake_stream)
    result = InstallDepsStep().apply(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.FAILED
    assert "timed out" in (result.error or "")


def test_fingerprint_stable_for_identical_inputs(
    tmp_path: Path, ctx_factory: Callable[..., StepContext]
) -> None:
    _seed_project(tmp_path)
    (tmp_path / "uv.lock").write_text("# v1\n", encoding="utf-8")
    ctx = ctx_factory(project_dir=tmp_path)
    step = InstallDepsStep()
    assert step.fingerprint(ctx) == step.fingerprint(ctx)


def test_fingerprint_changes_when_pyproject_changes(
    tmp_path: Path, ctx_factory: Callable[..., StepContext]
) -> None:
    _seed_project(tmp_path)
    (tmp_path / "uv.lock").write_text("# v1\n", encoding="utf-8")
    ctx = ctx_factory(project_dir=tmp_path)
    step = InstallDepsStep()
    fp_before = step.fingerprint(ctx)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\nversion='1'\n", encoding="utf-8")
    assert step.fingerprint(ctx) != fp_before
