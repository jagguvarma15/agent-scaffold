"""Tests for ``agent_scaffold.steps.seed``."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from agent_scaffold.orchestrator import StepContext, StepStatus
from agent_scaffold.steps import seed as seed_mod
from agent_scaffold.steps._subprocess import SubprocessResult
from agent_scaffold.steps.seed import SeedStep


def _make_seed_script(tmp_path: Path, ext: str = "py", body: str = "print('seeded')") -> Path:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    script = scripts / f"seed.{ext}"
    script.write_text(body, encoding="utf-8")
    return script


def test_detect_skipped_when_no_script(
    tmp_path: Path,
    ctx_factory: Callable[..., StepContext],
    recipe_factory: Callable[..., Any],
    patch_load_recipe: Callable[[Any], None],
) -> None:
    patch_load_recipe(recipe_factory())
    result = SeedStep().detect(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.SKIPPED


def test_detect_pending_with_python_script(
    tmp_path: Path,
    ctx_factory: Callable[..., StepContext],
    recipe_factory: Callable[..., Any],
    patch_load_recipe: Callable[[Any], None],
) -> None:
    _make_seed_script(tmp_path, "py")
    patch_load_recipe(recipe_factory())
    result = SeedStep().detect(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.PENDING
    assert "seed.py" in result.reason


def test_apply_invokes_uv_run_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
    recipe_factory: Callable[..., Any],
    patch_load_recipe: Callable[[Any], None],
) -> None:
    _make_seed_script(tmp_path, "py")
    patch_load_recipe(recipe_factory())
    monkeypatch.setattr(seed_mod.shutil, "which", lambda _name: "/usr/bin/uv")
    calls: list[list[str]] = []

    def fake_stream(cmd: list[str], **_kw: Any) -> SubprocessResult:
        calls.append(cmd)
        return SubprocessResult(0, "", False, 0.1)

    monkeypatch.setattr(seed_mod, "stream_subprocess", fake_stream)
    result = SeedStep().apply(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.DONE
    assert calls == [["uv", "run", "python", "scripts/seed.py"]]


def test_apply_invokes_bash_for_shell_script(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
    recipe_factory: Callable[..., Any],
    patch_load_recipe: Callable[[Any], None],
) -> None:
    _make_seed_script(tmp_path, "sh", body="#!/usr/bin/env bash\necho seeded\n")
    patch_load_recipe(recipe_factory())
    monkeypatch.setattr(seed_mod.shutil, "which", lambda _name: "/bin/bash")
    calls: list[list[str]] = []

    def fake_stream(cmd: list[str], **_kw: Any) -> SubprocessResult:
        calls.append(cmd)
        return SubprocessResult(0, "", False, 0.1)

    monkeypatch.setattr(seed_mod, "stream_subprocess", fake_stream)
    result = SeedStep().apply(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.DONE
    assert calls == [["bash", "scripts/seed.sh"]]


def test_apply_failed_propagates_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
    recipe_factory: Callable[..., Any],
    patch_load_recipe: Callable[[Any], None],
) -> None:
    _make_seed_script(tmp_path, "py")
    patch_load_recipe(recipe_factory())
    monkeypatch.setattr(seed_mod.shutil, "which", lambda _name: "/usr/bin/uv")
    monkeypatch.setattr(
        seed_mod,
        "stream_subprocess",
        lambda *_a, **_kw: SubprocessResult(1, "no such table: x", False, 0.1),
    )
    result = SeedStep().apply(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.FAILED
    assert "exit 1" in (result.error or "")
    assert "no such table" in (result.stderr_tail or "")


def test_fingerprint_changes_when_script_changes(
    tmp_path: Path,
    ctx_factory: Callable[..., StepContext],
    recipe_factory: Callable[..., Any],
    patch_load_recipe: Callable[[Any], None],
) -> None:
    _make_seed_script(tmp_path, "py", body="v1")
    patch_load_recipe(recipe_factory())
    ctx = ctx_factory(project_dir=tmp_path)
    fp_before = SeedStep().fingerprint(ctx)
    (tmp_path / "scripts" / "seed.py").write_text("v2", encoding="utf-8")
    assert SeedStep().fingerprint(ctx) != fp_before
