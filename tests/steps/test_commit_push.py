"""Tests for ``agent_scaffold.steps.commit_push``.

Safety invariants to lock in (load-bearing per the brief):

- We never ``git add .`` — only allow-listed paths.
- ``--no-verify`` never appears in any invocation.
- ``--yes`` alone does NOT skip the per-action prompts. The user must also
  pass ``--confirm-commit-push`` to truly automate.
- A state.json that records any FAILED step is never committed (it may
  contain stderr tails with hostnames or paths).
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from agent_scaffold.orchestrator import StepContext, StepStatus
from agent_scaffold.steps import commit_push as cp_mod
from agent_scaffold.steps.commit_push import CommitPushStep


def _completed(rc: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["git"], returncode=rc, stdout=stdout, stderr=stderr)


def test_detect_skipped_when_not_git_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
) -> None:
    monkeypatch.setattr(cp_mod, "_run_git", lambda args, **_kw: _completed(128, stderr="fatal"))
    result = CommitPushStep().detect(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.SKIPPED


def test_detect_skipped_when_nothing_to_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
) -> None:
    def fake(args: list[str], **_kw: Any) -> subprocess.CompletedProcess[str]:
        if args[0] == "rev-parse":
            return _completed(0, ".git\n")
        if args[0] == "status":
            return _completed(0, "")
        return _completed(0)

    monkeypatch.setattr(cp_mod, "_run_git", fake)
    result = CommitPushStep().detect(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.SKIPPED
    assert "nothing to commit" in result.reason


def test_detect_pending_when_allowlisted_paths_dirty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
) -> None:
    def fake(args: list[str], **_kw: Any) -> subprocess.CompletedProcess[str]:
        if args[0] == "rev-parse":
            return _completed(0, ".git\n")
        if args[0] == "status":
            return _completed(0, " M .scaffold/manifest.json\n M .env.example\n")
        return _completed(0)

    monkeypatch.setattr(cp_mod, "_run_git", fake)
    result = CommitPushStep().detect(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.PENDING
    assert ".scaffold/manifest.json" in result.reason


def test_apply_always_prompts_even_in_yes_without_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
) -> None:
    """``--yes`` alone (confirm_commit_push=False) MUST still prompt."""
    prompts: list[str] = []

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return "n"  # user declines

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(
        cp_mod,
        "_run_git",
        lambda args, **_kw: (
            _completed(0, ".git\n")
            if args[0] == "rev-parse"
            else _completed(0, " M .scaffold/manifest.json\n")
        ),
    )
    step = CommitPushStep(confirm_commit_push=False)
    result = step.apply(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.SKIPPED
    assert "declined" in (result.detail or "").lower()
    assert prompts, "expected at least one prompt"


def test_apply_skips_prompts_with_confirm_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
) -> None:
    invoked: list[list[str]] = []

    def fake(args: list[str], **_kw: Any) -> subprocess.CompletedProcess[str]:
        invoked.append(args)
        if args[0] == "rev-parse" and args[1:] == ["--git-dir"]:
            return _completed(0, ".git\n")
        if args[0] == "status":
            return _completed(0, " M .scaffold/manifest.json\n")
        if args[0] == "remote":
            return _completed(0, "")  # no origin → no push
        if args[0] == "add":
            return _completed(0)
        if args[0] == "commit":
            return _completed(0, "[main abc] chore\n")
        return _completed(0)

    monkeypatch.setattr(cp_mod, "_run_git", fake)
    monkeypatch.setattr("builtins.input", lambda _p: pytest.fail("must not prompt"))
    step = CommitPushStep(confirm_commit_push=True)
    result = step.apply(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.DONE
    # Verify allowlist behavior: `git add` got specific paths, never just ``.``.
    add_cmds = [args for args in invoked if args[0] == "add"]
    assert add_cmds, "git add was never called"
    for args in add_cmds:
        assert "." not in args[1:], f"git add . was used: {args}"
        for path in args[1:]:
            assert path.startswith((".scaffold/", ".env.", ".git")), f"unexpected path: {path}"


def test_commit_invocation_never_uses_no_verify(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
) -> None:
    invoked: list[list[str]] = []

    def fake(args: list[str], **_kw: Any) -> subprocess.CompletedProcess[str]:
        invoked.append(args)
        if args[0] == "rev-parse":
            return _completed(0, ".git\n")
        if args[0] == "status":
            return _completed(0, " M .scaffold/manifest.json\n")
        if args[0] == "remote":
            return _completed(0, "")
        if args[0] == "commit":
            return _completed(0)
        return _completed(0)

    monkeypatch.setattr(cp_mod, "_run_git", fake)
    CommitPushStep(confirm_commit_push=True).apply(ctx_factory(project_dir=tmp_path))
    commit_cmds = [args for args in invoked if args[0] == "commit"]
    assert commit_cmds, "git commit was never called"
    for args in commit_cmds:
        assert "--no-verify" not in args, f"--no-verify must never appear: {args}"


def test_state_json_with_failed_steps_is_stripped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
) -> None:
    state_dir = tmp_path / ".scaffold"
    state_dir.mkdir()
    (state_dir / "state.json").write_text(
        json.dumps({"steps": {"x": {"status": "failed", "error": "boom"}}}),
        encoding="utf-8",
    )
    added_paths: list[str] = []

    def fake(args: list[str], **_kw: Any) -> subprocess.CompletedProcess[str]:
        if args[0] == "rev-parse":
            return _completed(0, ".git\n")
        if args[0] == "status":
            return _completed(0, " M .scaffold/state.json\n M .scaffold/manifest.json\n")
        if args[0] == "remote":
            return _completed(0, "")
        if args[0] == "add":
            added_paths.extend(args[1:])
            return _completed(0)
        return _completed(0)

    monkeypatch.setattr(cp_mod, "_run_git", fake)
    CommitPushStep(confirm_commit_push=True).apply(ctx_factory(project_dir=tmp_path))
    assert ".scaffold/manifest.json" in added_paths
    assert ".scaffold/state.json" not in added_paths


def test_push_prompted_separately_decline_does_not_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
) -> None:
    answers = iter(["y", "n"])  # commit yes, push no

    monkeypatch.setattr("builtins.input", lambda _p: next(answers))

    invoked: list[list[str]] = []

    def fake(args: list[str], **_kw: Any) -> subprocess.CompletedProcess[str]:
        invoked.append(args)
        if args[0] == "rev-parse":
            return _completed(0, ".git\n" if args[1:] == ["--git-dir"] else "main\n")
        if args[0] == "status":
            return _completed(0, " M .scaffold/manifest.json\n")
        if args[0] == "remote":
            return _completed(0, "origin\n")
        if args[0] == "commit":
            return _completed(0)
        if args[0] == "push":
            pytest.fail("push must not happen when user declined")
        return _completed(0)

    monkeypatch.setattr(cp_mod, "_run_git", fake)
    result = CommitPushStep(confirm_commit_push=False).apply(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.DONE
    assert "push declined" in (result.detail or "").lower()


def test_grep_no_no_verify_string_literal_in_codebase() -> None:
    """``--no-verify`` never appears as a *Python string literal* in source.

    Comments and docstrings are allowed to mention the flag (the design doc
    in commit_push.py explicitly warns against using it). What we forbid is
    any actual subprocess-passable form of the string.
    """
    src = Path(__file__).resolve().parents[2] / "src" / "agent_scaffold"
    forbidden = ('"--no-verify"', "'--no-verify'")
    for py_file in src.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in text, (
                f"{needle} literal in {py_file} — never bypass hooks per universal rules"
            )
