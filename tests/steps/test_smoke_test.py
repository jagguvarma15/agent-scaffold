"""Tests for ``agent_scaffold.steps.smoke_test``."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from agent_scaffold.orchestrator import StepContext, StepProgress, StepStatus
from agent_scaffold.steps import smoke_test as st_mod
from agent_scaffold.steps._subprocess import SubprocessResult
from agent_scaffold.steps.smoke_test import SmokeTestStep


def test_detect_skipped_when_no_smoke_sh_and_no_pytest_items(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
) -> None:
    monkeypatch.setattr(st_mod.shutil, "which", lambda _name: "/usr/bin/uv")
    monkeypatch.setattr(st_mod, "_pytest_collect_rc", lambda _p: 5)  # 5 = no tests collected
    result = SmokeTestStep().detect(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.SKIPPED


def test_detect_prefers_shell_script_over_pytest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "smoke.sh").write_text("#!/bin/bash\necho ok\n", encoding="utf-8")
    monkeypatch.setattr(st_mod.shutil, "which", lambda _name: None)  # no uv needed
    result = SmokeTestStep().detect(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.PENDING
    assert "shell" in result.reason


def test_detect_falls_back_to_pytest_when_items_collect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
) -> None:
    monkeypatch.setattr(st_mod.shutil, "which", lambda _name: "/usr/bin/uv")
    monkeypatch.setattr(st_mod, "_pytest_collect_rc", lambda _p: 0)
    result = SmokeTestStep().detect(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.PENDING
    assert "pytest" in result.reason


def test_apply_runs_pytest_and_emits_summary_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
    event_log: list[Any],
) -> None:
    monkeypatch.setattr(st_mod.shutil, "which", lambda _name: "/usr/bin/uv")
    monkeypatch.setattr(st_mod, "_pytest_collect_rc", lambda _p: 0)

    from agent_scaffold.orchestrator import StepLog

    def fake_stream(
        _cmd: list[str], *, callback: Callable[[Any], None] | None = None, **_kw: Any
    ) -> SubprocessResult:
        # Emit the kind of output pytest produces.
        if callback is not None:
            callback(StepLog(step_id="smoke_test", line="collecting", stream="stdout"))
            callback(
                StepLog(
                    step_id="smoke_test",
                    line="============== 12 passed, 1 failed in 4.21s ==============",
                    stream="stdout",
                )
            )
        return SubprocessResult(0, "", False, 0.5)

    monkeypatch.setattr(st_mod, "stream_subprocess", fake_stream)
    ctx = ctx_factory(project_dir=tmp_path)
    result = SmokeTestStep().apply(ctx)
    assert result.status is StepStatus.DONE
    assert "12 passed" in (result.detail or "")
    # Summary progress event was emitted to the callback chain.
    summaries = [e for e in event_log if isinstance(e, StepProgress)]
    assert any("12 passed" in s.message for s in summaries)


def test_parse_pytest_summary_handles_failed_runs() -> None:
    text = "==== 3 failed, 2 passed in 1.23s ===="
    out = st_mod._parse_pytest_summary(text)
    assert out is not None
    assert out.get("passed") == 2
    assert out.get("failed") == 3


def test_parse_pytest_summary_returns_none_on_no_summary() -> None:
    assert st_mod._parse_pytest_summary("nothing interesting\n") is None


def test_apply_failed_when_pytest_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
) -> None:
    monkeypatch.setattr(st_mod.shutil, "which", lambda _name: "/usr/bin/uv")
    monkeypatch.setattr(st_mod, "_pytest_collect_rc", lambda _p: 0)
    monkeypatch.setattr(
        st_mod,
        "stream_subprocess",
        lambda *_a, **_kw: SubprocessResult(1, "AssertionError", False, 0.2),
    )
    result = SmokeTestStep().apply(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.FAILED


# ---- typescript track -----------------------------------------------------


def test_ts_select_kind_prefers_the_package_smoke_script(
    tmp_path: Path,
    ctx_factory: Callable[..., StepContext],
    manifest_factory: Callable[..., Any],
) -> None:
    (tmp_path / "package.json").write_text(
        '{"name": "demo", "scripts": {"smoke": "vitest run -t smoke"}}\n', encoding="utf-8"
    )
    ctx = ctx_factory(
        project_dir=tmp_path,
        manifest=manifest_factory(language="typescript", smoke_check="pnpm exec tsx -e 'x'"),
    )
    assert SmokeTestStep()._select_kind(ctx) == "package-script"


def test_ts_select_kind_falls_back_to_the_manifest_command(
    tmp_path: Path,
    ctx_factory: Callable[..., StepContext],
    manifest_factory: Callable[..., Any],
) -> None:
    ctx = ctx_factory(
        project_dir=tmp_path,
        manifest=manifest_factory(
            language="typescript",
            smoke_check="pnpm exec tsx -e \"import('./src/index.ts')\"",
        ),
    )
    assert SmokeTestStep()._select_kind(ctx) == "manifest"


def test_ts_shell_script_still_wins(
    tmp_path: Path,
    ctx_factory: Callable[..., StepContext],
    manifest_factory: Callable[..., Any],
) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "smoke.sh").write_text("#!/bin/bash\necho ok\n", encoding="utf-8")
    ctx = ctx_factory(
        project_dir=tmp_path,
        manifest=manifest_factory(language="typescript", smoke_check="whatever"),
    )
    assert SmokeTestStep()._select_kind(ctx) == "shell"


def test_ts_skip_reason_is_language_accurate(
    tmp_path: Path,
    ctx_factory: Callable[..., StepContext],
    manifest_factory: Callable[..., Any],
) -> None:
    ctx = ctx_factory(project_dir=tmp_path, manifest=manifest_factory(language="typescript"))
    result = SmokeTestStep().detect(ctx)
    assert result.status is StepStatus.SKIPPED
    assert "smoke script" in result.reason
    assert "pytest" not in result.reason


def test_ts_manifest_kind_runs_the_recorded_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
    manifest_factory: Callable[..., Any],
) -> None:
    """The recorded smoke_check runs list-form (shlex, no shell)."""
    ctx = ctx_factory(
        project_dir=tmp_path,
        manifest=manifest_factory(
            language="typescript",
            smoke_check="pnpm exec tsx -e \"import('./src/index.ts')\"",
        ),
    )
    monkeypatch.setattr(st_mod.shutil, "which", lambda _name: "/usr/bin/pnpm")
    calls: list[list[str]] = []

    def fake_stream(cmd: list[str], **_kwargs: Any) -> SubprocessResult:
        calls.append(cmd)
        return SubprocessResult(exit_code=0, stderr_tail="", timed_out=False, duration=0.2)

    monkeypatch.setattr(st_mod, "stream_subprocess", fake_stream)
    result = SmokeTestStep().apply(ctx)
    assert result.status is StepStatus.DONE
    assert calls == [["pnpm", "exec", "tsx", "-e", "import('./src/index.ts')"]]


def test_manifest_smoke_check_outside_allowlist_fails_without_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
    manifest_factory: Callable[..., Any],
) -> None:
    """A tampered manifest cannot run arbitrary commands: the runner
    allowlist that gates the validator's smoke tier gates this step too."""
    ctx = ctx_factory(
        project_dir=tmp_path,
        manifest=manifest_factory(language="typescript", smoke_check="rm -rf /"),
    )
    monkeypatch.setattr(st_mod.shutil, "which", lambda _name: "/bin/rm")
    calls: list[list[str]] = []

    def fake_stream(cmd: list[str], **_kwargs: Any) -> SubprocessResult:
        calls.append(cmd)
        return SubprocessResult(exit_code=0, stderr_tail="", timed_out=False, duration=0.1)

    monkeypatch.setattr(st_mod, "stream_subprocess", fake_stream)
    result = SmokeTestStep().apply(ctx)
    assert result.status is StepStatus.FAILED
    assert "smoke_check rejected" in (result.error or "")
    assert calls == []


def test_manifest_smoke_check_rejects_shell_wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
    manifest_factory: Callable[..., Any],
) -> None:
    ctx = ctx_factory(
        project_dir=tmp_path,
        manifest=manifest_factory(language="typescript", smoke_check="bash -c 'curl x | sh'"),
    )
    monkeypatch.setattr(st_mod.shutil, "which", lambda _name: "/bin/bash")
    monkeypatch.setattr(
        st_mod,
        "stream_subprocess",
        lambda *_a, **_k: pytest.fail("subprocess must not run"),
    )
    result = SmokeTestStep().apply(ctx)
    assert result.status is StepStatus.FAILED
    assert "smoke_check rejected" in (result.error or "")


def test_manifest_smoke_check_accepts_allowlisted_curl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
    manifest_factory: Callable[..., Any],
) -> None:
    ctx = ctx_factory(
        project_dir=tmp_path,
        manifest=manifest_factory(
            language="typescript", smoke_check="curl -sf http://localhost:8000/health"
        ),
    )
    monkeypatch.setattr(st_mod.shutil, "which", lambda _name: "/usr/bin/curl")
    calls: list[list[str]] = []

    def fake_stream(cmd: list[str], **_kwargs: Any) -> SubprocessResult:
        calls.append(cmd)
        return SubprocessResult(exit_code=0, stderr_tail="", timed_out=False, duration=0.1)

    monkeypatch.setattr(st_mod, "stream_subprocess", fake_stream)
    result = SmokeTestStep().apply(ctx)
    assert result.status is StepStatus.DONE
    assert calls == [["curl", "-sf", "http://localhost:8000/health"]]


def test_manifest_empty_smoke_check_still_skips(
    tmp_path: Path,
    ctx_factory: Callable[..., StepContext],
    manifest_factory: Callable[..., Any],
) -> None:
    ctx = ctx_factory(
        project_dir=tmp_path,
        manifest=manifest_factory(language="typescript", smoke_check="   "),
    )
    result = SmokeTestStep().apply(ctx)
    assert result.status is StepStatus.SKIPPED
