"""Direct tests for ``agent_scaffold.cli._run_up_inline``.

``_run_up_inline`` was factored out of ``cmd_up`` so the autorun flow can
share the orchestrator-run-plus-welcome-panel code path. These tests pin its
behavior independent of Typer plumbing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

import agent_scaffold.cli as cli_mod
from agent_scaffold._scaffold_dir import SCAFFOLD_DIR
from agent_scaffold.cli import StepFlags, _run_up_inline
from agent_scaffold.manifest import Manifest, write_manifest
from agent_scaffold.orchestrator import (
    DetectionResult,
    StepContext,
    StepResult,
    StepStatus,
    compute_fingerprint,
)


@dataclass
class _StubStep:
    id: str
    description: str = "stub"
    depends_on: tuple[str, ...] = ()
    detect_status: StepStatus = StepStatus.PENDING
    apply_status: StepStatus = StepStatus.DONE
    apply_error: str | None = None
    apply_calls: int = field(default=0, init=False)

    def detect(self, ctx: StepContext) -> DetectionResult:
        return DetectionResult(self.detect_status, reason="stub")

    def apply(self, ctx: StepContext) -> StepResult:
        self.apply_calls += 1
        return StepResult(self.apply_status, detail="ok", error=self.apply_error)

    def fingerprint(self, ctx: StepContext) -> str:
        return compute_fingerprint({"id": self.id})


def _manifest_in(project_dir: Path, *, language: str = "python") -> Manifest:
    manifest = Manifest(
        recipe="test-recipe",
        language=language,
        framework="none",
        model="claude-test",
        generated_at="2026-05-30T00:00:00+00:00",
    )
    write_manifest(project_dir, manifest)
    return manifest


def _install_steps(monkeypatch: pytest.MonkeyPatch, steps: list[Any]) -> None:
    monkeypatch.setattr(cli_mod, "default_steps_for", lambda *_a, **_kw: list(steps))


def _flags(*, yes: bool = True, plan_only: bool = False) -> StepFlags:
    return StepFlags(
        only=[],
        skip=[],
        force=[],
        retry=[],
        resume=False,
        plan_only=plan_only,
        yes=yes,
        debug=False,
    )


# ---------------------------------------------------------------------------
# Plan-only short-circuit
# ---------------------------------------------------------------------------


def test_plan_only_returns_zero_without_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest_in(tmp_path)
    step = _StubStep(id="s1")
    _install_steps(monkeypatch, [step])

    flags = _flags(plan_only=True)
    rc = _run_up_inline(
        project_dir=tmp_path,
        manifest=manifest,
        recipe=None,
        resolved_stack=None,
        flags=flags,
        interactive=False,
    )
    assert rc == 0
    assert step.apply_calls == 0


# ---------------------------------------------------------------------------
# interactive vs non-interactive
# ---------------------------------------------------------------------------


def test_interactive_false_skips_confirm_prompt_even_without_yes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`interactive=False` runs the plan without prompting, even when yes=False.

    This is the autorun path: the user already implicitly approved by typing
    ``new``, so the plan-confirm picker is the wrong thing to do.
    """
    manifest = _manifest_in(tmp_path)
    step = _StubStep(id="s1")
    _install_steps(monkeypatch, [step])

    select_called: dict[str, int] = {"count": 0}

    def fake_select(*_a: Any, **_kw: Any) -> str:
        select_called["count"] += 1
        return "yes"

    monkeypatch.setattr(cli_mod, "_interactive_select", fake_select)

    rc = _run_up_inline(
        project_dir=tmp_path,
        manifest=manifest,
        recipe=None,
        resolved_stack=None,
        flags=_flags(yes=False),
        interactive=False,
    )
    assert rc == 0
    assert select_called["count"] == 0  # no prompt
    assert step.apply_calls == 1  # but did run


def test_interactive_true_with_yes_still_skips_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``flags.yes=True`` short-circuits the prompt — preserves cmd_up's --yes shape."""
    manifest = _manifest_in(tmp_path)
    step = _StubStep(id="s1")
    _install_steps(monkeypatch, [step])

    select_called = {"count": 0}

    def fake_select(*_a: Any, **_kw: Any) -> str:
        select_called["count"] += 1
        return "yes"

    monkeypatch.setattr(cli_mod, "_interactive_select", fake_select)

    rc = _run_up_inline(
        project_dir=tmp_path,
        manifest=manifest,
        recipe=None,
        resolved_stack=None,
        flags=_flags(yes=True),
        interactive=True,
    )
    assert rc == 0
    assert select_called["count"] == 0
    assert step.apply_calls == 1


# ---------------------------------------------------------------------------
# Welcome panel
# ---------------------------------------------------------------------------


def test_welcome_panel_printed_on_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _manifest_in(tmp_path)
    _install_steps(monkeypatch, [_StubStep(id="s1")])

    rendered: list[Any] = []

    def fake_render(project_dir: Path, mf: Manifest, stack: Any | None) -> str:
        rendered.append((project_dir, mf, stack))
        return "WELCOME-PANEL-MARKER"

    monkeypatch.setattr("agent_scaffold.welcome.render_welcome_panel", fake_render)

    rc = _run_up_inline(
        project_dir=tmp_path,
        manifest=manifest,
        recipe=None,
        resolved_stack=None,
        flags=_flags(yes=True),
        interactive=False,
    )
    assert rc == 0
    assert len(rendered) == 1
    assert rendered[0][0] == tmp_path
    assert rendered[0][1] is manifest


def test_welcome_panel_suppressed_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed run skips the welcome panel — the user needs to see the failure first."""
    manifest = _manifest_in(tmp_path)
    bad = _StubStep(id="boom", apply_status=StepStatus.FAILED, apply_error="boom")
    _install_steps(monkeypatch, [bad])

    rendered: list[Any] = []
    monkeypatch.setattr(
        "agent_scaffold.welcome.render_welcome_panel",
        lambda *_a, **_kw: rendered.append(1) or "panel",
    )

    rc = _run_up_inline(
        project_dir=tmp_path,
        manifest=manifest,
        recipe=None,
        resolved_stack=None,
        flags=_flags(yes=True),
        interactive=False,
    )
    assert rc == 1
    assert rendered == []


# ---------------------------------------------------------------------------
# Step failure path
# ---------------------------------------------------------------------------


def test_step_failure_returns_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _manifest_in(tmp_path)
    bad = _StubStep(id="docker_up", apply_status=StepStatus.FAILED, apply_error="oom")
    _install_steps(monkeypatch, [bad])

    rc = _run_up_inline(
        project_dir=tmp_path,
        manifest=manifest,
        recipe=None,
        resolved_stack=None,
        flags=_flags(yes=True),
        interactive=False,
    )
    assert rc == 1


# ---------------------------------------------------------------------------
# _resolve_frontend_url + _autorun_after_new
# ---------------------------------------------------------------------------


def _write_pid_file(project_dir: Path, *, port: int = 3000, pid: int = 4321) -> None:
    pid_file = project_dir / SCAFFOLD_DIR / "frontend.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(
        json.dumps({"pid": pid, "port": port, "started_at": "2026-05-30T00:00:00+00:00"}),
        encoding="utf-8",
    )


def test_resolve_frontend_url_returns_url_when_pid_file_present(tmp_path: Path) -> None:
    _write_pid_file(tmp_path, port=4001)
    assert cli_mod._resolve_frontend_url(tmp_path) == "http://localhost:4001"


def test_resolve_frontend_url_none_when_missing(tmp_path: Path) -> None:
    assert cli_mod._resolve_frontend_url(tmp_path) is None


@pytest.mark.parametrize(
    "body",
    [
        "{not-json",
        json.dumps({"pid": 1}),  # no port
        json.dumps({"pid": 1, "port": "abc"}),
        json.dumps({"pid": 1, "port": -1}),
    ],
)
def test_resolve_frontend_url_none_on_malformed_pid_file(tmp_path: Path, body: str) -> None:
    pid_file = tmp_path / SCAFFOLD_DIR / "frontend.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(body, encoding="utf-8")
    assert cli_mod._resolve_frontend_url(tmp_path) is None


def test_autorun_after_new_with_autorun_yes_is_silent_ci_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """autorun_yes=True preserves the pre-Phase-4 'just do it' shape — yes=True,
    interactive=False, no confirm prompt fires."""
    _manifest_in(tmp_path)
    captured: dict[str, Any] = {}

    def fake_run_up_inline(**kwargs: Any) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(cli_mod, "_run_up_inline", fake_run_up_inline)

    rc = cli_mod._autorun_after_new(
        project_dir=tmp_path,
        recipe=None,
        resolved_stack=None,
        open_browser=False,
        autorun_yes=True,
    )
    assert rc == 0
    assert captured["flags"].yes is True
    assert captured["interactive"] is False
    assert captured["project_dir"] == tmp_path


def test_autorun_after_new_default_is_gated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default autorun (autorun_yes=False) calls into _run_up_inline with
    yes=False + interactive=True so the existing yes/edit/dry-run/no prompt fires."""
    _manifest_in(tmp_path)
    captured: dict[str, Any] = {}

    def fake_run_up_inline(**kwargs: Any) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(cli_mod, "_run_up_inline", fake_run_up_inline)

    rc = cli_mod._autorun_after_new(
        project_dir=tmp_path,
        recipe=None,
        resolved_stack=None,
        open_browser=False,
    )
    assert rc == 0
    assert captured["flags"].yes is False
    assert captured["interactive"] is True


def test_autorun_after_new_opens_browser_when_requested_with_autorun_yes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In CI shape (autorun_yes=True), browser opens immediately with no prompt."""
    _manifest_in(tmp_path)
    _write_pid_file(tmp_path, port=3000)
    monkeypatch.setattr(cli_mod, "_run_up_inline", lambda **_kw: 0)

    opens: list[str] = []
    monkeypatch.setattr(
        "agent_scaffold.welcome._open_browser_safe",
        lambda url: opens.append(url) or True,
    )

    rc = cli_mod._autorun_after_new(
        project_dir=tmp_path,
        recipe=None,
        resolved_stack=None,
        open_browser=True,
        autorun_yes=True,
    )
    assert rc == 0
    assert opens == ["http://localhost:3000"]


def test_autorun_after_new_browser_prompts_default_yes_when_interactive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Interactive default: prompts before opening; bare-Enter (default Yes) opens."""
    _manifest_in(tmp_path)
    _write_pid_file(tmp_path, port=3000)
    monkeypatch.setattr(cli_mod, "_run_up_inline", lambda **_kw: 0)

    opens: list[str] = []
    monkeypatch.setattr(
        "agent_scaffold.welcome._open_browser_safe",
        lambda url: opens.append(url) or True,
    )
    # Stub the console's input so the prompt resolves to bare-Enter (default-yes).
    from agent_scaffold.cli_shared import console as shared_console

    monkeypatch.setattr(shared_console, "input", lambda _prompt: "")

    rc = cli_mod._autorun_after_new(
        project_dir=tmp_path,
        recipe=None,
        resolved_stack=None,
        open_browser=True,
    )
    assert rc == 0
    assert opens == ["http://localhost:3000"]


def test_autorun_after_new_browser_decline_skips_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """User typing 'n' at the browser prompt → no open call, friendly hint."""
    _manifest_in(tmp_path)
    _write_pid_file(tmp_path, port=3000)
    monkeypatch.setattr(cli_mod, "_run_up_inline", lambda **_kw: 0)

    opens: list[str] = []
    monkeypatch.setattr(
        "agent_scaffold.welcome._open_browser_safe",
        lambda url: opens.append(url) or True,
    )
    from agent_scaffold.cli_shared import console as shared_console

    monkeypatch.setattr(shared_console, "input", lambda _prompt: "n")

    rc = cli_mod._autorun_after_new(
        project_dir=tmp_path,
        recipe=None,
        resolved_stack=None,
        open_browser=True,
    )
    assert rc == 0
    assert opens == []


def test_autorun_after_new_skips_browser_when_no_pid_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _manifest_in(tmp_path)
    monkeypatch.setattr(cli_mod, "_run_up_inline", lambda **_kw: 0)
    opens: list[str] = []
    monkeypatch.setattr(
        "agent_scaffold.welcome._open_browser_safe", lambda url: opens.append(url) or True
    )

    rc = cli_mod._autorun_after_new(
        project_dir=tmp_path,
        recipe=None,
        resolved_stack=None,
        open_browser=True,
        autorun_yes=True,
    )
    assert rc == 0
    assert opens == []


def test_autorun_after_new_propagates_up_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _manifest_in(tmp_path)
    monkeypatch.setattr(cli_mod, "_run_up_inline", lambda **_kw: 1)

    rc = cli_mod._autorun_after_new(
        project_dir=tmp_path,
        recipe=None,
        resolved_stack=None,
        open_browser=True,
        autorun_yes=True,
    )
    assert rc == 1


def test_autorun_after_new_skips_browser_on_up_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _manifest_in(tmp_path)
    _write_pid_file(tmp_path, port=3000)
    monkeypatch.setattr(cli_mod, "_run_up_inline", lambda **_kw: 1)
    opens: list[str] = []
    monkeypatch.setattr(
        "agent_scaffold.welcome._open_browser_safe", lambda url: opens.append(url) or True
    )

    cli_mod._autorun_after_new(
        project_dir=tmp_path,
        recipe=None,
        resolved_stack=None,
        open_browser=True,
        autorun_yes=True,
    )
    assert opens == []  # don't open a broken UI


def test_autorun_after_new_returns_zero_when_manifest_missing(tmp_path: Path) -> None:
    """A missing manifest only means autorun is a no-op — generation already succeeded."""
    rc = cli_mod._autorun_after_new(
        project_dir=tmp_path,
        recipe=None,
        resolved_stack=None,
        open_browser=False,
    )
    assert rc == 0
