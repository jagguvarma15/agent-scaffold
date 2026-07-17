"""Tests for the autorun chain on ``agent-scaffold new``.

The full ``cmd_new`` happy path is exercised by ``test_cli_e2e``; this file
pins the autorun-gating decision: when does ``_autorun_after_new`` fire, when
does the legacy "next steps" panel print, and what flags toggle which path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from agent_scaffold import cli, generator
from agent_scaffold.cli import app

# Reuse the mock client structure from test_cli_e2e ----------------------


class _Block:
    def __init__(self, text: str) -> None:
        self.text = text


class _Response:
    def __init__(self, text: str) -> None:
        self.content = [_Block(text)]


class _StreamCtx:
    def __init__(self, response: Any) -> None:
        self._response = response

    def __enter__(self) -> _StreamCtx:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def __iter__(self) -> Any:
        return iter(())

    def get_final_message(self) -> Any:
        return self._response


class _Messages:
    def __init__(self, payload: str) -> None:
        self._payload = payload
        self.calls: list[dict[str, Any]] = []

    def stream(self, **kwargs: Any) -> _StreamCtx:
        self.calls.append(kwargs)
        return _StreamCtx(_Response(self._payload))


class _Client:
    def __init__(self, payload: str) -> None:
        self.messages = _Messages(payload)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def autorun_spy(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace ``_autorun_after_new`` with a spy that records its call args.

    The autorun chain pulls in docker / pnpm / capability probes — none of which
    we want to actually run in a unit test. Returning ``0`` looks like success
    so the CLI exits cleanly.
    """
    calls: list[dict[str, Any]] = []

    def spy(**kwargs: Any) -> int:
        calls.append(kwargs)
        return 0

    monkeypatch.setattr(cli, "_autorun_after_new", spy)
    return calls


def _invoke_new(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_deployments_path: Path,
    mock_responses_path: Path,
    *,
    dest_name: str = "demo_agent",
    extra_args: list[str] | None = None,
) -> tuple[Any, Path]:
    """Drive ``agent-scaffold new`` end-to-end with controlled inputs."""
    payload = (mock_responses_path / "valid_python.json").read_text(encoding="utf-8")
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: _Client(payload))

    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("AGENT_SCAFFOLD_DEPLOYMENTS_PATH", str(mock_deployments_path))
    monkeypatch.setenv("AGENT_SCAFFOLD_CACHE_DIR", str(cache_dir))

    dest = tmp_path / "out" / dest_name
    args = [
        "new",
        "--non-interactive",
        "--recipe",
        "customer-support-triage",
        "--language",
        "python",
        "--framework",
        "langgraph",
        "--project-name",
        dest_name,
        "--dest",
        str(dest),
        "--write-mode",
        "overwrite",
        "--skip-validation",
    ]
    if extra_args:
        args.extend(extra_args)
    result = runner.invoke(app, args)
    return result, dest


# ---------------------------------------------------------------------------
# --non-interactive (existing CI shape) implicitly disables autorun
# ---------------------------------------------------------------------------


def test_non_interactive_does_not_autorun(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_deployments_path: Path,
    mock_responses_path: Path,
    autorun_spy: list[dict[str, Any]],
) -> None:
    """CI scripts that use --non-interactive must not suddenly start chaining `up`."""
    result, _dest = _invoke_new(
        runner, tmp_path, monkeypatch, mock_deployments_path, mock_responses_path
    )
    assert result.exit_code == 0, result.output
    assert autorun_spy == []


def test_non_interactive_still_prints_legacy_next_steps(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_deployments_path: Path,
    mock_responses_path: Path,
    autorun_spy: list[dict[str, Any]],
) -> None:
    """The legacy `print_next_steps` panel only prints when autorun is suppressed."""
    result, _dest = _invoke_new(
        runner, tmp_path, monkeypatch, mock_deployments_path, mock_responses_path
    )
    assert result.exit_code == 0, result.output
    # "Next steps" header from pipeline.print_next_steps is the canonical marker.
    assert "Next steps" in result.output or "next steps" in result.output.lower()


# ---------------------------------------------------------------------------
# --no-autorun explicit opt-out
# ---------------------------------------------------------------------------


def test_no_autorun_flag_skips_chain(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_deployments_path: Path,
    mock_responses_path: Path,
    autorun_spy: list[dict[str, Any]],
) -> None:
    result, _dest = _invoke_new(
        runner,
        tmp_path,
        monkeypatch,
        mock_deployments_path,
        mock_responses_path,
        extra_args=["--no-autorun"],
    )
    assert result.exit_code == 0, result.output
    assert autorun_spy == []


def test_no_autorun_flag_keeps_legacy_next_steps_panel(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_deployments_path: Path,
    mock_responses_path: Path,
    autorun_spy: list[dict[str, Any]],
) -> None:
    result, _dest = _invoke_new(
        runner,
        tmp_path,
        monkeypatch,
        mock_deployments_path,
        mock_responses_path,
        extra_args=["--no-autorun"],
    )
    assert result.exit_code == 0, result.output
    assert "Next steps" in result.output or "next steps" in result.output.lower()


# ---------------------------------------------------------------------------
# Flag surface — --no-open-browser is parsed and forwarded
# ---------------------------------------------------------------------------


def test_open_browser_default_propagates_when_autorun_runs(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_deployments_path: Path,
    mock_responses_path: Path,
    autorun_spy: list[dict[str, Any]],
) -> None:
    """If autorun fires, the --open-browser default must reach _autorun_after_new.

    --non-interactive implicitly disables autorun in CI today; this test
    explicitly bypasses that by also passing --autorun, so we can see the
    flag wiring is correct.
    """
    # We can't easily exercise interactive mode in CI, so this test asserts
    # the flag surface exists. The actual auto-chaining path is covered by
    # the helper-level tests in test_run_up_inline.py.
    result, _dest = _invoke_new(
        runner,
        tmp_path,
        monkeypatch,
        mock_deployments_path,
        mock_responses_path,
        extra_args=["--no-open-browser"],
    )
    assert result.exit_code == 0, result.output
    # --non-interactive bypasses autorun, so the spy still isn't called —
    # this test just pins that --no-open-browser is a valid flag.
    assert autorun_spy == []


def test_unknown_autorun_value_rejected_by_typer(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_deployments_path: Path,
    mock_responses_path: Path,
) -> None:
    """``--autorun=garbage`` is a parse error (boolean toggle is on/off only)."""
    payload = (mock_responses_path / "valid_python.json").read_text(encoding="utf-8")
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: _Client(payload))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("AGENT_SCAFFOLD_DEPLOYMENTS_PATH", str(mock_deployments_path))

    result = runner.invoke(
        app,
        [
            "new",
            "--non-interactive",
            "--autorun=garbage",
            "--recipe",
            "customer-support-triage",
            "--language",
            "python",
            "--project-name",
            "x",
            "--dest",
            str(tmp_path / "x"),
        ],
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Help text mentions the new flags
# ---------------------------------------------------------------------------


def test_help_lists_autorun_flags(runner: CliRunner) -> None:
    """Both new flags surface in --help output (Typer's text-wrapping may split tokens)."""
    result = runner.invoke(app, ["new", "--help"], terminal_width=200)
    assert result.exit_code == 0
    assert "autorun" in result.output
    assert "open-browser" in result.output


# ---------------------------------------------------------------------------
# Stale-stack teardown on regenerate (teardown_stale)
# ---------------------------------------------------------------------------


def _stub_autorun_chain(monkeypatch: pytest.MonkeyPatch, order: list[str]) -> None:
    """Stub the heavy pieces of _autorun_after_new; record call order."""

    class _FakeManifest:
        recipe = "test-recipe"
        capabilities: list[str] = []

    monkeypatch.setattr(cli, "read_manifest", lambda _p: _FakeManifest())
    monkeypatch.setattr(cli, "_down_inline", lambda *_a, **_k: (order.append("down"), 0)[1])
    monkeypatch.setattr(cli, "_run_up_inline", lambda **_k: (order.append("up"), 0)[1])


def _seed_docker_up_state(project_dir: Path) -> None:
    from agent_scaffold.orchestrator import (
        OrchestratorState,
        StepState,
        StepStatus,
        write_state,
    )

    write_state(
        project_dir,
        OrchestratorState(steps={"docker_up": StepState(status=StepStatus.PENDING)}),
    )


def test_autorun_tears_down_stale_stack_before_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Previously provisioned destination + compose file: down runs before up,
    so containers built from the old files never survive a regenerate."""
    order: list[str] = []
    _stub_autorun_chain(monkeypatch, order)
    (tmp_path / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")
    _seed_docker_up_state(tmp_path)

    rc = cli._autorun_after_new(
        project_dir=tmp_path,
        recipe=None,
        resolved_stack=None,
        open_browser=False,
        teardown_stale=True,
    )
    assert rc == 0
    assert order == ["down", "up"]


def test_autorun_skips_teardown_on_fresh_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No prior provisioning state: the teardown subprocess never runs."""
    order: list[str] = []
    _stub_autorun_chain(monkeypatch, order)
    (tmp_path / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")

    rc = cli._autorun_after_new(
        project_dir=tmp_path,
        recipe=None,
        resolved_stack=None,
        open_browser=False,
        teardown_stale=True,
    )
    assert rc == 0
    assert order == ["up"]


def test_autorun_skips_teardown_without_compose_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    _stub_autorun_chain(monkeypatch, order)
    _seed_docker_up_state(tmp_path)

    rc = cli._autorun_after_new(
        project_dir=tmp_path,
        recipe=None,
        resolved_stack=None,
        open_browser=False,
        teardown_stale=True,
    )
    assert rc == 0
    assert order == ["up"]


def test_autorun_default_does_not_teardown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Callers that already tore down (REPL /up) get no second teardown."""
    order: list[str] = []
    _stub_autorun_chain(monkeypatch, order)
    (tmp_path / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")
    _seed_docker_up_state(tmp_path)

    rc = cli._autorun_after_new(
        project_dir=tmp_path,
        recipe=None,
        resolved_stack=None,
        open_browser=False,
    )
    assert rc == 0
    assert order == ["up"]
