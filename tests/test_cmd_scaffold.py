"""Tests for first-launch Anthropic-key onboarding in ``cmd_scaffold``.

A fresh install with no env key and no stored credential should open the
secure paste form (browser, or hidden getpass when headless) and store the
key — instead of dead-ending on a ``MissingKeyError``. CI / non-interactive
sessions keep the hard exit: there the key must come from the environment.
"""

from __future__ import annotations

import getpass
import sys
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

from agent_scaffold import auth, auth_browser, cli
from agent_scaffold.cli import app
from agent_scaffold.config import MissingKeyError

# ---------------------------------------------------------------------------
# _capture_key_first_launch: browser form, getpass fallback
# ---------------------------------------------------------------------------


def test_capture_key_uses_browser_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_browser, "browser_available", lambda: True)
    monkeypatch.setattr(auth_browser, "browser_paste_flow", lambda *a, **k: "sk-ant-browser")
    assert cli._capture_key_first_launch() == "sk-ant-browser"


def test_capture_key_falls_back_to_getpass_when_no_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(auth_browser, "browser_available", lambda: False)
    # getpass.getpass (never input()) — the captured text is stripped.
    monkeypatch.setattr(getpass, "getpass", lambda *a, **k: "  sk-ant-typed  ")
    assert cli._capture_key_first_launch() == "sk-ant-typed"


def test_capture_key_getpass_after_empty_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    """Browser available but user submits nothing → fall through to getpass."""
    monkeypatch.setattr(auth_browser, "browser_available", lambda: True)
    monkeypatch.setattr(auth_browser, "browser_paste_flow", lambda *a, **k: None)
    monkeypatch.setattr(getpass, "getpass", lambda *a, **k: "sk-ant-typed")
    assert cli._capture_key_first_launch() == "sk-ant-typed"


def test_capture_key_returns_none_on_abort(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_browser, "browser_available", lambda: False)

    def _abort(*_a: object, **_k: object) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr(getpass, "getpass", _abort)
    assert cli._capture_key_first_launch() is None


# ---------------------------------------------------------------------------
# _onboard_key_or_exit: interactive stores + re-resolves; CI exits
# ---------------------------------------------------------------------------


def test_onboard_stores_key_to_file_and_reresolves(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_stdio_is_interactive", lambda: True)
    monkeypatch.setattr(cli, "_capture_key_first_launch", lambda: "sk-ant-new")

    stored: dict[str, object] = {}

    def fake_store(name: str, value: object, backend: str = "keyring") -> None:
        stored["name"] = name
        stored["backend"] = backend
        stored["value"] = value.get_secret_value()  # type: ignore[attr-defined]

    monkeypatch.setattr(auth, "store_key", fake_store)
    sentinel = object()
    monkeypatch.setattr(cli, "load_config", lambda: sentinel)

    result = cli._onboard_key_or_exit(MissingKeyError("no key"))

    assert result is sentinel
    assert stored == {"name": auth.DEFAULT_KEY_NAME, "backend": "file", "value": "sk-ant-new"}


def test_onboard_non_interactive_exits_without_prompting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "_stdio_is_interactive", lambda: False)
    captured = {"called": False}

    def _capture() -> str:
        captured["called"] = True
        return "sk-ant-should-not-happen"

    monkeypatch.setattr(cli, "_capture_key_first_launch", _capture)

    with pytest.raises(typer.Exit) as excinfo:
        cli._onboard_key_or_exit(MissingKeyError("no key"))

    assert excinfo.value.exit_code == 1
    assert captured["called"] is False  # CI never blocks on a prompt


def test_onboard_aborts_when_no_key_supplied(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_stdio_is_interactive", lambda: True)
    monkeypatch.setattr(cli, "_capture_key_first_launch", lambda: None)
    monkeypatch.setattr(
        auth, "store_key", lambda *a, **k: pytest.fail("must not store without a key")
    )
    with pytest.raises(typer.Exit):
        cli._onboard_key_or_exit(MissingKeyError("no key"))


# ---------------------------------------------------------------------------
# End-to-end: `scaffold` with no key in a non-interactive (CliRunner) session
# ---------------------------------------------------------------------------


def test_scaffold_no_key_non_interactive_exits_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    """CliRunner is non-tty, so a fresh `scaffold` with no key must exit 1 with
    the missing-key message — never hang waiting on a prompt."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(cli, "_capture_key_first_launch", lambda: pytest.fail("must not prompt"))

    result = CliRunner().invoke(app, ["scaffold"])

    assert result.exit_code == 1
    assert "No Anthropic key found" in result.output


def test_scaffold_interactive_onboards_then_opens_shell(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    """End-to-end wiring: an interactive `scaffold` with no key routes the
    MissingKeyError through onboarding (capture → store → re-resolve) and
    proceeds to open the shell. Guards the `except MissingKeyError ->
    _onboard_key_or_exit` block — deleting it makes this test fail (the run
    hard-exits before reaching run_shell)."""
    from agent_scaffold.repl import shell as repl_shell

    calls = {"load_config": 0, "store": 0, "run_shell": 0, "captured": False}
    fake_cfg = SimpleNamespace(cache_dir=tmp_path)

    def fake_load_config(*_a: object, **_k: object) -> object:
        calls["load_config"] += 1
        if calls["load_config"] == 1:
            raise MissingKeyError("No Anthropic key found.")
        return fake_cfg

    def fake_store(name: str, value: object, backend: str = "keyring") -> None:
        calls["store"] += 1
        assert backend == "file"

    def fake_capture() -> str:
        calls["captured"] = True
        return "sk-ant-test"

    def fake_run_shell(*_a: object, **_k: object) -> int:
        calls["run_shell"] += 1
        return 0

    monkeypatch.setattr(cli, "load_config", fake_load_config)
    monkeypatch.setattr(cli, "_stdio_is_interactive", lambda: True)
    monkeypatch.setattr(cli, "_capture_key_first_launch", fake_capture)
    monkeypatch.setattr(auth, "store_key", fake_store)
    monkeypatch.setattr(cli, "resolve_deployments", lambda **_k: SimpleNamespace(path=tmp_path))
    monkeypatch.setattr(cli, "resolve_blueprints", lambda **_k: SimpleNamespace(path=tmp_path))
    monkeypatch.setattr(repl_shell, "run_shell", fake_run_shell)

    result = CliRunner().invoke(app, ["scaffold"])

    assert result.exit_code == 0, result.output
    assert calls["captured"] is True  # onboarding form was reached
    assert calls["store"] == 1  # key persisted
    assert calls["load_config"] == 2  # re-resolved after storing
    assert calls["run_shell"] == 1  # proceeded into the shell


def test_stdio_is_interactive_requires_both_streams_to_be_ttys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CI safety gate is AND, not OR: a piped/redirected session (exactly
    one stream a TTY) must read as non-interactive so onboarding never prompts
    in CI."""

    class _Stream:
        def __init__(self, tty: bool) -> None:
            self._tty = tty

        def isatty(self) -> bool:
            return self._tty

    cases = [(True, True, True), (True, False, False), (False, True, False), (False, False, False)]
    for stdin_tty, stdout_tty, expected in cases:
        monkeypatch.setattr(sys, "stdin", _Stream(stdin_tty))
        monkeypatch.setattr(sys, "stdout", _Stream(stdout_tty))
        assert cli._stdio_is_interactive() is expected, (stdin_tty, stdout_tty)


def test_scaffold_positional_project_dir_passes_open_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    """`agent-scaffold scaffold <dir>` threads the directory into run_shell as
    open_dir so the shell attaches to the existing generated project."""
    from pathlib import Path

    from agent_scaffold.repl import shell as repl_shell

    project = Path(str(tmp_path)) / "existing-proj"
    project.mkdir()
    captured: dict[str, object] = {}

    def fake_run_shell(*_a: object, **kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(cli, "load_config", lambda: SimpleNamespace(cache_dir=tmp_path))
    monkeypatch.setattr(cli, "resolve_deployments", lambda **_k: SimpleNamespace(path=tmp_path))
    monkeypatch.setattr(cli, "resolve_blueprints", lambda **_k: SimpleNamespace(path=tmp_path))
    monkeypatch.setattr(repl_shell, "run_shell", fake_run_shell)

    result = CliRunner().invoke(app, ["scaffold", str(project)])

    assert result.exit_code == 0, result.output
    assert captured["open_dir"] == project


def test_scaffold_without_positional_passes_no_open_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    from agent_scaffold.repl import shell as repl_shell

    captured: dict[str, object] = {}

    def fake_run_shell(*_a: object, **kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(cli, "load_config", lambda: SimpleNamespace(cache_dir=tmp_path))
    monkeypatch.setattr(cli, "resolve_deployments", lambda **_k: SimpleNamespace(path=tmp_path))
    monkeypatch.setattr(cli, "resolve_blueprints", lambda **_k: SimpleNamespace(path=tmp_path))
    monkeypatch.setattr(repl_shell, "run_shell", fake_run_shell)

    result = CliRunner().invoke(app, ["scaffold"])

    assert result.exit_code == 0, result.output
    assert captured["open_dir"] is None


def test_scaffold_syncs_sources_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    from agent_scaffold.repl import shell as repl_shell

    captured: dict[str, dict] = {}

    monkeypatch.setattr(cli, "load_config", lambda: SimpleNamespace(cache_dir=tmp_path))
    monkeypatch.setattr(
        cli,
        "resolve_deployments",
        lambda **k: captured.setdefault("dep", k)
        and SimpleNamespace(path=tmp_path)
        or SimpleNamespace(path=tmp_path),
    )
    monkeypatch.setattr(
        cli,
        "resolve_blueprints",
        lambda **k: captured.setdefault("bp", k)
        and SimpleNamespace(path=tmp_path)
        or SimpleNamespace(path=tmp_path),
    )
    monkeypatch.setattr(repl_shell, "run_shell", lambda *a, **k: 0)

    result = CliRunner().invoke(app, ["scaffold"])
    assert result.exit_code == 0, result.output
    assert captured["dep"]["refresh"] is True
    assert captured["bp"]["refresh"] is True


def test_scaffold_no_sync_skips_refresh(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    from agent_scaffold.repl import shell as repl_shell

    captured: dict[str, dict] = {}

    monkeypatch.setattr(cli, "load_config", lambda: SimpleNamespace(cache_dir=tmp_path))
    monkeypatch.setattr(
        cli,
        "resolve_deployments",
        lambda **k: captured.setdefault("dep", k)
        and SimpleNamespace(path=tmp_path)
        or SimpleNamespace(path=tmp_path),
    )
    monkeypatch.setattr(
        cli,
        "resolve_blueprints",
        lambda **k: captured.setdefault("bp", k)
        and SimpleNamespace(path=tmp_path)
        or SimpleNamespace(path=tmp_path),
    )
    monkeypatch.setattr(repl_shell, "run_shell", lambda *a, **k: 0)

    result = CliRunner().invoke(app, ["scaffold", "--no-sync"])
    assert result.exit_code == 0, result.output
    assert captured["dep"]["refresh"] is False
    assert captured["bp"]["refresh"] is False
