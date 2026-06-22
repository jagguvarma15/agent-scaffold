"""Tests for first-launch Anthropic-key onboarding in ``cmd_scaffold``.

A fresh install with no env key and no stored credential should open the
secure paste form (browser, or hidden getpass when headless) and store the
key — instead of dead-ending on a ``MissingKeyError``. CI / non-interactive
sessions keep the hard exit: there the key must come from the environment.
"""

from __future__ import annotations

import getpass

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
