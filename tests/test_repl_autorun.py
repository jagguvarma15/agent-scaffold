"""Tests for the REPL autorun chain — ``/autorun`` slash + chain after ``/go``."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import agent_scaffold.repl.shell as shell_mod
from agent_scaffold.config import Config
from agent_scaffold.repl.commands import CommandError, CommandHandler
from agent_scaffold.repl.session import SessionState
from agent_scaffold.repl.shell import _autorun_after_repl_generate
from agent_scaffold.sources import DEPLOYMENTS_SPEC, ResolvedSource

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def _cfg(tmp_path: Path) -> Config:
    return Config(
        anthropic_api_key="test-key",
        cache_dir=tmp_path / "cache",
        failures_dir=tmp_path / "cache" / "failures",
    )


@pytest.fixture
def _source(tmp_path: Path) -> ResolvedSource:
    return ResolvedSource(
        spec=DEPLOYMENTS_SPEC,
        path=tmp_path / "deployments",
        label="test",
        kind="explicit-path",
        commit_sha=None,
    )


def _session_state(
    cfg: Config, src: ResolvedSource, *, autorun: bool = True, dest: Path | None = None
) -> SessionState:
    return SessionState(
        cfg=cfg,
        deployments=src,
        blueprints=src,
        autorun=autorun,
        dest=dest,
    )


# ---------------------------------------------------------------------------
# /autorun slash command
# ---------------------------------------------------------------------------


def test_autorun_default_on_session_state(_cfg: Config, _source: ResolvedSource) -> None:
    assert _session_state(_cfg, _source).autorun is True


def test_autorun_slash_off_disables(_cfg: Config, _source: ResolvedSource) -> None:
    state = _session_state(_cfg, _source, autorun=True)
    handler = CommandHandler(recipes=[])
    result = handler.cmd_autorun(["off"], state)
    assert result.new_state is not None
    assert result.new_state.autorun is False


def test_autorun_slash_on_enables(_cfg: Config, _source: ResolvedSource) -> None:
    state = _session_state(_cfg, _source, autorun=False)
    handler = CommandHandler(recipes=[])
    result = handler.cmd_autorun(["on"], state)
    assert result.new_state is not None
    assert result.new_state.autorun is True


@pytest.mark.parametrize("token", ["TRUE", "Yes", "1"])
def test_autorun_slash_accepts_truthy_variants(
    token: str, _cfg: Config, _source: ResolvedSource
) -> None:
    state = _session_state(_cfg, _source, autorun=False)
    handler = CommandHandler(recipes=[])
    result = handler.cmd_autorun([token], state)
    assert result.new_state is not None
    assert result.new_state.autorun is True


@pytest.mark.parametrize("token", ["FALSE", "No", "0"])
def test_autorun_slash_accepts_falsy_variants(
    token: str, _cfg: Config, _source: ResolvedSource
) -> None:
    state = _session_state(_cfg, _source, autorun=True)
    handler = CommandHandler(recipes=[])
    result = handler.cmd_autorun([token], state)
    assert result.new_state is not None
    assert result.new_state.autorun is False


def test_autorun_slash_no_args_toggles(_cfg: Config, _source: ResolvedSource) -> None:
    handler = CommandHandler(recipes=[])
    state = _session_state(_cfg, _source, autorun=True)
    result_off = handler.cmd_autorun([], state)
    assert result_off.new_state is not None
    assert result_off.new_state.autorun is False

    result_on = handler.cmd_autorun([], result_off.new_state)
    assert result_on.new_state is not None
    assert result_on.new_state.autorun is True


def test_autorun_slash_rejects_unknown_token(_cfg: Config, _source: ResolvedSource) -> None:
    handler = CommandHandler(recipes=[])
    state = _session_state(_cfg, _source)
    with pytest.raises(CommandError):
        handler.cmd_autorun(["maybe"], state)


# ---------------------------------------------------------------------------
# Dispatcher integration — /autorun is registered alongside /go
# ---------------------------------------------------------------------------


def test_autorun_command_dispatchable_through_handler(
    _cfg: Config, _source: ResolvedSource
) -> None:
    state = _session_state(_cfg, _source, autorun=True)
    handler = CommandHandler(recipes=[])
    # The dispatcher prefixes /; check the command exists in the registered set.
    result = handler.dispatch("/autorun off", state)
    assert result.new_state is not None
    assert result.new_state.autorun is False


# ---------------------------------------------------------------------------
# _autorun_after_repl_generate — chain after generation
# ---------------------------------------------------------------------------


def _stub_manifest_read(monkeypatch: pytest.MonkeyPatch, recipe_slug: str = "test-recipe") -> None:
    """Make ``read_manifest`` return a minimal in-memory manifest."""

    class FakeMf:
        recipe = recipe_slug

    monkeypatch.setattr(shell_mod, "read_manifest", lambda _p: FakeMf())


def test_autorun_after_repl_generate_calls_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_manifest_read(monkeypatch)
    monkeypatch.setattr("agent_scaffold.cli._resolve_recipe_silently", lambda _slug: None)
    monkeypatch.setattr("agent_scaffold.cli._resolve_capability_stack_silently", lambda _r: None)

    captured: dict[str, Any] = {}

    def fake_autorun_after_new(**kwargs: Any) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr("agent_scaffold.cli._autorun_after_new", fake_autorun_after_new)

    console = MagicMock()
    _autorun_after_repl_generate(tmp_path, console)

    assert captured["project_dir"] == tmp_path
    assert captured["open_browser"] is True


def test_autorun_after_repl_generate_skips_when_no_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing manifest is a soft skip in the REPL — print + return."""
    called: list[Any] = []
    monkeypatch.setattr("agent_scaffold.cli._autorun_after_new", lambda **_kw: called.append(1))
    console = MagicMock()
    _autorun_after_repl_generate(tmp_path, console)
    assert called == []
    console.print.assert_called()  # skipped message


def test_autorun_after_repl_generate_prints_warning_on_nonzero_rc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_manifest_read(monkeypatch)
    monkeypatch.setattr("agent_scaffold.cli._resolve_recipe_silently", lambda _slug: None)
    monkeypatch.setattr("agent_scaffold.cli._resolve_capability_stack_silently", lambda _r: None)
    monkeypatch.setattr("agent_scaffold.cli._autorun_after_new", lambda **_kw: 1)

    console = MagicMock()
    _autorun_after_repl_generate(tmp_path, console)

    # Inspect the printed text for the warning.
    printed_args = [c.args[0] if c.args else "" for c in console.print.call_args_list]
    assert any("exit code 1" in str(arg) for arg in printed_args)


# ---------------------------------------------------------------------------
# Toggling autorun off via slash command stops the chain
# ---------------------------------------------------------------------------


def test_session_state_autorun_off_persists(_cfg: Config, _source: ResolvedSource) -> None:
    """Sanity: once SessionState.autorun is False, that field stays False until re-toggled."""
    state = _session_state(_cfg, _source, autorun=True)
    handler = CommandHandler(recipes=[])
    state = handler.cmd_autorun(["off"], state).new_state  # type: ignore[assignment]
    assert state is not None
    assert state.autorun is False
