"""Tests for the REPL lifecycle slash commands."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from agent_scaffold.config import Config
from agent_scaffold.repl.commands import CommandError, CommandHandler
from agent_scaffold.repl.session import SessionState
from agent_scaffold.sources import DEPLOYMENTS_SPEC, ResolvedSource


def _state(dest: str | None = "/tmp/demo") -> SessionState:
    tmp = Path(tempfile.gettempdir())
    cfg = Config(
        anthropic_api_key="sk-test",
        cache_dir=tmp / "cache",
        failures_dir=tmp / "cache" / "failures",
    )
    src = ResolvedSource(
        spec=DEPLOYMENTS_SPEC,
        path=tmp,
        label="test",
        kind="bundled",
        commit_sha=None,
    )
    dest_path = Path(dest) if dest else None
    return SessionState(cfg=cfg, deployments=src, blueprints=src, dest=dest_path)


def _handler() -> CommandHandler:
    return CommandHandler(recipes=[])


def test_deploy_slash_requires_target() -> None:
    handler = _handler()
    with pytest.raises(CommandError) as exc:
        handler.cmd_deploy([], _state())
    assert "usage:" in str(exc.value)


def test_deploy_slash_requires_dest() -> None:
    handler = _handler()
    with pytest.raises(CommandError) as exc:
        handler.cmd_deploy(["vercel"], _state(dest=None))
    assert "dest" in str(exc.value)


def test_deploy_slash_unknown_target() -> None:
    handler = _handler()
    with pytest.raises(CommandError) as exc:
        handler.cmd_deploy(["aws-lambda"], _state())
    assert "unknown deploy target" in str(exc.value).lower()


def test_deploy_slash_returns_dry_run_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("agent_scaffold.deploy.vercel.cli_present", lambda _: False)
    handler = _handler()
    result = handler.cmd_deploy(["vercel"], _state(dest=str(tmp_path)))
    assert any("vercel CLI not found" in str(m) for m in result.messages)


def test_down_slash_renders_command() -> None:
    handler = _handler()
    result = handler.cmd_down([], _state())
    text = " ".join(str(m) for m in result.messages)
    assert "agent-scaffold down" in text
    assert "-v" not in text


def test_down_slash_with_v_flag() -> None:
    handler = _handler()
    result = handler.cmd_down(["-v"], _state())
    text = " ".join(str(m) for m in result.messages)
    assert "agent-scaffold down -v" in text


def test_status_slash_is_a_readiness_check_not_a_dest_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # /status is now a fast local readiness check (key + docker + stack env
    # vars), so it works without a dest — no CommandError.
    from agent_scaffold.repl import readiness

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(readiness, "docker_status", lambda **_k: (False, "not installed"))
    handler = _handler()
    result = handler.cmd_status([], _state(dest=None))
    from rich.console import Console

    console = Console(record=True, color_system=None, width=120)
    for msg in result.messages:
        console.print(msg)
    text = console.export_text()
    assert "ANTHROPIC_API_KEY" in text  # surfaces the missing key
    assert "Docker" in text
    assert "/config" in text  # points at the fix


def test_logs_slash_requires_service() -> None:
    handler = _handler()
    with pytest.raises(CommandError):
        handler.cmd_logs([], _state())


def test_logs_slash_renders_command() -> None:
    handler = _handler()
    result = handler.cmd_logs(["redis"], _state(dest="/tmp/x"))
    text = " ".join(str(m) for m in result.messages)
    assert "agent-scaffold logs redis" in text
    assert "/tmp/x" in text


def test_lifecycle_commands_registered_in_handler() -> None:
    handler = _handler()
    for name in ("deploy", "down", "status", "logs"):
        assert name in handler.commands, f"/{name} missing from handler"
