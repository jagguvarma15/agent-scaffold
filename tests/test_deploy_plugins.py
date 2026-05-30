"""Tests for ``agent_scaffold.deploy`` provider plugins."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_scaffold.deploy import fly, get_plugin, railway, vercel
from agent_scaffold.deploy._common import DeployResult


def test_get_plugin_resolves_known_targets() -> None:
    assert get_plugin("vercel").name == "vercel"
    assert get_plugin("fly").name == "fly"
    assert get_plugin("railway").name == "railway"


def test_get_plugin_raises_on_unknown_target() -> None:
    with pytest.raises(KeyError):
        get_plugin("aws-lambda")


def test_vercel_skip_when_cli_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent_scaffold.deploy.vercel.cli_present", lambda _: False)
    result = vercel.deploy(tmp_path, dry_run=False, yes=True)
    assert result.skipped is True
    assert result.skip_reason == "missing_cli"
    assert "npm i -g vercel" in result.summary


def test_vercel_skip_when_not_linked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent_scaffold.deploy.vercel.cli_present", lambda _: True)
    result = vercel.deploy(tmp_path, dry_run=True, yes=False)
    assert result.skipped is True
    assert result.skip_reason == "not_linked"


def test_vercel_dry_run_prints_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent_scaffold.deploy.vercel.cli_present", lambda _: True)
    (tmp_path / ".vercel").mkdir()
    (tmp_path / ".vercel" / "project.json").write_text("{}")

    result = vercel.deploy(tmp_path, dry_run=True, yes=False)
    assert result.skipped is True
    assert result.skip_reason == "dry_run"
    assert result.cmd_run == ["vercel", "deploy", "--prod"]


def test_vercel_yes_runs_provider_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent_scaffold.deploy.vercel.cli_present", lambda _: True)
    (tmp_path / ".vercel").mkdir()
    (tmp_path / ".vercel" / "project.json").write_text("{}")
    invocations: list[list[str]] = []

    def fake_run(cmd: list[str], cwd: Path, timeout: float | None = None) -> int:
        invocations.append(cmd)
        return 0

    monkeypatch.setattr("agent_scaffold.deploy.vercel.run_provider_cli", fake_run)
    result = vercel.deploy(tmp_path, dry_run=False, yes=True)
    assert invocations == [["vercel", "deploy", "--prod", "--yes"]]
    assert result.exit_code == 0
    assert result.skipped is False


def test_vercel_declines_without_yes_in_non_tty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("agent_scaffold.deploy.vercel.cli_present", lambda _: True)
    (tmp_path / ".vercel").mkdir()
    (tmp_path / ".vercel" / "project.json").write_text("{}")
    # confirm() returns False when stdin isn't a TTY → declined.
    monkeypatch.setattr("agent_scaffold.deploy.vercel.confirm", lambda _msg: False)
    invocations: list[list[str]] = []

    def fake_run(cmd: list[str], cwd: Path, timeout: float | None = None) -> int:
        invocations.append(cmd)
        return 0

    monkeypatch.setattr("agent_scaffold.deploy.vercel.run_provider_cli", fake_run)
    result = vercel.deploy(tmp_path, dry_run=False, yes=False)
    assert invocations == []
    assert result.skipped is True
    assert result.skip_reason == "declined"


def test_fly_skip_without_fly_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent_scaffold.deploy.fly.cli_present", lambda _: True)
    result = fly.deploy(tmp_path, dry_run=True, yes=False)
    assert result.skipped is True
    assert result.skip_reason == "not_launched"


def test_fly_dry_run_with_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent_scaffold.deploy.fly.cli_present", lambda _: True)
    (tmp_path / "fly.toml").write_text("app = 'demo'\n")
    result = fly.deploy(tmp_path, dry_run=True, yes=False)
    assert result.skipped is True
    assert result.skip_reason == "dry_run"
    assert "fly" in result.cmd_run[0]


def test_railway_skip_without_link(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent_scaffold.deploy.railway.cli_present", lambda _: True)
    result = railway.deploy(tmp_path, dry_run=True, yes=False)
    assert result.skipped is True
    assert result.skip_reason == "not_linked"


def test_railway_dry_run_with_link(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent_scaffold.deploy.railway.cli_present", lambda _: True)
    (tmp_path / ".railway").mkdir()
    (tmp_path / ".railway" / "config.json").write_text("{}")
    result = railway.deploy(tmp_path, dry_run=True, yes=False)
    assert result.skipped is True
    assert result.cmd_run == ["railway", "up"]


def test_deploy_result_shape_stable() -> None:
    # Sanity: DeployResult fields are what the CLI renderer expects.
    r = DeployResult(target="x", cmd_run=["echo", "hi"], summary="ok")
    assert r.target == "x"
    assert r.cmd_run == ["echo", "hi"]
    assert r.exit_code is None
    assert r.skipped is False


def test_confirm_returns_false_in_non_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from agent_scaffold.deploy._common import confirm

    class FakeStdin:
        @staticmethod
        def isatty() -> bool:
            return False

    monkeypatch.setattr(sys, "stdin", FakeStdin)
    assert confirm("prompt?") is False


def test_confirm_requires_literal_yes(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins
    import sys

    from agent_scaffold.deploy._common import confirm

    class FakeStdin:
        @staticmethod
        def isatty() -> bool:
            return True

    monkeypatch.setattr(sys, "stdin", FakeStdin)

    def fake_input(prompt: str) -> str:
        return "y"  # NOT "yes"

    monkeypatch.setattr(builtins, "input", fake_input)
    assert confirm("prompt?") is False

    def yes_input(prompt: str) -> str:
        return "YES"

    monkeypatch.setattr(builtins, "input", yes_input)
    assert confirm("prompt?") is True
