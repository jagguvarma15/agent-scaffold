"""Tests for ``agent-scaffold secrets purge`` + ``secrets list``."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent_scaffold.auth import StoredCredential, mask
from agent_scaffold.cli import app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def stub_list_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[list[StoredCredential]], None]:
    """Replace ``auth.list_credentials`` with a fake controllable per-test."""

    def install(creds: list[StoredCredential]) -> None:
        from agent_scaffold import cli as cli_mod

        monkeypatch.setattr(cli_mod, "list_credentials", lambda: list(creds))

    return install


@pytest.fixture
def stub_delete_key(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    """Record which names ``delete_key`` was called with; return True for each."""
    deleted: dict[str, list[str]] = {"names": []}

    def fake(name: str) -> bool:
        deleted["names"].append(name)
        return True

    from agent_scaffold import cli as cli_mod

    monkeypatch.setattr(cli_mod, "delete_key", fake)
    return deleted


def _cred(name: str, backend: str) -> StoredCredential:
    return StoredCredential(
        name=name, backend=backend, masked_value=mask("sk-ant-aaaaaaaaaaaaaaaa")
    )


def test_secrets_list_empty(
    runner: CliRunner,
    stub_list_credentials: Callable[[list[StoredCredential]], None],
) -> None:
    stub_list_credentials([])
    result = runner.invoke(app, ["secrets", "list"])
    assert result.exit_code == 0
    assert "No stored credentials" in result.output


def test_secrets_list_shows_credentials(
    runner: CliRunner,
    stub_list_credentials: Callable[[list[StoredCredential]], None],
) -> None:
    stub_list_credentials([_cred("anthropic", "keyring"), _cred("ci-prod", "file")])
    result = runner.invoke(app, ["secrets", "list"])
    assert result.exit_code == 0
    assert "anthropic" in result.output
    assert "ci-prod" in result.output
    assert "(keyring)" in result.output
    assert "(file)" in result.output


def test_secrets_list_json(
    runner: CliRunner,
    stub_list_credentials: Callable[[list[StoredCredential]], None],
) -> None:
    stub_list_credentials([_cred("anthropic", "keyring")])
    result = runner.invoke(app, ["secrets", "list", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema_version"] == 1
    assert payload["credentials"][0]["name"] == "anthropic"
    assert payload["credentials"][0]["backend"] == "keyring"


def test_purge_empty_inventory_exits_zero_without_action(
    runner: CliRunner,
    stub_list_credentials: Callable[[list[StoredCredential]], None],
    stub_delete_key: dict[str, list[str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_list_credentials([])
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["secrets", "purge", "--yes"])
    assert result.exit_code == 0
    assert "Nothing to purge" in result.output
    assert stub_delete_key["names"] == []


def test_purge_yes_skips_confirmation_and_deletes(
    runner: CliRunner,
    stub_list_credentials: Callable[[list[StoredCredential]], None],
    stub_delete_key: dict[str, list[str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_list_credentials([_cred("anthropic", "keyring"), _cred("ci-prod", "file")])
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["secrets", "purge", "--yes"])
    assert result.exit_code == 0
    assert "anthropic" in stub_delete_key["names"]
    assert "ci-prod" in stub_delete_key["names"]
    assert "Removed" in result.output


def test_purge_without_yes_prompts_and_respects_decline(
    runner: CliRunner,
    stub_list_credentials: Callable[[list[StoredCredential]], None],
    stub_delete_key: dict[str, list[str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_list_credentials([_cred("anthropic", "keyring")])
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["secrets", "purge"], input="n\n")
    assert result.exit_code == 0
    assert "Aborted" in result.output
    assert stub_delete_key["names"] == []


def test_purge_removes_env_local_in_cwd(
    runner: CliRunner,
    stub_list_credentials: Callable[[list[StoredCredential]], None],
    stub_delete_key: dict[str, list[str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_list_credentials([])
    env_local = tmp_path / ".env.local"
    env_local.write_text("REDIS_URL=x\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["secrets", "purge", "--yes"])
    assert result.exit_code == 0
    assert not env_local.exists(), ".env.local should be removed"


def test_purge_keep_env_local_leaves_file_alone(
    runner: CliRunner,
    stub_list_credentials: Callable[[list[StoredCredential]], None],
    stub_delete_key: dict[str, list[str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_list_credentials([_cred("anthropic", "keyring")])
    env_local = tmp_path / ".env.local"
    env_local.write_text("REDIS_URL=x\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["secrets", "purge", "--yes", "--keep-env-local"])
    assert result.exit_code == 0
    assert env_local.exists(), "--keep-env-local must not delete the file"
    assert "anthropic" in stub_delete_key["names"]
