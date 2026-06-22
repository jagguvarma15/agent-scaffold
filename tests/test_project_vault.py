"""Tests for the project-scoped encrypted secrets vault (auth.py + envfile.py)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr

from agent_scaffold.auth import (
    credentials_path,
    delete_project_secret,
    delete_project_secrets,
    list_project_namespaces,
    list_project_secret_names,
    load_project_secret,
    load_project_secrets,
    project_namespace,
    project_secret_name,
    store_project_secret,
)
from agent_scaffold.envfile import build_runtime_env

_NS = "demo-agent-1a2b3c4d"


def test_project_namespace_is_stable_and_path_scoped(tmp_path: Path) -> None:
    a = project_namespace("demo", tmp_path / "a")
    a_again = project_namespace("demo", tmp_path / "a")
    b = project_namespace("demo", tmp_path / "b")
    assert a == a_again
    assert a != b
    assert a.startswith("demo-")
    assert project_secret_name(a, "REDIS_URL") == f"project:{a}:REDIS_URL"


def test_store_load_delete_round_trip() -> None:
    stored = store_project_secret(_NS, "REDIS_URL", SecretStr("redis://localhost:6379"))
    assert stored.backend == "keyring"  # conftest fake classifies as native
    assert stored.name == "REDIS_URL"

    value = load_project_secret(_NS, "REDIS_URL")
    assert value is not None
    assert value.get_secret_value() == "redis://localhost:6379"

    assert delete_project_secret(_NS, "REDIS_URL")
    assert load_project_secret(_NS, "REDIS_URL") is None
    assert list_project_secret_names(_NS) == {}


def test_index_holds_names_only_never_values() -> None:
    store_project_secret(_NS, "LANGFUSE_SECRET_KEY", SecretStr("sk-lf-supersecret-value"))
    names = list_project_secret_names(_NS)
    assert names == {"LANGFUSE_SECRET_KEY": "keyring"}
    # The credentials file carries the index (names + backend) but not the value.
    text = credentials_path().read_text(encoding="utf-8")
    assert "LANGFUSE_SECRET_KEY" in text
    assert "supersecret" not in text
    delete_project_secrets(_NS)


def test_index_preserves_env_var_case() -> None:
    store_project_secret(_NS, "QDRANT_URL", SecretStr("http://localhost:6333"))
    assert list(list_project_secret_names(_NS)) == ["QDRANT_URL"]
    delete_project_secrets(_NS)


def test_keyring_refusal_falls_back_to_file(monkeypatch: pytest.MonkeyPatch) -> None:
    import keyring

    class PlaintextKeyring:  # refused by detect_backend
        pass

    monkeypatch.setattr(keyring, "get_keyring", lambda: PlaintextKeyring())
    stored = store_project_secret(_NS, "REDIS_URL", SecretStr("redis://x"))
    assert stored.backend == "file"
    value = load_project_secret(_NS, "REDIS_URL")
    assert value is not None
    assert value.get_secret_value() == "redis://x"
    assert list_project_secret_names(_NS) == {"REDIS_URL": "file"}
    delete_project_secrets(_NS)


def test_namespaces_and_batch_operations() -> None:
    store_project_secret("proj-a-11111111", "A_TOKEN", SecretStr("a"))
    store_project_secret("proj-b-22222222", "B_TOKEN", SecretStr("b"))
    store_project_secret("proj-b-22222222", "C_TOKEN", SecretStr("c"))

    assert list_project_namespaces() == ["proj-a-11111111", "proj-b-22222222"]
    secrets = load_project_secrets("proj-b-22222222")
    assert {k: v.get_secret_value() for k, v in secrets.items()} == {"B_TOKEN": "b", "C_TOKEN": "c"}

    assert delete_project_secrets("proj-b-22222222") == 2
    assert list_project_namespaces() == ["proj-a-11111111"]
    delete_project_secrets("proj-a-11111111")


def test_build_runtime_env_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """shell env > vault > .env.local — and vault values reach the dict."""
    (tmp_path / ".env.local").write_text(
        "FROM_FILE=file-value\nSHADOWED_BY_VAULT=file-loses\nSHADOWED_BY_SHELL=file-loses\n",
        encoding="utf-8",
    )
    store_project_secret(_NS, "FROM_VAULT", SecretStr("vault-value"))
    store_project_secret(_NS, "SHADOWED_BY_VAULT", SecretStr("vault-wins"))
    store_project_secret(_NS, "SHADOWED_BY_SHELL", SecretStr("vault-loses"))
    monkeypatch.setenv("SHADOWED_BY_SHELL", "shell-wins")

    env = build_runtime_env(tmp_path, _NS)

    assert env["FROM_FILE"] == "file-value"
    assert env["FROM_VAULT"] == "vault-value"
    assert env["SHADOWED_BY_VAULT"] == "vault-wins"
    assert env["SHADOWED_BY_SHELL"] == "shell-wins"
    delete_project_secrets(_NS)


def test_build_runtime_env_without_namespace(tmp_path: Path) -> None:
    env = build_runtime_env(tmp_path, None)
    assert "PATH" in env  # inherits the shell


def test_build_runtime_env_injects_resolved_anthropic_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Anthropic key stored via `auth login` (keyring/file, not the shell
    env or .env.local) is injected so generated agents can build their client."""
    from agent_scaffold.auth import ENV_API_KEY, store_key

    monkeypatch.delenv(ENV_API_KEY, raising=False)
    store_key("anthropic", SecretStr("sk-ant-from-keyring-0001"))

    env = build_runtime_env(tmp_path, None)

    assert env[ENV_API_KEY] == "sk-ant-from-keyring-0001"


def test_build_runtime_env_shell_anthropic_key_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A shell-exported ANTHROPIC_API_KEY is never overwritten by the keyring fallback."""
    from agent_scaffold.auth import ENV_API_KEY, store_key

    monkeypatch.setenv(ENV_API_KEY, "sk-ant-from-shell-0002")
    store_key("anthropic", SecretStr("sk-ant-from-keyring-0001"))

    env = build_runtime_env(tmp_path, None)

    assert env[ENV_API_KEY] == "sk-ant-from-shell-0002"


def test_build_runtime_env_no_anthropic_key_stays_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no key anywhere, ANTHROPIC_API_KEY is left unset (not blank-injected)."""
    from agent_scaffold.auth import ENV_API_KEY

    monkeypatch.delenv(ENV_API_KEY, raising=False)
    env = build_runtime_env(tmp_path, None)
    assert ENV_API_KEY not in env
