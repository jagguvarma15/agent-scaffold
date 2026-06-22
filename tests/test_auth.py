"""Tests for ``agent_scaffold.auth`` and the ``auth`` Typer command."""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

import keyring
import keyring.errors
import pytest
from pydantic import SecretStr
from typer.testing import CliRunner

from agent_scaffold import auth as auth_mod
from agent_scaffold.auth import (
    AuthError,
    credentials_path,
    delete_key,
    detect_backend,
    list_credentials,
    load_key,
    mask,
    resolve_active,
    store_key,
    validate_anthropic_key,
    write_credentials_file,
)
from agent_scaffold.cli import app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ---------------------------------------------------------------------------
# Per-test in-memory keyring + isolated XDG config dir
# ---------------------------------------------------------------------------


class _FakeKeyring:
    """Drop-in for keyring.{get,set,delete}_password.

    Each instance gets its own one-off subclass so ``__class__.__name__``
    can be overridden per-instance without bleeding across tests.
    """

    def __init__(self, class_name: str = "Keyring") -> None:
        self._store: dict[tuple[str, str], str] = {}
        self.rename(class_name)

    def rename(self, class_name: str) -> None:
        """Swap to a fresh subclass with ``__class__.__name__ == class_name``."""
        new_cls = type(class_name, (_FakeKeyring,), {})
        object.__setattr__(self, "__class__", new_cls)

    # Match the API the real keyring module exposes at the package level.
    def get_password(self, service: str, name: str) -> str | None:
        return self._store.get((service, name))

    def set_password(self, service: str, name: str, value: str) -> None:
        self._store[(service, name)] = value

    def delete_password(self, service: str, name: str) -> None:
        if (service, name) not in self._store:
            raise keyring.errors.PasswordDeleteError(f"no entry for {name}")
        del self._store[(service, name)]


@pytest.fixture
def fake_keyring(monkeypatch: pytest.MonkeyPatch) -> _FakeKeyring:
    # `auth` no longer imports keyring at module scope — it resolves the real
    # module lazily via ``auth._keyring()``. Patch the real module's functions
    # (which that helper returns) so every auth code path sees the fake store.
    fk = _FakeKeyring()
    monkeypatch.setattr(keyring, "get_keyring", lambda: fk)
    monkeypatch.setattr(keyring, "get_password", fk.get_password)
    monkeypatch.setattr(keyring, "set_password", fk.set_password)
    monkeypatch.setattr(keyring, "delete_password", fk.delete_password)
    return fk


@pytest.fixture
def isolated_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Route the credentials file under tmp_path/config and clear ANTHROPIC_API_KEY."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return tmp_path / "config" / "agent-scaffold" / "credentials"


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------


def test_detect_backend_accepts_macos_keyring(
    monkeypatch: pytest.MonkeyPatch, fake_keyring: _FakeKeyring
) -> None:
    fake_keyring.rename("Keyring")
    assert detect_backend() == "keyring"


def test_detect_backend_accepts_winvault(
    monkeypatch: pytest.MonkeyPatch, fake_keyring: _FakeKeyring
) -> None:
    fake_keyring.rename("WinVaultKeyring")
    assert detect_backend() == "keyring"


def test_detect_backend_accepts_secret_service(
    monkeypatch: pytest.MonkeyPatch, fake_keyring: _FakeKeyring
) -> None:
    fake_keyring.rename("SecretServiceKeyring")
    assert detect_backend() == "keyring"


def test_detect_backend_refuses_plaintext(
    monkeypatch: pytest.MonkeyPatch, fake_keyring: _FakeKeyring
) -> None:
    fake_keyring.rename("PlaintextKeyring")
    with pytest.raises(AuthError) as exc:
        detect_backend()
    assert "Refusing" in str(exc.value)


def test_detect_backend_refuses_null_backend(
    monkeypatch: pytest.MonkeyPatch, fake_keyring: _FakeKeyring
) -> None:
    fake_keyring.rename("Null")
    with pytest.raises(AuthError):
        detect_backend()


def test_detect_backend_chainer_with_native_inner(
    monkeypatch: pytest.MonkeyPatch, fake_keyring: _FakeKeyring
) -> None:
    inner = _FakeKeyring("Keyring")
    chainer = _FakeKeyring("ChainerBackend")
    chainer.backends = [inner]  # type: ignore[attr-defined]
    monkeypatch.setattr(keyring, "get_keyring", lambda: chainer)
    assert detect_backend() == "keyring"


def test_detect_backend_chainer_without_native_inner(
    monkeypatch: pytest.MonkeyPatch, fake_keyring: _FakeKeyring
) -> None:
    inner = _FakeKeyring("PlaintextKeyring")
    chainer = _FakeKeyring("ChainerBackend")
    chainer.backends = [inner]  # type: ignore[attr-defined]
    monkeypatch.setattr(keyring, "get_keyring", lambda: chainer)
    with pytest.raises(AuthError):
        detect_backend()


# ---------------------------------------------------------------------------
# Absent keyring (default install, no `keyring` extra) degrades gracefully
# ---------------------------------------------------------------------------


def test_keyring_helper_returns_none_when_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The lazy accessor degrades to None instead of raising when the extra is
    not installed (`sys.modules['keyring'] = None` makes the import fail)."""
    import sys

    monkeypatch.setitem(sys.modules, "keyring", None)
    assert auth_mod._keyring() is None


def test_load_key_degrades_to_file_when_keyring_absent(
    monkeypatch: pytest.MonkeyPatch, isolated_config: Path
) -> None:
    """No keyring + no env: resolution skips keyring (no prompt) and reads the
    file backend — returning None when empty, the stored value when present."""
    monkeypatch.setattr(auth_mod, "_keyring", lambda: None)
    assert load_key() is None  # nothing anywhere → None, never raises

    write_credentials_file(auth_mod.DEFAULT_KEY_NAME, SecretStr("sk-ant-file-key"))
    resolved = load_key()
    assert resolved is not None
    assert resolved.get_secret_value() == "sk-ant-file-key"


def test_resolve_active_reports_file_when_keyring_absent(
    monkeypatch: pytest.MonkeyPatch, isolated_config: Path
) -> None:
    monkeypatch.setattr(auth_mod, "_keyring", lambda: None)
    write_credentials_file(auth_mod.DEFAULT_KEY_NAME, SecretStr("sk-ant-file-key"))
    resolved = resolve_active()
    assert resolved is not None
    _, backend = resolved
    assert backend == "file"


def test_detect_backend_raises_when_keyring_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_mod, "_keyring", lambda: None)
    # The message must give the absent-specific guidance (install the extra /
    # --use-file), not the generic plaintext-refusal text.
    with pytest.raises(AuthError, match="not installed"):
        detect_backend()


def test_auth_module_imports_without_keyring_installed() -> None:
    """The headline guarantee: ``import agent_scaffold.auth`` must succeed on a
    machine with no keyring. Run in a subprocess because keyring is already
    imported in-process by this test module + conftest, so the only faithful
    check of the fresh-import path is a clean interpreter with keyring blocked.
    """
    import subprocess
    import sys

    code = "import sys; sys.modules['keyring'] = None; import agent_scaffold.auth"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_describe_backend_when_keyring_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_mod, "_keyring", lambda: None)
    assert "not installed" in auth_mod.describe_backend()


def test_store_project_secret_falls_back_to_file_when_keyring_absent(
    monkeypatch: pytest.MonkeyPatch, isolated_config: Path
) -> None:
    """Without keyring the vault still works — values land in the 0600 file."""
    monkeypatch.setattr(auth_mod, "_keyring", lambda: None)
    cred = auth_mod.store_project_secret("ns", "REDIS_URL", SecretStr("redis://x"))
    assert cred.backend == "file"
    loaded = auth_mod.load_project_secret("ns", "REDIS_URL")
    assert loaded is not None
    assert loaded.get_secret_value() == "redis://x"


# ---------------------------------------------------------------------------
# mask()
# ---------------------------------------------------------------------------


def test_mask_short_keys() -> None:
    assert mask("") == ""
    assert mask("abc") == "***abc"


def test_mask_anthropic_key_shows_tail() -> None:
    assert mask("sk-ant-api03-abc-def-4j2k") == "sk-ant-...4j2k"


def test_mask_other_keys() -> None:
    assert mask("abcdefghijklmnop") == "abc...mnop"


# ---------------------------------------------------------------------------
# Credentials file: chmod 0600, round-trip, XDG honoring, delete
# ---------------------------------------------------------------------------


def test_write_credentials_file_creates_mode_0600(isolated_config: Path) -> None:
    write_credentials_file("anthropic", SecretStr("sk-ant-aaaaaaaa1234"))
    assert isolated_config.is_file()
    mode = stat.S_IMODE(isolated_config.stat().st_mode)
    assert mode == 0o600


def test_credentials_file_round_trip(isolated_config: Path) -> None:
    write_credentials_file("anthropic", SecretStr("sk-ant-aaaaaaaa1234"))
    write_credentials_file("ci-prod", SecretStr("sk-ant-bbbbbbbb5678"))
    creds = sorted(auth_mod._list_credentials_file(), key=lambda c: c.name)
    assert [c.name for c in creds] == ["anthropic", "ci-prod"]
    assert all(c.backend == "file" for c in creds)
    assert auth_mod._load_from_credentials_file("anthropic") is not None


def test_credentials_path_honors_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert credentials_path() == tmp_path / "xdg" / "agent-scaffold" / "credentials"


def test_credentials_path_default_when_xdg_unset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))  # type: ignore[arg-type]
    assert credentials_path() == tmp_path / ".config" / "agent-scaffold" / "credentials"


def test_delete_from_credentials_file_removes_section(isolated_config: Path) -> None:
    write_credentials_file("anthropic", SecretStr("sk-ant-aaaaaaaa1234"))
    assert auth_mod._delete_from_credentials_file("anthropic") is True
    assert auth_mod._delete_from_credentials_file("anthropic") is False


# ---------------------------------------------------------------------------
# Store / load / delete / list across backends
# ---------------------------------------------------------------------------


def test_store_key_keyring_writes_and_loads(
    fake_keyring: _FakeKeyring, isolated_config: Path
) -> None:
    descriptor = store_key("anthropic", SecretStr("sk-ant-zzzzzzzz9999"))
    assert descriptor.backend == "keyring"
    loaded = load_key("anthropic")
    assert loaded is not None
    assert loaded.get_secret_value() == "sk-ant-zzzzzzzz9999"


def test_store_key_file_writes_to_credentials_file(
    fake_keyring: _FakeKeyring, isolated_config: Path
) -> None:
    descriptor = store_key("ci-prod", SecretStr("sk-ant-yyyyyyyy8888"), backend="file")
    assert descriptor.backend == "file"
    assert isolated_config.is_file()
    loaded = load_key("ci-prod")
    assert loaded is not None and loaded.get_secret_value() == "sk-ant-yyyyyyyy8888"


def test_store_key_env_does_not_persist(fake_keyring: _FakeKeyring, isolated_config: Path) -> None:
    descriptor = store_key("anthropic", SecretStr("sk-ant-xxxxxxxx7777"), backend="env")
    assert descriptor.backend == "env"
    # Not in keyring, not in file.
    assert load_key("anthropic") is None


def test_store_key_keyring_refuses_plaintext_backend(
    monkeypatch: pytest.MonkeyPatch, fake_keyring: _FakeKeyring, isolated_config: Path
) -> None:
    fake_keyring.rename("PlaintextKeyring")
    with pytest.raises(AuthError):
        store_key("anthropic", SecretStr("sk-ant-aaaaa1234567"))


def test_load_key_resolution_order_env_wins(
    monkeypatch: pytest.MonkeyPatch, fake_keyring: _FakeKeyring, isolated_config: Path
) -> None:
    store_key("anthropic", SecretStr("sk-ant-from-keyring00"))
    write_credentials_file("anthropic", SecretStr("sk-ant-from-file000000"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env0000000")
    loaded = load_key("anthropic")
    assert loaded is not None
    assert loaded.get_secret_value() == "sk-ant-from-env0000000"


def test_load_key_resolution_order_keyring_beats_file(
    fake_keyring: _FakeKeyring, isolated_config: Path
) -> None:
    store_key("anthropic", SecretStr("sk-ant-from-keyring00"))
    write_credentials_file("anthropic", SecretStr("sk-ant-from-file000000"))
    loaded = load_key("anthropic")
    assert loaded is not None
    assert loaded.get_secret_value() == "sk-ant-from-keyring00"


def test_load_key_falls_through_to_file(fake_keyring: _FakeKeyring, isolated_config: Path) -> None:
    write_credentials_file("anthropic", SecretStr("sk-ant-from-file000000"))
    loaded = load_key("anthropic")
    assert loaded is not None
    assert loaded.get_secret_value() == "sk-ant-from-file000000"


def test_load_key_none_when_nothing_set(fake_keyring: _FakeKeyring, isolated_config: Path) -> None:
    assert load_key("anthropic") is None


def test_resolve_active_reports_backend(fake_keyring: _FakeKeyring, isolated_config: Path) -> None:
    store_key("anthropic", SecretStr("sk-ant-from-keyring00"))
    active = resolve_active()
    assert active is not None
    _, backend = active
    assert backend == "keyring"


def test_delete_key_removes_from_both_backends(
    fake_keyring: _FakeKeyring, isolated_config: Path
) -> None:
    store_key("anthropic", SecretStr("sk-ant-aaaaa1234567"))
    write_credentials_file("anthropic", SecretStr("sk-ant-aaaaa1234567"))
    assert delete_key("anthropic") is True
    # Idempotent: second call is a no-op.
    assert delete_key("anthropic") is False


def test_list_credentials_combines_backends(
    fake_keyring: _FakeKeyring, isolated_config: Path
) -> None:
    store_key("anthropic", SecretStr("sk-ant-from-keyring00"))
    write_credentials_file("ci-prod", SecretStr("sk-ant-from-file000000"))
    creds = list_credentials()
    by_name = {c.name: c.backend for c in creds}
    assert by_name == {"anthropic": "keyring", "ci-prod": "file"}


# ---------------------------------------------------------------------------
# validate_anthropic_key
# ---------------------------------------------------------------------------


def test_validate_anthropic_key_rejects_bad_prefix() -> None:
    ok, msg = validate_anthropic_key(SecretStr("not-a-real-key"))
    assert ok is False
    assert "sk-ant-" in msg


def test_validate_anthropic_key_success(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeModels:
        @staticmethod
        def list(limit: int = 1) -> Any:
            return type("Page", (), {"data": [object(), object()]})

    class _FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            self.models = _FakeModels()

    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", _FakeClient)
    ok, msg = validate_anthropic_key(SecretStr("sk-ant-aaaa1234567890"))
    assert ok is True
    assert "validated" in msg


def test_validate_anthropic_key_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import anthropic
    import httpx

    response = httpx.Response(
        status_code=401, request=httpx.Request("GET", "https://api.anthropic.com/v1/models")
    )

    class _FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            self.models = self

        def list(self, limit: int = 1) -> Any:
            raise anthropic.AuthenticationError(
                message="bad key", response=response, body={"error": {"message": "bad key"}}
            )

    monkeypatch.setattr(anthropic, "Anthropic", _FakeClient)
    ok, msg = validate_anthropic_key(SecretStr("sk-ant-aaaa1234567890"))
    assert ok is False
    assert "401" in msg


def test_validate_anthropic_key_other_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    import anthropic

    class _FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            self.models = self

        def list(self, limit: int = 1) -> Any:
            raise RuntimeError("network down")

    monkeypatch.setattr(anthropic, "Anthropic", _FakeClient)
    ok, msg = validate_anthropic_key(SecretStr("sk-ant-aaaa1234567890"))
    assert ok is False
    assert "RuntimeError" in msg


# ---------------------------------------------------------------------------
# CLI: agent-scaffold auth login / status / logout / setup-token
# ---------------------------------------------------------------------------


def test_cli_auth_login_no_browser_uses_paste(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    fake_keyring: _FakeKeyring,
    isolated_config: Path,
) -> None:
    monkeypatch.setattr(
        "agent_scaffold.cli_auth._prompt_paste", lambda prompt="x": "sk-ant-pasted-via-getpass"
    )
    monkeypatch.setattr(
        "agent_scaffold.cli_auth.validate_anthropic_key",
        lambda key: (True, "validated (mocked)"),
    )
    res = runner.invoke(app, ["auth", "login", "--no-browser"])
    assert res.exit_code == 0, res.output
    assert "Stored" in res.output
    assert load_key("anthropic") is not None


def test_cli_auth_login_rejects_failed_probe(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    fake_keyring: _FakeKeyring,
    isolated_config: Path,
) -> None:
    monkeypatch.setattr(
        "agent_scaffold.cli_auth._prompt_paste", lambda prompt="x": "sk-ant-rotten-key0000"
    )
    monkeypatch.setattr(
        "agent_scaffold.cli_auth.validate_anthropic_key",
        lambda key: (False, "key rejected by Anthropic API (401)"),
    )
    res = runner.invoke(app, ["auth", "login", "--no-browser"])
    assert res.exit_code == 1
    assert "401" in res.output
    assert load_key("anthropic") is None


def test_cli_auth_login_skips_validation_when_flagged(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    fake_keyring: _FakeKeyring,
    isolated_config: Path,
) -> None:
    monkeypatch.setattr(
        "agent_scaffold.cli_auth._prompt_paste", lambda prompt="x": "sk-ant-skipped-probe1"
    )

    def fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("validate_anthropic_key should not be called")

    monkeypatch.setattr("agent_scaffold.cli_auth.validate_anthropic_key", fail)
    res = runner.invoke(app, ["auth", "login", "--no-browser", "--no-validate"])
    assert res.exit_code == 0, res.output


def test_cli_auth_login_mutually_exclusive_flags(runner: CliRunner) -> None:
    res = runner.invoke(app, ["auth", "login", "--use-keyring", "--use-file", "--no-browser"])
    assert res.exit_code != 0
    # Rich wraps the error across box lines, so check the two words separately.
    assert "mutually" in res.output and "exclusive" in res.output


def test_cli_auth_login_use_env_prints_export_line(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    fake_keyring: _FakeKeyring,
    isolated_config: Path,
) -> None:
    monkeypatch.setattr(
        "agent_scaffold.cli_auth._prompt_paste", lambda prompt="x": "sk-ant-printed-only0"
    )
    monkeypatch.setattr(
        "agent_scaffold.cli_auth.validate_anthropic_key", lambda key: (True, "validated (mocked)")
    )
    res = runner.invoke(app, ["auth", "login", "--no-browser", "--use-env"])
    assert res.exit_code == 0
    assert "export ANTHROPIC_API_KEY=" in res.output
    # Did not persist.
    assert load_key("anthropic") is None


def test_cli_auth_login_falls_back_to_file_when_no_native_backend(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    fake_keyring: _FakeKeyring,
    isolated_config: Path,
) -> None:
    fake_keyring.rename("PlaintextKeyring")
    monkeypatch.setattr(
        "agent_scaffold.cli_auth._prompt_paste", lambda prompt="x": "sk-ant-fell-back-file"
    )
    monkeypatch.setattr(
        "agent_scaffold.cli_auth.validate_anthropic_key", lambda key: (True, "validated (mocked)")
    )
    res = runner.invoke(app, ["auth", "login", "--no-browser"])
    assert res.exit_code == 0, res.output
    assert "falling back to mode-0600" in res.output
    assert isolated_config.is_file()


def test_cli_auth_status_text_output(
    runner: CliRunner, fake_keyring: _FakeKeyring, isolated_config: Path
) -> None:
    store_key("anthropic", SecretStr("sk-ant-status-test1234"))
    res = runner.invoke(app, ["auth", "status"])
    assert res.exit_code == 0, res.output
    assert "Backend:" in res.output
    assert "Resolution order:" in res.output
    assert "Currently resolved:" in res.output


def test_cli_auth_status_json(
    runner: CliRunner, fake_keyring: _FakeKeyring, isolated_config: Path
) -> None:
    store_key("anthropic", SecretStr("sk-ant-status-test1234"))
    res = runner.invoke(app, ["auth", "status", "--json"])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["schema_version"] == 1
    assert payload["backend_ok"] is True
    assert payload["active"]["backend"] == "keyring"
    assert payload["resolution_order"] == [
        "env (ANTHROPIC_API_KEY)",
        "keyring",
        "file",
    ]


def test_cli_auth_status_no_creds(
    runner: CliRunner, fake_keyring: _FakeKeyring, isolated_config: Path
) -> None:
    res = runner.invoke(app, ["auth", "status"])
    assert res.exit_code == 0
    assert "No stored credentials" in res.output
    assert "No key resolved" in res.output


def test_cli_auth_logout_named(
    runner: CliRunner, fake_keyring: _FakeKeyring, isolated_config: Path
) -> None:
    store_key("anthropic", SecretStr("sk-ant-to-logout12345"))
    res = runner.invoke(app, ["auth", "logout"])
    assert res.exit_code == 0
    assert "Removed" in res.output
    assert load_key("anthropic") is None


def test_cli_auth_logout_no_match_exits_one(
    runner: CliRunner, fake_keyring: _FakeKeyring, isolated_config: Path
) -> None:
    res = runner.invoke(app, ["auth", "logout", "--name", "nonexistent"])
    assert res.exit_code == 1


def test_cli_auth_logout_all_clears_everything(
    runner: CliRunner, fake_keyring: _FakeKeyring, isolated_config: Path
) -> None:
    store_key("anthropic", SecretStr("sk-ant-keyringed12345"))
    write_credentials_file("ci-prod", SecretStr("sk-ant-fileonly1234567"))
    res = runner.invoke(app, ["auth", "logout", "--all"])
    assert res.exit_code == 0
    assert load_key("anthropic") is None
    assert load_key("ci-prod") is None


def test_cli_auth_setup_token_stdin(
    runner: CliRunner, fake_keyring: _FakeKeyring, isolated_config: Path
) -> None:
    res = runner.invoke(
        app, ["auth", "setup-token", "ci-prod", "--stdin"], input="sk-ant-ci-token12345\n"
    )
    assert res.exit_code == 0, res.output
    loaded = load_key("ci-prod")
    assert loaded is not None
    assert loaded.get_secret_value() == "sk-ant-ci-token12345"


def test_cli_auth_setup_token_empty_input_exits_one(
    runner: CliRunner, fake_keyring: _FakeKeyring, isolated_config: Path
) -> None:
    res = runner.invoke(app, ["auth", "setup-token", "ci-prod", "--stdin"], input="\n")
    assert res.exit_code == 1


# ---------------------------------------------------------------------------
# config.load_config fallback to auth.load_key
# ---------------------------------------------------------------------------


def test_load_config_falls_back_to_keyring(
    monkeypatch: pytest.MonkeyPatch, fake_keyring: _FakeKeyring, isolated_config: Path
) -> None:
    from agent_scaffold import config as config_mod

    store_key("anthropic", SecretStr("sk-ant-from-keyring00"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("AGENT_SCAFFOLD_DEPLOYMENTS_PATH", str(isolated_config.parent))
    cfg = config_mod.load_config()
    assert cfg.anthropic_api_key == "sk-ant-from-keyring00"


def test_load_config_friendly_error_mentions_auth_login(
    monkeypatch: pytest.MonkeyPatch, fake_keyring: _FakeKeyring, isolated_config: Path
) -> None:
    from agent_scaffold import config as config_mod

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("AGENT_SCAFFOLD_DEPLOYMENTS_PATH", str(isolated_config.parent))
    with pytest.raises(config_mod.ConfigError) as exc:
        config_mod.load_config()
    assert "auth login" in str(exc.value)
