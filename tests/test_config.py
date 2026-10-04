"""Tests for agent_scaffold.config."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_scaffold.config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    ENV_API_KEY,
    ENV_CONFIG_PATH,
    ENV_DEPLOYMENTS_PATH,
    ENV_FREE_PORTS,
    ENV_MAX_TOKENS,
    ENV_MODEL,
    ConfigError,
    MissingKeyError,
    load_config,
    resolve_free_ports,
)


def test_load_config_from_env(tmp_path: Path) -> None:
    deployments = tmp_path / "deployments"
    deployments.mkdir()
    env = {
        ENV_API_KEY: "test-key-123",
        ENV_DEPLOYMENTS_PATH: str(deployments),
        ENV_MODEL: "claude-test-1",
    }
    cfg = load_config(env)
    assert cfg.anthropic_api_key.get_secret_value() == "test-key-123"
    assert cfg.deployments_path == deployments
    assert cfg.model == "claude-test-1"
    assert cfg.failures_dir == cfg.cache_dir / "failures"


def test_load_config_defaults_model(tmp_path: Path) -> None:
    deployments = tmp_path / "deployments"
    deployments.mkdir()
    env = {ENV_API_KEY: "k", ENV_DEPLOYMENTS_PATH: str(deployments)}
    cfg = load_config(env)
    assert cfg.model == DEFAULT_MODEL


def test_load_config_toml_fallback(tmp_path: Path) -> None:
    deployments = tmp_path / "deployments"
    deployments.mkdir()
    toml = tmp_path / "config.toml"
    toml.write_text(f'deployments_path = "{deployments}"\nmodel = "from-toml"\n', encoding="utf-8")
    env = {ENV_API_KEY: "k", ENV_CONFIG_PATH: str(toml)}
    cfg = load_config(env)
    assert cfg.deployments_path == deployments
    assert cfg.model == "from-toml"


def test_env_overrides_toml(tmp_path: Path) -> None:
    deployments_a = tmp_path / "a"
    deployments_b = tmp_path / "b"
    deployments_a.mkdir()
    deployments_b.mkdir()
    toml = tmp_path / "config.toml"
    toml.write_text(
        f'deployments_path = "{deployments_a}"\nmodel = "from-toml"\n', encoding="utf-8"
    )
    env = {
        ENV_API_KEY: "k",
        ENV_CONFIG_PATH: str(toml),
        ENV_DEPLOYMENTS_PATH: str(deployments_b),
        ENV_MODEL: "from-env",
    }
    cfg = load_config(env)
    assert cfg.deployments_path == deployments_b
    assert cfg.model == "from-env"


def test_missing_api_key_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No key anywhere → ``MissingKeyError`` *specifically* (not just the base
    ``ConfigError``). ``cmd_scaffold`` catches the subclass to trigger
    first-launch onboarding, so the exact type is load-bearing — a regression
    to a bare ``ConfigError`` here would silently disable onboarding."""
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    env = {ENV_DEPLOYMENTS_PATH: str(tmp_path)}
    with pytest.raises(MissingKeyError, match=ENV_API_KEY):
        load_config(env)
    # Back-compat contract: it must still be catchable as a plain ConfigError.
    with pytest.raises(ConfigError):
        load_config(env)


def test_missing_deployments_is_optional() -> None:
    """load_config no longer requires a deployments path — resolution is deferred
    to sources.resolve_deployments which auto-fetches with a bundled fallback.
    """
    env = {ENV_API_KEY: "k"}
    cfg = load_config(env)
    assert cfg.deployments_path is None
    assert cfg.blueprints_path is None
    assert cfg.deployments_source == "auto"
    assert cfg.blueprints_source == "auto"


def test_max_tokens_default(tmp_path: Path) -> None:
    deployments = tmp_path / "deployments"
    deployments.mkdir()
    env = {ENV_API_KEY: "k", ENV_DEPLOYMENTS_PATH: str(deployments)}
    cfg = load_config(env)
    assert cfg.max_tokens == DEFAULT_MAX_TOKENS


def test_max_tokens_env_override(tmp_path: Path) -> None:
    deployments = tmp_path / "deployments"
    deployments.mkdir()
    env = {
        ENV_API_KEY: "k",
        ENV_DEPLOYMENTS_PATH: str(deployments),
        ENV_MAX_TOKENS: "48000",
    }
    cfg = load_config(env)
    assert cfg.max_tokens == 48000


def test_max_tokens_invalid_raises(tmp_path: Path) -> None:
    deployments = tmp_path / "deployments"
    deployments.mkdir()
    env = {
        ENV_API_KEY: "k",
        ENV_DEPLOYMENTS_PATH: str(deployments),
        ENV_MAX_TOKENS: "not-an-int",
    }
    with pytest.raises(ConfigError, match=ENV_MAX_TOKENS):
        load_config(env)


def test_invalid_toml_raises(tmp_path: Path) -> None:
    toml = tmp_path / "broken.toml"
    toml.write_text("this is = not = valid toml", encoding="utf-8")
    env = {ENV_API_KEY: "k", ENV_CONFIG_PATH: str(toml)}
    with pytest.raises(ConfigError, match="Failed to parse"):
        load_config(env)


def test_cache_ttl_defaults_to_5m(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("AGENT_SCAFFOLD_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("AGENT_SCAFFOLD_CACHE_TTL", raising=False)
    assert load_config().cache_ttl == "5m"


def test_cache_ttl_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("AGENT_SCAFFOLD_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_SCAFFOLD_CACHE_TTL", "1h")
    assert load_config().cache_ttl == "1h"


def test_cache_ttl_rejects_bad_value(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("AGENT_SCAFFOLD_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_SCAFFOLD_CACHE_TTL", "7d")
    with pytest.raises(ConfigError):
        load_config()


def test_load_config_legacy_contract_env(tmp_path: Path) -> None:
    deployments = tmp_path / "deployments"
    deployments.mkdir()
    base = {ENV_API_KEY: "k", ENV_DEPLOYMENTS_PATH: str(deployments)}
    assert load_config(base).legacy_contract is False
    on = load_config({**base, "AGENT_SCAFFOLD_LEGACY_CONTRACT": "1"})
    assert on.legacy_contract is True
    off = load_config({**base, "AGENT_SCAFFOLD_LEGACY_CONTRACT": "0"})
    assert off.legacy_contract is False


def test_repair_model_defaults_to_sonnet(tmp_path: Path) -> None:
    deployments = tmp_path / "deployments"
    deployments.mkdir()
    cfg = load_config({ENV_API_KEY: "k", ENV_DEPLOYMENTS_PATH: str(deployments)})
    assert cfg.repair_model == "claude-sonnet-5"


def test_repair_model_env_override(tmp_path: Path) -> None:
    deployments = tmp_path / "deployments"
    deployments.mkdir()
    cfg = load_config(
        {
            ENV_API_KEY: "k",
            ENV_DEPLOYMENTS_PATH: str(deployments),
            "AGENT_SCAFFOLD_REPAIR_MODEL": "claude-opus-4-8",
        }
    )
    assert cfg.repair_model == "claude-opus-4-8"


# ---- free_ports: a destructive opt-in, resolved without an API key ----------


def _toml(tmp_path: Path, body: str) -> dict[str, str]:
    config = tmp_path / "config.toml"
    config.write_text(body, encoding="utf-8")
    return {ENV_CONFIG_PATH: str(config)}


def test_free_ports_defaults_off(tmp_path: Path) -> None:
    assert resolve_free_ports({ENV_CONFIG_PATH: str(tmp_path / "missing.toml")}) is False
    assert load_config({ENV_API_KEY: "k"}).free_ports is False


def test_free_ports_from_toml_bool(tmp_path: Path) -> None:
    assert resolve_free_ports(_toml(tmp_path, "free_ports = true\n")) is True


def test_free_ports_from_toml_string(tmp_path: Path) -> None:
    assert resolve_free_ports(_toml(tmp_path, 'free_ports = "yes"\n')) is True


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on"])
def test_free_ports_env_truthy(tmp_path: Path, raw: str) -> None:
    env = {ENV_CONFIG_PATH: str(tmp_path / "missing.toml"), ENV_FREE_PORTS: raw}
    assert resolve_free_ports(env) is True


@pytest.mark.parametrize("raw", ["0", "false", "no", "off"])
def test_free_ports_env_falsy(tmp_path: Path, raw: str) -> None:
    env = {ENV_CONFIG_PATH: str(tmp_path / "missing.toml"), ENV_FREE_PORTS: raw}
    assert resolve_free_ports(env) is False


def test_free_ports_env_zero_beats_toml_true(tmp_path: Path) -> None:
    """A shell must always be able to switch the destructive default off."""
    env = _toml(tmp_path, "free_ports = true\n")
    env[ENV_FREE_PORTS] = "0"
    assert resolve_free_ports(env) is False


def test_free_ports_blank_env_falls_through_to_toml(tmp_path: Path) -> None:
    env = _toml(tmp_path, "free_ports = true\n")
    env[ENV_FREE_PORTS] = "  "
    assert resolve_free_ports(env) is True


def test_free_ports_invalid_value_raises_instead_of_arming(tmp_path: Path) -> None:
    env = {ENV_CONFIG_PATH: str(tmp_path / "missing.toml"), ENV_FREE_PORTS: "maybe"}
    with pytest.raises(ConfigError, match=ENV_FREE_PORTS):
        resolve_free_ports(env)


def test_free_ports_invalid_toml_value_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="free_ports"):
        resolve_free_ports(_toml(tmp_path, 'free_ports = "perhaps"\n'))


def test_resolve_free_ports_needs_no_api_key(tmp_path: Path) -> None:
    """``up`` never calls load_config (it raises MissingKeyError without a key)."""
    env = _toml(tmp_path, "free_ports = true\n")
    with pytest.raises(MissingKeyError):
        load_config({**env, "HOME": str(tmp_path)})
    assert resolve_free_ports(env) is True


def test_load_config_carries_free_ports(tmp_path: Path) -> None:
    env = {ENV_API_KEY: "k", ENV_FREE_PORTS: "1", ENV_CONFIG_PATH: str(tmp_path / "x.toml")}
    assert load_config(env).free_ports is True
