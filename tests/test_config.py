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
    ENV_MAX_TOKENS,
    ENV_MODEL,
    ConfigError,
    load_config,
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
    assert cfg.anthropic_api_key == "test-key-123"
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


def test_missing_api_key_raises(tmp_path: Path) -> None:
    env = {ENV_DEPLOYMENTS_PATH: str(tmp_path)}
    with pytest.raises(ConfigError, match=ENV_API_KEY):
        load_config(env)


def test_missing_deployments_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate empty bundled deployments so the fallback still raises
    import agent_scaffold.config

    empty_dir = tmp_path / "empty_bundle"
    empty_dir.mkdir()
    monkeypatch.setattr(agent_scaffold.config, "bundled_docs_path", lambda: empty_dir)
    env = {ENV_API_KEY: "k"}
    with pytest.raises(ConfigError, match="deployments_path"):
        load_config(env)


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
