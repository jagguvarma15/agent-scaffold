"""Configuration loading for agent-forge.

Resolves ``Config`` from environment variables with a TOML config file at
``~/.config/agent-forge/config.toml`` as a fallback for ``deployments_path``
and ``model``.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

DEFAULT_MODEL = "claude-opus-4-5"
DEFAULT_MAX_TOKENS = 16000

ENV_API_KEY = "ANTHROPIC_API_KEY"
ENV_MODEL = "AGENT_FORGE_MODEL"
ENV_DEPLOYMENTS_PATH = "AGENT_FORGE_DEPLOYMENTS_PATH"
ENV_CACHE_DIR = "AGENT_FORGE_CACHE_DIR"
ENV_CONFIG_PATH = "AGENT_FORGE_CONFIG_PATH"

DEFAULT_CONFIG_RELATIVE = Path(".config/agent-forge/config.toml")
DEFAULT_CACHE_RELATIVE = Path(".cache/agent-forge")


class ConfigError(Exception):
    """Raised when configuration cannot be resolved."""


class Config(BaseModel):
    """Resolved runtime configuration."""

    deployments_path: Path
    anthropic_api_key: str
    model: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    cache_dir: Path
    failures_dir: Path = Field(
        description="Directory where raw LLM responses are written when contract parsing fails."
    )


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Failed to parse config file at {path}: {exc}") from exc


def _home() -> Path:
    return Path.home()


def load_config(env: dict[str, str] | None = None) -> Config:
    """Resolve a :class:`Config` from environment and optional TOML fallback.

    Precedence: explicit env > TOML config > built-in defaults.
    """
    src = os.environ if env is None else env

    api_key = src.get(ENV_API_KEY, "").strip()

    config_path_str = src.get(ENV_CONFIG_PATH)
    config_path = (
        Path(config_path_str).expanduser()
        if config_path_str
        else _home() / DEFAULT_CONFIG_RELATIVE
    )
    toml_data = _read_toml(config_path)

    deployments_raw = src.get(ENV_DEPLOYMENTS_PATH) or toml_data.get("deployments_path")
    model = src.get(ENV_MODEL) or toml_data.get("model") or DEFAULT_MODEL

    cache_dir_raw = src.get(ENV_CACHE_DIR) or toml_data.get("cache_dir")
    cache_dir = (
        Path(cache_dir_raw).expanduser() if cache_dir_raw else _home() / DEFAULT_CACHE_RELATIVE
    )
    failures_dir = cache_dir / "failures"

    if not api_key:
        raise ConfigError(
            f"Missing {ENV_API_KEY}. Set it in your environment or in a .env file. "
            "See .env.example for the expected format."
        )

    if not deployments_raw:
        raise ConfigError(
            f"Missing deployments_path. Set {ENV_DEPLOYMENTS_PATH} or add "
            f"deployments_path = \"...\" to {config_path}."
        )

    deployments_path = Path(deployments_raw).expanduser()

    return Config(
        deployments_path=deployments_path,
        anthropic_api_key=api_key,
        model=str(model),
        cache_dir=cache_dir,
        failures_dir=failures_dir,
    )
