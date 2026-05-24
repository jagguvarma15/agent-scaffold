"""Configuration loading for agent-scaffold.

Resolves ``Config`` from environment variables with a TOML config file at
``~/.config/agent-scaffold/config.toml`` as a fallback for ``deployments_path``
and ``model``.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agent_scaffold._bundled_deployments import bundled_docs_path

DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_MAX_TOKENS = 32000
DEFAULT_MAX_CONTEXT_TOKENS = 60_000
DEFAULT_MAX_LINK_DEPTH = 2
DEFAULT_MAX_TOKENS_PER_DOC = 8_000

ENV_API_KEY = "ANTHROPIC_API_KEY"
ENV_MODEL = "AGENT_SCAFFOLD_MODEL"
ENV_MAX_TOKENS = "AGENT_SCAFFOLD_MAX_TOKENS"
ENV_THINKING_BUDGET = "AGENT_SCAFFOLD_THINKING_BUDGET"
ENV_EFFORT = "AGENT_SCAFFOLD_EFFORT"
ENV_DEPLOYMENTS_PATH = "AGENT_SCAFFOLD_DEPLOYMENTS_PATH"
ENV_CACHE_DIR = "AGENT_SCAFFOLD_CACHE_DIR"
ENV_CONFIG_PATH = "AGENT_SCAFFOLD_CONFIG_PATH"
ENV_MAX_CONTEXT_TOKENS = "AGENT_SCAFFOLD_MAX_CONTEXT_TOKENS"
ENV_MAX_LINK_DEPTH = "AGENT_SCAFFOLD_MAX_LINK_DEPTH"
ENV_MAX_TOKENS_PER_DOC = "AGENT_SCAFFOLD_MAX_TOKENS_PER_DOC"

DEFAULT_CONFIG_RELATIVE = Path(".config/agent-scaffold/config.toml")
DEFAULT_CACHE_RELATIVE = Path(".cache/agent-scaffold")


class ConfigError(Exception):
    """Raised when configuration cannot be resolved."""


class Config(BaseModel):
    """Resolved runtime configuration."""

    deployments_path: Path
    anthropic_api_key: str
    model: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    thinking_budget: int | None = None
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS
    max_link_depth: int = DEFAULT_MAX_LINK_DEPTH
    max_tokens_per_doc: int = DEFAULT_MAX_TOKENS_PER_DOC
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
        Path(config_path_str).expanduser() if config_path_str else _home() / DEFAULT_CONFIG_RELATIVE
    )
    toml_data = _read_toml(config_path)

    deployments_raw = src.get(ENV_DEPLOYMENTS_PATH) or toml_data.get("deployments_path")
    model = src.get(ENV_MODEL) or toml_data.get("model") or DEFAULT_MODEL

    max_tokens_raw = src.get(ENV_MAX_TOKENS) or toml_data.get("max_tokens")
    if max_tokens_raw is None:
        max_tokens = DEFAULT_MAX_TOKENS
    else:
        try:
            max_tokens = int(max_tokens_raw)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"Invalid {ENV_MAX_TOKENS}: {max_tokens_raw!r} (expected an integer)"
            ) from exc

    thinking_raw = src.get(ENV_THINKING_BUDGET) or toml_data.get("thinking_budget")
    if thinking_raw is None or thinking_raw == "":
        thinking_budget: int | None = None
    else:
        try:
            thinking_budget = int(thinking_raw)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"Invalid {ENV_THINKING_BUDGET}: {thinking_raw!r} (expected an integer)"
            ) from exc

    def _int_env(env_var: str, toml_key: str, default: int) -> int:
        raw = src.get(env_var) or toml_data.get(toml_key)
        if raw is None or raw == "":
            return default
        try:
            return int(raw)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"Invalid {env_var}: {raw!r} (expected an integer)"
            ) from exc

    max_context_tokens = _int_env(
        ENV_MAX_CONTEXT_TOKENS, "max_context_tokens", DEFAULT_MAX_CONTEXT_TOKENS
    )
    max_link_depth = _int_env(ENV_MAX_LINK_DEPTH, "max_link_depth", DEFAULT_MAX_LINK_DEPTH)
    max_tokens_per_doc = _int_env(
        ENV_MAX_TOKENS_PER_DOC, "max_tokens_per_doc", DEFAULT_MAX_TOKENS_PER_DOC
    )

    cache_dir_raw = src.get(ENV_CACHE_DIR) or toml_data.get("cache_dir")
    cache_dir = (
        Path(cache_dir_raw).expanduser() if cache_dir_raw else _home() / DEFAULT_CACHE_RELATIVE
    )
    failures_dir = cache_dir / "failures"

    if not api_key:
        raise ConfigError(
            f"Missing {ENV_API_KEY}. Set it in your environment or in a .env file.\n"
            "  export ANTHROPIC_API_KEY='sk-ant-...'\n"
            "See .env.example for the expected format."
        )

    if not deployments_raw:
        # Fall back to bundled deployments data (populated at build time)
        bundled = bundled_docs_path()
        docs_dir = bundled / "docs"
        if docs_dir.is_dir() and any(docs_dir.iterdir()):
            deployments_path = bundled
        else:
            raise ConfigError(
                f"Missing deployments_path. Set {ENV_DEPLOYMENTS_PATH} or add "
                f'deployments_path = "..." to {config_path}.\n'
                "  export AGENT_SCAFFOLD_DEPLOYMENTS_PATH='/path/to/agent-deployments'"
            )
    else:
        deployments_path = Path(deployments_raw).expanduser()

    return Config(
        deployments_path=deployments_path,
        anthropic_api_key=api_key,
        model=str(model),
        max_tokens=max_tokens,
        thinking_budget=thinking_budget,
        max_context_tokens=max_context_tokens,
        max_link_depth=max_link_depth,
        max_tokens_per_doc=max_tokens_per_doc,
        cache_dir=cache_dir,
        failures_dir=failures_dir,
    )
