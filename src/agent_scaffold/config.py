"""Configuration loading for agent-scaffold.

Resolves ``Config`` from environment variables with a TOML config file at
``~/.config/agent-scaffold/config.toml`` as a fallback for ``deployments_path``,
``blueprints_path``, and ``model``.

Deployments and blueprints paths are **optional hints** stored here. Actual
resolution (with auto-fetch from GitHub, cache lookup, and fallback) lives
in :mod:`agent_scaffold.sources` so ``load_config`` stays free of network
I/O — commands that don't need the deployments tree (``config``, ``auth``,
``secrets``) can run instantly.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr

from agent_scaffold.models import DEFAULT_MODEL

DEFAULT_MAX_TOKENS = 32000
DEFAULT_MAX_CONTEXT_TOKENS = 60_000
DEFAULT_MAX_LINK_DEPTH = 2
DEFAULT_MAX_TOKENS_PER_DOC = 8_000
DEFAULT_CACHE_TTL: Literal["5m", "1h"] = "5m"
CACHE_TTLS: tuple[str, ...] = ("5m", "1h")
DEFAULT_DEPLOYMENTS_SOURCE: Literal["auto"] = "auto"
DEFAULT_BLUEPRINTS_SOURCE: Literal["auto", "skip"] = "auto"

ENV_API_KEY = "ANTHROPIC_API_KEY"
ENV_MODEL = "AGENT_SCAFFOLD_MODEL"
ENV_MAX_TOKENS = "AGENT_SCAFFOLD_MAX_TOKENS"
ENV_THINKING_BUDGET = "AGENT_SCAFFOLD_THINKING_BUDGET"
ENV_EFFORT = "AGENT_SCAFFOLD_EFFORT"
ENV_DEPLOYMENTS_PATH = "AGENT_SCAFFOLD_DEPLOYMENTS_PATH"
ENV_BLUEPRINTS_PATH = "AGENT_SCAFFOLD_BLUEPRINTS_PATH"
ENV_DEPLOYMENTS_SOURCE = "AGENT_SCAFFOLD_DEPLOYMENTS_SOURCE"
ENV_BLUEPRINTS_SOURCE = "AGENT_SCAFFOLD_BLUEPRINTS_SOURCE"
ENV_CATALOG_URL = "AGENT_SCAFFOLD_CATALOG_URL"
ENV_CACHE_DIR = "AGENT_SCAFFOLD_CACHE_DIR"
ENV_CONFIG_PATH = "AGENT_SCAFFOLD_CONFIG_PATH"
ENV_MAX_CONTEXT_TOKENS = "AGENT_SCAFFOLD_MAX_CONTEXT_TOKENS"
ENV_MAX_LINK_DEPTH = "AGENT_SCAFFOLD_MAX_LINK_DEPTH"
ENV_MAX_TOKENS_PER_DOC = "AGENT_SCAFFOLD_MAX_TOKENS_PER_DOC"
ENV_CACHE_TTL = "AGENT_SCAFFOLD_CACHE_TTL"
ENV_LEGACY_CONTRACT = "AGENT_SCAFFOLD_LEGACY_CONTRACT"

DEPLOYMENTS_SOURCES: tuple[str, ...] = ("auto",)
BLUEPRINTS_SOURCES: tuple[str, ...] = ("auto", "skip")

DEFAULT_CONFIG_RELATIVE = Path(".config/agent-scaffold/config.toml")
DEFAULT_CACHE_RELATIVE = Path(".cache/agent-scaffold")


class ConfigError(Exception):
    """Raised when configuration cannot be resolved."""


class MissingKeyError(ConfigError):
    """No Anthropic key resolvable from env, keyring, or file.

    A :class:`ConfigError` subclass so existing ``except ConfigError`` handlers
    still catch it; ``cmd_scaffold`` catches it specifically to offer
    first-launch onboarding (open the secure paste form, store, continue)
    instead of a hard exit.
    """


class Config(BaseModel):
    """Resolved runtime configuration.

    ``deployments_path`` and ``blueprints_path`` are optional **hints** —
    explicit overrides via env var or TOML. If unset, the CLI's source
    resolver (:mod:`agent_scaffold.sources`) fetches the latest commit
    from GitHub and falls back to the bundled copy / skip as appropriate.
    """

    deployments_path: Path | None = None
    blueprints_path: Path | None = None
    deployments_source: Literal["auto"] = DEFAULT_DEPLOYMENTS_SOURCE
    blueprints_source: Literal["auto", "skip"] = DEFAULT_BLUEPRINTS_SOURCE
    catalog_url: str | None = None
    """Override for the deployments catalog URL. None means use
    ``catalog.DEFAULT_CATALOG_URL``. Resolved by :func:`catalog.load_catalog`
    at runtime — config just carries the override string."""
    anthropic_api_key: SecretStr
    """Typed ``SecretStr`` per docs/design/security.md rule 3: ``repr()`` /
    ``str()`` of the config masks the key. Unwrap with ``.get_secret_value()``
    only at the SDK client constructors."""
    model: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    thinking_budget: int | None = None
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS
    max_link_depth: int = DEFAULT_MAX_LINK_DEPTH
    max_tokens_per_doc: int = DEFAULT_MAX_TOKENS_PER_DOC
    cache_ttl: Literal["5m", "1h"] = DEFAULT_CACHE_TTL
    """Prompt-cache TTL for the stable prefix (system + hot context blocks).
    ``5m`` (the default) writes at the cheaper 5-minute rate; a one-shot
    generation reads nothing back, so the longer TTL is pure overhead. Opt
    into ``1h`` for REPL sessions that regenerate the same recipe within the
    hour and want the prefix to stay warm across runs."""
    legacy_contract: bool = False
    """Escape hatch (``AGENT_SCAFFOLD_LEGACY_CONTRACT=1``): skip the
    structured-outputs ``output_config.format`` on the generation call and
    restore the free-form response path. Exists in case a catalog or recipe
    combination ever trips a server-side grammar limit; delete after one
    release if unused."""
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
    # Fall back to the auth module's keyring/file resolution. Imported lazily
    # so config doesn't pull in `keyring` until it's actually needed (and so
    # tests that don't touch the API can run without the dep installed).
    if not api_key:
        try:
            from agent_scaffold.auth import load_key

            secret = load_key()
            if secret is not None:
                api_key = secret.get_secret_value().strip()
        except ImportError:
            pass

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
            raise ConfigError(f"Invalid {env_var}: {raw!r} (expected an integer)") from exc

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
        raise MissingKeyError(
            "No Anthropic key found.\n"
            "  - Set ANTHROPIC_API_KEY in your shell, or\n"
            "  - Run `agent-scaffold auth login` to store one in your keychain, or\n"
            "  - Run `agent-scaffold auth setup-token <name> --stdin` for a CI token."
        )

    # Deployments / blueprints paths are optional hints. None means "let the
    # source resolver decide" (auto-fetch + bundled / skip fallback).
    deployments_path = Path(deployments_raw).expanduser() if deployments_raw else None
    blueprints_raw = src.get(ENV_BLUEPRINTS_PATH) or toml_data.get("blueprints_path")
    blueprints_path = Path(blueprints_raw).expanduser() if blueprints_raw else None

    deployments_source_raw = (
        src.get(ENV_DEPLOYMENTS_SOURCE)
        or toml_data.get("deployments_source")
        or DEFAULT_DEPLOYMENTS_SOURCE
    )
    if deployments_source_raw not in DEPLOYMENTS_SOURCES:
        raise ConfigError(
            f"Invalid {ENV_DEPLOYMENTS_SOURCE}: {deployments_source_raw!r} "
            f"(expected one of {DEPLOYMENTS_SOURCES})"
        )
    blueprints_source_raw = (
        src.get(ENV_BLUEPRINTS_SOURCE)
        or toml_data.get("blueprints_source")
        or DEFAULT_BLUEPRINTS_SOURCE
    )
    if blueprints_source_raw not in BLUEPRINTS_SOURCES:
        raise ConfigError(
            f"Invalid {ENV_BLUEPRINTS_SOURCE}: {blueprints_source_raw!r} "
            f"(expected one of {BLUEPRINTS_SOURCES})"
        )

    catalog_url_raw = src.get(ENV_CATALOG_URL) or toml_data.get("catalog_url")
    catalog_url = str(catalog_url_raw).strip() if catalog_url_raw else None

    cache_ttl_raw = src.get(ENV_CACHE_TTL) or toml_data.get("cache_ttl") or DEFAULT_CACHE_TTL
    cache_ttl = str(cache_ttl_raw).strip().lower()
    if cache_ttl not in CACHE_TTLS:
        raise ConfigError(
            f"Invalid {ENV_CACHE_TTL}: {cache_ttl_raw!r} (expected one of {CACHE_TTLS})"
        )

    legacy_contract_raw = src.get(ENV_LEGACY_CONTRACT) or toml_data.get("legacy_contract") or ""
    legacy_contract = str(legacy_contract_raw).strip().lower() in ("1", "true", "yes")

    return Config(
        deployments_path=deployments_path,
        blueprints_path=blueprints_path,
        deployments_source=deployments_source_raw,
        blueprints_source=blueprints_source_raw,
        catalog_url=catalog_url,
        anthropic_api_key=api_key,
        model=str(model),
        max_tokens=max_tokens,
        thinking_budget=thinking_budget,
        max_context_tokens=max_context_tokens,
        max_link_depth=max_link_depth,
        max_tokens_per_doc=max_tokens_per_doc,
        cache_ttl=cache_ttl,  # validated against CACHE_TTLS above
        legacy_contract=legacy_contract,
        cache_dir=cache_dir,
        failures_dir=failures_dir,
    )
