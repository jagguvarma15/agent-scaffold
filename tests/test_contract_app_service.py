"""Tests for ``normalize_app_service`` in ``contract`` — the deterministic pass
that guarantees the backend (app) compose service can actually boot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from agent_scaffold.capabilities import Capability, ResolvedStack
from agent_scaffold.contract import (
    GeneratedFile,
    GenerationResult,
    normalize_app_service,
)

# A compose with a locally-built ``app`` (backend) plus an ``image:`` postgres
# whose own config vars must NOT leak onto the app, mirroring a real generation.
_COMPOSE = """\
services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    env_file: .env
    environment:
      DATABASE_URL: postgresql://${POSTGRES_USER:-agent}@postgres:5432/db
      REDIS_URL: redis://redis:6379
    depends_on:
      postgres:
        condition: service_healthy
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-agent}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-agent}
volumes:
  postgres_data:
"""


def _result(compose: str | None) -> GenerationResult:
    files = [GeneratedFile(path="README.md", content="hi")]
    if compose is not None:
        files.append(GeneratedFile(path="docker-compose.yml", content=compose))
    return GenerationResult(
        project_name="demo", language="python", files=files, smoke_check="pytest"
    )


def _stack(env_vars: list[str]) -> ResolvedStack:
    return ResolvedStack(
        capabilities=[
            Capability(id="obs.langsmith", kind="obs", path=Path("/x.md"), env_vars=env_vars)
        ]
    )


def _app_env(result: GenerationResult) -> dict[str, Any]:
    compose = next(f for f in result.files if f.path == "docker-compose.yml")
    data = yaml.safe_load(compose.content)
    return data["services"]["app"]["environment"]


def test_injects_anthropic_key_passthrough_and_preserves_app_values() -> None:
    out = normalize_app_service(_result(_COMPOSE), _stack([]))
    env = _app_env(out)
    # Agent key forwarded via the same ${VAR:-} interpolation the fragments use.
    assert env["ANTHROPIC_API_KEY"] == "${ANTHROPIC_API_KEY:-}"
    # The LLM's explicit in-network values are untouched.
    assert env["DATABASE_URL"] == "postgresql://${POSTGRES_USER:-agent}@postgres:5432/db"
    assert env["REDIS_URL"] == "redis://redis:6379"


def test_does_not_copy_other_service_env_keys_onto_app() -> None:
    # POSTGRES_* belong to the postgres service even if a capability lists them.
    out = normalize_app_service(_result(_COMPOSE), _stack(["POSTGRES_USER", "POSTGRES_PASSWORD"]))
    env = _app_env(out)
    assert "POSTGRES_USER" not in env
    assert "POSTGRES_PASSWORD" not in env


def test_capability_secret_var_is_forwarded() -> None:
    out = normalize_app_service(_result(_COMPOSE), _stack(["LANGCHAIN_API_KEY"]))
    env = _app_env(out)
    assert env["LANGCHAIN_API_KEY"] == "${LANGCHAIN_API_KEY:-}"


def test_env_file_made_required_false() -> None:
    out = normalize_app_service(_result(_COMPOSE), _stack([]))
    compose = next(f for f in out.files if f.path == "docker-compose.yml")
    app = yaml.safe_load(compose.content)["services"]["app"]
    assert app["env_file"] == [{"path": ".env", "required": False}]


def test_existing_anthropic_key_is_not_clobbered() -> None:
    compose = _COMPOSE.replace(
        "      REDIS_URL: redis://redis:6379",
        "      REDIS_URL: redis://redis:6379\n      ANTHROPIC_API_KEY: sk-ant-literal",
    )
    out = normalize_app_service(_result(compose), _stack([]))
    assert _app_env(out)["ANTHROPIC_API_KEY"] == "sk-ant-literal"


def test_app_resolved_by_conventional_name_without_build() -> None:
    compose = """\
services:
  api:
    image: ghcr.io/acme/api:1.0
    environment:
      LOG_LEVEL: INFO
"""
    out = normalize_app_service(_result(compose), _stack([]))
    data = yaml.safe_load(next(f for f in out.files if f.path == "docker-compose.yml").content)
    assert data["services"]["api"]["environment"]["ANTHROPIC_API_KEY"] == "${ANTHROPIC_API_KEY:-}"


def test_no_op_without_compose_file() -> None:
    r = _result(None)
    assert normalize_app_service(r, _stack([])) is r


def test_no_op_when_no_app_service() -> None:
    # Only an image-based infra service, no build and no conventional app name.
    compose = "services:\n  postgres:\n    image: postgres:16-alpine\n"
    r = _result(compose)
    assert normalize_app_service(r, _stack([])) is r


def test_is_idempotent() -> None:
    once = normalize_app_service(_result(_COMPOSE), _stack(["LANGCHAIN_API_KEY"]))
    twice = normalize_app_service(once, _stack(["LANGCHAIN_API_KEY"]))
    once_compose = next(f for f in once.files if f.path == "docker-compose.yml").content
    twice_compose = next(f for f in twice.files if f.path == "docker-compose.yml").content
    assert once_compose == twice_compose
