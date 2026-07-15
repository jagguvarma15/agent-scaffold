"""Tests for the pure helpers in ``agent_scaffold.integrations``."""

from __future__ import annotations

import yaml

from agent_scaffold import integrations
from agent_scaffold.integrations import (
    PROVIDER_EXTRAS,
    UpstashDatabase,
    find_literal_env_entries,
    parse_upstash_start_response,
    rewrite_literal_env,
    validate_redis_url,
)

COMPOSE = """services:
  app:
    build:
      context: .
    environment:
      ANTHROPIC_API_KEY: null
      REDIS_URL: redis://redis:6379
      LANGCHAIN_TRACING_V2: 'false'
      LANGCHAIN_PROJECT: research-assistant
      LANGCHAIN_ENDPOINT: ${LANGCHAIN_ENDPOINT:-}
      AGENT_SETUP_FIELDS: '[{"name":"ANTHROPIC_API_KEY","required":true,"hint":"console
        settings API keys"}]'
    ports:
    - 8000:8000
  redis:
    # the local sandbox container
    image: redis:7-alpine
    environment:
      REDIS_URL: untouched
"""


# ---- registry invariants ---------------------------------------------------


def test_provider_extras_cover_the_special_cased_providers() -> None:
    assert set(PROVIDER_EXTRAS) == {"langsmith", "redis"}
    langsmith = PROVIDER_EXTRAS["langsmith"]
    assert langsmith.validate is not None
    assert langsmith.companion is not None
    assert langsmith.closing is not None
    redis = PROVIDER_EXTRAS["redis"]
    assert redis.provision is not None
    assert redis.validate is not None


# ---- parse_upstash_start_response -------------------------------------------


def test_parse_upstash_full_url() -> None:
    db = parse_upstash_start_response(
        {"url": "rediss://:pw@fly-x.upstash.io:6379", "claim_url": "https://u/claim/1"}
    )
    assert db == UpstashDatabase(
        url="rediss://:pw@fly-x.upstash.io:6379", claim_url="https://u/claim/1"
    )


def test_parse_upstash_parts() -> None:
    db = parse_upstash_start_response(
        {"endpoint": "usw1-x.upstash.io", "port": "6380", "password": "pw"}
    )
    assert db is not None
    assert db.url == "rediss://:pw@usw1-x.upstash.io:6380"
    assert db.claim_url is None


def test_parse_upstash_token_alias_and_default_port() -> None:
    db = parse_upstash_start_response({"host": "h.upstash.io", "token": "tk"})
    assert db is not None
    assert db.url == "rediss://:tk@h.upstash.io:6379"


def test_parse_upstash_missing_password_is_none() -> None:
    assert parse_upstash_start_response({"endpoint": "h.upstash.io", "port": 6379}) is None


def test_parse_upstash_garbage_is_none() -> None:
    assert parse_upstash_start_response({"weird": True}) is None
    assert parse_upstash_start_response({"url": "http://not-redis"}) is None


# ---- find_literal_env_entries -----------------------------------------------


def test_find_literals_dict_form() -> None:
    found = find_literal_env_entries(
        COMPOSE, "app", ("REDIS_URL", "LANGCHAIN_TRACING_V2", "LANGCHAIN_ENDPOINT")
    )
    assert found == {
        "REDIS_URL": "redis://redis:6379",
        "LANGCHAIN_TRACING_V2": "false",
    }  # the ${...} entry is already interpolated and excluded


def test_find_literals_list_form() -> None:
    compose = "services:\n  app:\n    environment:\n    - REDIS_URL=redis://redis:6379\n"
    assert find_literal_env_entries(compose, "app", ("REDIS_URL",)) == {
        "REDIS_URL": "redis://redis:6379"
    }


def test_find_literals_other_service_ignored() -> None:
    assert find_literal_env_entries(COMPOSE, "redis", ("LANGCHAIN_TRACING_V2",)) == {}


def test_find_literals_malformed_yaml() -> None:
    assert find_literal_env_entries("services: [", "app", ("REDIS_URL",)) == {}


# ---- rewrite_literal_env ------------------------------------------------------


def test_rewrite_dict_form_preserves_literal_as_default() -> None:
    new_text, rewritten = rewrite_literal_env(COMPOSE, "app", ("LANGCHAIN_TRACING_V2", "REDIS_URL"))
    assert rewritten == ["LANGCHAIN_TRACING_V2", "REDIS_URL"]
    assert "LANGCHAIN_TRACING_V2: ${LANGCHAIN_TRACING_V2:-false}" in new_text
    assert "REDIS_URL: ${REDIS_URL:-redis://redis:6379}" in new_text
    # comments and unrelated services survive
    assert "# the local sandbox container" in new_text
    assert "REDIS_URL: untouched" in new_text
    # the multi-line setup-fields value is intact and the file still parses
    data = yaml.safe_load(new_text)
    assert "ANTHROPIC_API_KEY" in data["services"]["app"]["environment"]["AGENT_SETUP_FIELDS"]


def test_rewrite_list_form() -> None:
    compose = "services:\n  app:\n    environment:\n    - REDIS_URL=redis://redis:6379\n"
    new_text, rewritten = rewrite_literal_env(compose, "app", ("REDIS_URL",))
    assert rewritten == ["REDIS_URL"]
    assert "- REDIS_URL=${REDIS_URL:-redis://redis:6379}" in new_text


def test_rewrite_noop_when_already_interpolated() -> None:
    new_text, rewritten = rewrite_literal_env(COMPOSE, "app", ("LANGCHAIN_ENDPOINT",))
    assert rewritten == []
    assert new_text == COMPOSE


def test_rewrite_noop_for_unknown_service() -> None:
    new_text, rewritten = rewrite_literal_env(COMPOSE, "ghost", ("REDIS_URL",))
    assert rewritten == []
    assert new_text == COMPOSE


def test_rewrite_never_touches_other_services_var() -> None:
    new_text, rewritten = rewrite_literal_env(COMPOSE, "app", ("REDIS_URL",))
    assert rewritten == ["REDIS_URL"]
    assert "REDIS_URL: untouched" in new_text


def test_rewrite_reparse_verification_reverts(monkeypatch: object) -> None:
    # Force the verification step to see a mismatch: rewrite a var whose line
    # the scanner matched but whose parsed value would not carry the marker.
    import pytest

    mp = monkeypatch
    assert isinstance(mp, pytest.MonkeyPatch)
    original = integrations._service_env

    calls = {"n": 0}

    def flaky(data: object, service: str) -> dict[str, str] | None:
        calls["n"] += 1
        # First call: detection (real). Second call: verification (sabotaged).
        if calls["n"] >= 2:
            return {}
        return original(data, service)

    mp.setattr(integrations, "_service_env", flaky)
    new_text, rewritten = rewrite_literal_env(COMPOSE, "app", ("REDIS_URL",))
    assert rewritten == []
    assert new_text == COMPOSE


# ---- validate_redis_url --------------------------------------------------------


def test_validate_redis_url_maps_ping(monkeypatch: object) -> None:
    import pytest

    from agent_scaffold.probes import RedisPingResult

    mp = monkeypatch
    assert isinstance(mp, pytest.MonkeyPatch)
    mp.setattr(
        integrations,
        "redis_ping_url",
        lambda raw, timeout: RedisPingResult(
            False, "auth", "auth rejected (h:1 tls)", "-WRONGPASS"
        ),
    )
    verdict = validate_redis_url("rediss://:pw@h:1", 1.0)
    assert verdict.ok is False
    assert verdict.auth_failure is True
    assert "auth rejected" in verdict.detail
