"""Tests for ``agent_scaffold.probes``."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent_scaffold import probes
from agent_scaffold.discovery import ExternalService
from agent_scaffold.doctor import CheckStatus
from agent_scaffold.probes import (
    PROBES,
    probe_anthropic_list_models,
    probe_external_services,
    probe_kafka_metadata,
    probe_langfuse_health,
    probe_postgres_select_one,
    probe_redis_ping,
    resolve_endpoint,
    run_probe,
)


def _svc(**overrides: Any) -> ExternalService:
    base: dict[str, Any] = {"id": "demo"}
    base.update(overrides)
    return ExternalService(**base)


# ---------------------------------------------------------------------------
# resolve_endpoint
# ---------------------------------------------------------------------------


def test_resolve_endpoint_env_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://env-host:6380")
    svc = _svc(env_vars=["REDIS_URL"], default_local="redis://default-host:6379")
    endpoint = resolve_endpoint(svc)
    assert endpoint is not None
    assert endpoint.source == "REDIS_URL"
    assert endpoint.raw == "redis://env-host:6380"


def test_resolve_endpoint_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REDIS_URL", raising=False)
    svc = _svc(env_vars=["REDIS_URL"], default_local="redis://default-host:6379")
    endpoint = resolve_endpoint(svc)
    assert endpoint is not None
    assert endpoint.source == "default_local"


def test_resolve_endpoint_none_when_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NOPE", raising=False)
    svc = _svc(env_vars=["NOPE"])
    assert resolve_endpoint(svc) is None


def test_resolve_endpoint_skips_empty_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "   ")
    svc = _svc(env_vars=["REDIS_URL"], default_local="redis://fallback:6379")
    endpoint = resolve_endpoint(svc)
    assert endpoint is not None
    assert endpoint.source == "default_local"


# ---------------------------------------------------------------------------
# hostport parser
# ---------------------------------------------------------------------------


def test_hostport_from_url_handles_schemes() -> None:
    assert probes._hostport_from_url("redis://example.com:7000", default_port=6379) == (
        "example.com",
        7000,
    )
    assert probes._hostport_from_url("postgresql://u:p@db:5433/x", default_port=5432) == (
        "db",
        5433,
    )
    assert probes._hostport_from_url("https://lf.example.com", default_port=443) == (
        "lf.example.com",
        443,
    )


def test_hostport_from_url_handles_bare_hostport() -> None:
    assert probes._hostport_from_url("kafka:9092", default_port=9092) == ("kafka", 9092)
    assert probes._hostport_from_url("kafka", default_port=9092) == ("kafka", 9092)
    assert probes._hostport_from_url("kafka:bogus", default_port=9092) == ("kafka", 9092)


# ---------------------------------------------------------------------------
# Redis raw-socket PING
# ---------------------------------------------------------------------------


class _FakeSocket:
    def __init__(self, reply: bytes) -> None:
        self.sent: list[bytes] = []
        self._reply = reply

    def sendall(self, data: bytes) -> None:
        self.sent.append(data)

    def recv(self, _n: int) -> bytes:
        return self._reply

    def __enter__(self) -> _FakeSocket:
        return self

    def __exit__(self, *args: Any) -> None:
        return None


@contextmanager
def _patch_socket(monkeypatch: pytest.MonkeyPatch, factory: Any) -> Any:
    monkeypatch.setattr(probes.socket, "create_connection", factory)
    yield


def test_probe_redis_ping_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
    fake = _FakeSocket(b"+PONG\r\n")
    with _patch_socket(monkeypatch, lambda addr, timeout: fake):
        result = probe_redis_ping(_svc(env_vars=["REDIS_URL"], probe="redis_ping"), timeout=1.0)
    assert result.status == CheckStatus.OK
    assert fake.sent == [b"*1\r\n$4\r\nPING\r\n"]


def test_probe_redis_ping_no_address(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REDIS_URL", raising=False)
    result = probe_redis_ping(_svc(env_vars=["REDIS_URL"]), timeout=1.0)
    # `required` defaults to True so the missing address is a FAIL.
    assert result.status == CheckStatus.FAIL


def test_probe_redis_ping_connection_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")

    def boom(addr: Any, timeout: float) -> Any:
        raise ConnectionRefusedError("Connection refused")

    with _patch_socket(monkeypatch, boom):
        result = probe_redis_ping(_svc(env_vars=["REDIS_URL"]), timeout=1.0)
    assert result.status == CheckStatus.FAIL
    assert "Connection refused" in result.detail


def test_probe_redis_ping_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")

    def slow(addr: Any, timeout: float) -> Any:
        raise TimeoutError("timed out")

    with _patch_socket(monkeypatch, slow):
        result = probe_redis_ping(_svc(env_vars=["REDIS_URL"]), timeout=1.0)
    assert result.status == CheckStatus.FAIL
    assert "timed out" in result.detail


def test_probe_redis_ping_noauth_is_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
    fake = _FakeSocket(b"-NOAUTH Authentication required.\r\n")
    with _patch_socket(monkeypatch, lambda addr, timeout: fake):
        result = probe_redis_ping(_svc(env_vars=["REDIS_URL"]), timeout=1.0)
    assert result.status == CheckStatus.FAIL
    assert "auth required" in result.title


def test_probe_redis_ping_unexpected_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
    fake = _FakeSocket(b"-ERR something\r\n")
    with _patch_socket(monkeypatch, lambda addr, timeout: fake):
        result = probe_redis_ping(_svc(env_vars=["REDIS_URL"]), timeout=1.0)
    assert result.status == CheckStatus.FAIL
    assert "unexpected response" in result.title


# ---------------------------------------------------------------------------
# Postgres
# ---------------------------------------------------------------------------


def test_probe_postgres_no_address(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    result = probe_postgres_select_one(_svc(env_vars=["DATABASE_URL"]), timeout=1.0)
    assert result.status == CheckStatus.FAIL


def test_probe_postgres_tcp_fallback_when_psycopg_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When psycopg is not importable, fall back to a TCP-only check returning WARN."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:5432/db")
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "psycopg":
            raise ImportError("no psycopg in this env")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with _patch_socket(monkeypatch, lambda addr, timeout: _FakeSocket(b"")):
        result = probe_postgres_select_one(_svc(env_vars=["DATABASE_URL"]), timeout=1.0)
    assert result.status == CheckStatus.WARN
    assert "TCP-only" in result.title


def test_probe_postgres_tcp_fallback_connection_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:5432/db")
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "psycopg":
            raise ImportError("no psycopg in this env")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    def refused(addr: Any, timeout: float) -> Any:
        raise ConnectionRefusedError("nope")

    with _patch_socket(monkeypatch, refused):
        result = probe_postgres_select_one(_svc(env_vars=["DATABASE_URL"]), timeout=1.0)
    assert result.status == CheckStatus.FAIL


# ---------------------------------------------------------------------------
# Langfuse
# ---------------------------------------------------------------------------


def test_probe_langfuse_no_address(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    result = probe_langfuse_health(_svc(env_vars=["LANGFUSE_HOST"], required=False), timeout=1.0)
    assert result.status == CheckStatus.SKIP


def test_probe_langfuse_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    import httpx

    class _Resp:
        status_code = 200
        text = '{"status":"OK"}'

    def fake_get(url: str, timeout: float) -> Any:
        assert url == "https://cloud.langfuse.com/api/public/health"
        return _Resp()

    monkeypatch.setattr(httpx, "get", fake_get)
    result = probe_langfuse_health(_svc(env_vars=["LANGFUSE_HOST"]), timeout=1.0)
    assert result.status == CheckStatus.OK


def test_probe_langfuse_non_200(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    import httpx

    class _Resp:
        status_code = 503
        text = "Service Unavailable"

    monkeypatch.setattr(httpx, "get", lambda url, timeout: _Resp())
    result = probe_langfuse_health(_svc(env_vars=["LANGFUSE_HOST"]), timeout=1.0)
    assert result.status == CheckStatus.FAIL
    assert "503" in result.title


def test_probe_langfuse_connect_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    import httpx

    def boom(url: str, timeout: float) -> Any:
        raise httpx.ConnectError("dns failure")

    monkeypatch.setattr(httpx, "get", boom)
    result = probe_langfuse_health(_svc(env_vars=["LANGFUSE_HOST"]), timeout=1.0)
    assert result.status == CheckStatus.FAIL


def test_probe_langfuse_adds_https_when_scheme_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_HOST", "cloud.langfuse.com")
    import httpx

    captured: dict[str, str] = {}

    def fake_get(url: str, timeout: float) -> Any:
        captured["url"] = url
        return MagicMock(status_code=200, text="")

    monkeypatch.setattr(httpx, "get", fake_get)
    probe_langfuse_health(_svc(env_vars=["LANGFUSE_HOST"]), timeout=1.0)
    assert captured["url"].startswith("https://")


# ---------------------------------------------------------------------------
# Kafka
# ---------------------------------------------------------------------------


def test_probe_kafka_no_address(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
    result = probe_kafka_metadata(_svc(env_vars=["KAFKA_BOOTSTRAP_SERVERS"]), timeout=1.0)
    assert result.status == CheckStatus.FAIL


def test_probe_kafka_tcp_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "kafka":
            raise ImportError("no kafka in this env")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with _patch_socket(monkeypatch, lambda addr, timeout: _FakeSocket(b"")):
        result = probe_kafka_metadata(_svc(env_vars=["KAFKA_BOOTSTRAP_SERVERS"]), timeout=1.0)
    assert result.status == CheckStatus.WARN


def test_probe_kafka_tcp_fallback_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "kafka":
            raise ImportError("no kafka in this env")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    def refused(addr: Any, timeout: float) -> Any:
        raise OSError("connection refused")

    with _patch_socket(monkeypatch, refused):
        result = probe_kafka_metadata(_svc(env_vars=["KAFKA_BOOTSTRAP_SERVERS"]), timeout=1.0)
    assert result.status == CheckStatus.FAIL


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


def test_probe_anthropic_no_key_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_scaffold import auth

    monkeypatch.setattr(auth, "load_key", lambda name="anthropic": None)
    result = probe_anthropic_list_models(_svc(id="anthropic"), timeout=1.0)
    assert result.status == CheckStatus.FAIL
    assert "no API key" in result.title


def test_probe_anthropic_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic import SecretStr

    from agent_scaffold import auth

    monkeypatch.setattr(
        auth, "load_key", lambda name="anthropic": SecretStr("sk-ant-test-key-1234")
    )

    import anthropic

    class _Models:
        def list(self, limit: int = 1) -> Any:
            return MagicMock(data=[object(), object()])

    class _FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            self.models = _Models()

    monkeypatch.setattr(anthropic, "Anthropic", _FakeClient)
    result = probe_anthropic_list_models(_svc(id="anthropic"), timeout=1.0)
    assert result.status == CheckStatus.OK
    assert "anthropic" in result.title


def test_probe_anthropic_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic import SecretStr

    from agent_scaffold import auth

    monkeypatch.setattr(auth, "load_key", lambda name="anthropic": SecretStr("sk-ant-bad-key-1234"))

    import anthropic
    import httpx

    response = httpx.Response(
        status_code=401, request=httpx.Request("GET", "https://api.anthropic.com/v1/models")
    )

    class _Models:
        def list(self, limit: int = 1) -> Any:
            raise anthropic.AuthenticationError(message="bad", response=response, body=None)

    class _FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            self.models = _Models()

    monkeypatch.setattr(anthropic, "Anthropic", _FakeClient)
    result = probe_anthropic_list_models(_svc(id="anthropic"), timeout=1.0)
    assert result.status == CheckStatus.FAIL
    assert "401" in result.title


# ---------------------------------------------------------------------------
# run_probe dispatcher
# ---------------------------------------------------------------------------


def test_run_probe_skip_when_disabled() -> None:
    result = run_probe(_svc(probe="redis_ping"), skip=True)
    assert result.status == CheckStatus.SKIP
    assert "disabled" in result.title


def test_run_probe_skip_when_no_probe_configured() -> None:
    result = run_probe(_svc())  # probe=None
    assert result.status == CheckStatus.SKIP
    assert "no probe configured" in result.title


def test_run_probe_skip_when_unknown_probe() -> None:
    result = run_probe(_svc(probe="does-not-exist"))
    assert result.status == CheckStatus.SKIP
    assert "unknown probe" in result.title


def test_run_probe_catches_unhandled_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(svc: ExternalService, timeout: float) -> Any:
        raise RuntimeError("uncaught")

    monkeypatch.setitem(PROBES, "boom", boom)
    result = run_probe(_svc(probe="boom"))
    assert result.status == CheckStatus.FAIL
    assert "crashed" in result.title
    assert "RuntimeError" in result.detail


def test_probe_registry_has_all_documented_probes() -> None:
    expected = {
        "anthropic_list_models",
        "redis_ping",
        "postgres_select_one",
        "langfuse_health",
        "kafka_metadata",
    }
    assert expected <= set(PROBES.keys())


# ---------------------------------------------------------------------------
# Sanity: every probe returns a CheckResult on the "no address" path.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# probe_external_services — the parallel runner used by `/recipe` and `/plan`.
# ---------------------------------------------------------------------------


def test_probe_external_services_empty_returns_empty_list() -> None:
    """No services → no work — caller can skip the readiness section entirely."""
    assert probe_external_services([], timeout=1.0) == []


def test_probe_external_services_runs_in_parallel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Total wall time is bounded by ~max(probe_time), not sum(probe_time).

    Replace ``run_probe`` with a stub that sleeps ``timeout`` seconds and
    returns a fixed OK result; with 4 services each "taking" 0.2s the
    parallel runner finishes in well under 0.8s.
    """
    import time

    def slow_probe(svc: ExternalService, timeout: float, skip: bool = False) -> Any:
        time.sleep(0.2)
        from agent_scaffold.doctor import CheckResult

        return CheckResult(
            id=svc.id,
            category="service",
            status=CheckStatus.OK,
            title=f"{svc.id}: ok",
            detail="0ms",
        )

    monkeypatch.setattr(probes, "run_probe", slow_probe)

    services = [_svc(id=f"svc_{i}") for i in range(4)]
    started = time.monotonic()
    results = probe_external_services(services, timeout=1.0, max_workers=4)
    elapsed = time.monotonic() - started

    assert len(results) == 4
    assert all(r.status == CheckStatus.OK for r in results)
    # Sequential would be ~0.8s; parallel should be ~0.2s with a generous
    # safety margin for CI flakiness.
    assert elapsed < 0.55, f"parallel runner appears sequential ({elapsed:.2f}s)"


def test_probe_external_services_preserves_input_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Results come back in the same order as the input services list."""

    def fast_probe(svc: ExternalService, timeout: float, skip: bool = False) -> Any:
        from agent_scaffold.doctor import CheckResult

        return CheckResult(
            id=svc.id,
            category="service",
            status=CheckStatus.OK,
            title=f"{svc.id}: ok",
        )

    monkeypatch.setattr(probes, "run_probe", fast_probe)

    services = [_svc(id="alpha"), _svc(id="bravo"), _svc(id="charlie")]
    results = probe_external_services(services, timeout=1.0)
    assert [r.id for r in results] == ["alpha", "bravo", "charlie"]


def test_probe_external_services_unknown_probe_becomes_skip() -> None:
    """A service with an unknown probe name is reported as SKIP, not a crash.

    Same contract as ``run_probe`` — the helper should never raise. The REPL
    relies on this to keep `/recipe` non-blocking when a recipe references
    a probe the scaffold doesn't ship yet.
    """
    results = probe_external_services(
        [_svc(id="mystery", probe="does_not_exist")],
        timeout=1.0,
    )
    assert len(results) == 1
    assert results[0].status == CheckStatus.SKIP
    assert "unknown probe" in results[0].title.lower()


def test_no_probe_ever_throws_on_missing_address(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every probe must return a CheckResult — not raise — when nothing resolves."""
    for env_var in ("REDIS_URL", "DATABASE_URL", "LANGFUSE_HOST", "KAFKA_BOOTSTRAP_SERVERS"):
        monkeypatch.delenv(env_var, raising=False)
    for name, probe in PROBES.items():
        if name == "anthropic_list_models":
            from agent_scaffold import auth

            monkeypatch.setattr(auth, "load_key", lambda n="anthropic": None)
            result = probe(_svc(id="anthropic"), 1.0)
        else:
            result = probe(_svc(env_vars=[]), 1.0)
        assert result.status in {
            CheckStatus.OK,
            CheckStatus.WARN,
            CheckStatus.FAIL,
            CheckStatus.SKIP,
        }
