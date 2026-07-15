"""Tests for the capability-driven probes added to ``probes.PROBES``."""

from __future__ import annotations

from typing import Any

import pytest

from agent_scaffold.discovery import ExternalService
from agent_scaffold.doctor import CheckStatus
from agent_scaffold.probes import (
    PROBES,
    probe_chroma_heartbeat,
    probe_grafana_health,
    probe_kafka_topic_list,
    probe_langsmith_workspace,
    probe_qdrant_collections,
)


def _svc(name: str, env_var: str, default_local: str | None = None) -> ExternalService:
    return ExternalService(
        id=name,
        env_vars=[env_var],
        default_local=default_local,
    )


def test_all_phase2_probes_registered() -> None:
    for name in (
        "qdrant_collections",
        "chroma_heartbeat",
        "kafka_topic_list",
        "grafana_health",
        "langsmith_workspace",
    ):
        assert name in PROBES


def test_qdrant_probe_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        status_code = 200

        def json(self) -> dict[str, Any]:
            return {"result": {"collections": [{"name": "docs"}, {"name": "memos"}]}}

        text = ""

    class FakeHttpx:
        @staticmethod
        def get(url: str, **_kw: Any) -> FakeResponse:
            return FakeResponse()

        TimeoutException = Exception
        ConnectError = Exception

    monkeypatch.setitem(__import__("sys").modules, "httpx", FakeHttpx)
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    result = probe_qdrant_collections(_svc("qdrant", "QDRANT_URL"))
    assert result.status is CheckStatus.OK
    assert "2 collection" in result.title


def test_qdrant_probe_fails_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        status_code = 500
        text = "internal error"

    class FakeHttpx:
        @staticmethod
        def get(url: str, **_kw: Any) -> FakeResponse:
            return FakeResponse()

        TimeoutException = Exception
        ConnectError = Exception

    monkeypatch.setitem(__import__("sys").modules, "httpx", FakeHttpx)
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    result = probe_qdrant_collections(_svc("qdrant", "QDRANT_URL"))
    assert result.status is CheckStatus.FAIL


def test_chroma_probe_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        status_code = 200
        text = "{}"

    class FakeHttpx:
        @staticmethod
        def get(url: str, **_kw: Any) -> FakeResponse:
            return FakeResponse()

        TimeoutException = Exception
        ConnectError = Exception

    monkeypatch.setitem(__import__("sys").modules, "httpx", FakeHttpx)
    monkeypatch.setenv("CHROMA_URL", "http://localhost:8000")
    assert probe_chroma_heartbeat(_svc("chroma", "CHROMA_URL")).status is CheckStatus.OK


def test_kafka_topic_list_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeAdmin:
        def __init__(self, **_kw: Any) -> None:
            pass

        def list_topics(self) -> list[str]:
            return ["a", "b", "c"]

        def close(self) -> None:
            pass

    monkeypatch.setitem(
        __import__("sys").modules, "kafka", type("M", (), {"KafkaAdminClient": FakeAdmin})
    )
    svc = _svc("kafka", "KAFKA_BOOTSTRAP_SERVERS", default_local="localhost:9092")
    result = probe_kafka_topic_list(svc)
    assert result.status is CheckStatus.OK
    assert "3 topic" in result.title


def test_grafana_health_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        status_code = 200

        def json(self) -> dict[str, str]:
            return {"database": "ok"}

        text = ""

    class FakeHttpx:
        @staticmethod
        def get(url: str, **_kw: Any) -> FakeResponse:
            return FakeResponse()

        TimeoutException = Exception
        ConnectError = Exception

    monkeypatch.setitem(__import__("sys").modules, "httpx", FakeHttpx)
    monkeypatch.setenv("GRAFANA_URL", "http://localhost:3002")
    assert probe_grafana_health(_svc("grafana", "GRAFANA_URL")).status is CheckStatus.OK


def test_grafana_health_warns_when_db_not_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        status_code = 200

        def json(self) -> dict[str, str]:
            return {"database": "warning"}

        text = ""

    class FakeHttpx:
        @staticmethod
        def get(url: str, **_kw: Any) -> FakeResponse:
            return FakeResponse()

        TimeoutException = Exception
        ConnectError = Exception

    monkeypatch.setitem(__import__("sys").modules, "httpx", FakeHttpx)
    monkeypatch.setenv("GRAFANA_URL", "http://localhost:3002")
    result = probe_grafana_health(_svc("grafana", "GRAFANA_URL"))
    assert result.status is CheckStatus.WARN


def test_langsmith_probe_fail_when_required_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    # required=True (default): missing key is FAIL, not SKIP.
    result = probe_langsmith_workspace(_svc("langsmith", "LANGCHAIN_API_KEY"))
    assert result.status is CheckStatus.FAIL
    assert "agent-scaffold connect langsmith" in result.fix_hint


def test_qdrant_probe_env_overlay_supplies_url_and_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeResponse:
        status_code = 200

        def json(self) -> dict[str, Any]:
            return {"result": {"collections": []}}

        text = ""

    class FakeHttpx:
        @staticmethod
        def get(url: str, **kw: Any) -> FakeResponse:
            captured["url"] = url
            captured["headers"] = kw.get("headers") or {}
            return FakeResponse()

        TimeoutException = Exception
        ConnectError = Exception

    monkeypatch.setitem(__import__("sys").modules, "httpx", FakeHttpx)
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    result = probe_qdrant_collections(
        _svc("qdrant", "QDRANT_URL"),
        env={
            "QDRANT_URL": "https://x.cloud.qdrant.io:6333",
            "QDRANT_API_KEY": "qd-key",
        },
    )
    assert result.status is CheckStatus.OK
    assert captured["url"].startswith("https://x.cloud.qdrant.io:6333")
    assert captured["headers"].get("api-key") == "qd-key"


def test_langsmith_probe_skip_for_optional_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    svc = ExternalService(id="langsmith", env_vars=["LANGCHAIN_API_KEY"], required=False)
    result = probe_langsmith_workspace(svc)
    assert result.status is CheckStatus.SKIP


def test_langsmith_probe_ok_with_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self, **_kw: Any) -> None:
            pass

        def info(self) -> dict[str, str]:
            return {"tenant_handle": "acme"}

    monkeypatch.setitem(
        __import__("sys").modules, "langsmith", type("M", (), {"Client": FakeClient})
    )
    monkeypatch.setenv("LANGCHAIN_API_KEY", "ls__test")
    result = probe_langsmith_workspace(_svc("langsmith", "LANGCHAIN_API_KEY"))
    assert result.status is CheckStatus.OK
    assert "acme" in result.title


def test_langsmith_probe_skip_when_sdk_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGCHAIN_API_KEY", "ls__test")
    sys = __import__("sys")
    monkeypatch.delitem(sys.modules, "langsmith", raising=False)

    real_import = (
        __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__
    )

    def fake_import(name: str, *args: Any, **kw: Any) -> Any:
        if name == "langsmith":
            raise ImportError("no module named langsmith")
        return real_import(name, *args, **kw)

    monkeypatch.setattr("builtins.__import__", fake_import)
    result = probe_langsmith_workspace(_svc("langsmith", "LANGCHAIN_API_KEY"))
    assert result.status is CheckStatus.SKIP
