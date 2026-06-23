"""Tests for ``agent_scaffold.steps.bootstrap_vector_db``."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from agent_scaffold.capabilities import Capability, ResolvedStack
from agent_scaffold.orchestrator import StepContext, StepStatus
from agent_scaffold.steps import bootstrap_vector_db as bvd
from agent_scaffold.steps.bootstrap_vector_db import BootstrapVectorDbStep


def _cap(name: str, tmp_path: Path) -> Capability:
    return Capability(
        id=f"vector_db.{name}",
        kind="vector_db",
        path=tmp_path / f"{name}.md",
        env_vars=[],
    )


def _stack(*caps: Capability) -> ResolvedStack:
    return ResolvedStack(capabilities=list(caps))


def test_pgvector_extension_from_bootstrap_inputs(tmp_path: Path) -> None:
    """The Postgres extension is read from the capability's bootstrap_inputs —
    the single source of truth, not a hard-coded step constant."""
    cap = Capability(
        id="vector_db.pgvector",
        kind="vector_db",
        path=tmp_path / "pgvector.md",
        bootstrap_inputs={"vector_extension": "vectorscale"},
    )
    assert bvd._pgvector_extension(cap) == "vectorscale"


def test_pgvector_extension_defaults_and_rejects_unsafe(tmp_path: Path) -> None:
    # Absent input → default 'vector'.
    assert bvd._pgvector_extension(_cap("pgvector", tmp_path)) == "vector"
    # Unsafe (non-identifier) value → falls back to the default, never interpolated
    # into CREATE EXTENSION.
    unsafe = Capability(
        id="vector_db.pgvector",
        kind="vector_db",
        path=tmp_path / "p.md",
        bootstrap_inputs={"vector_extension": "vector; DROP TABLE users"},
    )
    assert bvd._pgvector_extension(unsafe) == "vector"


def test_detect_skipped_without_capability(
    ctx_factory: Callable[..., StepContext],
) -> None:
    result = BootstrapVectorDbStep().detect(ctx_factory())
    assert result.status is StepStatus.SKIPPED
    assert "no vector_db" in result.reason


def test_detect_pending_with_capability(
    ctx_factory: Callable[..., StepContext], tmp_path: Path
) -> None:
    stack = _stack(_cap("qdrant", tmp_path))
    result = BootstrapVectorDbStep().detect(ctx_factory(resolved_stack=stack))
    assert result.status is StepStatus.PENDING
    assert "vector_db.qdrant" in result.reason


def test_apply_skipped_without_capability(
    ctx_factory: Callable[..., StepContext],
) -> None:
    result = BootstrapVectorDbStep().apply(ctx_factory())
    assert result.status is StepStatus.SKIPPED


def test_apply_qdrant_creates_missing_collections(
    ctx_factory: Callable[..., StepContext],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mocked qdrant-client: empty cluster → create the default collection."""
    stack = _stack(_cap("qdrant", tmp_path))
    created: list[str] = []

    class FakeCollections:
        def __init__(self) -> None:
            self.collections: list[Any] = []  # empty cluster

    class FakeClient:
        def __init__(self, **_kw: Any) -> None:
            pass

        def get_collections(self) -> FakeCollections:
            return FakeCollections()

        def create_collection(self, **kw: Any) -> None:
            created.append(kw["collection_name"])

    class FakeModels:
        class Distance:
            COSINE = "cosine"
            EUCLID = "euclid"
            DOT = "dot"

        class VectorParams:
            def __init__(self, **kw: Any) -> None:
                self.kw = kw

    fake_qc = type("M", (), {"QdrantClient": FakeClient})
    fake_http = type("M", (), {"models": FakeModels})
    monkeypatch.setitem(__import__("sys").modules, "qdrant_client", fake_qc)
    monkeypatch.setitem(__import__("sys").modules, "qdrant_client.http", fake_http)
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")

    result = BootstrapVectorDbStep().apply(ctx_factory(resolved_stack=stack))
    assert result.status is StepStatus.DONE
    assert created == ["documents"]
    assert "created 1" in result.detail


def test_apply_qdrant_idempotent_when_all_exist(
    ctx_factory: Callable[..., StepContext],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing collections → DONE without create_collection calls."""
    stack = _stack(_cap("qdrant", tmp_path))

    class Col:
        def __init__(self, name: str) -> None:
            self.name = name

    class FakeCollections:
        collections = [Col("documents")]

    create_calls: list[str] = []

    class FakeClient:
        def __init__(self, **_kw: Any) -> None:
            pass

        def get_collections(self) -> FakeCollections:
            return FakeCollections()

        def create_collection(self, **kw: Any) -> None:
            create_calls.append(kw["collection_name"])

    class FakeModels:
        class Distance:
            COSINE = "cosine"
            EUCLID = "euclid"
            DOT = "dot"

        class VectorParams:
            def __init__(self, **kw: Any) -> None:
                pass

    monkeypatch.setitem(
        __import__("sys").modules, "qdrant_client", type("M", (), {"QdrantClient": FakeClient})
    )
    monkeypatch.setitem(
        __import__("sys").modules, "qdrant_client.http", type("M", (), {"models": FakeModels})
    )

    result = BootstrapVectorDbStep().apply(ctx_factory(resolved_stack=stack))
    assert result.status is StepStatus.DONE
    assert create_calls == []
    assert "already exist" in result.detail


def test_apply_qdrant_skipped_when_sdk_missing(
    ctx_factory: Callable[..., StepContext],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = _stack(_cap("qdrant", tmp_path))
    sys = __import__("sys")
    # Ensure the import fails.
    monkeypatch.delitem(sys.modules, "qdrant_client", raising=False)

    real_import = (
        __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__
    )

    def fake_import(name: str, *args: Any, **kw: Any) -> Any:
        if name == "qdrant_client":
            raise ImportError("no module named qdrant_client")
        return real_import(name, *args, **kw)

    monkeypatch.setattr("builtins.__import__", fake_import)
    result = BootstrapVectorDbStep().apply(ctx_factory(resolved_stack=stack))
    assert result.status is StepStatus.SKIPPED
    assert "qdrant-client not installed" in result.detail


def test_apply_chroma_creates_via_http(
    ctx_factory: Callable[..., StepContext],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = _stack(_cap("chroma", tmp_path))
    posted: list[str] = []

    class FakeResponse:
        def __init__(self, code: int, text: str = "") -> None:
            self.status_code = code
            self.text = text

    class FakeHttpx:
        @staticmethod
        def post(url: str, **kw: Any) -> FakeResponse:
            payload = kw.get("json", {})
            posted.append(payload["name"])
            return FakeResponse(200)

        TimeoutException = Exception
        ConnectError = Exception

    monkeypatch.setitem(__import__("sys").modules, "httpx", FakeHttpx)
    monkeypatch.setenv("CHROMA_URL", "http://localhost:8000")
    result = BootstrapVectorDbStep().apply(ctx_factory(resolved_stack=stack))
    assert result.status is StepStatus.DONE
    assert posted == ["documents"]


def test_apply_chroma_idempotent_on_409(
    ctx_factory: Callable[..., StepContext],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = _stack(_cap("chroma", tmp_path))

    class FakeResponse:
        status_code = 409
        text = "collection already exists"

    class FakeHttpx:
        @staticmethod
        def post(url: str, **kw: Any) -> FakeResponse:
            return FakeResponse()

        TimeoutException = Exception
        ConnectError = Exception

    monkeypatch.setitem(__import__("sys").modules, "httpx", FakeHttpx)
    result = BootstrapVectorDbStep().apply(ctx_factory(resolved_stack=stack))
    assert result.status is StepStatus.DONE
    assert "already exist" in result.detail


def test_apply_unknown_variant_skipped(
    ctx_factory: Callable[..., StepContext], tmp_path: Path
) -> None:
    weird = Capability(
        id="vector_db.weaviate",
        kind="vector_db",
        path=tmp_path / "weaviate.md",
    )
    stack = _stack(weird)
    result = BootstrapVectorDbStep().apply(ctx_factory(resolved_stack=stack))
    assert result.status is StepStatus.SKIPPED
    assert "unknown" in result.detail


def test_fingerprint_changes_with_capability_set(
    ctx_factory: Callable[..., StepContext], tmp_path: Path
) -> None:
    step = BootstrapVectorDbStep()
    a = ctx_factory(resolved_stack=_stack(_cap("qdrant", tmp_path)))
    b = ctx_factory(resolved_stack=_stack(_cap("qdrant", tmp_path), _cap("chroma", tmp_path)))
    assert step.fingerprint(a) != step.fingerprint(b)


def test_resolve_collections_uses_manifest_override(
    ctx_factory: Callable[..., StepContext], tmp_path: Path
) -> None:
    from agent_scaffold.manifest import Manifest

    custom = Manifest(
        recipe="x",
        language="python",
        framework="none",
        model="m",
        generated_at="2026-01-01T00:00:00+00:00",
        answers={
            "vector_collections": ('[{"name": "memories", "vector_size": 768, "distance": "l2"}]')
        },
    )
    ctx = ctx_factory(manifest=custom)
    collections = bvd._resolve_collections(ctx)
    assert collections == [{"name": "memories", "vector_size": 768, "distance": "l2"}]


def test_resolve_collections_falls_back_on_bad_json(
    ctx_factory: Callable[..., StepContext],
) -> None:
    from agent_scaffold.manifest import Manifest

    bad = Manifest(
        recipe="x",
        language="python",
        framework="none",
        model="m",
        generated_at="2026-01-01T00:00:00+00:00",
        answers={"vector_collections": "not valid json"},
    )
    collections = bvd._resolve_collections(ctx_factory(manifest=bad))
    assert collections[0]["name"] == "documents"
