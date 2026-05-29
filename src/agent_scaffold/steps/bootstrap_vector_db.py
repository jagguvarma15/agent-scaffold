"""``bootstrap_vector_db`` step: create vector DB collections post docker_up.

Driven by the resolved capability set on ``StepContext.resolved_stack``
(Phase 1b) plus an optional recipe-frontmatter ``vector_collections:``
block. For each vector_db capability:

- ``vector_db.qdrant``  → ``qdrant-client`` ``recreate_collection`` (idempotent
  via ``get_collections()`` first).
- ``vector_db.chroma``  → HTTP ``POST /api/v1/collections`` (idempotent on 409).
- ``vector_db.pgvector`` → ``CREATE EXTENSION IF NOT EXISTS vector;`` plus
  optional ``CREATE TABLE IF NOT EXISTS <name> (...)`` per declared collection.

Edge cases (mirrors :mod:`agent_scaffold.steps.docker_up`):

- No vector_db capability on the recipe → ``SKIPPED``.
- ``ctx.resolved_stack`` is None → ``SKIPPED``.
- Optional SDK missing (no ``qdrant-client`` / ``chromadb`` / ``psycopg``) →
  ``SKIPPED`` with a fix_hint pointing at the ``vector`` extra, **not**
  ``FAILED`` (the user may swap implementations between runs).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from agent_scaffold.orchestrator import (
    DetectionResult,
    StepContext,
    StepLog,
    StepProgress,
    StepResult,
    StepStatus,
    compute_fingerprint,
)

_DEFAULT_TIMEOUT = 60.0
_DEFAULT_COLLECTION_NAME = "documents"
_DEFAULT_VECTOR_SIZE = 1536
_DEFAULT_DISTANCE = "cosine"


@dataclass
class BootstrapVectorDbStep:
    """Initialize vector DB collections declared by the recipe."""

    id: str = "bootstrap_vector_db"
    description: str = "Initialize vector DB collections declared by the recipe"
    depends_on: tuple[str, ...] = ("docker_up",)
    timeout: float = _DEFAULT_TIMEOUT
    troubleshoot: dict[str, str] = field(
        default_factory=lambda: {
            "Connection refused": (
                "vector DB container not reachable — re-run "
                "`agent-scaffold up --retry docker_up` first"
            ),
            "Unauthorized": (
                "API key rejected — re-run `agent-scaffold up --retry wire_credentials` "
                "and paste a fresh key"
            ),
            "must be installed": (
                'install the vector extra: pip install "agent-scaffold-cli[vector]"'
            ),
        }
    )

    # ---- detection ----------------------------------------------------

    def detect(self, ctx: StepContext) -> DetectionResult:
        caps = self._vector_capabilities(ctx)
        if not caps:
            return DetectionResult(
                StepStatus.SKIPPED,
                reason="recipe declares no vector_db.* capability",
            )
        # detect() doesn't probe — that would be a network call on every plan.
        # Always report PENDING so apply() runs (apply() is idempotent).
        cap_ids = ", ".join(c.id for c in caps)
        return DetectionResult(StepStatus.PENDING, reason=f"initialize: {cap_ids}")

    # ---- apply --------------------------------------------------------

    def apply(self, ctx: StepContext) -> StepResult:
        caps = self._vector_capabilities(ctx)
        if not caps:
            return StepResult(StepStatus.SKIPPED, detail="no vector_db.* capability")
        collections = _resolve_collections(ctx)
        summary: list[str] = []
        for cap in caps:
            name = cap.id.split(".", 1)[1]
            ctx.emit(
                StepProgress(step_id=self.id, message=f"initializing {cap.id}")
            )
            try:
                if name == "qdrant":
                    detail = _init_qdrant(cap, collections, ctx)
                elif name == "chroma":
                    detail = _init_chroma(cap, collections, ctx)
                elif name == "pgvector":
                    detail = _init_pgvector(collections, ctx)
                else:
                    return StepResult(
                        StepStatus.SKIPPED,
                        detail=f"unknown vector_db variant {cap.id!r}",
                    )
            except _BootstrapSkip as skip:
                return StepResult(StepStatus.SKIPPED, detail=skip.reason)
            except _BootstrapFail as fail:
                return StepResult(
                    StepStatus.FAILED,
                    error=fail.reason,
                    stderr_tail=fail.stderr_tail,
                )
            summary.append(f"{cap.id}: {detail}")
        return StepResult(StepStatus.DONE, detail="; ".join(summary))

    # ---- fingerprint --------------------------------------------------

    def fingerprint(self, ctx: StepContext) -> str:
        caps = self._vector_capabilities(ctx)
        collections = _resolve_collections(ctx)
        return compute_fingerprint(
            {
                "capabilities": sorted(c.id for c in caps),
                "collections": collections,
            }
        )

    # ---- helpers ------------------------------------------------------

    def _vector_capabilities(self, ctx: StepContext) -> list[Any]:
        stack = ctx.resolved_stack
        if stack is None:
            return []
        return [c for c in stack.capabilities if c.kind == "vector_db"]


# ---------------------------------------------------------------------------
# Per-variant initialization helpers
# ---------------------------------------------------------------------------


class _BootstrapSkip(Exception):
    """Step should report SKIPPED with a friendly reason."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _BootstrapFail(Exception):
    """Step should report FAILED with the given reason + optional stderr tail."""

    def __init__(self, reason: str, stderr_tail: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.stderr_tail = stderr_tail


def _resolve_collections(ctx: StepContext) -> list[dict[str, Any]]:
    """Resolve the list of collections to create.

    Order of precedence:
    1. ``ctx.manifest.answers["vector_collections"]`` if set (JSON-encoded list).
    2. Single default collection: ``documents`` / 1536 dims / cosine.

    Each entry: ``{"name": str, "vector_size": int, "distance": "cosine"|"l2"|"dot"}``.
    """
    raw = ctx.manifest.answers.get("vector_collections", "") if ctx.manifest else ""
    if raw:
        import json

        try:
            data = json.loads(raw)
        except ValueError:
            data = []
        if isinstance(data, list) and data:
            cleaned: list[dict[str, Any]] = []
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                cleaned.append(
                    {
                        "name": str(entry.get("name") or _DEFAULT_COLLECTION_NAME),
                        "vector_size": int(entry.get("vector_size") or _DEFAULT_VECTOR_SIZE),
                        "distance": str(entry.get("distance") or _DEFAULT_DISTANCE).lower(),
                    }
                )
            if cleaned:
                return cleaned
    return [
        {
            "name": _DEFAULT_COLLECTION_NAME,
            "vector_size": _DEFAULT_VECTOR_SIZE,
            "distance": _DEFAULT_DISTANCE,
        }
    ]


def _qdrant_url() -> str:
    return os.environ.get("QDRANT_URL", "http://localhost:6333").rstrip("/")


def _init_qdrant(
    cap: Any, collections: list[dict[str, Any]], ctx: StepContext
) -> str:
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qm
    except ImportError as exc:
        raise _BootstrapSkip(
            'qdrant-client not installed — pip install "agent-scaffold-cli[vector]"'
        ) from exc
    url = _qdrant_url()
    api_key = os.environ.get("QDRANT_API_KEY") or None
    try:
        client = QdrantClient(url=url, api_key=api_key, timeout=int(_DEFAULT_TIMEOUT))
        existing = {c.name for c in client.get_collections().collections}
    except Exception as exc:  # noqa: BLE001 — surface any network/SDK error
        raise _BootstrapFail(f"qdrant: get_collections failed ({url}): {exc}") from exc
    _ = cap  # silence linter; cap is the resolved capability for future hooks
    distance_map = {
        "cosine": qm.Distance.COSINE,
        "l2": qm.Distance.EUCLID,
        "dot": qm.Distance.DOT,
    }
    created: list[str] = []
    for col in collections:
        if col["name"] in existing:
            continue
        distance = distance_map.get(col["distance"], qm.Distance.COSINE)
        try:
            client.create_collection(
                collection_name=col["name"],
                vectors_config=qm.VectorParams(size=col["vector_size"], distance=distance),
            )
        except Exception as exc:  # noqa: BLE001
            raise _BootstrapFail(
                f"qdrant: create_collection({col['name']!r}) failed: {exc}"
            ) from exc
        created.append(col["name"])
        ctx.emit(StepLog(step_id="bootstrap_vector_db", line=f"qdrant: created {col['name']}"))
    if not created:
        return f"all {len(collections)} collection(s) already exist"
    return f"created {len(created)} collection(s): {', '.join(created)}"


def _init_chroma(
    cap: Any, collections: list[dict[str, Any]], ctx: StepContext
) -> str:
    try:
        import httpx
    except ImportError as exc:
        raise _BootstrapSkip(
            'httpx not installed — pip install "agent-scaffold-cli[vector]"'
        ) from exc
    base = os.environ.get("CHROMA_URL", "http://localhost:8000").rstrip("/")
    _ = cap
    created: list[str] = []
    for col in collections:
        try:
            response = httpx.post(
                f"{base}/api/v1/collections",
                json={
                    "name": col["name"],
                    "metadata": {"hnsw:space": col["distance"]},
                },
                timeout=_DEFAULT_TIMEOUT,
            )
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            raise _BootstrapFail(f"chroma: cannot reach {base}: {exc}") from exc
        # Chroma returns 200 on create + 409 (or 500 with "already exists") on dupe.
        if response.status_code == 200:
            created.append(col["name"])
            ctx.emit(
                StepLog(step_id="bootstrap_vector_db", line=f"chroma: created {col['name']}")
            )
            continue
        body_text = response.text.lower()
        if response.status_code == 409 or "already exists" in body_text:
            continue
        raise _BootstrapFail(
            f"chroma: create({col['name']!r}) returned {response.status_code}",
            stderr_tail=response.text[:200],
        )
    if not created:
        return f"all {len(collections)} collection(s) already exist"
    return f"created {len(created)} collection(s): {', '.join(created)}"


def _init_pgvector(collections: list[dict[str, Any]], ctx: StepContext) -> str:
    try:
        import psycopg  # type: ignore[import-untyped]
    except ImportError as exc:
        raise _BootstrapSkip(
            'psycopg not installed — pip install "agent-scaffold-cli[vector]"'
        ) from exc
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        raise _BootstrapSkip("DATABASE_URL not set — pgvector needs Postgres connection")
    try:
        with psycopg.connect(database_url, autocommit=True) as conn, conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            ctx.emit(
                StepLog(step_id="bootstrap_vector_db", line="pgvector: extension ready")
            )
            created: list[str] = []
            for col in collections:
                # Postgres identifier quoting: collection name must already be a
                # safe identifier (alphanumeric + underscore). Reject otherwise.
                name = col["name"]
                if not name.replace("_", "").isalnum():
                    raise _BootstrapFail(
                        f"pgvector: collection name {name!r} is not a safe SQL identifier"
                    )
                cur.execute(
                    f'CREATE TABLE IF NOT EXISTS "{name}" ('
                    "  id BIGSERIAL PRIMARY KEY,"
                    f"  embedding vector({int(col['vector_size'])}),"
                    "  payload JSONB,"
                    "  created_at TIMESTAMPTZ DEFAULT NOW()"
                    ");"
                )
                created.append(name)
    except psycopg.Error as exc:
        raise _BootstrapFail(f"pgvector: SQL failed: {exc}") from exc
    return f"ensured {len(created)} table(s): {', '.join(created)}"


__all__ = ["BootstrapVectorDbStep"]
