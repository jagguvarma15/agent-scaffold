"""Tests for the stack-option derivation layer."""

from __future__ import annotations

from typing import Any

from agent_scaffold.catalog import Catalog
from agent_scaffold.stack_options import (
    MODE_CLOUD,
    MODE_INTERNAL,
    MODE_INTERNAL_OVERRIDABLE,
    OVERRIDABLE_URL_VARS,
    derive_stack_options,
    missing_credentials,
    option_by_id,
    service_for_option,
)


def _catalog(capabilities: list[dict[str, Any]]) -> Catalog:
    return Catalog.model_validate(
        {
            "schema_version": 1,
            "blueprints": {"repo": "example/blueprints", "branch": "main"},
            "capabilities": capabilities,
        }
    )


CAPS = [
    {
        "id": "cache.redis",
        "kind": "cache",
        "path": "docs/capabilities/cache/redis.md",
        "env_vars": ["REDIS_URL"],
        "docker_service": "redis",
        "probe": "redis_ping",
        "card": {"name": "Redis", "description": "In-memory store."},
        "verification": {"tier": "T2", "delivery": "self-hosted"},
    },
    {
        "id": "queue.redis-streams",
        "kind": "queue",
        "path": "docs/capabilities/queue/redis-streams.md",
        "env_vars": ["REDIS_URL"],
        "probe": "redis_ping",
        "card": {"name": "Redis Streams", "description": "Queue on Redis."},
    },
    {
        "id": "relational.postgres",
        "kind": "relational",
        "path": "docs/capabilities/relational/postgres.md",
        "env_vars": ["DATABASE_URL", "POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"],
        "docker_service": "postgres",
        "probe": "postgres_select_one",
        "card": {"name": "Postgres", "description": "Relational database."},
        "verification": {"tier": "T2", "delivery": "self-hosted"},
    },
    {
        "id": "obs.langsmith",
        "kind": "obs",
        "path": "docs/capabilities/obs/langsmith.md",
        "env_vars": [
            "LANGCHAIN_API_KEY",
            "LANGCHAIN_TRACING_V2",
            "LANGCHAIN_PROJECT",
            "LANGCHAIN_ENDPOINT",
        ],
        "bootstrap_step": "bootstrap_langsmith",
        "probe": "langsmith_workspace",
        "card": {
            "name": "LangSmith",
            "description": "Hosted LLM observability.",
            "required_credentials": ["LANGCHAIN_API_KEY"],
        },
        "verification": {"tier": "T1", "delivery": "managed"},
    },
    {
        "id": "obs.langfuse",
        "kind": "obs",
        "path": "docs/capabilities/obs/langfuse.md",
        "env_vars": ["LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"],
        "docker_service": "langfuse",
        "bootstrap_step": "bootstrap_langfuse",
        "probe": "langfuse_health",
        "card": {
            "name": "Langfuse",
            "description": "LLM observability.",
            "required_credentials": ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"],
        },
    },
    {
        "id": "vector_db.qdrant",
        "kind": "vector_db",
        "path": "docs/capabilities/vector_db/qdrant.md",
        "env_vars": ["QDRANT_URL", "QDRANT_API_KEY"],
        "docker_service": "qdrant",
        "bootstrap_step": "bootstrap_vector_db",
        "probe": "qdrant_collections",
        "card": {"name": "Qdrant", "description": "Vector database."},
    },
    {
        "id": "live_data.tavily",
        "kind": "live_data",
        "path": "docs/capabilities/live_data/tavily.md",
        "env_vars": ["TAVILY_API_KEY"],
        "probe": "tavily_search_ping",
        "card": {
            "name": "Tavily",
            "description": "Web search API.",
            "required_credentials": ["TAVILY_API_KEY"],
        },
    },
    {
        "id": "frontend.minimal-chat",
        "kind": "frontend",
        "path": "docs/capabilities/frontend/minimal-chat.md",
        "env_vars": ["VITE_AGENT_URL"],
        "card": {"name": "Minimal Chat UI", "description": "Chat frontend."},
    },
    {
        "id": "auth.key-bootstrap",
        "kind": "auth",
        "path": "docs/capabilities/auth/key-bootstrap.md",
        "card": {"name": "Key bootstrap", "description": "Runtime key form."},
    },
    {
        "id": "eval.promptfoo",
        "kind": "eval",
        "path": "docs/capabilities/eval/promptfoo.md",
        "env_vars": ["ANTHROPIC_API_KEY"],
        "card": {
            "name": "Promptfoo",
            "description": "Eval harness.",
            "required_credentials": ["ANTHROPIC_API_KEY"],
        },
    },
]


def test_delivery_from_catalog_drives_mode() -> None:
    catalog = _catalog(CAPS)
    options = derive_stack_options(["cache.redis", "obs.langsmith", "relational.postgres"], catalog)
    by_id = {o.id: o for o in options}
    assert by_id["redis"].mode == MODE_INTERNAL_OVERRIDABLE
    assert by_id["postgres"].mode == MODE_INTERNAL_OVERRIDABLE
    assert by_id["langsmith"].mode == MODE_CLOUD
    assert by_id["langsmith"].cloud_capable
    assert by_id["redis"].cloud_capable


def test_delivery_inference_without_verification() -> None:
    """No verification block: docker_service means self-hosted, else managed."""
    catalog = _catalog(
        [
            {
                "id": "vector_db.qdrant",
                "kind": "vector_db",
                "path": "p",
                "env_vars": ["QDRANT_URL"],
                "docker_service": "qdrant",
                "probe": "qdrant_collections",
            },
            {
                "id": "obs.langsmith",
                "kind": "obs",
                "path": "p",
                "env_vars": ["LANGCHAIN_API_KEY"],
                "probe": "langsmith_workspace",
            },
        ]
    )
    options = derive_stack_options(["vector_db.qdrant", "obs.langsmith"], catalog)
    by_id = {o.id: o for o in options}
    assert by_id["qdrant"].mode == MODE_INTERNAL_OVERRIDABLE
    assert by_id["langsmith"].mode == MODE_CLOUD


def test_redis_and_redis_streams_collapse_into_one_option() -> None:
    catalog = _catalog(CAPS)
    options = derive_stack_options(["cache.redis", "queue.redis-streams"], catalog)
    assert len(options) == 1
    option = options[0]
    assert option.id == "redis"
    assert option.capability_ids == frozenset({"cache.redis", "queue.redis-streams"})
    assert option.docker_service == "redis"
    assert option.probe == "redis_ping"


def test_excluded_kinds_and_generation_time_caps_are_skipped() -> None:
    catalog = _catalog(CAPS)
    options = derive_stack_options(
        ["frontend.minimal-chat", "auth.key-bootstrap", "eval.promptfoo", "cache.redis"],
        catalog,
    )
    assert [o.id for o in options] == ["redis"]


def test_unknown_capability_degrades_to_minimal_internal() -> None:
    catalog = _catalog(CAPS)
    options = derive_stack_options(["memory_store.zep"], catalog)
    assert len(options) == 1
    option = options[0]
    assert option.id == "zep"
    assert option.mode == MODE_INTERNAL
    assert option.kind == "unknown"
    assert option.probe is None
    assert option.credentials == ()


def test_credential_specs_per_provider() -> None:
    catalog = _catalog(CAPS)
    options = derive_stack_options(
        ["obs.langsmith", "obs.langfuse", "vector_db.qdrant", "relational.postgres"],
        catalog,
    )
    by_id = {o.id: o for o in options}
    assert [c.var for c in by_id["langsmith"].credentials] == ["LANGCHAIN_API_KEY"]
    assert [c.var for c in by_id["langfuse"].credentials] == [
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
    ]
    langfuse = {c.var: c for c in by_id["langfuse"].credentials}
    assert langfuse["LANGFUSE_SECRET_KEY"].secret is True
    assert langfuse["LANGFUSE_PUBLIC_KEY"].secret is False
    assert langfuse["LANGFUSE_HOST"].optional is True
    qdrant = {c.var: c for c in by_id["qdrant"].credentials}
    assert qdrant["QDRANT_URL"].secret is False
    assert qdrant["QDRANT_API_KEY"].optional is True
    assert [c.var for c in by_id["postgres"].credentials] == ["DATABASE_URL"]


def test_generic_cloud_provider_uses_declared_credentials() -> None:
    """A provider without a details table entry gets card credentials."""
    catalog = _catalog(CAPS)
    options = derive_stack_options(["live_data.tavily"], catalog)
    assert len(options) == 1
    option = options[0]
    assert option.id == "tavily"
    assert option.mode == MODE_CLOUD
    assert [c.var for c in option.credentials] == ["TAVILY_API_KEY"]
    assert option.credentials[0].secret is True


def test_service_for_option_bridge() -> None:
    catalog = _catalog(CAPS)
    option = derive_stack_options(["cache.redis"], catalog)[0]
    service = service_for_option(option)
    assert service.id == "redis"
    assert service.required is False
    assert service.env_vars == ["REDIS_URL"]
    assert service.default_local == "redis://localhost:6379"
    assert service.docker_service == "redis"
    assert service.probe == "redis_ping"


def test_option_by_id_and_missing_credentials() -> None:
    catalog = _catalog(CAPS)
    options = derive_stack_options(["obs.langsmith", "cache.redis"], catalog)
    langsmith = option_by_id(options, "langsmith")
    assert langsmith is not None
    assert option_by_id(options, "nope") is None
    assert missing_credentials(langsmith, {}) == ["LANGCHAIN_API_KEY"]
    assert missing_credentials(langsmith, {"LANGCHAIN_API_KEY": "lsv2_x"}) == []
    assert missing_credentials(langsmith, {"LANGCHAIN_API_KEY": "   "}) == ["LANGCHAIN_API_KEY"]


def test_overridable_set_covers_the_documented_swaps() -> None:
    assert {"REDIS_URL", "DATABASE_URL", "QDRANT_URL", "LANGFUSE_HOST"} <= set(OVERRIDABLE_URL_VARS)


def test_manifest_order_is_preserved() -> None:
    catalog = _catalog(CAPS)
    options = derive_stack_options(["relational.postgres", "cache.redis", "obs.langsmith"], catalog)
    assert [o.id for o in options] == ["postgres", "redis", "langsmith"]
