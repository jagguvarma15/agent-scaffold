"""Tests for catalog Pydantic parsing of the additive 2026-SOTA recipe fields
and new capability kinds. Catalog stays on ``schema_version: 1``; the new
fields land as optional + the new kind strings land as additive enum values
(catalog ``kind`` is a ``str`` so unknown kinds also degrade gracefully on
older scaffold builds — that's the consumer-side forward-compat story).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import yaml

from agent_scaffold.capabilities import _KNOWN_KINDS, LAYER_ORDER, CapabilityKind
from agent_scaffold.catalog import (
    Catalog,
    MCPServerRef,
    RecipeEntry,
    SkillRef,
    load_catalog,
)


def _mock_response(body: str, etag: str | None = None):
    class _Resp:
        def __init__(self) -> None:
            self.headers = {"ETag": etag} if etag else {}
            self._body = body.encode("utf-8")

        def read(self) -> bytes:
            return self._body

        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *_: object) -> None:
            pass

    return _Resp()


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_recipe_entry_advanced_fields_round_trip() -> None:
    """RecipeEntry parses mcp_servers / skills / guardrails / sandbox / durable_workflow."""
    entry = RecipeEntry.model_validate(
        {
            "slug": "research-assistant",
            "path": "docs/recipes/research-assistant.md",
            "title": "Recipe: Research Assistant",
            "mcp_servers": [
                {
                    "id": "tavily",
                    "capability": "mcp.tavily",
                    "transport": "streamable_http",
                    "env": {"TAVILY_API_KEY": "required"},
                }
            ],
            "skills": [
                {
                    "id": "web-search-loop",
                    "path": "skills/web-search-loop/SKILL.md",
                    "triggers": ["research", "investigate"],
                }
            ],
            "guardrails": ["guardrail.llama-guard"],
            "sandbox": "sandbox.e2b",
            "durable_workflow": "durable.temporal",
        }
    )
    assert entry.mcp_servers == [
        MCPServerRef(
            id="tavily",
            capability="mcp.tavily",
            transport="streamable_http",
            env={"TAVILY_API_KEY": "required"},
        )
    ]
    assert entry.skills == [
        SkillRef(
            id="web-search-loop",
            path="skills/web-search-loop/SKILL.md",
            triggers=["research", "investigate"],
        )
    ]
    assert entry.guardrails == ["guardrail.llama-guard"]
    assert entry.sandbox == "sandbox.e2b"
    assert entry.durable_workflow == "durable.temporal"


def test_recipe_entry_defaults_empty_when_fields_absent() -> None:
    """RecipeEntry parses cleanly with none of the new fields present."""
    entry = RecipeEntry.model_validate(
        {
            "slug": "minimal",
            "path": "docs/recipes/minimal.md",
            "title": "Minimal",
        }
    )
    assert entry.mcp_servers == []
    assert entry.skills == []
    assert entry.guardrails == []
    assert entry.sandbox is None
    assert entry.durable_workflow is None


def test_mcp_server_ref_transport_default_is_stdio() -> None:
    ref = MCPServerRef.model_validate({"id": "filesystem", "capability": "mcp.filesystem"})
    assert ref.transport == "stdio"
    assert ref.env == {}


def test_skill_ref_triggers_default_empty() -> None:
    ref = SkillRef.model_validate({"id": "my-skill", "path": "skills/my-skill/SKILL.md"})
    assert ref.triggers == []


def test_recipe_entry_ignores_unknown_fields() -> None:
    """The ``extra='ignore'`` ConfigDict means unknown fields pass through silently."""
    entry = RecipeEntry.model_validate(
        {
            "slug": "future",
            "path": "docs/recipes/future.md",
            "title": "Future",
            "tool_runtime": "browserbase",  # future kind we haven't modeled
            "mystery_field": [1, 2, 3],
        }
    )
    # No error raised; the field just doesn't appear on the model.
    assert not hasattr(entry, "tool_runtime")


def test_catalog_load_with_advanced_recipe_fields(tmp_path: Path) -> None:
    """Full catalog round-trip parses a recipes[] block carrying advanced fields."""
    fixture_body = (FIXTURE_DIR / "catalog_minimal.yaml").read_text(encoding="utf-8")
    data = yaml.safe_load(fixture_body)
    data["recipes"].append(
        {
            "slug": "advanced",
            "path": "docs/recipes/advanced.md",
            "title": "Advanced",
            "mcp_servers": [
                {"id": "tavily", "capability": "mcp.tavily", "transport": "streamable_http"}
            ],
            "skills": [{"id": "loop", "path": "skills/loop/SKILL.md"}],
            "guardrails": ["guardrail.llama-guard"],
            "sandbox": "sandbox.e2b",
            "durable_workflow": "durable.temporal",
        }
    )
    body = yaml.dump(data, sort_keys=False)

    with patch("urllib.request.urlopen", return_value=_mock_response(body)):
        catalog = load_catalog(url="https://example.com/c.yaml", cache_dir=tmp_path)

    assert isinstance(catalog, Catalog)
    advanced = next(r for r in catalog.recipes if r.slug == "advanced")
    assert advanced.mcp_servers[0].transport == "streamable_http"
    assert advanced.sandbox == "sandbox.e2b"


def test_catalog_doc_indexes_accept_tagged_mapping_entries(tmp_path: Path) -> None:
    """Catalog generator 1.3+ publishes stack/cross-cutting/pattern doc indexes
    as ``{path, tags, when_to_load}`` mappings instead of bare path strings.
    Both shapes must load; mapping entries normalize to their path."""
    fixture_body = (FIXTURE_DIR / "catalog_minimal.yaml").read_text(encoding="utf-8")
    data = yaml.safe_load(fixture_body)
    data["stack"] = [
        "docs/stack/legacy-plain-string.md",
        {
            "path": "docs/stack/api-fastapi.md",
            "tags": ["python", "web-api"],
            "when_to_load": "recipe.language == 'python'",
        },
        {"tags": ["orphan-without-path"]},  # unusable → dropped, not fatal
    ]
    data["cross_cutting_docs"] = [{"path": "docs/cross-cutting/auth.md", "tags": ["auth"]}]
    body = yaml.dump(data, sort_keys=False)

    with patch("urllib.request.urlopen", return_value=_mock_response(body)):
        catalog = load_catalog(url="https://example.com/c.yaml", cache_dir=tmp_path)

    assert catalog.stack == [
        "docs/stack/legacy-plain-string.md",
        "docs/stack/api-fastapi.md",
    ]
    assert catalog.cross_cutting_docs == ["docs/cross-cutting/auth.md"]


def test_catalog_load_with_new_capability_kinds(tmp_path: Path) -> None:
    """Catalog parses capabilities[] entries with the 8 new kind strings."""
    fixture_body = (FIXTURE_DIR / "catalog_minimal.yaml").read_text(encoding="utf-8")
    data = yaml.safe_load(fixture_body)
    new_kind_capabilities = [
        {"id": "mcp.tavily", "kind": "mcp", "path": "docs/capabilities/mcp/tavily.md"},
        {"id": "sandbox.e2b", "kind": "sandbox", "path": "docs/capabilities/sandbox/e2b.md"},
        {
            "id": "durable.temporal",
            "kind": "durable",
            "path": "docs/capabilities/durable/temporal.md",
        },
        {
            "id": "memory_store.mem0",
            "kind": "memory_store",
            "path": "docs/capabilities/memory_store/mem0.md",
        },
        {
            "id": "guardrail.llama-guard",
            "kind": "guardrail",
            "path": "docs/capabilities/guardrail/llama-guard.md",
        },
        {
            "id": "embedding.voyage",
            "kind": "embedding",
            "path": "docs/capabilities/embedding/voyage.md",
        },
        {
            "id": "live_data.tavily",
            "kind": "live_data",
            "path": "docs/capabilities/live_data/tavily.md",
        },
        {
            "id": "rerank.cohere-rerank",
            "kind": "rerank",
            "path": "docs/capabilities/rerank/cohere-rerank.md",
        },
    ]
    data["capabilities"].extend(new_kind_capabilities)
    body = yaml.dump(data, sort_keys=False)

    with patch("urllib.request.urlopen", return_value=_mock_response(body)):
        catalog = load_catalog(url="https://example.com/c.yaml", cache_dir=tmp_path)

    kinds_present = {c.kind for c in catalog.capabilities}
    for kind in (
        "mcp",
        "sandbox",
        "durable",
        "memory_store",
        "guardrail",
        "embedding",
        "live_data",
        "rerank",
    ):
        assert kind in kinds_present


def test_layer_order_covers_all_known_kinds() -> None:
    """``LAYER_ORDER`` must enumerate every entry in ``_KNOWN_KINDS`` exactly once."""
    assert set(LAYER_ORDER) == _KNOWN_KINDS
    assert len(LAYER_ORDER) == len(_KNOWN_KINDS)


def test_capability_kind_literal_matches_known_kinds() -> None:
    """``CapabilityKind`` Literal members must match ``_KNOWN_KINDS`` exactly."""
    import typing

    literal_members = set(typing.get_args(CapabilityKind))
    assert literal_members == _KNOWN_KINDS
