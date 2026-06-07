"""Tests for :mod:`agent_scaffold.catalog`.

Covers the public surface: load_catalog's resolution + cache + fallback chain,
schema-version enforcement, malformed-input refusal, min_alias_length safety
knob, and the derived-view helpers (alias_lookup, cross_cutting_lookup,
build_secondary_url_re, framework_doc_paths).

HTTP is mocked at urlopen so tests don't touch the network. The fixture
catalog at ``tests/fixtures/catalog_minimal.yaml`` provides a stable input
shape — its content is deliberately small (one of everything) so test
expectations stay readable.
"""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from agent_scaffold.catalog import (
    DEFAULT_CATALOG_URL,
    SCAFFOLD_CATALOG_SCHEMA_VERSION_MAX,
    Catalog,
    CatalogSchemaError,
    CatalogUnavailable,
    CatalogVersionTooHigh,
    alias_lookup,
    build_secondary_url_re,
    cross_cutting_lookup,
    framework_doc_paths,
    load_catalog,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "catalog_minimal.yaml"


def _fixture_text() -> str:
    return FIXTURE_PATH.read_text(encoding="utf-8")


def _mock_response(body: str, etag: str | None = None, status: int = 200):
    """Return a context-manager mock that mimics ``urlopen``'s return."""

    class _Resp:
        def __init__(self) -> None:
            self.headers = {"ETag": etag} if etag else {}
            self._body = body.encode("utf-8")
            self.status = status

        def read(self) -> bytes:
            return self._body

        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *_: Any) -> None:
            pass

    return _Resp()


# ---------------------------------------------------------------------------
# Fetch + cache + fallback
# ---------------------------------------------------------------------------


def test_load_catalog_happy_path(tmp_path: Path) -> None:
    """Fresh fetch → parse → return Catalog with all sections populated."""
    body = _fixture_text()
    with patch("urllib.request.urlopen", return_value=_mock_response(body, etag='"abc123"')):
        catalog = load_catalog(url="https://example.com/c.yaml", cache_dir=tmp_path)

    assert isinstance(catalog, Catalog)
    assert catalog.schema_version == 1
    assert catalog.blueprints.repo == "jagguvarma15/agent-blueprints"
    assert len(catalog.recipes) == 1
    assert catalog.recipes[0].slug == "docs-rag-qa"
    assert "react" in catalog.aliases


def test_load_catalog_writes_cache(tmp_path: Path) -> None:
    """Successful fetch persists the body + ETag for the next call."""
    body = _fixture_text()
    url = "https://example.com/c.yaml"
    with patch("urllib.request.urlopen", return_value=_mock_response(body, etag='"v1"')):
        load_catalog(url=url, cache_dir=tmp_path)

    # The cache dir layout is owned by catalog.py — assert via the public
    # behavior: cached files should now exist under cache_dir/catalog/.
    cache_files = list((tmp_path / "catalog").iterdir())
    assert any(f.suffix == ".yaml" for f in cache_files)
    assert any(f.suffix == ".etag" for f in cache_files)


def test_load_catalog_falls_back_to_cache_on_network_error(tmp_path: Path) -> None:
    """First call seeds cache; second call with network down uses cached body."""
    body = _fixture_text()
    url = "https://example.com/c.yaml"

    # Seed the cache.
    with patch("urllib.request.urlopen", return_value=_mock_response(body, etag='"v1"')):
        load_catalog(url=url, cache_dir=tmp_path)

    # Simulate network failure.
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("offline")):
        catalog = load_catalog(url=url, cache_dir=tmp_path)
    assert catalog.recipes[0].slug == "docs-rag-qa"


def test_load_catalog_falls_back_to_embedded(tmp_path: Path) -> None:
    """No cache + network failure → embedded JSON fallback."""
    url = "https://example.com/c.yaml"
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("offline")):
        catalog = load_catalog(url=url, cache_dir=tmp_path)
    # The embedded catalog ships in the wheel; just confirm we got a valid
    # Catalog (specific content depends on what was baked at build time).
    assert isinstance(catalog, Catalog)
    assert catalog.schema_version <= SCAFFOLD_CATALOG_SCHEMA_VERSION_MAX


def test_load_catalog_handles_304_with_cache(tmp_path: Path) -> None:
    """HTTP 304 + a cached body → serve from cache."""
    body = _fixture_text()
    url = "https://example.com/c.yaml"

    with patch("urllib.request.urlopen", return_value=_mock_response(body, etag='"v1"')):
        load_catalog(url=url, cache_dir=tmp_path)

    http_304 = urllib.error.HTTPError(url, 304, "Not Modified", {}, io.BytesIO(b""))
    with patch("urllib.request.urlopen", side_effect=http_304):
        catalog = load_catalog(url=url, cache_dir=tmp_path)
    assert catalog.recipes[0].slug == "docs-rag-qa"


def test_load_catalog_resolves_env_var(tmp_path: Path) -> None:
    """``$AGENT_SCAFFOLD_CATALOG_URL`` overrides the default URL."""
    body = _fixture_text()
    custom = "https://fork.example.com/catalog.yaml"

    captured: dict[str, str] = {}

    def _fake_urlopen(req, **_):
        captured["url"] = req.get_full_url()
        return _mock_response(body)

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        load_catalog(url=None, cache_dir=tmp_path, env={"AGENT_SCAFFOLD_CATALOG_URL": custom})
    assert captured["url"] == custom


def test_load_catalog_defaults_to_default_url(tmp_path: Path) -> None:
    body = _fixture_text()
    captured: dict[str, str] = {}

    def _fake_urlopen(req, **_):
        captured["url"] = req.get_full_url()
        return _mock_response(body)

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        load_catalog(url=None, cache_dir=tmp_path, env={})
    assert captured["url"] == DEFAULT_CATALOG_URL


def test_load_catalog_file_url(tmp_path: Path) -> None:
    """file:// URLs are read directly, no network call."""
    catalog = load_catalog(url=f"file://{FIXTURE_PATH}", cache_dir=tmp_path)
    assert catalog.recipes[0].slug == "docs-rag-qa"


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_load_catalog_refuses_higher_schema_version(tmp_path: Path) -> None:
    """``schema_version`` above our max raises CatalogVersionTooHigh."""
    data = yaml.safe_load(_fixture_text())
    data["schema_version"] = SCAFFOLD_CATALOG_SCHEMA_VERSION_MAX + 99
    body = yaml.safe_dump(data)
    with patch("urllib.request.urlopen", return_value=_mock_response(body)):
        with pytest.raises(CatalogVersionTooHigh) as exc:
            load_catalog(url="https://example.com/c.yaml", cache_dir=tmp_path)
    assert exc.value.got == SCAFFOLD_CATALOG_SCHEMA_VERSION_MAX + 99
    assert exc.value.max_supported == SCAFFOLD_CATALOG_SCHEMA_VERSION_MAX


def test_load_catalog_refuses_malformed_yaml(tmp_path: Path) -> None:
    body = "this: is: not: valid: yaml: [unclosed"
    with patch("urllib.request.urlopen", return_value=_mock_response(body)):
        with pytest.raises(CatalogSchemaError):
            load_catalog(url="https://example.com/c.yaml", cache_dir=tmp_path)


def test_load_catalog_refuses_non_mapping_body(tmp_path: Path) -> None:
    body = "- list\n- not\n- a\n- mapping\n"
    with patch("urllib.request.urlopen", return_value=_mock_response(body)):
        with pytest.raises(CatalogSchemaError):
            load_catalog(url="https://example.com/c.yaml", cache_dir=tmp_path)


def test_load_catalog_raises_when_all_fallbacks_fail(tmp_path: Path) -> None:
    """No cache, no embedded available, and network down → CatalogUnavailable."""
    # Force the embedded reader to return None too.
    with (
        patch("urllib.request.urlopen", side_effect=urllib.error.URLError("offline")),
        patch("agent_scaffold.catalog._read_embedded", return_value=None),
    ):
        with pytest.raises(CatalogUnavailable):
            load_catalog(url="https://example.com/c.yaml", cache_dir=tmp_path)


# ---------------------------------------------------------------------------
# min_alias_length safety knob
# ---------------------------------------------------------------------------


def test_min_alias_length_drops_short_aliases(tmp_path: Path) -> None:
    """The fixture includes ``ab: ...`` which should drop at min_alias_length=3."""
    catalog = load_catalog(url=f"file://{FIXTURE_PATH}", cache_dir=tmp_path)
    assert "ab" not in catalog.aliases
    assert "react" in catalog.aliases


# ---------------------------------------------------------------------------
# Derived views
# ---------------------------------------------------------------------------


@pytest.fixture()
def catalog(tmp_path: Path) -> Catalog:
    return load_catalog(url=f"file://{FIXTURE_PATH}", cache_dir=tmp_path)


def test_alias_lookup_matches_word_boundaries(catalog: Catalog) -> None:
    hits = alias_lookup(catalog, "We use the ReAct loop with Qdrant for retrieval.")
    keys = [k for k, _ in hits]
    assert "react" in keys
    assert "qdrant" in keys


def test_alias_lookup_skips_substring_inside_word(catalog: Catalog) -> None:
    # 'react' should match the "ReAct" mention via word boundaries, but NOT
    # the "reactor" substring. Confirm by asserting against a text with only
    # 'reactor' (no standalone "ReAct").
    text_only_reactor = alias_lookup(catalog, "the reactor pattern")
    assert not any(k == "react" for k, _ in text_only_reactor)
    # And confirm the word-boundary positive case fires when "ReAct" stands alone.
    text_with_react = alias_lookup(catalog, "we use the ReAct loop")
    assert any(k == "react" for k, _ in text_with_react)


def test_cross_cutting_lookup_matches_categories(catalog: Catalog) -> None:
    hits = cross_cutting_lookup(catalog, "Use JWT auth and structured logging.")
    keys = [k for k, _ in hits]
    assert "auth" in keys
    assert "logging" in keys


def test_framework_doc_paths_includes_language(catalog: Catalog) -> None:
    paths = framework_doc_paths(catalog)
    assert paths["docs/frameworks/pydantic-ai.md"] == {
        "id": "pydantic_ai",
        "language": "python",
    }
    assert paths["docs/frameworks/vercel-ai-sdk.md"]["language"] == "typescript"


def test_build_secondary_url_re_matches_tree(catalog: Catalog) -> None:
    pattern = build_secondary_url_re(catalog)
    match = pattern.match(
        "https://github.com/jagguvarma15/agent-blueprints/tree/main/patterns/react"
    )
    assert match is not None
    assert match.group("path") == "patterns/react"


def test_build_secondary_url_re_matches_blob(catalog: Catalog) -> None:
    pattern = build_secondary_url_re(catalog)
    match = pattern.match(
        "https://github.com/jagguvarma15/agent-blueprints/blob/main/patterns/react/design.md"
    )
    assert match is not None
    assert match.group("path") == "patterns/react/design.md"


def test_build_secondary_url_re_rejects_wrong_repo(catalog: Catalog) -> None:
    pattern = build_secondary_url_re(catalog)
    assert pattern.match("https://github.com/other-owner/agent-blueprints/tree/main/x") is None


# ---------------------------------------------------------------------------
# Smoke test: the catalog round-trips through Pydantic without losing data.
# ---------------------------------------------------------------------------


def test_catalog_round_trip_via_json(tmp_path: Path) -> None:
    """JSON-serialize the loaded catalog and reload — same Catalog."""
    catalog = load_catalog(url=f"file://{FIXTURE_PATH}", cache_dir=tmp_path)
    as_json = catalog.model_dump_json()
    reparsed = Catalog.model_validate(json.loads(as_json))
    assert reparsed.recipes[0].slug == catalog.recipes[0].slug
    assert reparsed.blueprints.repo == catalog.blueprints.repo
