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
import os
import time
import urllib.error
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml
from pydantic import ValidationError

from agent_scaffold.catalog import (
    DEFAULT_CATALOG_URL,
    SCAFFOLD_CATALOG_SCHEMA_VERSION_MAX,
    CapabilityCard,
    CapabilityEntry,
    Catalog,
    CatalogSchemaError,
    CatalogUnavailable,
    CatalogURLError,
    CatalogVersionTooHigh,
    EnvContractEntry,
    _reset_catalog_memo,
    alias_lookup,
    build_secondary_url_re,
    cross_cutting_lookup,
    framework_doc_paths,
    load_catalog,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "catalog_minimal.yaml"


def _fixture_text() -> str:
    return FIXTURE_PATH.read_text(encoding="utf-8")


def _age_cache(cache_dir: Path, seconds: float = 3600.0) -> None:
    """Backdate the cached catalog files past the freshness TTL so the next
    load attempts a real fetch instead of serving the fresh cache. The
    in-process memo shares the same TTL; it cannot be backdated via mtime,
    so expire it explicitly — production expiry is equivalent."""
    stamp = time.time() - seconds
    for f in (cache_dir / "catalog").iterdir():
        os.utime(f, (stamp, stamp))
    _reset_catalog_memo()


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

    # Seed the cache, then age it past the TTL so the fetch really runs.
    with patch("urllib.request.urlopen", return_value=_mock_response(body, etag='"v1"')):
        load_catalog(url=url, cache_dir=tmp_path)
    _age_cache(tmp_path)

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


def test_cached_fallback_warning_prints_once_per_process(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """load_catalog runs several times per command; the offline-fallback
    warning must not repeat for the same URL + error."""
    body = _fixture_text()
    url = "https://example.com/c.yaml"
    with patch("urllib.request.urlopen", return_value=_mock_response(body, etag='"v1"')):
        load_catalog(url=url, cache_dir=tmp_path)
    _age_cache(tmp_path)

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("offline")):
        load_catalog(url=url, cache_dir=tmp_path)
        load_catalog(url=url, cache_dir=tmp_path)
    assert capsys.readouterr().err.count("using cached catalog") == 1


def test_embedded_fallback_warning_prints_once_per_process(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    url = "https://example.com/c.yaml"
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("offline")):
        load_catalog(url=url, cache_dir=tmp_path)
        load_catalog(url=url, cache_dir=tmp_path)
    # Second call serves from the cache written by the first embedded load (or
    # re-falls back); either way the embedded warning must appear exactly once.
    assert capsys.readouterr().err.count("using embedded catalog fallback") == 1


def test_load_catalog_handles_304_with_cache(tmp_path: Path) -> None:
    """HTTP 304 + a cached body → serve from cache."""
    body = _fixture_text()
    url = "https://example.com/c.yaml"

    with patch("urllib.request.urlopen", return_value=_mock_response(body, etag='"v1"')):
        load_catalog(url=url, cache_dir=tmp_path)
    _age_cache(tmp_path)

    http_304 = urllib.error.HTTPError(url, 304, "Not Modified", {}, io.BytesIO(b""))
    with patch("urllib.request.urlopen", side_effect=http_304):
        catalog = load_catalog(url=url, cache_dir=tmp_path)
    assert catalog.recipes[0].slug == "docs-rag-qa"


def test_fetch_retries_transient_urlerror(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One transient URLError then success — no warning, no stale cache."""
    body = _fixture_text()
    url = "https://example.com/c.yaml"
    with patch(
        "urllib.request.urlopen",
        side_effect=[urllib.error.URLError("reset"), _mock_response(body)],
    ) as mock_open:
        catalog = load_catalog(url=url, cache_dir=tmp_path)
    assert mock_open.call_count == 2
    assert catalog.recipes[0].slug == "docs-rag-qa"
    err = capsys.readouterr().err
    assert "using cached catalog" not in err
    assert "using embedded catalog" not in err


def test_fetch_does_not_retry_http_404(tmp_path: Path) -> None:
    """Deterministic 4xx fails immediately — one attempt, then the fallback chain."""
    url = "https://example.com/c.yaml"
    http_404 = urllib.error.HTTPError(url, 404, "Not Found", {}, io.BytesIO(b""))
    with patch("urllib.request.urlopen", side_effect=http_404) as mock_open:
        catalog = load_catalog(url=url, cache_dir=tmp_path)  # embedded fallback
    assert mock_open.call_count == 1
    assert isinstance(catalog, Catalog)


def test_fetch_exhausts_retries_then_falls_back_to_cache(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every attempt fails — stale cache serves, warning printed once."""
    body = _fixture_text()
    url = "https://example.com/c.yaml"
    with patch("urllib.request.urlopen", return_value=_mock_response(body, etag='"v1"')):
        load_catalog(url=url, cache_dir=tmp_path)
    _age_cache(tmp_path)

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("offline")) as mock_open:
        catalog = load_catalog(url=url, cache_dir=tmp_path)
    assert mock_open.call_count == 2  # FETCH_ATTEMPTS
    assert catalog.recipes[0].slug == "docs-rag-qa"
    assert capsys.readouterr().err.count("using cached catalog") == 1


def test_load_catalog_uses_fresh_cache_without_network(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A cache younger than the TTL serves directly — no network, no warning."""
    body = _fixture_text()
    url = "https://example.com/c.yaml"
    with patch("urllib.request.urlopen", return_value=_mock_response(body, etag='"v1"')):
        load_catalog(url=url, cache_dir=tmp_path)

    with patch("urllib.request.urlopen", side_effect=AssertionError("network hit")):
        catalog = load_catalog(url=url, cache_dir=tmp_path)
    assert catalog.recipes[0].slug == "docs-rag-qa"
    err = capsys.readouterr().err
    assert "using cached catalog" not in err
    assert "using embedded catalog" not in err


def test_load_catalog_refetches_when_cache_stale(tmp_path: Path) -> None:
    """A cache older than the TTL goes back to the network."""
    body = _fixture_text()
    url = "https://example.com/c.yaml"
    with patch("urllib.request.urlopen", return_value=_mock_response(body, etag='"v1"')):
        load_catalog(url=url, cache_dir=tmp_path)
    _age_cache(tmp_path)

    with patch("urllib.request.urlopen", return_value=_mock_response(body)) as mock_open:
        load_catalog(url=url, cache_dir=tmp_path)
    assert mock_open.call_count == 1


def test_304_refreshes_freshness_ttl(tmp_path: Path) -> None:
    """A conditional hit restarts the TTL: the next load is network-free."""
    body = _fixture_text()
    url = "https://example.com/c.yaml"
    with patch("urllib.request.urlopen", return_value=_mock_response(body, etag='"v1"')):
        load_catalog(url=url, cache_dir=tmp_path)
    _age_cache(tmp_path)

    http_304 = urllib.error.HTTPError(url, 304, "Not Modified", {}, io.BytesIO(b""))
    with patch("urllib.request.urlopen", side_effect=http_304):
        load_catalog(url=url, cache_dir=tmp_path)

    with patch("urllib.request.urlopen", side_effect=AssertionError("network hit")):
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


def test_load_catalog_rejects_plain_http(tmp_path: Path) -> None:
    """Plain http (and any non-https scheme) is refused before any fetch."""
    with patch("urllib.request.urlopen", side_effect=AssertionError("network hit")):
        for url in ("http://example.com/catalog.yaml", "ftp://example.com/catalog.yaml"):
            with pytest.raises(CatalogURLError, match="https"):
                load_catalog(url=url, cache_dir=tmp_path)


def test_load_catalog_rejects_http_via_env_var(tmp_path: Path) -> None:
    with pytest.raises(CatalogURLError):
        load_catalog(
            url=None,
            cache_dir=tmp_path,
            env={"AGENT_SCAFFOLD_CATALOG_URL": "http://fork.example.com/catalog.yaml"},
        )


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
# Capability entry/card modeling — typed surface, additive-forward-compatible
# ---------------------------------------------------------------------------


def test_capability_entry_models_full_published_key_set() -> None:
    """Every key the deployments catalog publishes on capabilities[] parses into
    a typed field (instead of being dropped)."""
    entry = CapabilityEntry(
        id="cache.redis",
        kind="cache",
        path="docs/capabilities/cache/redis.md",
        env_vars=["REDIS_URL"],
        docker_service="redis",
        probe="redis_ping",
        layer="infrastructure",
        requires=[],
        bootstrap_inputs={},
        card={"name": "Redis", "description": "in-memory store"},
        cost_tier="free",
        est_tokens=650,
        provisioning_time="instant",
        when_to_load="recipe declares cache.redis",
        tags=["cache"],
    )
    assert entry.kind == "cache"
    assert entry.card is not None and entry.card.name == "Redis"


def test_capability_card_requires_name_and_description() -> None:
    # name + description are producer-guaranteed (generate_catalog hard-enforces
    # them), so the consumer mirrors that requirement.
    CapabilityCard(name="Redis", description="in-memory store")  # ok
    with pytest.raises(ValidationError):
        CapabilityCard(name="Redis")  # missing description


def test_leaf_models_tolerate_additive_keys() -> None:
    """The catalog index mirrors a producer schema that evolves additively, and
    load_catalog has no embedded fallback for a schema error — so the leaf entry
    models must NOT forbid unknown keys (a forbidden extra would brick the load).
    A future capability/card/env_contract field is dropped, not rejected."""
    # An additive capability key (e.g. a future `provides`) is tolerated.
    entry = CapabilityEntry.model_validate(
        {"id": "cache.redis", "kind": "cache", "path": "p.md", "a_future_key": 1}
    )
    assert entry.id == "cache.redis"
    # A future kind degrades gracefully (free string, not a hard enum reject).
    assert (
        CapabilityEntry.model_validate(
            {"id": "kg.neo4j", "kind": "knowledge_graph", "path": "p.md"}
        ).kind
        == "knowledge_graph"
    )
    # Additive card + env_contract keys are tolerated too.
    assert CapabilityCard.model_validate({"name": "R", "description": "d", "icon": "x"}).name == "R"
    assert EnvContractEntry.model_validate({"name": "X", "required": True}).name == "X"


def test_top_level_models_tolerate_unknown_keys() -> None:
    """Catalog / RecipeEntry stay extra="ignore" so an additive future field
    doesn't break older scaffold builds parsing a newer catalog."""
    from agent_scaffold.catalog import RecipeEntry

    entry = RecipeEntry.model_validate(
        {"slug": "r", "path": "docs/recipes/r.md", "title": "R", "a_future_field": 42}
    )
    assert entry.slug == "r"
    # The Catalog container itself tolerates an additive top-level key.
    raw = yaml.safe_load(_fixture_text())
    raw["a_future_top_level_field"] = 1
    assert Catalog.model_validate(raw).schema_version == raw["schema_version"]


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


# ---------------------------------------------------------------------------
# Process-level memoization
# ---------------------------------------------------------------------------


def test_load_catalog_memoizes_within_ttl(tmp_path: Path) -> None:
    """A second load inside the TTL returns the same parsed instance with no
    network and no disk re-parse (the disk cache alone would re-validate)."""
    body = _fixture_text()
    calls = {"n": 0}

    def _fake_urlopen(req, **_):
        calls["n"] += 1
        return _mock_response(body)

    url = "https://example.com/catalog.yaml"
    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        first = load_catalog(url=url, cache_dir=tmp_path)
    with patch("urllib.request.urlopen", side_effect=AssertionError("network hit")):
        second = load_catalog(url=url, cache_dir=tmp_path)
    assert second is first
    assert calls["n"] == 1


def test_load_catalog_memo_expires_after_ttl(tmp_path: Path) -> None:
    """An aged memo entry re-resolves (served by the fresh disk cache)."""
    from agent_scaffold.catalog import _CATALOG_MEMO, CATALOG_FRESH_TTL_SECONDS

    body = _fixture_text()
    url = "https://example.com/catalog.yaml"
    with patch("urllib.request.urlopen", side_effect=lambda *a, **k: _mock_response(body)):
        first = load_catalog(url=url, cache_dir=tmp_path)

    key = (url, str(tmp_path))
    loaded_at, memoized = _CATALOG_MEMO[key]
    _CATALOG_MEMO[key] = (loaded_at - CATALOG_FRESH_TTL_SECONDS - 1, memoized)

    with patch("urllib.request.urlopen", side_effect=AssertionError("network hit")):
        again = load_catalog(url=url, cache_dir=tmp_path)
    assert again is not first
    assert again.recipes[0].slug == first.recipes[0].slug


def test_load_catalog_file_url_never_memoized(tmp_path: Path) -> None:
    """Local catalogs re-read every call so an edited file is always seen."""
    a = load_catalog(url=f"file://{FIXTURE_PATH}", cache_dir=tmp_path)
    b = load_catalog(url=f"file://{FIXTURE_PATH}", cache_dir=tmp_path)
    assert a is not b


def _age_fallback_memo(key: tuple[str, str]) -> None:
    """Expire a negative-memo entry — the production 60s TTL equivalent."""
    from agent_scaffold.catalog import _CATALOG_FALLBACK_MEMO, CATALOG_FALLBACK_TTL_SECONDS

    failed_at, degraded = _CATALOG_FALLBACK_MEMO[key]
    _CATALOG_FALLBACK_MEMO[key] = (failed_at - CATALOG_FALLBACK_TTL_SECONDS - 1, degraded)


def test_load_catalog_fallbacks_are_not_memoized(tmp_path: Path) -> None:
    """A stale-cache or embedded fallback must never land in the HEALTHY memo
    (which would pin the degraded copy for the full fresh TTL). Fallbacks go
    to the short negative memo instead; once that expires — sub-minute — the
    network is retried and recovery re-memoizes normally."""
    from agent_scaffold.catalog import _CATALOG_MEMO

    body = _fixture_text()
    url = "https://example.com/catalog.yaml"
    key = (url, str(tmp_path))

    # Embedded fallback (no cache, network down): healthy memo stays empty.
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("offline")):
        load_catalog(url=url, cache_dir=tmp_path)
    assert key not in _CATALOG_MEMO
    _age_fallback_memo(key)

    # Stale-cache fallback: seed + age the disk cache, then fail the fetch.
    with patch("urllib.request.urlopen", return_value=_mock_response(body, etag='"v1"')):
        load_catalog(url=url, cache_dir=tmp_path)
    _age_cache(tmp_path)
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("offline")):
        stale = load_catalog(url=url, cache_dir=tmp_path)
    assert stale.recipes[0].slug == "docs-rag-qa"
    assert key not in _CATALOG_MEMO

    # Network restored + negative TTL expired: the next call really
    # refetches and memoizes again.
    _age_fallback_memo(key)
    calls = {"n": 0}

    def _fake_urlopen(req, **_):
        calls["n"] += 1
        return _mock_response(body, etag='"v2"')

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        recovered = load_catalog(url=url, cache_dir=tmp_path)
    assert calls["n"] == 1
    assert recovered.recipes[0].slug == "docs-rag-qa"
    assert key in _CATALOG_MEMO


def test_offline_calls_inside_negative_ttl_skip_the_network(tmp_path: Path) -> None:
    """Within 60s of a failed fetch, repeated loads serve the degraded copy
    with ZERO network attempts — an offline wizard must not stall up to ~16s
    on every /plan, /status, and step render."""
    from agent_scaffold.catalog import _CATALOG_FALLBACK_MEMO, _CATALOG_MEMO

    url = "https://example.com/catalog.yaml"
    key = (url, str(tmp_path))
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("offline")):
        first = load_catalog(url=url, cache_dir=tmp_path)
    assert key in _CATALOG_FALLBACK_MEMO

    with patch("urllib.request.urlopen", side_effect=AssertionError("network hit")):
        second = load_catalog(url=url, cache_dir=tmp_path)
    assert second is first
    assert key not in _CATALOG_MEMO


def test_healthy_load_clears_the_negative_entry(tmp_path: Path) -> None:
    from agent_scaffold.catalog import _CATALOG_FALLBACK_MEMO

    body = _fixture_text()
    url = "https://example.com/catalog.yaml"
    key = (url, str(tmp_path))
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("offline")):
        load_catalog(url=url, cache_dir=tmp_path)
    assert key in _CATALOG_FALLBACK_MEMO

    _age_fallback_memo(key)
    with patch("urllib.request.urlopen", return_value=_mock_response(body, etag='"v1"')):
        load_catalog(url=url, cache_dir=tmp_path)
    assert key not in _CATALOG_FALLBACK_MEMO


def test_reset_catalog_memo_clears_both_memos(tmp_path: Path) -> None:
    from agent_scaffold.catalog import (
        _CATALOG_FALLBACK_MEMO,
        _CATALOG_MEMO,
        _reset_catalog_memo,
    )

    url = "https://example.com/catalog.yaml"
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("offline")):
        load_catalog(url=url, cache_dir=tmp_path)
    assert _CATALOG_FALLBACK_MEMO
    _reset_catalog_memo()
    assert not _CATALOG_FALLBACK_MEMO
    assert not _CATALOG_MEMO


def test_file_urls_never_touch_the_negative_memo(tmp_path: Path) -> None:
    from agent_scaffold.catalog import _CATALOG_FALLBACK_MEMO

    load_catalog(url=f"file://{FIXTURE_PATH}", cache_dir=tmp_path)
    assert not _CATALOG_FALLBACK_MEMO


# ---------------------------------------------------------------------------
# Synced-tree resolution (catalog.yaml from the deployments cache)
# ---------------------------------------------------------------------------

_TREE_SHA = "a" * 40


def _plant_synced_tree(cache_dir: Path, body: str, sha: str = _TREE_SHA) -> None:
    """Lay down the sources-cache layout: HEAD.sha + <sha>/catalog.yaml."""
    root = cache_dir / "deployments"
    (root / sha).mkdir(parents=True)
    (root / "HEAD.sha").write_text(sha, encoding="utf-8")
    (root / sha / "catalog.yaml").write_text(body, encoding="utf-8")


def test_synced_tree_serves_the_catalog_without_network(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """With a synced deployments tree, the default-URL load never fetches
    and never warns — this was a per-launch fetch that warned on every
    transient network blip."""
    _plant_synced_tree(tmp_path, _fixture_text())
    with patch("urllib.request.urlopen", side_effect=AssertionError("network touched")):
        catalog = load_catalog(cache_dir=tmp_path)
    assert catalog.recipes
    assert "using cached catalog" not in capsys.readouterr().err


def test_synced_tree_wins_over_the_fetch_cache(tmp_path: Path) -> None:
    """The tree copy is at the same commit as the served docs, so it beats a
    previously fetched network copy even when that cache is fresh."""
    with patch("urllib.request.urlopen", return_value=_mock_response(_fixture_text())):
        load_catalog(cache_dir=tmp_path)
    _reset_catalog_memo()
    tree_data = yaml.safe_load(_fixture_text())
    tree_data["recipes"][0]["slug"] = "tree-only-recipe"
    _plant_synced_tree(tmp_path, yaml.safe_dump(tree_data))
    with patch("urllib.request.urlopen", side_effect=AssertionError("network touched")):
        catalog = load_catalog(cache_dir=tmp_path)
    assert any(r.slug == "tree-only-recipe" for r in catalog.recipes)


def test_explicit_url_override_ignores_the_synced_tree(tmp_path: Path) -> None:
    """An explicit catalog URL means the user wants THAT catalog — the tree
    shortcut only applies to the default resolution."""
    tree_data = yaml.safe_load(_fixture_text())
    tree_data["recipes"][0]["slug"] = "tree-only-recipe"
    _plant_synced_tree(tmp_path, yaml.safe_dump(tree_data))
    with patch("urllib.request.urlopen", return_value=_mock_response(_fixture_text())):
        catalog = load_catalog(url="https://example.com/catalog.yaml", cache_dir=tmp_path)
    assert not any(r.slug == "tree-only-recipe" for r in catalog.recipes)


def test_env_override_ignores_the_synced_tree(tmp_path: Path) -> None:
    tree_data = yaml.safe_load(_fixture_text())
    tree_data["recipes"][0]["slug"] = "tree-only-recipe"
    _plant_synced_tree(tmp_path, yaml.safe_dump(tree_data))
    env = {"AGENT_SCAFFOLD_CATALOG_URL": "https://example.com/catalog.yaml"}
    with patch("urllib.request.urlopen", return_value=_mock_response(_fixture_text())):
        catalog = load_catalog(cache_dir=tmp_path, env=env)
    assert not any(r.slug == "tree-only-recipe" for r in catalog.recipes)


def test_tree_without_catalog_falls_back_to_fetch(tmp_path: Path) -> None:
    """A synced tree from before the catalog existed (or a torn extraction)
    quietly falls through to the network path."""
    root = tmp_path / "deployments"
    (root / _TREE_SHA).mkdir(parents=True)
    (root / "HEAD.sha").write_text(_TREE_SHA, encoding="utf-8")
    with patch("urllib.request.urlopen", return_value=_mock_response(_fixture_text())):
        catalog = load_catalog(cache_dir=tmp_path)
    assert catalog.recipes


def test_synced_tree_load_memoizes_as_healthy(tmp_path: Path) -> None:
    from agent_scaffold.catalog import _CATALOG_FALLBACK_MEMO, _CATALOG_MEMO

    _plant_synced_tree(tmp_path, _fixture_text())
    with patch("urllib.request.urlopen", side_effect=AssertionError("network touched")):
        load_catalog(cache_dir=tmp_path)
    assert _CATALOG_MEMO
    assert not _CATALOG_FALLBACK_MEMO
