"""Tests for cache-tier segments: discovery parsing, assembly grouping,
alias-scan demotion, and the generator's tiered cache breakpoints."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import pytest
import yaml

from agent_scaffold.catalog import Catalog
from agent_scaffold.context import AssembledContext, ContextSegment
from agent_scaffold.context import assemble as _real_assemble
from agent_scaffold.discovery import default_cache_tier, discover_recipes
from agent_scaffold.generator import (
    CACHE_SPLIT_WARM_MARKER,
    GenerationRequest,
    _build_user_content,
    _context_for_prompt,
    _system_blocks,
)

_TEST_CATALOG_PATH = Path(__file__).parent / "fixtures" / "catalog_minimal.yaml"
_TEST_CATALOG: Catalog = Catalog.model_validate(
    yaml.safe_load(_TEST_CATALOG_PATH.read_text(encoding="utf-8"))
)
assemble = partial(_real_assemble, catalog=_TEST_CATALOG)

# Comfortably above the Opus cache minimum (4096 tokens ≈ 16 KB at 4 chars/token).
_BIG = "x" * 20_000
_MODEL = "claude-opus-4-8"


def _recipe(deployments: Path, slug: str) -> Any:
    return next(r for r in discover_recipes(deployments) if r.slug == slug)


# ---------------------------------------------------------------------------
# default_cache_tier — parity with the deployments generator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("../../vendored/blueprints/patterns/react/overview.md", "hot"),
        ("blueprints/patterns/react/overview.md", "hot"),  # scaffold display form
        ("../frameworks/pydantic-ai.md", "hot"),
        ("../stack/llm-claude.md", "hot"),
        ("../cross-cutting/project-layout.md", "hot"),
        ("../cross-cutting/observability.md", "warm"),
        ("../capabilities/obs/langfuse.md", "warm"),
        ("recipes/docs-rag-qa.md", "warm"),
        ("../patterns/react.md", "dynamic"),
        ("something/else.md", "dynamic"),
    ],
)
def test_default_cache_tier_mapping(path: str, expected: str) -> None:
    assert default_cache_tier(path) == expected


# ---------------------------------------------------------------------------
# discovery — cache_tier frontmatter parsing
# ---------------------------------------------------------------------------


def test_load_list_cache_tier_parsed_and_validated(
    mock_deployments_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    recipe = _recipe(mock_deployments_path, "with-load-list-and-prose")
    by_path = {e.path: e for e in recipe.load_list}
    assert by_path["../patterns/react.md"].cache_tier == "hot"  # authored
    assert by_path["../cross-cutting/logging-structured.md"].cache_tier is None  # defaulted later


def test_load_list_invalid_cache_tier_warns_and_defaults(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent_scaffold.discovery import _coerce_load_list

    entries = _coerce_load_list(
        [{"path": "../patterns/react.md", "required": True, "cache_tier": "blazing"}],
        "r.md",
    )
    assert len(entries) == 1
    assert entries[0].cache_tier is None
    assert "cache_tier='blazing'" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# assemble — segments + alias-scan demotion
# ---------------------------------------------------------------------------


def test_assemble_builds_segments_for_load_list_recipe(mock_deployments_path: Path) -> None:
    recipe = _recipe(mock_deployments_path, "with-load-list")
    out = assemble(
        recipe, language="python", framework="pydantic_ai", deployments_path=mock_deployments_path
    )
    assert out.segments, "load_list recipe must produce segments"
    tiers = [s.cache_tier for s in out.segments]
    assert tiers == ["hot", "warm"] or tiers == ["warm"]
    # Framework doc defaults hot; recipe body leads the warm segment.
    hot_text = "".join(s.text for s in out.segments if s.cache_tier == "hot")
    warm_text = "".join(s.text for s in out.segments if s.cache_tier == "warm")
    assert "frameworks/pydantic-ai.md" in hot_text
    assert warm_text.lstrip().startswith(
        "# With Load List"
    )  # recipe body leads (frontmatter stripped from the shipped recipe doc)
    # Hot precedes warm; same docs as the legacy body (content parity).
    for marker_line in (line for line in out.body.splitlines() if "referenced:" in line):
        assert marker_line in hot_text or marker_line in warm_text


def test_assemble_no_segments_without_load_list(mock_deployments_path: Path) -> None:
    recipe = _recipe(mock_deployments_path, "customer-support-triage")
    out = assemble(
        recipe, language="python", framework="langgraph", deployments_path=mock_deployments_path
    )
    assert out.segments == []


def test_assemble_builds_segments_for_manifest_only_recipe(tmp_path: Path) -> None:
    """A recipe with no load_list but a catalog context_manifest still gets
    cache-tier segments — the manifest is a curated doc set, so the hot prefix
    is deterministic enough to cache. Without this, manifest-only recipes (the
    shape recipes take after the v0.2→v0.3 cutover) silently fall back to
    single-block caching."""
    from agent_scaffold.catalog import ContextManifest, ManifestDoc, RecipeEntry

    docs = tmp_path / "docs"
    (docs / "recipes").mkdir(parents=True)
    (docs / "frameworks").mkdir()
    (docs / "frameworks" / "pydantic-ai.md").write_text("# Pydantic AI\n", encoding="utf-8")
    (docs / "recipes" / "r.md").write_text(
        "---\nlanguages: [python]\n---\n\n# R\n",  # no load_list
        encoding="utf-8",
    )
    recipe = _recipe(tmp_path, "r")
    assert not recipe.load_list  # precondition: manifest is the only curated source
    cat = _TEST_CATALOG.model_copy(
        update={
            "recipes": [
                RecipeEntry(slug="r", path="docs/recipes/r.md", title="R").model_copy(
                    update={
                        "context_manifest": ContextManifest(
                            docs=[ManifestDoc(path="../frameworks/pydantic-ai.md", required=True)]
                        )
                    }
                )
            ]
        }
    )
    out = _real_assemble(recipe, "python", "pydantic_ai", tmp_path, catalog=cat)
    assert out.segments, "manifest-only recipe must still produce cache-tier segments"


def test_load_list_recipe_skips_alias_and_cross_cutting_prose_scans(
    mock_deployments_path: Path,
) -> None:
    """The recipe prose mentions 'qdrant' and 'rate limiting' — bait the
    prose scanners would normally load. With a load_list declared, the
    author's curation wins: only declared docs (+ transitives from them,
    which stay enabled by design) load."""
    recipe = _recipe(mock_deployments_path, "with-load-list-and-prose")
    out = assemble(
        recipe, language="python", framework="none", deployments_path=mock_deployments_path
    )
    names = {p.name for p in out.referenced_paths}
    assert "react.md" in names  # declared
    assert "logging-structured.md" in names  # declared
    assert "vector-qdrant.md" not in names  # alias bait, not declared
    assert "rate-limiting.md" not in names  # cross-cutting bait, not declared


def test_context_manifest_skips_transitive_walk(tmp_path: Path) -> None:
    """A catalog context_manifest with docs marks the recipe's context as closed,
    so the speculative transitive link walk is skipped; without it, the walk
    pulls transitively-linked docs."""
    from agent_scaffold.catalog import ContextManifest, ManifestDoc, RecipeEntry

    docs = tmp_path / "docs"
    (docs / "recipes").mkdir(parents=True)
    (docs / "x").mkdir()
    (docs / "x" / "a.md").write_text("# A\n\nSee [B](b.md).\n", encoding="utf-8")
    (docs / "x" / "b.md").write_text("# B (reachable only transitively)\n", encoding="utf-8")
    (docs / "recipes" / "r.md").write_text(
        "---\nlanguages: [python]\nload_list:\n  - {path: ../x/a.md, required: true}\n---\n\n# R\n",
        encoding="utf-8",
    )
    recipe = _recipe(tmp_path, "r")
    base = RecipeEntry(slug="r", path="docs/recipes/r.md", title="R")
    cat_without = _TEST_CATALOG.model_copy(update={"recipes": [base]})
    cat_with = _TEST_CATALOG.model_copy(
        update={
            "recipes": [
                base.model_copy(
                    update={
                        "context_manifest": ContextManifest(
                            docs=[ManifestDoc(path="../x/a.md", required=True)]
                        )
                    }
                )
            ]
        }
    )
    out_without = _real_assemble(recipe, "python", "none", tmp_path, catalog=cat_without)
    out_with = _real_assemble(recipe, "python", "none", tmp_path, catalog=cat_with)
    names_without = {p.name for p in out_without.referenced_paths}
    names_with = {p.name for p in out_with.referenced_paths}
    assert "a.md" in names_without and "b.md" in names_without  # transitive walk pulls b
    assert "a.md" in names_with and "b.md" not in names_with  # manifest skips the walk


def test_context_manifest_drives_load_beyond_load_list(tmp_path: Path) -> None:
    """A doc that's only in `context_manifest.docs` (not the recipe's `load_list`)
    — e.g. a resolved pattern level or adapter stack doc — is loaded via the
    manifest, which the consumer treats as the authoritative menu."""
    from agent_scaffold.catalog import ContextManifest, ManifestDoc, RecipeEntry

    docs = tmp_path / "docs"
    (docs / "recipes").mkdir(parents=True)
    (docs / "x").mkdir()
    (docs / "x" / "a.md").write_text("# A\n", encoding="utf-8")
    (docs / "x" / "extra.md").write_text("# Extra (manifest-only)\n", encoding="utf-8")
    (docs / "recipes" / "r.md").write_text(
        "---\nlanguages: [python]\nload_list:\n  - {path: ../x/a.md, required: true}\n---\n\n# R\n",
        encoding="utf-8",
    )
    recipe = _recipe(tmp_path, "r")
    base = RecipeEntry(slug="r", path="docs/recipes/r.md", title="R")
    cat = _TEST_CATALOG.model_copy(
        update={
            "recipes": [
                base.model_copy(
                    update={
                        "context_manifest": ContextManifest(
                            docs=[
                                ManifestDoc(path="../x/a.md", required=True),
                                ManifestDoc(path="../x/extra.md", required=True),
                            ]
                        )
                    }
                )
            ]
        }
    )
    names = {
        p.name
        for p in _real_assemble(recipe, "python", "none", tmp_path, catalog=cat).referenced_paths
    }
    assert "a.md" in names and "extra.md" in names  # manifest (not just load_list) drives the load


# ---------------------------------------------------------------------------
# generator — tiered breakpoints
# ---------------------------------------------------------------------------


def _ctx(segments: list[ContextSegment]) -> AssembledContext:
    return AssembledContext(
        recipe_path=Path("/r.md"),
        referenced_paths=[],
        body="".join(s.text for s in segments),
        token_estimate=0,
        segments=segments,
    )


def test_context_for_prompt_inserts_warm_marker() -> None:
    ctx = _ctx(
        [
            ContextSegment(cache_tier="hot", text="HOT DOCS\n"),
            ContextSegment(cache_tier="warm", text="WARM DOCS\n"),
        ]
    )
    rendered = _context_for_prompt(ctx)
    assert rendered.index("HOT DOCS") < rendered.index(CACHE_SPLIT_WARM_MARKER)
    assert rendered.index(CACHE_SPLIT_WARM_MARKER) < rendered.index("WARM DOCS")


def test_context_for_prompt_without_segments_uses_body() -> None:
    ctx = AssembledContext(
        recipe_path=Path("/r.md"), referenced_paths=[], body="BODY", token_estimate=0
    )
    assert _context_for_prompt(ctx) == "BODY"
    assert CACHE_SPLIT_WARM_MARKER not in _context_for_prompt(ctx)


def test_build_user_content_three_tiered_blocks_default_5m() -> None:
    """Default TTL is 5m (plain ephemeral) for every cached block — a one-shot
    generation reads nothing back, so the cheaper write wins."""
    context = f"hints+hot {_BIG}\n{CACHE_SPLIT_WARM_MARKER}\nwarm {_BIG}\n"
    blocks = _build_user_content(context, "tail", _MODEL)
    assert len(blocks) == 3
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert blocks[1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in blocks[2]
    assert CACHE_SPLIT_WARM_MARKER not in blocks[0]["text"] + blocks[1]["text"]
    system = _system_blocks()
    total = sum(1 for b in [*system, *blocks] if "cache_control" in b)
    assert total == 3
    assert system[0]["cache_control"] == {"type": "ephemeral"}


def test_build_user_content_hot_ttl_opt_in() -> None:
    """cache_ttl=1h promotes the hot prefix (system + hot block) to the 1h
    tier while the warm block stays 5m — the API's precedence order holds."""
    context = f"hints+hot {_BIG}\n{CACHE_SPLIT_WARM_MARKER}\nwarm {_BIG}\n"
    blocks = _build_user_content(context, "tail", _MODEL, hot_ttl_1h=True)
    assert blocks[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert blocks[1]["cache_control"] == {"type": "ephemeral"}
    assert _system_blocks(ttl_1h=True)[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_build_user_content_small_hot_folds_into_warm() -> None:
    context = f"tiny-hot\n{CACHE_SPLIT_WARM_MARKER}\nwarm {_BIG}\n"
    blocks = _build_user_content(context, "tail", _MODEL)
    assert len(blocks) == 2
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert "tiny-hot" in blocks[0]["text"] and "warm" in blocks[0]["text"]


def test_build_user_content_small_warm_folds_into_hot() -> None:
    context = f"hot {_BIG}\n{CACHE_SPLIT_WARM_MARKER}\ntiny-warm\n"
    blocks = _build_user_content(context, "tail", _MODEL, hot_ttl_1h=True)
    assert len(blocks) == 2
    assert blocks[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert "tiny-warm" in blocks[0]["text"]


def test_build_user_content_legacy_path_unchanged() -> None:
    blocks = _build_user_content(f"context {_BIG}", "tail", _MODEL)
    assert len(blocks) == 2
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}


def test_build_user_content_respects_per_model_cache_minimum() -> None:
    # A ~2500-token context caches on Sonnet 4.5 (1024-token minimum) but not on
    # Opus (4096-token minimum) — the block folds into a single uncached block.
    context = "y" * 10_000
    assert len(_build_user_content(context, "tail", "claude-sonnet-4-5")) == 2
    assert len(_build_user_content(context, "tail", "claude-opus-4-8")) == 1


def test_generate_request_tiered_end_to_end(
    monkeypatch: pytest.MonkeyPatch, mock_deployments_path: Path
) -> None:
    """Full generate() against a fake client with the default 5m TTL: tiered
    context yields a 5m system + 5m hot + 5m warm + uncached tail, ≤4
    breakpoints. A 1h TTL is opt-in (see the sibling test)."""
    import json

    from agent_scaffold import generator as gen_mod
    from agent_scaffold.config import load_config

    captured: list[dict[str, Any]] = []

    class _Stream:
        def __enter__(self) -> Any:
            return self

        def __exit__(self, *a: Any) -> None:
            return None

        def __iter__(self) -> Any:
            return iter(())

        def get_final_message(self) -> Any:
            class _R:
                content = [type("B", (), {"text": "{}"})()]

            return _R()

    class _Client:
        class messages:  # noqa: N801
            @staticmethod
            def stream(**kwargs: Any) -> Any:
                captured.append(kwargs)
                return _Stream()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(gen_mod, "_make_client", lambda _cfg: _Client())

    ctx = _ctx(
        [
            ContextSegment(cache_tier="hot", text=f"HOT {_BIG}\n"),
            ContextSegment(cache_tier="warm", text=f"WARM {_BIG}\n"),
        ]
    )
    req = GenerationRequest(
        project_name="demo",
        target_language="python",
        framework="none",
        assembled_context=ctx,
        language_hints={"language": "python"},
        extra_required=[],
        strict=False,
    )
    gen_mod.generate(req, load_config())

    kwargs = captured[0]
    cc = [
        b.get("cache_control")
        for b in [*kwargs["system"], *kwargs["messages"][0]["content"]]
        if b.get("cache_control")
    ]
    assert len(cc) == 3  # one spare under the 4-breakpoint limit
    assert cc == [
        {"type": "ephemeral"},  # system
        {"type": "ephemeral"},  # hot
        {"type": "ephemeral"},  # warm
    ]
    # No 1h entries at the default TTL, so the API precedence rule is trivial.
    payload = json.dumps([*kwargs["system"], *kwargs["messages"][0]["content"]])
    assert '"1h"' not in payload

    # cache_ttl=1h promotes the system + hot prefix, warm stays 5m.
    captured.clear()
    gen_mod.generate(req, load_config().model_copy(update={"cache_ttl": "1h"}))
    kwargs = captured[0]
    cc = [
        b.get("cache_control")
        for b in [*kwargs["system"], *kwargs["messages"][0]["content"]]
        if b.get("cache_control")
    ]
    assert cc == [
        {"type": "ephemeral", "ttl": "1h"},  # system
        {"type": "ephemeral", "ttl": "1h"},  # hot
        {"type": "ephemeral"},  # warm
    ]
    payload = json.dumps([*kwargs["system"], *kwargs["messages"][0]["content"]])
    assert payload.index('"1h"') < payload.index('{"type": "ephemeral"}')
