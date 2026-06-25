"""Tests for agent_scaffold.content_lint.

Covers the parity guard (the shared constants must not drift from the canonical
set the deployments producer enforces) and one positive + negative case per
content-drift rule, built on throwaway deployments trees in ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, get_args

import yaml

from agent_scaffold.capabilities import _KNOWN_KINDS, CapabilityKind
from agent_scaffold.content_lint import (
    ADVERTISED_PROVIDERS,
    ENTRY_POINT_BASENAMES,
    VALID_CAPABILITY_KINDS,
    VALID_TOPOLOGIES,
    ContentLintError,
    errors,
    lint_content,
    summarize,
)
from agent_scaffold.steps.launch_backend import _ENTRY_CANDIDATES

# --- Parity guard ------------------------------------------------------------

# A vendored copy of agent-deployments' canonical capability `kind` list
# (docs/recipes/SCHEMA.md → `Allowed kinds`, docs/capabilities/README.md). It's
# duplicated here because the deployments repo isn't importable from the
# scaffold test tree. The authoritative SCHEMA.md ↔ list tie is machine-enforced
# cross-repo by agent-deployments' `test_canonical_capability_kinds_match_schema_doc`
# (which reads SCHEMA.md directly). When the canonical list changes, update BOTH
# this copy and `capabilities._KNOWN_KINDS` / the `CapabilityKind` Literal.
CANONICAL_CAPABILITY_KINDS = frozenset(
    {
        # v0.2 infrastructure cohort.
        "vector_db",
        "cache",
        "relational",
        "queue",
        "obs",
        "eval",
        "frontend",
        "host",
        # 2026-SOTA agent-native cohort.
        "mcp",
        "sandbox",
        "durable",
        "memory_store",
        "guardrail",
        "embedding",
        "live_data",
        "rerank",
        # Runtime key bootstrap.
        "auth",
    }
)


def test_known_kinds_match_canonical_set() -> None:
    """Mirror guard: the scaffold's _KNOWN_KINDS must equal the vendored
    canonical list — the producer/consumer shared contract."""
    assert _KNOWN_KINDS == CANONICAL_CAPABILITY_KINDS
    assert VALID_CAPABILITY_KINDS == CANONICAL_CAPABILITY_KINDS


def test_known_kinds_match_capability_kind_literal() -> None:
    """The frozenset and the typed Literal in capabilities.py must agree, so a
    kind that type-checks is also a kind the lint accepts (and vice versa)."""
    assert set(get_args(CapabilityKind)) == _KNOWN_KINDS


# Vendored copies of the producer's lint constants (agent-deployments
# scripts/generate_catalog.py). The deployments repo isn't importable from this
# test tree, so the cross-repo tie is: this copy + the deployments-side parity
# test (test_lint_constants_match_canonical) both pin their repo's constant to
# the same literal. A one-sided edit fails one repo's CI. When a constant
# changes, update BOTH repos' constants AND both vendored copies.
CANONICAL_ENTRY_POINT_BASENAMES = frozenset(
    {
        "main.py",
        "app.py",
        "server.py",
        "api.py",
        "asgi.py",
        "__main__.py",
        "index.ts",
        "index.js",
        "main.ts",
        "server.ts",
        "app.ts",
    }
)
CANONICAL_ADVERTISED_PROVIDERS = {
    "qdrant": ("qdrant", "qdrant"),
    "chroma": ("chroma", "chroma"),
    "pgvector": ("pgvector", "pgvector"),
    "openai": ("embedding.openai", "openai"),
    "cohere": ("rerank.cohere", "cohere"),
    "zep": ("memory_store.zep", "zep"),
}
CANONICAL_TOPOLOGIES = frozenset(
    {
        "single",
        "chain",
        "parallel",
        "event-driven",
        "multi-agent-flat",
        "multi-agent-hierarchical",
    }
)


def test_entry_point_basenames_match_canonical() -> None:
    """Mirror guard: the consumer's ENTRY_POINT_BASENAMES must equal the producer
    copy (vendored above) so the entry-point lint rule can't drift across repos."""
    assert ENTRY_POINT_BASENAMES == CANONICAL_ENTRY_POINT_BASENAMES


def test_entry_point_basenames_superset_of_launch_candidates() -> None:
    """Anything run's launch_backend can boot (its _ENTRY_CANDIDATES) must be a
    valid declared entry point, or a recipe that ships a launchable layout is
    falsely rejected. The Python subset of ENTRY_POINT_BASENAMES must therefore
    cover every launch candidate."""
    assert set(_ENTRY_CANDIDATES) <= ENTRY_POINT_BASENAMES


def test_advertised_providers_match_canonical() -> None:
    """Mirror guard for the advisory advertisement-coherence provider map."""
    assert ADVERTISED_PROVIDERS == CANONICAL_ADVERTISED_PROVIDERS


def test_topologies_match_canonical() -> None:
    """Mirror guard: the lint's topology set must equal the canonical list (the
    same set the Topology enum and SCHEMA.md are pinned to)."""
    assert VALID_TOPOLOGIES == CANONICAL_TOPOLOGIES


# --- Tree builders -----------------------------------------------------------


def _write_md(path: Path, frontmatter: dict[str, Any], body: str = "# Title\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "---\n" + yaml.safe_dump(frontmatter, sort_keys=False) + "---\n\n" + body
    path.write_text(text, encoding="utf-8")


def _capability(dep: Path, kind: str, name: str, **overrides: Any) -> None:
    fm: dict[str, Any] = {
        "id": f"{kind}.{name}",
        "kind": kind,
        "card": {"name": name.title(), "description": f"{name} capability."},
    }
    fm.update(overrides)
    _write_md(dep / "docs" / "capabilities" / kind / f"{name}.md", fm)


def _recipe(dep: Path, slug: str, **fm: Any) -> None:
    _write_md(dep / "docs" / "recipes" / f"{slug}.md", fm)


def _clean_tree(tmp_path: Path) -> Path:
    """A minimal deployments tree that lints clean (no errors)."""
    dep = tmp_path / "dep"
    (dep / "docs" / "recipes").mkdir(parents=True)
    (dep / "docs" / "capabilities").mkdir(parents=True)
    (dep / "docs" / "cross-cutting").mkdir(parents=True)
    _write_md(dep / "docs" / "cross-cutting" / "layout.md", {}, body="# Layout\n")
    _capability(dep, "cache", "redis", docker={"ports": ["6379:6379"]})
    _capability(dep, "relational", "postgres", docker={"ports": ["5432:5432"]})
    _recipe(
        dep,
        "demo",
        topology="single",
        capabilities=["cache.redis", "relational.postgres"],
        required_files=["Dockerfile", "app/main.py"],
        load_list=[{"path": "../cross-cutting/layout.md", "required": True}],
        runtime_modes={"default": {"description": "Claude + Postgres + Redis."}},
        recipe_dependencies={"python": {"redis": ">=5", "asyncpg": ">=0.29"}},
    )
    return dep


# --- Rules -------------------------------------------------------------------


def test_clean_source_passes(tmp_path: Path) -> None:
    findings = lint_content(_clean_tree(tmp_path))
    assert errors(findings) == [], [f.format() for f in findings]


def test_missing_recipes_dir_raises(tmp_path: Path) -> None:
    try:
        lint_content(tmp_path)
    except ContentLintError:
        pass
    else:
        raise AssertionError("a tree without docs/recipes/ must raise ContentLintError")


def test_bad_capability_kind(tmp_path: Path) -> None:
    dep = _clean_tree(tmp_path)
    _write_md(
        dep / "docs" / "capabilities" / "cache" / "ghost.md",
        {
            "id": "cache.ghost",
            "kind": "ghost_kind",
            "card": {"name": "Ghost", "description": "g"},
        },
    )
    findings = lint_content(dep)
    assert any(f.rule == "capability-kind" and "ghost_kind" in f.message for f in errors(findings))


def test_missing_card_is_advisory(tmp_path: Path) -> None:
    # A capability with no card at all is a soft warning (mirrors the producer's
    # soft_errors path), not a hard error.
    dep = _clean_tree(tmp_path)
    _write_md(
        dep / "docs" / "capabilities" / "obs" / "nocard.md",
        {"id": "obs.nocard", "kind": "obs"},
    )
    findings = lint_content(dep)
    assert any(
        f.rule == "capability-card" and f.severity == "warn" and "obs/nocard" in f.location
        for f in findings
    )
    assert not any(
        f.rule == "capability-card" and "obs/nocard" in f.location for f in errors(findings)
    )


def test_incomplete_card_is_error(tmp_path: Path) -> None:
    # A present-but-incomplete card (empty name) is a hard error — once you
    # declare a card, it must be complete.
    dep = _clean_tree(tmp_path)
    _write_md(
        dep / "docs" / "capabilities" / "obs" / "partial.md",
        {"id": "obs.partial", "kind": "obs", "card": {"name": "", "description": "x"}},
    )
    findings = lint_content(dep)
    assert any(
        f.rule == "capability-card" and "obs/partial" in f.location and "name" in f.message
        for f in errors(findings)
    )


def test_unresolved_capability_ref(tmp_path: Path) -> None:
    dep = _clean_tree(tmp_path)
    _recipe(
        dep,
        "bad-ref",
        capabilities=["cache.redis", "vector_db.nope"],
        required_files=["app/main.py"],
    )
    findings = lint_content(dep)
    assert any(
        f.rule == "capability-ref" and "vector_db.nope" in f.message for f in errors(findings)
    )


def test_required_files_entry_point(tmp_path: Path) -> None:
    dep = _clean_tree(tmp_path)
    _recipe(
        dep,
        "no-entry",
        capabilities=[],
        required_files=["Dockerfile", "docker-compose.yml", "tests/unit/test_x.py"],
    )
    findings = lint_content(dep)
    assert any(
        f.rule == "required-files-entry" and "no-entry" in f.location for f in errors(findings)
    )

    # A recipe that DOES name an entry point produces no such error.
    assert not any(
        f.rule == "required-files-entry" and "demo" in f.location for f in errors(findings)
    )


def test_port_collision(tmp_path: Path) -> None:
    dep = _clean_tree(tmp_path)
    # Two capabilities that both bind host port 7000.
    _capability(dep, "obs", "a", docker={"ports": ["7000:7000"]})
    _capability(dep, "obs", "b", docker={"ports": ["7000:7000"]})
    _recipe(
        dep,
        "collide",
        capabilities=["obs.a", "obs.b"],
        required_files=["app/main.py"],
    )
    findings = lint_content(dep)
    assert any(f.rule == "port-collision" and "7000" in f.message for f in errors(findings))


def test_port_collision_via_transitive_requires(tmp_path: Path) -> None:
    dep = _clean_tree(tmp_path)
    # obs.dep requires relational.postgres, which is already 5432 — and obs.dep
    # also claims 5432, so the resolved stack collides even though the recipe
    # only declares obs.dep.
    _capability(
        dep, "obs", "dep", docker={"ports": ["5432:5432"]}, requires=["relational.postgres"]
    )
    _recipe(dep, "transitive", capabilities=["obs.dep"], required_files=["app/main.py"])
    findings = lint_content(dep)
    assert any(f.rule == "port-collision" and "5432" in f.message for f in errors(findings))


def test_load_list_dead_link(tmp_path: Path) -> None:
    dep = _clean_tree(tmp_path)
    _recipe(
        dep,
        "dead",
        capabilities=[],
        required_files=["app/main.py"],
        load_list=[{"path": "../cross-cutting/ghost.md", "required": True}],
    )
    findings = lint_content(dep)
    assert any(f.rule == "load-list-link" and "ghost.md" in f.message for f in errors(findings))


def test_invalid_topology(tmp_path: Path) -> None:
    dep = _clean_tree(tmp_path)
    _recipe(dep, "weird", topology="swarm", capabilities=[], required_files=["app/main.py"])
    findings = lint_content(dep)
    assert any(f.rule == "topology" and "swarm" in f.message for f in errors(findings))


def test_advertisement_warning(tmp_path: Path) -> None:
    dep = _clean_tree(tmp_path)
    _recipe(
        dep,
        "advertised",
        capabilities=["cache.redis"],
        required_files=["app/main.py"],
        runtime_modes={"default": {"description": "Claude + Zep for memory."}},
        recipe_dependencies={"python": {"redis": ">=5"}},
    )
    findings = lint_content(dep)
    warns = [f for f in findings if f.severity == "warn"]
    assert any(
        f.rule == "advertisement" and "zep" in f.message and "advertised" in f.location
        for f in warns
    )
    # Advertisement findings are advisory, never errors.
    assert not any(f.rule == "advertisement" for f in errors(findings))


def test_advertisement_backed_provider_is_clean(tmp_path: Path) -> None:
    dep = _clean_tree(tmp_path)
    _capability(dep, "vector_db", "qdrant", docker={"ports": ["6333:6333"]})
    _recipe(
        dep,
        "backed",
        capabilities=["vector_db.qdrant"],
        required_files=["app/main.py"],
        runtime_modes={"default": {"description": "Claude + Qdrant retrieval."}},
        recipe_dependencies={"python": {"qdrant-client": ">=1"}},
    )
    findings = lint_content(dep)
    assert not any(f.rule == "advertisement" and "backed" in f.location for f in findings)


def test_agent_pattern_resolution_and_orphans(tmp_path: Path) -> None:
    dep = _clean_tree(tmp_path)
    # Provide a catalog so pattern resolution + orphan detection activate.
    (dep / "catalog.yaml").write_text(
        yaml.safe_dump({"patterns": [{"id": "rag"}, {"id": "saga"}]}), encoding="utf-8"
    )
    _recipe(
        dep,
        "patterned",
        agent_pattern="rag",
        capabilities=[],
        required_files=["app/main.py"],
    )
    # A recipe whose agent_pattern doesn't resolve.
    _recipe(
        dep,
        "ghost-pattern",
        agent_pattern="does-not-exist",
        capabilities=[],
        required_files=["app/main.py"],
    )
    findings = lint_content(dep)
    assert any(
        f.rule == "agent-pattern" and "does-not-exist" in f.message for f in errors(findings)
    )
    # 'saga' is in the catalog but no recipe selects it → advisory orphan warning.
    warns = [f for f in findings if f.severity == "warn"]
    assert any(f.rule == "orphan-pattern" and "saga" in f.message for f in warns)


def test_required_files_empty_is_not_checked(tmp_path: Path) -> None:
    # The entry-point rule fires only once a recipe declares files. An empty
    # list (or an absent key) must not be flagged — negative control for the guard.
    dep = _clean_tree(tmp_path)
    _recipe(dep, "empty-rf", capabilities=[], required_files=[])
    _recipe(dep, "absent-rf", capabilities=[])
    findings = lint_content(dep)
    assert not any(
        f.rule == "required-files-entry" and ("empty-rf" in f.location or "absent-rf" in f.location)
        for f in findings
    )


def test_advertisement_backed_by_capability_only(tmp_path: Path) -> None:
    # Backed by a capability but NOT a matching dependency → still clean (OR
    # semantics). Pins that the cap branch alone suppresses the warning.
    dep = _clean_tree(tmp_path)
    _capability(dep, "vector_db", "qdrant", docker={"ports": ["6333:6333"]})
    _recipe(
        dep,
        "cap-only",
        capabilities=["vector_db.qdrant"],
        required_files=["app/main.py"],
        runtime_modes={"default": {"description": "Claude + Qdrant retrieval."}},
        recipe_dependencies={"python": {"fastapi": ">=0.110"}},
    )
    findings = lint_content(dep)
    assert not any(f.rule == "advertisement" and "cap-only" in f.location for f in findings)


def test_advertisement_backed_by_dependency_only(tmp_path: Path) -> None:
    # Backed by a dependency but NOT a capability → still clean (OR semantics).
    dep = _clean_tree(tmp_path)
    _recipe(
        dep,
        "dep-only",
        capabilities=["cache.redis"],
        required_files=["app/main.py"],
        runtime_modes={"default": {"description": "Claude + Qdrant retrieval."}},
        recipe_dependencies={"python": {"qdrant-client": ">=1"}},
    )
    findings = lint_content(dep)
    assert not any(f.rule == "advertisement" and "dep-only" in f.location for f in findings)


def test_orphan_framework_warns(tmp_path: Path) -> None:
    dep = _clean_tree(tmp_path)
    (dep / "docs" / "frameworks").mkdir(parents=True)
    _write_md(
        dep / "docs" / "frameworks" / "lonely.md",
        {"id": "lonely", "language": "python"},
    )
    _write_md(
        dep / "docs" / "frameworks" / "used.md",
        {"id": "used", "language": "python"},
    )
    # A recipe that references 'used' in its load_list (so it is not orphaned).
    _recipe(
        dep,
        "fw-user",
        capabilities=[],
        required_files=["app/main.py"],
        load_list=[{"path": "../frameworks/used.md", "required": True}],
    )
    findings = lint_content(dep)
    warns = [f for f in findings if f.severity == "warn"]
    assert any(f.rule == "orphan-framework" and "lonely" in f.message for f in warns)
    assert not any(f.rule == "orphan-framework" and "used" in f.message for f in warns)


def test_orphan_framework_skips_non_framework_docs(tmp_path: Path) -> None:
    # A docs/frameworks/*.md without id+language (e.g. comparison.md) is not a
    # framework and must not be flagged as an orphan.
    dep = _clean_tree(tmp_path)
    (dep / "docs" / "frameworks").mkdir(parents=True)
    _write_md(dep / "docs" / "frameworks" / "comparison.md", {"title": "Comparison"})
    findings = lint_content(dep)
    assert not any(f.rule == "orphan-framework" and "comparison" in f.location for f in findings)


def test_summarize_counts() -> None:
    from agent_scaffold.content_lint import Finding

    findings = [
        Finding("error", "r", "loc", "m"),
        Finding("warn", "r", "loc", "m"),
        Finding("warn", "r", "loc", "m"),
    ]
    assert summarize(findings) == "1 error, 2 warnings"
    assert len(errors(findings)) == 1


# --- CLI exit-code contract --------------------------------------------------


def test_cli_lint_content_exit_codes(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from agent_scaffold.cli import app

    runner = CliRunner()

    # Clean tree → exit 0.
    clean = _clean_tree(tmp_path / "clean")
    res = runner.invoke(app, ["lint-content", "--deployments-path", str(clean)])
    assert res.exit_code == 0, res.output

    # Drifted tree (bad kind) → exit 1.
    drift = _clean_tree(tmp_path / "drift")
    _write_md(
        drift / "docs" / "capabilities" / "cache" / "ghost.md",
        {"id": "cache.ghost", "kind": "ghost_kind", "card": {"name": "G", "description": "g"}},
    )
    res = runner.invoke(app, ["lint-content", "--deployments-path", str(drift)])
    assert res.exit_code == 1, res.output

    # A tree with no docs/recipes/ → exit 1 (ContentLintError).
    empty = tmp_path / "empty"
    empty.mkdir()
    res = runner.invoke(app, ["lint-content", "--deployments-path", str(empty)])
    assert res.exit_code == 1, res.output


def test_cli_warnings_as_errors(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from agent_scaffold.cli import app

    runner = CliRunner()
    # A tree with a warning (unbacked advertisement) but no errors.
    dep = _clean_tree(tmp_path)
    _recipe(
        dep,
        "warned",
        capabilities=["cache.redis"],
        required_files=["app/main.py"],
        runtime_modes={"default": {"description": "Claude + Zep for memory."}},
        recipe_dependencies={"python": {"redis": ">=5"}},
    )
    # Without -W: warnings don't fail.
    res = runner.invoke(app, ["lint-content", "--deployments-path", str(dep)])
    assert res.exit_code == 0, res.output
    # With -W: warnings fail.
    res = runner.invoke(app, ["lint-content", "--deployments-path", str(dep), "-W"])
    assert res.exit_code == 1, res.output
