"""Tests for the capability catalog loader + resolver."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_scaffold.capabilities import (
    Capability,
    DockerFragment,
    ResolvedStack,
    _reset_warn_dedupe,
    load_capabilities,
    resolve,
)
from agent_scaffold.discovery import Recipe, discover_recipes


def test_load_capabilities_discovers_all_valid(mock_deployments_path: Path) -> None:
    catalog = load_capabilities(mock_deployments_path)
    # README + malformed entries excluded; the rest of the fixture catalog
    # is open-ended so additional caps can be added without churning this test.
    assert {
        "vector_db.qdrant",
        "cache.redis",
        "host.vercel",
    } <= set(catalog)
    assert all(
        cap_id not in catalog for cap_id in ("malformed.no_frontmatter", "malformed.wrong_path")
    )


def test_load_capabilities_accepts_eval_kind(mock_deployments_path: Path) -> None:
    """The `eval` kind was added when bootstrap_evals shipped. Before this fix,
    eval.promptfoo (and siblings) were rejected with `kind 'eval' must be one of [...]`
    because _KNOWN_KINDS hadn't been updated."""
    catalog = load_capabilities(mock_deployments_path)
    assert "eval.promptfoo" in catalog
    cap = catalog["eval.promptfoo"]
    assert cap.kind == "eval"
    assert cap.bootstrap_step == "bootstrap_evals"


def test_load_capabilities_skips_readme(
    mock_deployments_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    load_capabilities(mock_deployments_path)
    err = capsys.readouterr().err
    # README.md must NOT trigger a "missing 'id'" warning — it's intentionally skipped.
    assert "README" not in err


def test_load_capabilities_skips_doc_files(
    mock_deployments_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCHEMA.md sits next to capability files but documents the schema itself.

    It has a valid H1 but no capability frontmatter — must be skipped by name,
    same as README.md.
    """
    catalog = load_capabilities(mock_deployments_path)
    err = capsys.readouterr().err
    assert "SCHEMA" not in err
    assert "SCHEMA" not in catalog
    assert "schema" not in catalog


def test_load_capabilities_warns_on_no_frontmatter(
    mock_deployments_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    load_capabilities(mock_deployments_path)
    err = capsys.readouterr().err
    assert "no_frontmatter.md" in err
    assert "missing frontmatter" in err


def test_load_capabilities_rejects_path_mismatch(
    mock_deployments_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    load_capabilities(mock_deployments_path)
    err = capsys.readouterr().err
    assert "wrong_path.md" in err
    assert "does not match path" in err


def test_load_capabilities_missing_catalog_returns_empty(tmp_path: Path) -> None:
    # Deployments root with no docs/capabilities/ at all.
    (tmp_path / "docs" / "recipes").mkdir(parents=True)
    catalog = load_capabilities(tmp_path)
    assert catalog == {}


def test_capability_docker_fragment_parsed(mock_deployments_path: Path) -> None:
    catalog = load_capabilities(mock_deployments_path)
    qdrant = catalog["vector_db.qdrant"]
    assert qdrant.kind == "vector_db"
    assert qdrant.env_vars == ["QDRANT_URL", "QDRANT_API_KEY"]
    assert qdrant.docker is not None
    assert qdrant.docker.service == "qdrant"
    assert qdrant.docker.image == "qdrant/qdrant:v1.12.0"
    assert qdrant.docker.ports == ["6333:6333"]
    assert qdrant.docker.environment == {"QDRANT__LOG_LEVEL": "INFO"}
    assert qdrant.docker.healthcheck is not None
    assert qdrant.probe == "qdrant_collections"
    assert qdrant.bootstrap_step == "bootstrap_vector_db"
    assert "## Local setup" in qdrant.body


def test_capability_deploy_configs_parsed(mock_deployments_path: Path) -> None:
    catalog = load_capabilities(mock_deployments_path)
    vercel = catalog["host.vercel"]
    assert vercel.kind == "host"
    assert vercel.docker is None
    assert len(vercel.emit_files) == 1
    assert vercel.emit_files[0].source == "templates/vercel.json"
    assert vercel.emit_files[0].dest == "vercel.json"
    assert len(vercel.deploy_configs) == 1
    cfg = vercel.deploy_configs[0]
    assert cfg.target == "vercel"
    assert cfg.cli_cmd == "vercel deploy --prod"
    assert cfg.dashboard_url == "https://vercel.com/dashboard"
    assert cfg.config_file == "vercel.json"


def _recipe(slug: str, capabilities: list[str], tmp_path: Path) -> Recipe:
    return Recipe(slug=slug, title="t", path=tmp_path / f"{slug}.md", capabilities=capabilities)


def test_resolve_preserves_order_and_unresolved(
    mock_deployments_path: Path, tmp_path: Path
) -> None:
    catalog = load_capabilities(mock_deployments_path)
    recipe = _recipe("demo", ["vector_db.qdrant", "vector_db.absent", "cache.redis"], tmp_path)
    stack = resolve(recipe, catalog)
    assert stack.ids() == ["vector_db.qdrant", "cache.redis"]
    assert stack.unresolved == ["vector_db.absent"]


def test_resolve_deduplicates_with_warning(
    mock_deployments_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    catalog = load_capabilities(mock_deployments_path)
    recipe = _recipe("demo", ["cache.redis", "cache.redis"], tmp_path)
    stack = resolve(recipe, catalog)
    assert stack.ids() == ["cache.redis"]
    err = capsys.readouterr().err
    assert "declared twice" in err


# ---------------------------------------------------------------------------
# default_frontend — every agent ships a UI
# ---------------------------------------------------------------------------


def _frontend_catalog() -> dict[str, Capability]:
    return {
        "frontend.minimal-chat": Capability(
            id="frontend.minimal-chat", kind="frontend", path=Path("/f.md"), serve_in_container=True
        ),
        "relational.postgres": Capability(
            id="relational.postgres", kind="relational", path=Path("/p.md")
        ),
        "frontend.nextjs-chat": Capability(
            id="frontend.nextjs-chat", kind="frontend", path=Path("/n.md")
        ),
    }


def test_default_frontend_added_when_recipe_has_none(tmp_path: Path) -> None:
    recipe = _recipe("a", ["relational.postgres"], tmp_path)
    stack = resolve(recipe, _frontend_catalog(), default_frontend=True)
    assert "frontend.minimal-chat" in stack.ids()


def test_default_frontend_not_added_when_recipe_declares_a_frontend(tmp_path: Path) -> None:
    recipe = _recipe("b", ["frontend.nextjs-chat"], tmp_path)
    stack = resolve(recipe, _frontend_catalog(), default_frontend=True)
    assert stack.ids() == ["frontend.nextjs-chat"]  # respects the recipe's own UI


def test_default_frontend_inert_when_not_in_catalog(tmp_path: Path) -> None:
    # Until deployments ships the capability, the auto-include is a safe no-op —
    # no add, no `unresolved` warning.
    recipe = _recipe("c", ["relational.postgres"], tmp_path)
    catalog = {"relational.postgres": _frontend_catalog()["relational.postgres"]}
    stack = resolve(recipe, catalog, default_frontend=True)
    assert stack.ids() == ["relational.postgres"]
    assert stack.unresolved == []


def test_default_frontend_off_by_default(tmp_path: Path) -> None:
    recipe = _recipe("d", ["relational.postgres"], tmp_path)
    stack = resolve(recipe, _frontend_catalog())  # default_frontend defaults False
    assert "frontend.minimal-chat" not in stack.ids()


# ---------------------------------------------------------------------------
# default_key_bootstrap — runtime API-key capture pairs with the chat UI
# ---------------------------------------------------------------------------


def _chat_catalog() -> dict[str, Capability]:
    catalog = _frontend_catalog()
    catalog["auth.key-bootstrap"] = Capability(
        id="auth.key-bootstrap", kind="auth", path=Path("/k.md")
    )
    return catalog


def test_key_bootstrap_added_alongside_default_frontend(tmp_path: Path) -> None:
    recipe = _recipe("a", ["relational.postgres"], tmp_path)
    stack = resolve(recipe, _chat_catalog(), default_frontend=True, default_key_bootstrap=True)
    assert "frontend.minimal-chat" in stack.ids()
    assert "auth.key-bootstrap" in stack.ids()


def test_key_bootstrap_added_when_recipe_has_own_frontend(tmp_path: Path) -> None:
    recipe = _recipe("b", ["frontend.nextjs-chat"], tmp_path)
    stack = resolve(recipe, _chat_catalog(), default_key_bootstrap=True)
    assert "auth.key-bootstrap" in stack.ids()  # any active frontend triggers it


def test_key_bootstrap_not_added_without_a_frontend(tmp_path: Path) -> None:
    # No default frontend and no recipe frontend → no chat surface → no bootstrap.
    recipe = _recipe("c", ["relational.postgres"], tmp_path)
    stack = resolve(recipe, _chat_catalog(), default_key_bootstrap=True)
    assert "auth.key-bootstrap" not in stack.ids()


def test_key_bootstrap_inert_when_not_in_catalog(tmp_path: Path) -> None:
    recipe = _recipe("d", ["relational.postgres"], tmp_path)
    stack = resolve(recipe, _frontend_catalog(), default_frontend=True, default_key_bootstrap=True)
    assert "auth.key-bootstrap" not in stack.ids()  # absent from _frontend_catalog
    assert stack.unresolved == []


def test_key_bootstrap_off_by_default(tmp_path: Path) -> None:
    recipe = _recipe("e", ["relational.postgres"], tmp_path)
    stack = resolve(recipe, _chat_catalog(), default_frontend=True)  # bootstrap defaults False
    assert "auth.key-bootstrap" not in stack.ids()


# ---------------------------------------------------------------------------
# requires — auto-add capability dependencies (langfuse → postgres)
# ---------------------------------------------------------------------------


def _requires_catalog() -> dict[str, Capability]:
    return {
        "obs.langfuse": Capability(
            id="obs.langfuse", kind="obs", path=Path("/lf.md"), requires=["relational.postgres"]
        ),
        "relational.postgres": Capability(
            id="relational.postgres", kind="relational", path=Path("/pg.md")
        ),
    }


def test_requires_auto_added(tmp_path: Path) -> None:
    recipe = _recipe("a", ["obs.langfuse"], tmp_path)
    stack = resolve(recipe, _requires_catalog())
    assert "relational.postgres" in stack.ids()  # depends_on no longer dangles


def test_requires_not_duplicated_when_already_declared(tmp_path: Path) -> None:
    recipe = _recipe("b", ["relational.postgres", "obs.langfuse"], tmp_path)
    stack = resolve(recipe, _requires_catalog())
    assert stack.ids().count("relational.postgres") == 1


def test_requires_honors_explicit_removal(tmp_path: Path) -> None:
    # A user who explicitly drops the dep keeps it dropped (no silent re-add).
    recipe = _recipe("c", ["obs.langfuse"], tmp_path)
    stack = resolve(recipe, _requires_catalog(), remove_capabilities={"relational.postgres"})
    assert "relational.postgres" not in stack.ids()


def test_requires_unknown_dep_falls_to_unresolved(tmp_path: Path) -> None:
    catalog = {
        "obs.langfuse": Capability(
            id="obs.langfuse", kind="obs", path=Path("/lf.md"), requires=["relational.absent"]
        )
    }
    recipe = _recipe("d", ["obs.langfuse"], tmp_path)
    stack = resolve(recipe, catalog)
    assert "relational.absent" in stack.unresolved


def test_resolved_stack_helpers(mock_deployments_path: Path, tmp_path: Path) -> None:
    catalog = load_capabilities(mock_deployments_path)
    recipe = _recipe("demo", ["cache.redis", "vector_db.qdrant", "host.vercel"], tmp_path)
    stack = resolve(recipe, catalog)

    docker_services = stack.docker_services()
    assert [d.service for d in docker_services] == ["redis", "qdrant"]

    env_vars = stack.env_vars()
    assert env_vars == ["REDIS_URL", "QDRANT_URL", "QDRANT_API_KEY", "VERCEL_TOKEN"]

    bootstrap = stack.bootstrap_steps()
    # cache.redis has no bootstrap_step, qdrant + vercel do.
    assert bootstrap == ["bootstrap_vector_db", "emit_deploy_configs"]

    targets = stack.deploy_targets()
    assert targets == ["vercel"]


def test_resolved_stack_env_vars_dedupe(tmp_path: Path) -> None:
    # Hand-build two caps sharing an env var to verify dedup beyond fixture coverage.
    cap_a = Capability(
        id="cache.redis",
        kind="cache",
        path=tmp_path / "a",
        env_vars=["REDIS_URL", "SHARED"],
    )
    cap_b = Capability(
        id="queue.redis-streams",
        kind="queue",
        path=tmp_path / "b",
        env_vars=["REDIS_URL", "SHARED", "NEW"],
    )
    stack = ResolvedStack(capabilities=[cap_a, cap_b])
    assert stack.env_vars() == ["REDIS_URL", "SHARED", "NEW"]


def test_discovery_recipe_capabilities_round_trip(mock_deployments_path: Path) -> None:
    recipes = {r.slug: r for r in discover_recipes(mock_deployments_path)}
    catalog = load_capabilities(mock_deployments_path)
    rcp = recipes["with-capabilities"]
    stack = resolve(rcp, catalog)
    assert stack.ids() == ["cache.redis", "vector_db.qdrant", "host.vercel"]
    assert stack.unresolved == ["vector_db.nonexistent"]


def test_docker_fragment_validation_drops_bad_block(tmp_path: Path) -> None:
    # Capability with docker missing service field -> docker becomes None, capability still loads.
    bad = tmp_path / "docs" / "capabilities" / "cache" / "bad.md"
    bad.parent.mkdir(parents=True)
    bad.write_text(
        "---\n"
        "id: cache.bad\n"
        "kind: cache\n"
        "env_vars: [URL]\n"
        "docker:\n"
        "  image: redis:7\n"  # missing service
        "---\n\n# x\n",
        encoding="utf-8",
    )
    catalog = load_capabilities(tmp_path)
    assert "cache.bad" in catalog
    assert catalog["cache.bad"].docker is None


def test_docker_fragment_model_directly() -> None:
    # Sanity check the model accepts canonical input shape.
    d = DockerFragment(
        service="redis",
        image="redis:7-alpine",
        ports=["6379:6379"],
        environment={"FOO": "bar"},
    )
    assert d.environment == {"FOO": "bar"}
    assert d.healthcheck is None


def test_capability_catalog_metadata_keys_load_without_warnings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catalog-schema fields (card/cost_tier/layer/tags/…) and docker depends_on
    are part of the deployments capability schema; valid extras must not warn."""
    _reset_warn_dedupe()
    cap = tmp_path / "docs" / "capabilities" / "vector_db" / "qdrant.md"
    cap.parent.mkdir(parents=True)
    cap.write_text(
        "---\n"
        "id: vector_db.qdrant\n"
        "kind: vector_db\n"
        "layer: data\n"
        "provides: [embeddings_store]\n"
        "requires: []\n"
        "bootstrap_inputs: {}\n"
        "env_vars: [QDRANT_URL]\n"
        "docker:\n"
        "  service: qdrant\n"
        "  image: qdrant/qdrant:v1.12.0\n"
        "  depends_on: [postgres]\n"
        "probe: qdrant_collections\n"
        "bootstrap_step: bootstrap_vector_db\n"
        "provisioning_time: ~10s\n"
        "cost_tier: free\n"
        "est_tokens: 450\n"
        "card:\n"
        "  name: Qdrant\n"
        "tags: [vector-search, retrieval]\n"
        "when_to_load: recipe declares vector_db.qdrant\n"
        "---\n\n# Capability: vector_db.qdrant\n",
        encoding="utf-8",
    )
    catalog = load_capabilities(tmp_path)
    assert "vector_db.qdrant" in catalog
    assert catalog["vector_db.qdrant"].docker is not None  # docker fragment still parsed
    err = capsys.readouterr().err
    assert "unknown keys" not in err
    assert "depends_on" not in err


def test_capability_port_registry_keys_load_without_warnings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """implements and stack_docs are published catalog-schema fields (port
    registry + adapter doc paths); the per-file parser must accept them
    silently instead of warning once per capability."""
    _reset_warn_dedupe()
    cap = tmp_path / "docs" / "capabilities" / "obs" / "langsmith.md"
    cap.parent.mkdir(parents=True)
    cap.write_text(
        "---\n"
        "id: obs.langsmith\n"
        "kind: obs\n"
        "implements:\n"
        "  port: obs\n"
        '  interface_version: "1.0"\n'
        "provides: [tracing]\n"
        "env_vars: [LANGCHAIN_API_KEY]\n"
        "probe: langsmith_workspace\n"
        "stack_docs:\n"
        "  - stack/tracing-langfuse.md\n"
        "---\n\n# Capability: obs.langsmith\n",
        encoding="utf-8",
    )
    catalog = load_capabilities(tmp_path)
    assert "obs.langsmith" in catalog
    err = capsys.readouterr().err
    assert "unknown keys" not in err


def test_load_capabilities_parses_bootstrap_inputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """bootstrap_inputs is now a consumed key: it hydrates onto the Capability
    (preserving non-string values) so a bootstrap step can read it."""
    _reset_warn_dedupe()
    cap = tmp_path / "docs" / "capabilities" / "vector_db" / "pgvector.md"
    cap.parent.mkdir(parents=True)
    cap.write_text(
        "---\n"
        "id: vector_db.pgvector\n"
        "kind: vector_db\n"
        "env_vars: [DATABASE_URL]\n"
        "bootstrap_inputs:\n"
        "  vector_extension: vector\n"
        "  default_table_name: chunks\n"
        "  default_vector_size: 1536\n"
        "---\n\n# pgvector\n",
        encoding="utf-8",
    )
    bi = load_capabilities(tmp_path)["vector_db.pgvector"].bootstrap_inputs
    assert bi == {
        "vector_extension": "vector",
        "default_table_name": "chunks",
        "default_vector_size": 1536,
    }
    # Non-string values are preserved (not stringified) for the step to use.
    assert isinstance(bi["default_vector_size"], int)
    assert "unknown keys" not in capsys.readouterr().err


def test_load_capabilities_skips_templates_doc(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """TEMPLATES.md is a schema doc that lives in the capabilities dir, not a
    capability — it must be skipped by name, not warned about as malformed."""
    _reset_warn_dedupe()
    caps = tmp_path / "docs" / "capabilities"
    (caps / "cache").mkdir(parents=True)
    (caps / "TEMPLATES.md").write_text(
        "# Capability templates\n\nnot a capability\n", encoding="utf-8"
    )
    (caps / "cache" / "redis.md").write_text(
        "---\nid: cache.redis\nkind: cache\nenv_vars: [REDIS_URL]\n---\n\n# redis\n",
        encoding="utf-8",
    )
    catalog = load_capabilities(tmp_path)
    assert "cache.redis" in catalog  # the real capability still loads
    err = capsys.readouterr().err
    assert "TEMPLATES" not in err
    assert "missing frontmatter" not in err


def test_apply_hosting_overrides_drops_docker_for_cloud(mock_deployments_path: Path) -> None:
    """Cloud mode keeps the capability but nulls its docker fragment, so the
    compose merge and readiness gating treat it as a managed service."""
    from agent_scaffold.capabilities import apply_hosting_overrides, resolve

    catalog = load_capabilities(mock_deployments_path)
    recipe = next(
        r for r in discover_recipes(mock_deployments_path) if r.slug == "customer-support-triage"
    )
    stack = resolve(recipe, catalog, add_capabilities=["cache.redis"])
    dockered = [c.id for c in stack.capabilities if c.docker is not None]
    assert dockered, "fixture stack should have at least one docker capability"
    target = dockered[0]

    overridden = apply_hosting_overrides(stack, {target: "cloud"})
    by_id = {c.id: c for c in overridden.capabilities}
    assert by_id[target].docker is None
    assert overridden.ids() == stack.ids()
    # env vars survive the override — they are how the cloud endpoint is wired.
    assert by_id[target].env_vars == next(c for c in stack.capabilities if c.id == target).env_vars
    # the original stack is untouched.
    assert next(c for c in stack.capabilities if c.id == target).docker is not None


def test_apply_hosting_overrides_docker_and_unknown_are_noops(
    mock_deployments_path: Path,
) -> None:
    from agent_scaffold.capabilities import apply_hosting_overrides, resolve

    catalog = load_capabilities(mock_deployments_path)
    recipe = next(
        r for r in discover_recipes(mock_deployments_path) if r.slug == "customer-support-triage"
    )
    stack = resolve(recipe, catalog, add_capabilities=["cache.redis"])
    unchanged = apply_hosting_overrides(stack, {})
    assert unchanged is stack
    dockered = [c.id for c in stack.capabilities if c.docker is not None]
    assert dockered
    kept = apply_hosting_overrides(stack, {dockered[0]: "docker", "obs.nope": "cloud"})
    assert [c.id for c in kept.capabilities if c.docker is not None] == dockered
