"""Tests for the capability catalog loader + resolver."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_scaffold.capabilities import (
    Capability,
    DockerFragment,
    ResolvedStack,
    load_capabilities,
    resolve,
)
from agent_scaffold.discovery import Recipe, discover_recipes


def test_load_capabilities_discovers_all_valid(mock_deployments_path: Path) -> None:
    catalog = load_capabilities(mock_deployments_path)
    # Three valid capabilities; README + malformed entries should be excluded.
    assert set(catalog) == {"vector_db.qdrant", "cache.redis", "host.vercel"}


def test_load_capabilities_skips_readme(
    mock_deployments_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    load_capabilities(mock_deployments_path)
    err = capsys.readouterr().err
    # README.md must NOT trigger a "missing 'id'" warning — it's intentionally skipped.
    assert "README" not in err


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
