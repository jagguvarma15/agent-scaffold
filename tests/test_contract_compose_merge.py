"""Tests for ``merge_capability_fragments`` in ``contract``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from agent_scaffold.capabilities import (
    Capability,
    DockerFragment,
    ResolvedStack,
)
from agent_scaffold.contract import (
    GeneratedFile,
    GenerationResult,
    merge_capability_fragments,
)


def _cap(
    name: str,
    *,
    service: str,
    image: str,
    ports: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> Capability:
    return Capability(
        id=name,
        kind="cache" if name.startswith("cache") else "vector_db",
        path=Path(f"/fake/{name}.md"),
        docker=DockerFragment(
            service=service,
            image=image,
            ports=ports or [],
            environment=env or {},
        ),
    )


def _result(files: list[tuple[str, str]]) -> GenerationResult:
    return GenerationResult(
        project_name="demo",
        language="python",
        files=[GeneratedFile(path=p, content=c) for p, c in files],
        smoke_check="pytest",
    )


def test_no_op_when_stack_is_none() -> None:
    r = _result([("README.md", "hi")])
    out = merge_capability_fragments(r, None)
    assert out is r


def test_no_op_when_no_docker_caps() -> None:
    stack = ResolvedStack(
        capabilities=[Capability(id="obs.langsmith", kind="obs", path=Path("/x.md"))]
    )
    r = _result([("README.md", "hi")])
    out = merge_capability_fragments(r, stack)
    assert out is r


def test_inserts_missing_service_into_existing_compose() -> None:
    existing = yaml.safe_dump(
        {"services": {"redis": {"image": "redis:7-alpine", "ports": ["6379:6379"]}}}
    )
    stack = ResolvedStack(
        capabilities=[
            _cap("cache.redis", service="redis", image="redis:7-alpine", ports=["6379:6379"]),
            _cap(
                "vector_db.qdrant",
                service="qdrant",
                image="qdrant/qdrant:v1.12.0",
                ports=["6333:6333"],
            ),
        ]
    )
    r = _result([("docker-compose.yml", existing)])
    out = merge_capability_fragments(r, stack)
    compose = next(f for f in out.files if f.path == "docker-compose.yml")
    data = yaml.safe_load(compose.content)
    assert set(data["services"]) == {"redis", "qdrant"}
    assert data["services"]["qdrant"]["image"] == "qdrant/qdrant:v1.12.0"
    assert data["services"]["qdrant"]["ports"] == ["6333:6333"]


def test_creates_compose_when_missing() -> None:
    stack = ResolvedStack(
        capabilities=[_cap("cache.redis", service="redis", image="redis:7-alpine")]
    )
    r = _result([("README.md", "hi")])
    out = merge_capability_fragments(r, stack)
    paths = {f.path for f in out.files}
    assert "docker-compose.yml" in paths
    compose = next(f for f in out.files if f.path == "docker-compose.yml")
    assert "redis" in yaml.safe_load(compose.content)["services"]


def test_capability_image_pin_wins_on_conflict() -> None:
    existing = yaml.safe_dump({"services": {"redis": {"image": "redis:latest"}}})
    stack = ResolvedStack(
        capabilities=[_cap("cache.redis", service="redis", image="redis:7-alpine")]
    )
    r = _result([("docker-compose.yml", existing)])
    out = merge_capability_fragments(r, stack)
    compose = next(f for f in out.files if f.path == "docker-compose.yml")
    assert yaml.safe_load(compose.content)["services"]["redis"]["image"] == "redis:7-alpine"


def test_existing_service_with_matching_image_untouched() -> None:
    existing = yaml.safe_dump(
        {
            "services": {
                "redis": {
                    "image": "redis:7-alpine",
                    "volumes": ["custom_vol:/data"],
                    "container_name": "user_chose_this",
                }
            }
        }
    )
    stack = ResolvedStack(
        capabilities=[_cap("cache.redis", service="redis", image="redis:7-alpine")]
    )
    r = _result([("docker-compose.yml", existing)])
    out = merge_capability_fragments(r, stack)
    compose = next(f for f in out.files if f.path == "docker-compose.yml")
    data = yaml.safe_load(compose.content)
    # We didn't touch the model's redis block (image matches, so no override needed).
    assert data["services"]["redis"]["container_name"] == "user_chose_this"
    assert data["services"]["redis"]["volumes"] == ["custom_vol:/data"]


def test_merge_is_idempotent() -> None:
    stack = ResolvedStack(
        capabilities=[
            _cap("cache.redis", service="redis", image="redis:7-alpine"),
            _cap("vector_db.qdrant", service="qdrant", image="qdrant/qdrant:v1.12.0"),
        ]
    )
    r = _result([("README.md", "hi")])
    once = merge_capability_fragments(r, stack)
    twice = merge_capability_fragments(once, stack)
    one_compose = next(f.content for f in once.files if f.path == "docker-compose.yml")
    two_compose = next(f.content for f in twice.files if f.path == "docker-compose.yml")
    assert one_compose == two_compose


def test_services_alphabetised_deterministically() -> None:
    stack = ResolvedStack(
        capabilities=[
            _cap("vector_db.qdrant", service="qdrant", image="qdrant/qdrant:v1.12.0"),
            _cap("cache.redis", service="redis", image="redis:7-alpine"),
        ]
    )
    r = _result([("README.md", "hi")])
    out = merge_capability_fragments(r, stack)
    compose = next(f for f in out.files if f.path == "docker-compose.yml")
    data = yaml.safe_load(compose.content)
    assert list(data["services"]) == ["qdrant", "redis"]


def test_malformed_existing_compose_is_replaced() -> None:
    r = _result([("docker-compose.yml", "not: valid: yaml:")])
    stack = ResolvedStack(
        capabilities=[_cap("cache.redis", service="redis", image="redis:7-alpine")]
    )
    out = merge_capability_fragments(r, stack)
    compose = next(f for f in out.files if f.path == "docker-compose.yml")
    data: dict[str, Any] = yaml.safe_load(compose.content)
    assert "redis" in data["services"]


def test_platform_carries_into_the_merged_service() -> None:
    """Single-arch images (TEI's CPU builds are amd64-only) need compose's
    platform key to run emulated on arm64 hosts — the fragment used to drop
    it silently, and docker compose hard-failed the pull on Apple Silicon."""
    cap = Capability(
        id="guardrail.injection-classifier",
        kind="guardrail",
        path=Path("/fake/g.md"),
        docker=DockerFragment(
            service="guardrail-classifier",
            image="ghcr.io/huggingface/text-embeddings-inference:cpu-1.6",
            ports=["8081:80"],
            platform="linux/amd64",
        ),
    )
    stack = ResolvedStack(capabilities=[cap])
    result = merge_capability_fragments(_result([("README.md", "hi")]), stack)
    compose = next(f for f in result.files if f.path == "docker-compose.yml")
    services = yaml.safe_load(compose.content)["services"]
    assert services["guardrail-classifier"]["platform"] == "linux/amd64"


def test_platform_absent_when_fragment_declares_none() -> None:
    cap = _cap("cache.redis", service="redis", image="redis:7-alpine")
    stack = ResolvedStack(capabilities=[cap])
    result = merge_capability_fragments(_result([("README.md", "hi")]), stack)
    compose = next(f for f in result.files if f.path == "docker-compose.yml")
    services = yaml.safe_load(compose.content)["services"]
    assert "platform" not in services["redis"]
