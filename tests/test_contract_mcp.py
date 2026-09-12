"""Tests for the compose-volumes and MCP registry contract passes."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agent_scaffold.capabilities import Capability, DockerFragment, ResolvedStack
from agent_scaffold.contract import (
    ContractParseError,
    GeneratedFile,
    GenerationResult,
    assert_mcp_wiring,
    merge_capability_fragments,
    normalize_compose_volumes,
    normalize_mcp_registry_mount,
)
from agent_scaffold.discovery import MCPServerSpec


def _result(files: list[tuple[str, str]], language: str = "python") -> GenerationResult:
    return GenerationResult(
        project_name="demo",
        language=language,
        files=[GeneratedFile(path=p, content=c) for p, c in files],
        smoke_check="pytest",
    )


def _compose_of(result: GenerationResult) -> dict:
    for f in result.files:
        if f.path == "docker-compose.yml":
            return yaml.safe_load(f.content)
    raise AssertionError("no compose file in result")


def _arrowhead_cap() -> Capability:
    return Capability(
        id="mcp.arrowhead",
        kind="mcp",
        path=Path("/fake/arrowhead.md"),
        transport="streamable_http",
        endpoint="http://127.0.0.1:8004/mcp",
        docker=DockerFragment(
            service="arrowhead",
            image="ghcr.io/example/arrowhead:0.2.832",
            ports=["127.0.0.1:8004:8000"],
            volumes=["arrowhead_corpus:/app/documents"],
            environment={"ARROWHEAD_PROFILE": "${ARROWHEAD_PROFILE:-docs}"},
        ),
    )


def _arrowhead_server(env: dict[str, str] | None = None) -> MCPServerSpec:
    return MCPServerSpec(
        id="arrowhead",
        capability="mcp.arrowhead",
        transport="streamable_http",
        env=env or {},
    )


# ---------------------------------------------------------------------------
# normalize_compose_volumes
# ---------------------------------------------------------------------------


def test_fragment_merged_named_volume_gets_a_top_level_declaration() -> None:
    """The live bug: a capability fragment's named volume never reached top level."""
    compose = yaml.safe_dump({"services": {"app": {"build": "."}}})
    result = _result([("docker-compose.yml", compose)])
    merged = merge_capability_fragments(result, ResolvedStack(capabilities=[_arrowhead_cap()]))
    fixed = normalize_compose_volumes(merged)
    data = _compose_of(fixed)
    assert data["services"]["arrowhead"]["volumes"] == ["arrowhead_corpus:/app/documents"]
    assert data["volumes"] == {"arrowhead_corpus": {}}


def test_path_like_and_anonymous_sources_are_not_declared() -> None:
    compose = yaml.safe_dump(
        {
            "services": {
                "app": {
                    "build": ".",
                    "volumes": [
                        "./mcp.json:/app/mcp.json:ro",
                        "/var/run/docker.sock:/var/run/docker.sock",
                        "~/data:/data",
                        "$HOME/x:/x",
                        "/cache",  # anonymous volume: no colon
                        "pgdata:/var/lib/postgresql/data",
                    ],
                }
            }
        }
    )
    result = normalize_compose_volumes(_result([("docker-compose.yml", compose)]))
    assert _compose_of(result)["volumes"] == {"pgdata": {}}


def test_long_form_entries_follow_their_type() -> None:
    compose = yaml.safe_dump(
        {
            "services": {
                "app": {
                    "build": ".",
                    "volumes": [
                        {"type": "volume", "source": "modeldata", "target": "/models"},
                        {"type": "bind", "source": "./conf", "target": "/conf"},
                        {"source": "scratch", "target": "/scratch"},
                    ],
                }
            }
        }
    )
    result = normalize_compose_volumes(_result([("docker-compose.yml", compose)]))
    assert _compose_of(result)["volumes"] == {"modeldata": {}, "scratch": {}}


def test_existing_top_level_definitions_are_preserved() -> None:
    compose = yaml.safe_dump(
        {
            "services": {"db": {"image": "postgres:16", "volumes": ["pgdata:/var/lib/pg"]}},
            "volumes": {"pgdata": {"driver": "local"}},
        }
    )
    original = _result([("docker-compose.yml", compose)])
    result = normalize_compose_volumes(original)
    # Already declared: nothing to add, same object back.
    assert result is original


def test_normalize_compose_volumes_is_idempotent() -> None:
    compose = yaml.safe_dump(
        {"services": {"db": {"image": "postgres:16", "volumes": ["pgdata:/var/lib/pg"]}}}
    )
    once = normalize_compose_volumes(_result([("docker-compose.yml", compose)]))
    twice = normalize_compose_volumes(once)
    assert twice is once


def test_normalize_compose_volumes_noop_without_compose() -> None:
    original = _result([("README.md", "hi")])
    assert normalize_compose_volumes(original) is original


# ---------------------------------------------------------------------------
# normalize_mcp_registry_mount
# ---------------------------------------------------------------------------


def test_registry_mount_added_to_the_app_service() -> None:
    compose = yaml.safe_dump({"services": {"app": {"build": "."}}})
    result = normalize_mcp_registry_mount(
        _result([("docker-compose.yml", compose)]),
        ResolvedStack(capabilities=[_arrowhead_cap()]),
        [_arrowhead_server()],
    )
    volumes = _compose_of(result)["services"]["app"]["volumes"]
    assert "./mcp.json:/app/mcp.json:ro" in volumes


def test_registry_mount_targets_the_dockerfile_workdir() -> None:
    compose = yaml.safe_dump({"services": {"app": {"build": "."}}})
    dockerfile = "FROM python:3.11-slim\nWORKDIR /srv\nCOPY . .\n"
    result = normalize_mcp_registry_mount(
        _result([("docker-compose.yml", compose), ("Dockerfile", dockerfile)]),
        None,
        [_arrowhead_server()],
    )
    volumes = _compose_of(result)["services"]["app"]["volumes"]
    assert "./mcp.json:/srv/mcp.json:ro" in volumes


def test_existing_registry_mount_is_left_alone() -> None:
    compose = yaml.safe_dump(
        {"services": {"app": {"build": ".", "volumes": ["./mcp.json:/opt/mcp.json:ro"]}}}
    )
    original = _result([("docker-compose.yml", compose)])
    result = normalize_mcp_registry_mount(original, None, [_arrowhead_server()])
    assert result is original


def test_registry_mount_noop_without_servers() -> None:
    compose = yaml.safe_dump({"services": {"app": {"build": "."}}})
    original = _result([("docker-compose.yml", compose)])
    assert normalize_mcp_registry_mount(original, None, []) is original


def test_recipe_env_default_pins_the_bound_service_env() -> None:
    """A non-sentinel env value becomes ${VAR:-value} on the capability's service."""
    compose = yaml.safe_dump({"services": {"app": {"build": "."}}})
    result = _result([("docker-compose.yml", compose)])
    merged = merge_capability_fragments(result, ResolvedStack(capabilities=[_arrowhead_cap()]))
    rewritten = normalize_mcp_registry_mount(
        merged,
        ResolvedStack(capabilities=[_arrowhead_cap()]),
        [_arrowhead_server(env={"ARROWHEAD_PROFILE": "coding", "ARROWHEAD_API_KEY": "required"})],
    )
    env = _compose_of(rewritten)["services"]["arrowhead"]["environment"]
    # The recipe default wins over the fragment's authored default...
    assert env["ARROWHEAD_PROFILE"] == "${ARROWHEAD_PROFILE:-coding}"
    # ...and sentinels never turn into compose entries.
    assert "ARROWHEAD_API_KEY" not in env


# ---------------------------------------------------------------------------
# assert_mcp_wiring
# ---------------------------------------------------------------------------


def test_wiring_passes_when_python_source_reads_the_registry() -> None:
    files = [("app/main.py", 'REGISTRY = Path("mcp.json")\n')]
    assert_mcp_wiring(_result(files), [_arrowhead_server()])


def test_wiring_passes_when_typescript_source_reads_the_registry() -> None:
    files = [("src/index.ts", 'const registry = "mcp.json";\n')]
    assert_mcp_wiring(_result(files, language="typescript"), [_arrowhead_server()])


def test_wiring_fails_when_only_non_source_files_mention_the_registry() -> None:
    files = [
        ("README.md", "The agent reads mcp.json at boot."),
        ("docker-compose.yml", "services:\n  app:\n    volumes:\n      - ./mcp.json:/app:ro\n"),
        ("app/main.py", "print('hello')\n"),
    ]
    with pytest.raises(ContractParseError) as excinfo:
        assert_mcp_wiring(_result(files), [_arrowhead_server()])
    assert excinfo.value.tier == "required-files"
    assert "arrowhead" in excinfo.value.reason
    assert "mcp.json" in excinfo.value.reason


def test_wiring_noop_without_servers() -> None:
    assert_mcp_wiring(_result([("app/main.py", "print('hi')\n")]), [])


def test_registry_mount_skips_non_root_builds() -> None:
    """The mount target path comes from the root Dockerfile, so only services
    built from the project root get the registry — a ./frontend build never
    reads it."""
    compose = yaml.safe_dump(
        {
            "services": {
                "app": {"build": "."},
                "frontend": {"build": {"context": "./frontend", "dockerfile": "Dockerfile"}},
            }
        }
    )
    result = normalize_mcp_registry_mount(
        _result([("docker-compose.yml", compose)]),
        ResolvedStack(capabilities=[_arrowhead_cap()]),
        [_arrowhead_server()],
    )
    services = _compose_of(result)["services"]
    assert services["app"]["volumes"] == ["./mcp.json:/app/mcp.json:ro"]
    assert "volumes" not in services["frontend"]


def test_registry_mount_falls_back_to_conventional_names() -> None:
    compose = yaml.safe_dump(
        {
            "services": {
                "app": {"image": "demo:latest"},
                "redis": {"image": "redis:7-alpine"},
            }
        }
    )
    result = normalize_mcp_registry_mount(
        _result([("docker-compose.yml", compose)]),
        ResolvedStack(capabilities=[_arrowhead_cap()]),
        [_arrowhead_server()],
    )
    services = _compose_of(result)["services"]
    assert services["app"]["volumes"] == ["./mcp.json:/app/mcp.json:ro"]
    assert "volumes" not in services["redis"]
