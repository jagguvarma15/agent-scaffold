"""Tests for ``harden_scaffold_services`` — the compose hardening pass over
the app and scaffold-added frontend services."""

from __future__ import annotations

from pathlib import Path

import yaml

from agent_scaffold.capabilities import Capability, ResolvedStack
from agent_scaffold.contract import (
    GeneratedFile,
    GenerationResult,
    harden_scaffold_services,
    normalize_frontend_service,
)

_COMPOSE = """\
services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
  postgres:
    image: postgres:16-alpine
    ports:
      - "5432:5432"
"""


def _result(compose: str) -> GenerationResult:
    return GenerationResult(
        project_name="demo",
        language="python",
        files=[
            GeneratedFile(path="README.md", content="hi"),
            GeneratedFile(path="docker-compose.yml", content=compose),
        ],
        smoke_check="pytest",
    )


def _services(result: GenerationResult) -> dict:
    compose = next(f for f in result.files if f.path == "docker-compose.yml")
    return yaml.safe_load(compose.content)["services"]


def test_hardens_app_and_binds_loopback_but_leaves_capability_services() -> None:
    services = _services(harden_scaffold_services(_result(_COMPOSE), None))
    app = services["app"]
    assert app["security_opt"] == ["no-new-privileges:true"]
    assert app["cap_drop"] == ["ALL"]
    assert app["ports"] == ["127.0.0.1:8000:8000"]
    # Capability-authored fragments keep their authored shape.
    postgres = services["postgres"]
    assert "security_opt" not in postgres
    assert "cap_drop" not in postgres
    assert postgres["ports"] == ["5432:5432"]


def test_pins_every_port_shape_the_compose_syntax_allows() -> None:
    # An int, a bare string, a range, a protocol suffix, and the long-form
    # mapping all publish on 0.0.0.0 — the first pass only caught "H:C"
    # strings, leaving the rest wide open while AGENTS.md claimed loopback.
    compose = """\
services:
  app:
    build: {context: .}
    ports:
      - 8080
      - "9090"
      - "7000/udp"
      - "8000-8010"
      - target: 6000
        published: 6000
"""
    services = _services(harden_scaffold_services(_result(compose), None))
    ports = services["app"]["ports"]
    assert ports[0] == "127.0.0.1:8080:8080"
    assert ports[1] == "127.0.0.1:9090:9090"
    assert ports[2] == "127.0.0.1:7000:7000/udp"
    assert ports[3] == "127.0.0.1:8000-8010:8000-8010"
    assert ports[4] == {"target": 6000, "published": 6000, "host_ip": "127.0.0.1"}


def test_long_form_port_with_host_ip_is_respected() -> None:
    compose = """\
services:
  app:
    build: {context: .}
    ports:
      - target: 6000
        published: 6000
        host_ip: 0.0.0.0
"""
    services = _services(harden_scaffold_services(_result(compose), None))
    assert services["app"]["ports"][0]["host_ip"] == "0.0.0.0"


def test_respects_author_set_values_and_existing_host_ip() -> None:
    compose = """\
services:
  app:
    build:
      context: .
    security_opt:
      - seccomp:custom.json
    cap_drop:
      - NET_RAW
    ports:
      - "0.0.0.0:8000:8000"
"""
    app = _services(harden_scaffold_services(_result(compose), None))["app"]
    # Author intent wins: additive only, never replaced.
    assert app["security_opt"] == ["seccomp:custom.json"]
    assert app["cap_drop"] == ["NET_RAW"]
    # An entry that already names a host ip is respected.
    assert app["ports"] == ["0.0.0.0:8000:8000"]


def test_frontend_gets_the_minimal_nginx_cap_set() -> None:
    frontend_cap = Capability(
        id="frontend.minimal-chat",
        kind="frontend",
        path=Path("/x.md"),
        env_vars=["VITE_AGENT_URL"],
        serve_in_container=True,
    )
    stack = ResolvedStack(capabilities=[frontend_cap])
    with_frontend = normalize_frontend_service(_result(_COMPOSE), stack)
    services = _services(harden_scaffold_services(with_frontend, stack))
    frontend = services["frontend"]
    assert frontend["cap_drop"] == ["ALL"]
    assert frontend["cap_add"] == ["CHOWN", "SETGID", "SETUID", "NET_BIND_SERVICE"]
    assert frontend["security_opt"] == ["no-new-privileges:true"]
    assert frontend["ports"] == ["127.0.0.1:3000:3000"]


def test_noop_without_compose() -> None:
    result = GenerationResult(
        project_name="demo",
        language="python",
        files=[GeneratedFile(path="README.md", content="hi")],
        smoke_check="pytest",
    )
    assert harden_scaffold_services(result, None) is result
