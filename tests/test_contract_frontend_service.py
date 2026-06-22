"""Tests for ``normalize_frontend_service`` — the pass that containerizes the
frontend into the docker sandbox when a frontend capability opts in."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agent_scaffold.capabilities import Capability, ResolvedStack
from agent_scaffold.contract import (
    ContractParseError,
    GeneratedFile,
    GenerationResult,
    assert_chat_endpoint,
    assert_cors,
    normalize_frontend_service,
)

_COMPOSE = """\
services:
  app:
    build: .
    ports:
      - "8000:8000"
  postgres:
    image: postgres:16-alpine
"""


def _result(compose: str | None) -> GenerationResult:
    files = [GeneratedFile(path="README.md", content="hi")]
    if compose is not None:
        files.append(GeneratedFile(path="docker-compose.yml", content=compose))
    return GenerationResult(
        project_name="demo", language="python", files=files, smoke_check="pytest"
    )


def _frontend_cap(*, serve_in_container: bool, env_vars: list[str] | None = None) -> Capability:
    return Capability(
        id="frontend.minimal-chat",
        kind="frontend",
        path=Path("/f.md"),
        env_vars=env_vars if env_vars is not None else ["NEXT_PUBLIC_AGENT_URL"],
        serve_in_container=serve_in_container,
    )


def _frontend_service(result: GenerationResult) -> dict | None:
    compose = next(f for f in result.files if f.path == "docker-compose.yml")
    return yaml.safe_load(compose.content)["services"].get("frontend")


def test_adds_built_frontend_service_wired_to_backend() -> None:
    stack = ResolvedStack(capabilities=[_frontend_cap(serve_in_container=True)])
    svc = _frontend_service(normalize_frontend_service(_result(_COMPOSE), stack))
    assert svc is not None
    assert svc["build"]["context"] == "./frontend"
    assert svc["build"]["dockerfile"] == "Dockerfile"
    assert svc["ports"] == ["3000:3000"]
    # Host-mapped backend URL passed as a BUILD ARG — the browser runs on the host
    # (not the compose net), and a static Vite build bakes VITE_* at build time,
    # so nginx serving it ignores runtime env (there must be no `environment`).
    assert svc["build"]["args"]["NEXT_PUBLIC_AGENT_URL"] == "http://localhost:8000"
    assert "environment" not in svc
    assert svc["depends_on"] == ["app"]


def test_inert_without_serve_in_container() -> None:
    # A frontend capability that runs locally (no Dockerfile) must NOT get a
    # compose service — otherwise `build: ./frontend` would reference a missing
    # Dockerfile and break `docker compose up`.
    stack = ResolvedStack(capabilities=[_frontend_cap(serve_in_container=False)])
    assert _frontend_service(normalize_frontend_service(_result(_COMPOSE), stack)) is None


def test_no_op_without_frontend_capability() -> None:
    stack = ResolvedStack(
        capabilities=[Capability(id="cache.redis", kind="cache", path=Path("/r.md"))]
    )
    r = _result(_COMPOSE)
    assert normalize_frontend_service(r, stack) is r  # unchanged object


def test_no_op_when_frontend_service_already_present() -> None:
    compose = _COMPOSE + "  frontend:\n    build: ./frontend\n"
    stack = ResolvedStack(capabilities=[_frontend_cap(serve_in_container=True)])
    out = normalize_frontend_service(_result(compose), stack)
    # Existing service untouched (no overwrite, no second service).
    data = yaml.safe_load(next(f for f in out.files if f.path == "docker-compose.yml").content)
    assert data["services"]["frontend"] == {"build": "./frontend"}


def test_backend_url_uses_host_mapped_port() -> None:
    compose = _COMPOSE.replace('"8000:8000"', '"8080:8000"')  # host 8080 → container 8000
    stack = ResolvedStack(capabilities=[_frontend_cap(serve_in_container=True)])
    svc = _frontend_service(normalize_frontend_service(_result(compose), stack))
    assert svc is not None
    assert svc["build"]["args"]["NEXT_PUBLIC_AGENT_URL"] == "http://localhost:8080"


def test_streamlit_url_var() -> None:
    stack = ResolvedStack(
        capabilities=[_frontend_cap(serve_in_container=True, env_vars=["AGENT_URL"])]
    )
    svc = _frontend_service(normalize_frontend_service(_result(_COMPOSE), stack))
    assert svc is not None
    assert svc["build"]["args"]["AGENT_URL"] == "http://localhost:8000"


def test_no_op_without_compose_file() -> None:
    stack = ResolvedStack(capabilities=[_frontend_cap(serve_in_container=True)])
    r = _result(None)
    assert normalize_frontend_service(r, stack) is r


def test_no_op_when_stack_is_none() -> None:
    r = _result(_COMPOSE)
    assert normalize_frontend_service(r, None) is r


def test_agent_title_passed_as_build_arg() -> None:
    stack = ResolvedStack(capabilities=[_frontend_cap(serve_in_container=True)])
    svc = _frontend_service(
        normalize_frontend_service(_result(_COMPOSE), stack, agent_title="Docs Q&A")
    )
    assert svc is not None
    # Title is a BUILD arg (baked into the static bundle), alongside the URL.
    assert svc["build"]["args"]["VITE_AGENT_TITLE"] == "Docs Q&A"
    assert svc["build"]["args"]["NEXT_PUBLIC_AGENT_URL"] == "http://localhost:8000"
    assert "environment" not in svc


def test_no_title_arg_when_unset() -> None:
    stack = ResolvedStack(capabilities=[_frontend_cap(serve_in_container=True)])
    svc = _frontend_service(normalize_frontend_service(_result(_COMPOSE), stack))
    assert svc is not None
    assert "VITE_AGENT_TITLE" not in svc["build"]["args"]


def _gen(content: str) -> GenerationResult:
    return GenerationResult(
        project_name="demo",
        language="python",
        files=[GeneratedFile(path="app/main.py", content=content)],
        smoke_check="pytest",
    )


def test_assert_chat_endpoint_raises_when_route_missing() -> None:
    stack = ResolvedStack(capabilities=[_frontend_cap(serve_in_container=True)])
    with pytest.raises(ContractParseError, match="/chat"):
        assert_chat_endpoint(_gen("print('no route here')"), stack)


def test_assert_chat_endpoint_ok_when_route_present() -> None:
    stack = ResolvedStack(capabilities=[_frontend_cap(serve_in_container=True)])
    assert_chat_endpoint(_gen('@app.post("/chat")\ndef chat(): ...'), stack)  # no raise


def test_assert_chat_endpoint_noop_without_container_frontend() -> None:
    # A local (non-container) frontend doesn't call the sandbox /chat — no gate.
    stack = ResolvedStack(capabilities=[_frontend_cap(serve_in_container=False)])
    assert_chat_endpoint(_gen("no chat route"), stack)  # no raise
    assert_chat_endpoint(_gen("no chat route"), None)  # no raise (no stack)


def test_assert_cors_raises_when_absent() -> None:
    stack = ResolvedStack(capabilities=[_frontend_cap(serve_in_container=True)])
    with pytest.raises(ContractParseError, match="CORS"):
        assert_cors(_gen('@app.post("/chat")\ndef chat(): ...'), stack)


def test_assert_cors_ok_when_middleware_present() -> None:
    stack = ResolvedStack(capabilities=[_frontend_cap(serve_in_container=True)])
    assert_cors(_gen('app.add_middleware(CORSMiddleware, allow_origins=["*"])'), stack)  # no raise


def test_assert_cors_noop_without_container_frontend() -> None:
    # No cross-origin chat UI → no CORS requirement.
    stack = ResolvedStack(capabilities=[_frontend_cap(serve_in_container=False)])
    assert_cors(_gen("no cors here"), stack)  # no raise
    assert_cors(_gen("no cors here"), None)  # no raise (no stack)
