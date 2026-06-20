"""Shared readiness checks for the REPL: the Anthropic key, Docker, and the
selected stack's env vars.

``/config`` (fill the gaps), ``/status`` (show them), and the generate gate all
route through :func:`config_requirements` so the three agree on what "configured"
means. The gate is the only *blocking* surface — generation refuses to spend
tokens until :func:`required_gaps` is empty.

The minimal docker sandbox needs **only the Anthropic key**. Everything else is
optional "connect later":

- Docker-provided infra (a postgres/redis capability with a ``docker:`` fragment,
  or an external service with a ``docker_service``) is supplied by the sandbox
  containers — shown ✓ "in sandbox", never asked for.
- External/cloud credentials (hosted observability, cloud vector DBs, search APIs)
  are shown ○ optional — the agent runs without them; set them via ``/config``
  whenever you want to connect that service.

So :func:`required_gaps` returns at most ``ANTHROPIC_API_KEY`` — the agent's key is
the only thing generation truly can't run without.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from agent_scaffold.auth import ENV_API_KEY, resolve_active
from agent_scaffold.capabilities import ResolvedStack
from agent_scaffold.discovery import Recipe
from agent_scaffold.preflight import EnvRequirement, collect_env_requirements
from agent_scaffold.repl._capabilities import resolve_stack_for_session
from agent_scaffold.repl.session import SessionState


def config_requirements(state: SessionState) -> list[EnvRequirement]:
    """Env vars for the current selections, Anthropic key first.

    Only the Anthropic key is ``required`` — it's the one thing generation can't
    run without. Every other var is ``optional`` (``required=False``): docker-
    provided infra is marked satisfied + "in sandbox" (the containers supply it),
    and external/cloud credentials stay unsatisfied + optional ("connect later").
    """
    key_ok = resolve_active() is not None
    reqs: list[EnvRequirement] = [
        EnvRequirement(name=ENV_API_KEY, source="agent", required=True, satisfied=key_ok)
    ]
    if state.recipe is None:
        return reqs

    project_dir = state.dest or Path.cwd()
    stack = resolve_stack_for_session(state)
    provided = _docker_provided_vars(state.recipe, stack)
    for req in collect_env_requirements(state.recipe, None, stack, project_dir):
        if req.name == ENV_API_KEY:
            continue
        in_docker = req.name in provided
        reqs.append(
            replace(
                req,
                source=f"{req.source} — in sandbox" if in_docker else req.source,
                satisfied=req.satisfied or in_docker,
                required=False,  # only the Anthropic key blocks; the rest connect later
            )
        )
    return reqs


def required_gaps(state: SessionState) -> list[str]:
    """Names of required, unsatisfied config values — the blocking set."""
    return [r.name for r in config_requirements(state) if r.required and not r.satisfied]


def docker_status(*, timeout: float = 3.0) -> tuple[bool, str]:
    """``(ok, reason)`` from ``docker info`` — for the /status readiness line.

    Short timeout so /status stays snappy even when the daemon is hung.
    """
    from agent_scaffold.steps.docker_up import docker_available

    return docker_available(timeout=timeout)


def _docker_provided_vars(recipe: Recipe, stack: ResolvedStack | None) -> set[str]:
    """Env vars supplied by a docker container — excluded from the gate."""
    provided: set[str] = set()
    for svc in recipe.external_services:
        if svc.docker_service:
            provided.update(svc.env_vars)
    if stack is not None:
        for cap in stack.capabilities:
            if cap.docker is not None:
                provided.update(cap.env_vars)
    return provided


__all__ = ["config_requirements", "docker_status", "required_gaps"]
