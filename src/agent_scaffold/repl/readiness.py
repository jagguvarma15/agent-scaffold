"""Shared readiness checks for the REPL: the Anthropic key, Docker, and the
selected stack's env vars.

``/config`` (fill the gaps), ``/status`` (show them), and the generate gate all
route through :func:`config_requirements` so the three agree on what "configured"
means. The gate is the only *blocking* surface — generation refuses to spend
tokens until :func:`required_gaps` is empty.

Docker-provided infra (a postgres/redis capability with a ``docker:`` fragment,
or an external service with a ``docker_service``) is deliberately **not** gated:
``up`` wires those container env vars, so asking the user to set ``DATABASE_URL``
by hand would be wrong. Only the agent key and genuinely-external credentials
(cloud vector DBs, hosted observability, search APIs) block.
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
    """Env vars the current selections need, Anthropic key first.

    The key is always required (generation can't run without it). When a recipe
    is selected, each external-service / capability var is added; vars a docker
    container provides are marked satisfied + non-required so they never gate.
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
                satisfied=req.satisfied or in_docker,
                required=req.required and not in_docker,
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
