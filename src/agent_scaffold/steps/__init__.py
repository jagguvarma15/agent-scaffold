"""Concrete orchestrator steps and the factory that selects them per project.

Q6 ships the first three steps:

- :class:`InstallDepsStep`     — Python ``uv lock`` + ``uv sync``
- :class:`DockerUpStep`        — ``docker compose up -d`` declared services
- :class:`WireCredentialsStep` — prompt for missing env vars, store safely

Q7 will append migrations / seed / smoke / commit_push / open_editor to
:data:`ALL_STEP_CLASSES` and extend :func:`default_steps_for` to wire them
in dependency order. The orchestrator framework (:mod:`agent_scaffold.orchestrator`)
does not change — that's the whole point of Q5's design.
"""

from __future__ import annotations

from agent_scaffold.discovery import Recipe
from agent_scaffold.manifest import Manifest
from agent_scaffold.orchestrator import Step
from agent_scaffold.steps.docker_up import DockerUpStep
from agent_scaffold.steps.install_deps import InstallDepsStep
from agent_scaffold.steps.wire_credentials import WireCredentialsStep

ALL_STEP_CLASSES: tuple[type, ...] = (
    InstallDepsStep,
    DockerUpStep,
    WireCredentialsStep,
)


def default_steps_for(
    manifest: Manifest,
    recipe: Recipe | None,
    *,
    yes: bool = False,
) -> list[Step]:
    """Return the configured step instances for this project, in declaration order.

    The orchestrator topo-sorts on ``depends_on`` so the order here is only a
    tiebreaker for unrelated steps. Q7's additional steps will slot in by
    declaration order between the three below.

    ``recipe`` may be ``None`` if discovery failed; the step instances are
    still constructed so ``detect()`` can surface the SKIP/PENDING reason
    instead of the plan panel showing nothing.
    """
    del recipe  # currently only used by detect/apply via ctx, not by selection
    return [
        InstallDepsStep(),
        DockerUpStep(),
        WireCredentialsStep(yes=yes),
    ]


def step_class_by_id(step_id: str) -> type | None:
    """Resolve a step id (``"install_deps"``) to its class, for introspection/tests."""
    for cls in ALL_STEP_CLASSES:
        # Each step class sets ``id`` as a class default on the dataclass.
        if getattr(cls, "id", None) == step_id:
            return cls
    return None


__all__ = [
    "ALL_STEP_CLASSES",
    "DockerUpStep",
    "InstallDepsStep",
    "Step",
    "WireCredentialsStep",
    "default_steps_for",
    "step_class_by_id",
]
