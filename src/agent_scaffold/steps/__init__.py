"""Concrete orchestrator steps and the factory that selects them per project.

Steps shipped:

- :class:`InstallDepsStep`     — Python ``uv lock`` + ``uv sync``                  (Q6)
- :class:`DockerUpStep`        — ``docker compose up -d`` declared services        (Q6)
- :class:`WireCredentialsStep` — prompt for missing env vars, store safely         (Q6)
- :class:`MigrationsStep`      — ``alembic upgrade head`` per migrating service    (Q7)
- :class:`SeedStep`            — run ``scripts/seed.py`` / ``scripts/seed.sh``     (Q7)
- :class:`SmokeTestStep`       — ``scripts/smoke.sh`` or ``pytest -m smoke``       (Q7)
- :class:`CommitPushStep`      — opt-in commit + push of provisioning artifacts    (Q7)
- :class:`OpenEditorStep`      — open README in ``$EDITOR`` when done              (Q7)

The orchestrator framework (:mod:`agent_scaffold.orchestrator`) is unchanged by
Q7 — adding a step is one class + one entry in :data:`ALL_STEP_CLASSES`.
"""

from __future__ import annotations

from agent_scaffold.discovery import Recipe
from agent_scaffold.manifest import Manifest
from agent_scaffold.orchestrator import Step
from agent_scaffold.steps.commit_push import CommitPushStep
from agent_scaffold.steps.docker_up import DockerUpStep
from agent_scaffold.steps.install_deps import InstallDepsStep
from agent_scaffold.steps.migrations import MigrationsStep
from agent_scaffold.steps.open_editor import OpenEditorStep
from agent_scaffold.steps.seed import SeedStep
from agent_scaffold.steps.smoke_test import SmokeTestStep
from agent_scaffold.steps.wire_credentials import WireCredentialsStep

ALL_STEP_CLASSES: tuple[type, ...] = (
    InstallDepsStep,
    DockerUpStep,
    WireCredentialsStep,
    MigrationsStep,
    SeedStep,
    SmokeTestStep,
    CommitPushStep,
    OpenEditorStep,
)


def default_steps_for(
    manifest: Manifest,
    recipe: Recipe | None,
    *,
    yes: bool = False,
    confirm_commit_push: bool = False,
) -> list[Step]:
    """Return the configured step instances for this project, in declaration order.

    ``commit_push`` is included only when the recipe's (future) ``setup_steps``
    field opts in. ``open_editor`` always lives in the registry; its ``detect()``
    handles the ``--yes``-mode silent-skip itself.

    ``recipe`` may be ``None`` if discovery failed; the step instances are
    still constructed so ``detect()`` can surface the SKIP/PENDING reason
    instead of an empty plan panel.
    """
    setup_steps = _recipe_setup_steps(recipe)
    steps: list[Step] = [
        InstallDepsStep(),
        DockerUpStep(),
        WireCredentialsStep(yes=yes),
        MigrationsStep(),
        SeedStep(),
        SmokeTestStep(),
    ]
    if "commit_push" in setup_steps:
        steps.append(CommitPushStep(confirm_commit_push=confirm_commit_push))
    steps.append(OpenEditorStep(yes=yes))
    return steps


def _recipe_setup_steps(recipe: Recipe | None) -> frozenset[str]:
    """Read ``setup_steps`` off the recipe if available; tolerate missing field.

    Discovery's :class:`Recipe` model doesn't (yet) carry ``setup_steps`` — Q3
    schema only formalised ``external_services``. Until the discovery layer
    grows the field, recipes can still drop a sibling marker; for now we just
    return the empty set so the opt-in steps stay off unless ``--only`` forces.
    """
    del recipe  # field not yet on Recipe; keep the call site stable for forward-compat
    return frozenset()


def step_class_by_id(step_id: str) -> type | None:
    """Resolve a step id (``"install_deps"``) to its class, for introspection/tests."""
    for cls in ALL_STEP_CLASSES:
        if getattr(cls, "id", None) == step_id:
            return cls
    return None


__all__ = [
    "ALL_STEP_CLASSES",
    "CommitPushStep",
    "DockerUpStep",
    "InstallDepsStep",
    "MigrationsStep",
    "OpenEditorStep",
    "SeedStep",
    "SmokeTestStep",
    "Step",
    "WireCredentialsStep",
    "default_steps_for",
    "step_class_by_id",
]
