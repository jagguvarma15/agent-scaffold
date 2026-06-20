"""Concrete orchestrator steps and the factory that selects them per project.

Steps shipped:

- :class:`InstallDepsStep`             — Python ``uv lock`` + ``uv sync``
- :class:`DockerUpStep`                — ``docker compose up -d`` declared services
- :class:`WireCredentialsStep`         — prompt for missing env vars, store safely
- :class:`BootstrapVectorDbStep`       — init Qdrant / Chroma / pgvector collections
- :class:`BootstrapKafkaStep`          — create Kafka topics + Redis Stream groups
- :class:`MigrationsStep`              — ``alembic upgrade head`` per migrating service
- :class:`BootstrapLangSmithStep`      — create LangSmith project + write tracing env
- :class:`BootstrapObservabilityStep`  — provision Grafana datasources + dashboards
- :class:`SeedStep`                    — run ``scripts/seed.py`` / ``scripts/seed.sh``
- :class:`SmokeTestStep`               — ``scripts/smoke.sh`` or ``pytest -m smoke``
- :class:`BootstrapEvalsStep`          — run the eval suite once + store the baseline
- :class:`EmitDeployConfigsStep`       — write cloud-deploy configs from host.* caps
- :class:`LaunchFrontendStep`          — spawn frontend dev server in the background
- :class:`CommitPushStep`              — opt-in commit + push of provisioning artifacts
- :class:`OpenEditorStep`              — open README in ``$EDITOR`` when done

Adding a step is one class + one entry in :data:`ALL_STEP_CLASSES`.
"""

from __future__ import annotations

from agent_scaffold.discovery import Recipe
from agent_scaffold.manifest import Manifest
from agent_scaffold.orchestrator import Step
from agent_scaffold.steps.bootstrap_evals import BootstrapEvalsStep
from agent_scaffold.steps.bootstrap_kafka import BootstrapKafkaStep
from agent_scaffold.steps.bootstrap_langfuse import BootstrapLangfuseStep
from agent_scaffold.steps.bootstrap_langsmith import BootstrapLangSmithStep
from agent_scaffold.steps.bootstrap_observability import BootstrapObservabilityStep
from agent_scaffold.steps.bootstrap_vector_db import BootstrapVectorDbStep
from agent_scaffold.steps.commit_push import CommitPushStep
from agent_scaffold.steps.docker_up import DockerUpStep
from agent_scaffold.steps.emit_deploy_configs import EmitDeployConfigsStep
from agent_scaffold.steps.install_deps import InstallDepsStep
from agent_scaffold.steps.launch_backend import LaunchBackendStep
from agent_scaffold.steps.launch_frontend import LaunchFrontendStep
from agent_scaffold.steps.migrations import MigrationsStep
from agent_scaffold.steps.open_editor import OpenEditorStep
from agent_scaffold.steps.seed import SeedStep
from agent_scaffold.steps.smoke_test import SmokeTestStep
from agent_scaffold.steps.wire_credentials import WireCredentialsStep

ALL_STEP_CLASSES: tuple[type, ...] = (
    InstallDepsStep,
    DockerUpStep,
    WireCredentialsStep,
    BootstrapVectorDbStep,
    BootstrapKafkaStep,
    MigrationsStep,
    BootstrapLangSmithStep,
    BootstrapLangfuseStep,
    BootstrapObservabilityStep,
    SeedStep,
    SmokeTestStep,
    BootstrapEvalsStep,
    EmitDeployConfigsStep,
    LaunchBackendStep,
    LaunchFrontendStep,
    CommitPushStep,
    OpenEditorStep,
)


def default_steps_for(
    manifest: Manifest,
    recipe: Recipe | None,
    *,
    yes: bool = False,
    confirm_commit_push: bool = False,
    with_evals: bool = False,
    use_docker: bool = False,
) -> list[Step]:
    """Return the configured step instances for this project, in declaration order.

    Order brings infrastructure + servers up first (docker → migrations →
    service bootstraps → backend/frontend), then runs the slower quality steps
    (smoke, deploy-config emit). Combined with the orchestrator's
    dependency-aware skip, a failure in a best-effort step never blocks the
    servers reaching the user.

    The eval baseline (``bootstrap_evals``) is **opt-in**: it makes real LLM
    calls and is slow, so it's kept out of the default ``up``/autorun chain.
    Pass ``with_evals=True`` (the ``--with-evals`` flag) to re-include it, or
    run ``agent-scaffold eval --update-baseline`` on demand.

    Docker is **opt-in** too (``use_docker``, the ``--docker`` flag / prompt).
    ``use_docker=True`` enables ``docker_up`` (run the backend + services as
    containers) and skips the local ``launch_backend`` (the compose ``app``
    container serves it). Default (``False``) skips ``docker_up`` and runs the
    backend as a local process.

    ``commit_push`` is included only when the recipe's (future) ``setup_steps``
    field opts in. ``open_editor`` always lives in the registry; its ``detect()``
    handles the ``--yes``-mode silent-skip itself. The capability-driven
    ``bootstrap_*`` / ``emit_deploy_configs`` steps are always included and
    ``detect()``-skip when the recipe declares no matching capability.

    ``recipe`` may be ``None`` if discovery failed; the step instances are
    still constructed so ``detect()`` can surface the SKIP/PENDING reason
    instead of an empty plan panel.
    """
    setup_steps = _recipe_setup_steps(recipe)
    steps: list[Step] = [
        InstallDepsStep(),
        DockerUpStep(enabled=use_docker),
        WireCredentialsStep(yes=yes),
        MigrationsStep(),
        BootstrapVectorDbStep(),
        BootstrapKafkaStep(),
        BootstrapLangSmithStep(),
        BootstrapLangfuseStep(),
        BootstrapObservabilityStep(),
        SeedStep(),
        LaunchBackendStep(served_by_docker=use_docker),
        LaunchFrontendStep(served_by_docker=use_docker),
        SmokeTestStep(),
        EmitDeployConfigsStep(),
    ]
    if with_evals:
        steps.append(BootstrapEvalsStep())
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
    "BootstrapEvalsStep",
    "BootstrapKafkaStep",
    "BootstrapLangSmithStep",
    "BootstrapObservabilityStep",
    "BootstrapVectorDbStep",
    "CommitPushStep",
    "DockerUpStep",
    "EmitDeployConfigsStep",
    "InstallDepsStep",
    "LaunchBackendStep",
    "LaunchFrontendStep",
    "MigrationsStep",
    "OpenEditorStep",
    "SeedStep",
    "SmokeTestStep",
    "Step",
    "WireCredentialsStep",
    "default_steps_for",
    "step_class_by_id",
]
