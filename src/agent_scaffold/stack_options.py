"""Per-project stack options: every service a generated project depends on,
how it is delivered (internal docker container vs cloud hosted platform), and
what connecting it requires.

A :class:`StackOption` is derived from the manifest's capability ids joined
with the deployments catalog, unifying three previously disjoint views of the
same service (the catalog entry, the capability record, and the connect
integration). The ``connect`` command, its no-argument dashboard, ``status``,
and the post-run display texts all consume this one shape.

Delivery classification comes from the catalog's ``verification.delivery``
(``managed`` or ``self-hosted``) with an inference fallback for older
catalogs: a ``docker_service`` means self-hosted, anything else is managed.
Self-hosted options whose environment contract carries a swappable URL
(:data:`OVERRIDABLE_URL_VARS`) can additionally be pointed at a managed
instance without code changes — the capability docs guarantee the swap is
environment-only.

This module must not import :mod:`agent_scaffold.cli` (the REPL runs connect
through it without the CLI layer).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from agent_scaffold.catalog import CapabilityEntry, Catalog
from agent_scaffold.discovery import ExternalService
from agent_scaffold.preflight import hint_for, is_credential

MODE_INTERNAL = "internal"
MODE_CLOUD = "cloud"
MODE_INTERNAL_OVERRIDABLE = "internal-overridable"

# Service URLs a self-hosted option can point at a managed instance instead.
# Deliberately broader than the in-app setup form's set (preflight keeps
# DATABASE_URL as internal wiring there): connect owns the full swap story.
OVERRIDABLE_URL_VARS = frozenset({"REDIS_URL", "DATABASE_URL", "QDRANT_URL", "LANGFUSE_HOST"})

# Capability kinds that never become a connect option: they are either the
# generated app itself (frontend), generation-time concerns (core, eval), the
# Anthropic key flow (auth — owned by `agent-scaffold auth`), or deploy
# targets (host — owned by `agent-scaffold deploy`).
_EXCLUDED_KINDS = frozenset({"frontend", "auth", "core", "eval", "host"})

# The Anthropic key is wired by `agent-scaffold auth login`, never by connect.
_EXCLUDED_CREDENTIALS = frozenset({"ANTHROPIC_API_KEY"})

# Host-side default endpoints for the local docker containers, so probes work
# from the developer's shell even before any env var is written.
_DEFAULT_LOCAL: dict[str, str] = {
    "redis": "redis://localhost:6379",
    "postgres": "postgresql://agent:agent@localhost:5432/agent_db",
    "qdrant": "http://localhost:6333",
}


@dataclass(frozen=True)
class CredentialSpec:
    """One value captured while connecting an option, in capture order."""

    var: str
    secret: bool = True
    optional: bool = False
    placeholder: str = ""
    hint: str = ""


@dataclass(frozen=True)
class StackOption:
    """One connectable service of a generated project."""

    id: str
    title: str
    capability_ids: frozenset[str]
    kind: str
    mode: str
    credentials: tuple[CredentialSpec, ...]
    managed_vars: tuple[str, ...]
    docker_service: str | None
    probe: str | None
    bootstrap_step: str | None
    key_page_url: str | None

    @property
    def cloud_capable(self) -> bool:
        """True when the option can talk to a cloud hosted instance."""
        return self.mode in (MODE_CLOUD, MODE_INTERNAL_OVERRIDABLE)


# Per-provider capture details the catalog cannot carry: capture order,
# which values are secrets, placeholders, and provider dashboards.
_CREDENTIAL_DETAILS: dict[str, tuple[CredentialSpec, ...]] = {
    "langsmith": (
        CredentialSpec(
            var="LANGCHAIN_API_KEY",
            placeholder="lsv2_...",
            hint="smith.langchain.com, Settings, API Keys",
        ),
    ),
    "redis": (
        CredentialSpec(
            var="REDIS_URL",
            placeholder="rediss://:<password>@<host>:6380",
            hint="managed Redis URL (Upstash, ElastiCache, Redis Cloud)",
        ),
    ),
    "postgres": (
        CredentialSpec(
            var="DATABASE_URL",
            placeholder="postgresql://user:password@host:5432/dbname",
            hint="managed Postgres connection string (Neon, Supabase, RDS)",
        ),
    ),
    "qdrant": (
        CredentialSpec(
            var="QDRANT_URL",
            secret=False,
            placeholder="https://<cluster>.cloud.qdrant.io:6333",
            hint="Qdrant Cloud cluster URL",
        ),
        CredentialSpec(
            var="QDRANT_API_KEY",
            optional=True,
            hint="Qdrant Cloud API key (skip for an unsecured instance)",
        ),
    ),
    "langfuse": (
        CredentialSpec(
            var="LANGFUSE_PUBLIC_KEY",
            secret=False,
            placeholder="pk-lf-...",
            hint="cloud.langfuse.com, Project Settings, API Keys",
        ),
        CredentialSpec(
            var="LANGFUSE_SECRET_KEY",
            placeholder="sk-lf-...",
            hint="cloud.langfuse.com, Project Settings, API Keys",
        ),
        CredentialSpec(
            var="LANGFUSE_HOST",
            secret=False,
            optional=True,
            placeholder="https://cloud.langfuse.com",
            hint="Langfuse host (defaults to cloud.langfuse.com)",
        ),
    ),
}

_KEY_PAGES: dict[str, str] = {
    "langsmith": "https://smith.langchain.com/settings",
    "redis": "https://console.upstash.com",
    "postgres": "https://console.neon.tech",
    "qdrant": "https://cloud.qdrant.io",
    "langfuse": "https://cloud.langfuse.com",
}


def derive_stack_options(capability_ids: Sequence[str], catalog: Catalog) -> list[StackOption]:
    """Join the project's capability ids with the catalog into stack options.

    Capabilities sharing a primary env var collapse into one option (e.g.
    ``cache.redis`` + ``queue.redis-streams`` are both the ``redis`` service).
    Ids the catalog doesn't know degrade to a minimal internal option instead
    of erroring — an older embedded catalog must never brick the dashboard.
    """
    entries = {entry.id: entry for entry in catalog.capabilities}
    order: list[str] = []
    groups: dict[str, list[tuple[str, CapabilityEntry | None]]] = {}
    for cap_id in capability_ids:
        entry = entries.get(cap_id)
        if entry is not None and entry.kind in _EXCLUDED_KINDS:
            continue
        key = entry.env_vars[0] if entry is not None and entry.env_vars else cap_id
        if key not in groups:
            order.append(key)
            groups[key] = []
        groups[key].append((cap_id, entry))
    options: list[StackOption] = []
    for key in order:
        option = _build_option(groups[key])
        if option is not None:
            options.append(option)
    return options


def option_by_id(options: Sequence[StackOption], option_id: str) -> StackOption | None:
    """Look up an option by its connect handle."""
    for option in options:
        if option.id == option_id:
            return option
    return None


def _load_catalog_safe() -> Catalog:
    """Load the catalog via the resolved config; degrade to empty, never raise.

    The catalog load is ETag-cached with an embedded offline fallback, so this
    is cheap; if even that fails, an empty catalog makes every capability
    degrade to a minimal internal option instead of a crash.
    """
    try:
        from agent_scaffold.catalog import load_catalog
        from agent_scaffold.config import load_config

        cfg = load_config()
        return load_catalog(cache_dir=cfg.cache_dir)
    except Exception:  # noqa: BLE001 — the dashboard must degrade, not brick
        return Catalog.model_validate(
            {"schema_version": 1, "blueprints": {"repo": "", "branch": ""}}
        )


def known_provider_capabilities(option_id: str) -> list[str]:
    """Catalog capability ids whose adapter stem matches ``option_id``.

    Lets the CLI distinguish "known provider, capability not in this project"
    (actionable: regenerate with the capability) from a plain typo.
    """
    catalog = _load_catalog_safe()
    return [entry.id for entry in catalog.capabilities if _stem(entry.id) == option_id]


def load_stack_options(capability_ids: Sequence[str]) -> list[StackOption]:
    """Derive options against the resolved config's catalog; never raises."""
    return derive_stack_options(capability_ids, _load_catalog_safe())


def service_for_option(option: StackOption) -> ExternalService:
    """Bridge a stack option into the probeable ``ExternalService`` shape."""
    return ExternalService(
        id=option.id,
        required=False,
        env_vars=list(option.managed_vars),
        default_local=_DEFAULT_LOCAL.get(option.id),
        docker_service=option.docker_service,
        probe=option.probe,
    )


def missing_credentials(option: StackOption, env: Mapping[str, str]) -> list[str]:
    """Names of the option's required credential vars absent from ``env``."""
    return [
        spec.var
        for spec in option.credentials
        if not spec.optional and not (env.get(spec.var) or "").strip()
    ]


def _stem(cap_id: str) -> str:
    return cap_id.split(".", 1)[1] if "." in cap_id else cap_id


def _build_option(members: list[tuple[str, CapabilityEntry | None]]) -> StackOption | None:
    known = [(cap_id, entry) for cap_id, entry in members if entry is not None]
    cap_ids = frozenset(cap_id for cap_id, _ in members)
    if not known:
        # Unknown to this catalog (older embedded fallback): keep a minimal
        # internal row so the dashboard can say so instead of hiding it.
        cap_id = members[0][0]
        return StackOption(
            id=_stem(cap_id),
            title=_stem(cap_id),
            capability_ids=cap_ids,
            kind="unknown",
            mode=MODE_INTERNAL,
            credentials=(),
            managed_vars=(),
            docker_service=None,
            probe=None,
            bootstrap_step=None,
            key_page_url=None,
        )

    primary_id, primary = next(
        ((cap_id, entry) for cap_id, entry in known if entry.docker_service),
        known[0],
    )
    managed_vars: list[str] = []
    for _, entry in known:
        for var in entry.env_vars:
            if var not in managed_vars:
                managed_vars.append(var)
    probe = next((entry.probe for _, entry in known if entry.probe), None)
    bootstrap = next((entry.bootstrap_step for _, entry in known if entry.bootstrap_step), None)
    option_id = _stem(primary_id)
    mode = _classify_mode(primary, managed_vars)
    credentials = _credentials_for(option_id, known, mode)
    if probe is None and primary.docker_service is None and not credentials:
        return None
    title = primary.card.name if primary.card is not None else option_id
    return StackOption(
        id=option_id,
        title=title,
        capability_ids=cap_ids,
        kind=primary.kind,
        mode=mode,
        credentials=credentials,
        managed_vars=tuple(managed_vars),
        docker_service=primary.docker_service,
        probe=probe,
        bootstrap_step=bootstrap,
        key_page_url=_KEY_PAGES.get(option_id),
    )


def _classify_mode(primary: CapabilityEntry, managed_vars: list[str]) -> str:
    delivery = primary.verification.delivery if primary.verification is not None else None
    if delivery == "managed":
        return MODE_CLOUD
    if delivery is None and primary.docker_service is None:
        return MODE_CLOUD
    if set(managed_vars) & OVERRIDABLE_URL_VARS:
        return MODE_INTERNAL_OVERRIDABLE
    return MODE_INTERNAL


def _credentials_for(
    option_id: str,
    known: Sequence[tuple[str, CapabilityEntry | None]],
    mode: str,
) -> tuple[CredentialSpec, ...]:
    details = _CREDENTIAL_DETAILS.get(option_id)
    if details is not None:
        return details
    declared: list[str] = []
    for _, entry in known:
        if entry is None or entry.card is None:
            continue
        for var in entry.card.required_credentials:
            if var not in declared and var not in _EXCLUDED_CREDENTIALS:
                declared.append(var)
    if not declared and mode == MODE_CLOUD:
        for _, entry in known:
            if entry is None:
                continue
            for var in entry.env_vars:
                if is_credential(var) and var not in declared:
                    if var not in _EXCLUDED_CREDENTIALS:
                        declared.append(var)
    if not declared and mode == MODE_INTERNAL_OVERRIDABLE:
        for _, entry in known:
            if entry is None:
                continue
            for var in entry.env_vars:
                if var in OVERRIDABLE_URL_VARS:
                    declared.append(var)
                    break
            if declared:
                break
    return tuple(CredentialSpec(var=var, hint=hint_for(var) or "") for var in declared)


__all__ = [
    "MODE_CLOUD",
    "MODE_INTERNAL",
    "MODE_INTERNAL_OVERRIDABLE",
    "OVERRIDABLE_URL_VARS",
    "CredentialSpec",
    "StackOption",
    "derive_stack_options",
    "known_provider_capabilities",
    "load_stack_options",
    "missing_credentials",
    "option_by_id",
    "service_for_option",
]
