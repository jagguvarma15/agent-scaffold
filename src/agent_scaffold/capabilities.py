"""Capability catalog loader + recipe resolver.

A *capability* is a provisioning contract for a single infra need
(``vector_db.qdrant``, ``cache.redis``, ``host.vercel``, ...) shipped as a
markdown file under ``{deployments_path}/docs/capabilities/<kind>/<name>.md``.
The frontmatter carries everything ``agent-scaffold`` needs to provision the
service (docker fragment, env vars, post-up bootstrap step, optional file
templates, cloud-deploy hints); the body is the human/LLM-readable docs.

Recipes opt in via a ``capabilities: [...]`` frontmatter field
(``discovery.Recipe.capabilities``). At generation time,
:func:`load_capabilities` walks the catalog and :func:`resolve` turns the
recipe's id list into a typed :class:`ResolvedStack` — consumed by the
context assembler (``context.assemble``), the manifest, the orchestrator
bootstrap steps, and the template copier.

The loader treats unknown frontmatter keys as warnings (forward-compat) and
unknown capability ids as :attr:`ResolvedStack.unresolved` entries — never
fatal. If the catalog directory doesn't exist on the deployments source,
:func:`load_capabilities` returns an empty dict and logs once.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from agent_scaffold.discovery import _NON_RECIPE_STEMS, Recipe

CAPABILITIES_SUBDIR = ("docs", "capabilities")

CapabilityKind = Literal[
    # Original eight kinds (v0.2).
    "vector_db", "cache", "relational", "queue", "obs", "frontend", "host", "eval",
    # Additive 2026-SOTA kinds. Capability docs for these may not yet exist
    # under ``docs/capabilities/`` — that's fine; ``resolve()`` carries
    # missing ids in ``ResolvedStack.unresolved`` rather than raising. The
    # kinds become real when the matching ``docs/capabilities/<kind>/<name>.md``
    # files land (per the C-* batches in the roadmap).
    "mcp", "sandbox", "durable", "memory_store",
    "guardrail", "embedding", "live_data", "rerank",
]

_KNOWN_KINDS: frozenset[str] = frozenset(
    {
        "vector_db", "cache", "relational", "queue", "obs", "frontend", "host", "eval",
        "mcp", "sandbox", "durable", "memory_store",
        "guardrail", "embedding", "live_data", "rerank",
    }
)

LAYER_ORDER: tuple[CapabilityKind, ...] = (
    # Data-layer (provisioned first; agent runtime depends on these).
    "relational",
    "cache",
    "vector_db",
    "embedding",
    "rerank",
    "memory_store",
    # Tool / connectivity layer.
    "live_data",
    "mcp",
    # Runtime layer.
    "sandbox",
    "durable",
    "queue",
    # Safety layer (wraps the agent's tool-call surface).
    "guardrail",
    # Instrumentation + evaluation.
    "obs",
    "eval",
    # Surface + hosting (provisioned last).
    "frontend",
    "host",
)
"""Stable presentation order for the wizard's layer-walk and the report's
Layers section. Order encodes provisioning dependency: data-layer
(persistence → retrieval → memory) first, then tool / connectivity, then
runtime (sandbox + durable + queue), then safety wrapper, then
instrumentation, then user-facing surface and hosting. Consumers iterating
over capabilities in this order honor the natural setup sequence."""

_CAPABILITY_ID_RE = re.compile(r"^[a-z_]+\.[a-z0-9_-]+$")

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)

_CAPABILITY_KNOWN_KEYS: frozenset[str] = frozenset(
    {
        "id",
        "kind",
        "provides",
        "env_vars",
        "docker",
        "probe",
        "bootstrap_step",
        "emit_files",
        "deploy_configs",
        "docs",
    }
)

_DOCKER_KNOWN_KEYS: frozenset[str] = frozenset(
    {"service", "image", "ports", "volumes", "environment", "healthcheck"}
)

_DEPLOY_CONFIG_KNOWN_KEYS: frozenset[str] = frozenset(
    {"target", "cli_cmd", "dashboard_url", "config_file"}
)

_EMIT_FILE_KNOWN_KEYS: frozenset[str] = frozenset({"source", "dest"})


# Process-level dedupe set: bootstrap steps re-call `load_capabilities` per
# orchestrator run; without this, every capability schema warning prints once
# per call. Authors still see each unique warning once.
_WARN_SEEN: set[str] = set()


def _warn(msg: str) -> None:
    if msg in _WARN_SEEN:
        return
    _WARN_SEEN.add(msg)
    print(f"agent-scaffold: warning: {msg}", file=sys.stderr)


def _info(msg: str) -> None:
    print(f"agent-scaffold: info: {msg}", file=sys.stderr)


def _reset_warn_dedupe() -> None:
    """Test seam — clears the warning dedupe set between test runs."""
    _WARN_SEEN.clear()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class DockerFragment(BaseModel):
    """Capability docker-compose fragment.

    Phase 5 merges these into the generated ``docker-compose.yml``; this
    module only parses + validates them.
    """

    service: str
    image: str
    ports: list[str] = Field(default_factory=list)
    volumes: list[str] = Field(default_factory=list)
    environment: dict[str, str] = Field(default_factory=dict)
    healthcheck: dict[str, Any] | None = None


class EmitFile(BaseModel):
    """One ``source → dest`` mapping in ``capability.emit_files``.

    Phase 3b's copier walks these. ``source`` is relative to the capability's
    directory; ``dest`` is relative to the generated project root. ``source``
    may be a glob ending in ``**``.
    """

    source: str
    dest: str


class DeployConfig(BaseModel):
    """Cloud-deploy hint for a ``host.*`` capability.

    Consumed by Phase 4's ``agent-scaffold deploy --target`` verb.
    """

    target: str
    cli_cmd: str
    dashboard_url: str | None = None
    config_file: str | None = None


class Capability(BaseModel):
    """One resolved capability — a typed view of a ``docs/capabilities/`` file.

    ``body`` carries the markdown body (no frontmatter) so the context
    assembler can inject it under a ``## Capability:`` header without
    re-reading the file.
    """

    id: str
    kind: CapabilityKind
    path: Path
    provides: list[str] = Field(default_factory=list)
    env_vars: list[str] = Field(default_factory=list)
    docker: DockerFragment | None = None
    probe: str | None = None
    bootstrap_step: str | None = None
    emit_files: list[EmitFile] = Field(default_factory=list)
    deploy_configs: list[DeployConfig] = Field(default_factory=list)
    docs: str = ""
    body: str = ""


class ResolvedStack(BaseModel):
    """The result of resolving a recipe's ``capabilities:`` list.

    ``capabilities`` preserves declaration order — downstream consumers
    (context assembler, orchestrator) iterate in that order so the user can
    influence priority by reordering the recipe frontmatter.
    ``unresolved`` carries ids the recipe declared but the catalog didn't
    contain (typically: the user is on an older deployments fork without
    yet-to-be-merged catalog). They surface as WARN in ``doctor``.
    """

    capabilities: list[Capability] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)

    def ids(self) -> list[str]:
        return [c.id for c in self.capabilities]

    def docker_services(self) -> list[DockerFragment]:
        """Every capability's docker fragment, in declaration order."""
        return [c.docker for c in self.capabilities if c.docker is not None]

    def env_vars(self) -> list[str]:
        """Union of every capability's env_vars, deduped, first-seen order."""
        seen: list[str] = []
        for cap in self.capabilities:
            for var in cap.env_vars:
                if var not in seen:
                    seen.append(var)
        return seen

    def bootstrap_steps(self) -> list[str]:
        """Step ids declared by capabilities (in declaration order, deduped)."""
        seen: list[str] = []
        for cap in self.capabilities:
            if cap.bootstrap_step and cap.bootstrap_step not in seen:
                seen.append(cap.bootstrap_step)
        return seen

    def deploy_targets(self) -> list[str]:
        """Cloud-deploy targets declared by host.* capabilities."""
        return [cfg.target for cap in self.capabilities for cfg in cap.deploy_configs]

    def by_kind(self) -> dict[CapabilityKind, list[Capability]]:
        """Group capabilities by ``kind``, preserving within-kind declaration order.

        Used by the wizard's customize-mode layer walk and by the post-gen
        report's Layers section so consumers don't have to re-derive the
        grouping. Iteration order of the returned dict matches the order in
        which each kind first appeared in ``capabilities``; pair with
        :data:`LAYER_ORDER` when you need a stable presentation order.
        """
        groups: dict[CapabilityKind, list[Capability]] = {}
        for cap in self.capabilities:
            groups.setdefault(cap.kind, []).append(cap)
        return groups


# ---------------------------------------------------------------------------
# Frontmatter coercion
# ---------------------------------------------------------------------------


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    raw = match.group(1)
    try:
        loaded = yaml.safe_load(raw) or {}
    except yaml.YAMLError:
        return {}, text[match.end() :]
    if not isinstance(loaded, dict):
        return {}, text[match.end() :]
    return loaded, text[match.end() :]


def _coerce_str_list(value: Any, *, context: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out: list[str] = []
        for entry in value:
            if entry is None:
                continue
            out.append(str(entry))
        return out
    _warn(f"{context}: expected list of strings, got {type(value).__name__}; ignoring")
    return []


def _coerce_str_map(value: Any, *, context: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        _warn(f"{context}: expected mapping, got {type(value).__name__}; ignoring")
        return {}
    out: dict[str, str] = {}
    for key, val in value.items():
        if not isinstance(key, str):
            _warn(f"{context}: non-string key {key!r}; dropping")
            continue
        out[key] = str(val) if val is not None else ""
    return out


def _coerce_docker(value: Any, *, capability_id: str) -> DockerFragment | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        _warn(
            f"capability {capability_id!r}: docker must be a mapping; "
            f"got {type(value).__name__}; ignoring"
        )
        return None
    unknown = set(value) - _DOCKER_KNOWN_KEYS
    if unknown:
        _warn(
            f"capability {capability_id!r}: docker has unknown keys " f"{sorted(unknown)}; ignored"
        )
    service = value.get("service")
    image = value.get("image")
    if not isinstance(service, str) or not service.strip():
        _warn(
            f"capability {capability_id!r}: docker.service missing or not a string; "
            "dropping docker fragment"
        )
        return None
    if not isinstance(image, str) or not image.strip():
        _warn(
            f"capability {capability_id!r}: docker.image missing or not a string; "
            "dropping docker fragment"
        )
        return None
    healthcheck_raw = value.get("healthcheck")
    healthcheck: dict[str, Any] | None
    if healthcheck_raw is None:
        healthcheck = None
    elif isinstance(healthcheck_raw, dict):
        healthcheck = dict(healthcheck_raw)
    else:
        _warn(f"capability {capability_id!r}: docker.healthcheck must be a mapping; " "ignoring")
        healthcheck = None
    return DockerFragment(
        service=service.strip(),
        image=image.strip(),
        ports=_coerce_str_list(
            value.get("ports"), context=f"capability {capability_id!r}: docker.ports"
        ),
        volumes=_coerce_str_list(
            value.get("volumes"), context=f"capability {capability_id!r}: docker.volumes"
        ),
        environment=_coerce_str_map(
            value.get("environment"),
            context=f"capability {capability_id!r}: docker.environment",
        ),
        healthcheck=healthcheck,
    )


def _coerce_emit_files(value: Any, *, capability_id: str) -> list[EmitFile]:
    if value is None:
        return []
    if not isinstance(value, list):
        _warn(
            f"capability {capability_id!r}: emit_files must be a list; "
            f"got {type(value).__name__}; ignoring"
        )
        return []
    out: list[EmitFile] = []
    for idx, raw in enumerate(value):
        if not isinstance(raw, dict):
            _warn(
                f"capability {capability_id!r}: emit_files[{idx}] expected mapping, "
                f"got {type(raw).__name__}; dropping"
            )
            continue
        unknown = set(raw) - _EMIT_FILE_KNOWN_KEYS
        if unknown:
            _warn(
                f"capability {capability_id!r}: emit_files[{idx}] has unknown keys "
                f"{sorted(unknown)}; ignored"
            )
        source = raw.get("source")
        dest = raw.get("dest")
        if not isinstance(source, str) or not source.strip():
            _warn(f"capability {capability_id!r}: emit_files[{idx}].source missing; dropping")
            continue
        if not isinstance(dest, str) or not dest.strip():
            _warn(f"capability {capability_id!r}: emit_files[{idx}].dest missing; dropping")
            continue
        out.append(EmitFile(source=source.strip(), dest=dest.strip()))
    return out


def _coerce_deploy_configs(value: Any, *, capability_id: str) -> list[DeployConfig]:
    if value is None:
        return []
    if not isinstance(value, list):
        _warn(
            f"capability {capability_id!r}: deploy_configs must be a list; "
            f"got {type(value).__name__}; ignoring"
        )
        return []
    out: list[DeployConfig] = []
    for idx, raw in enumerate(value):
        if not isinstance(raw, dict):
            _warn(
                f"capability {capability_id!r}: deploy_configs[{idx}] expected mapping, "
                f"got {type(raw).__name__}; dropping"
            )
            continue
        unknown = set(raw) - _DEPLOY_CONFIG_KNOWN_KEYS
        if unknown:
            _warn(
                f"capability {capability_id!r}: deploy_configs[{idx}] has unknown keys "
                f"{sorted(unknown)}; ignored"
            )
        target = raw.get("target")
        cli_cmd = raw.get("cli_cmd")
        if not isinstance(target, str) or not target.strip():
            _warn(f"capability {capability_id!r}: deploy_configs[{idx}].target missing; dropping")
            continue
        if not isinstance(cli_cmd, str) or not cli_cmd.strip():
            _warn(f"capability {capability_id!r}: deploy_configs[{idx}].cli_cmd missing; dropping")
            continue
        dashboard_raw = raw.get("dashboard_url")
        config_raw = raw.get("config_file")
        dashboard = str(dashboard_raw).strip() if isinstance(dashboard_raw, str) else None
        config_file = str(config_raw).strip() if isinstance(config_raw, str) else None
        out.append(
            DeployConfig(
                target=target.strip(),
                cli_cmd=cli_cmd.strip(),
                dashboard_url=dashboard or None,
                config_file=config_file or None,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Loader + resolver
# ---------------------------------------------------------------------------


def _expected_id_from_path(path: Path, root: Path) -> str | None:
    """Derive the expected capability id from a file path under the catalog root.

    ``vector_db/qdrant.md`` → ``vector_db.qdrant``.
    Returns ``None`` if the path isn't shaped like ``<kind>/<name>.md``.
    """
    try:
        rel = path.relative_to(root).with_suffix("")
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) != 2:
        return None
    return f"{parts[0]}.{parts[1]}"


def _parse_capability_file(path: Path, *, root: Path) -> Capability | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        _warn(f"could not read {path}: {exc}")
        return None

    frontmatter, body = _parse_frontmatter(text)
    if not frontmatter:
        _warn(f"capability {path.name}: missing frontmatter; skipping")
        return None

    capability_id = frontmatter.get("id")
    if not isinstance(capability_id, str) or not capability_id.strip():
        _warn(f"capability {path.name}: missing/empty 'id'; skipping")
        return None
    capability_id = capability_id.strip()

    if not _CAPABILITY_ID_RE.match(capability_id):
        _warn(
            f"capability {path.name}: id {capability_id!r} must match "
            f"^<kind>.<name>$ (lowercase, dotted); skipping"
        )
        return None

    expected_id = _expected_id_from_path(path, root)
    if expected_id is not None and expected_id != capability_id:
        _warn(
            f"capability {path.name}: id {capability_id!r} does not match path "
            f"(expected {expected_id!r}); skipping"
        )
        return None

    kind = frontmatter.get("kind")
    if not isinstance(kind, str) or kind not in _KNOWN_KINDS:
        _warn(
            f"capability {capability_id!r}: kind {kind!r} must be one of "
            f"{sorted(_KNOWN_KINDS)}; skipping"
        )
        return None

    if capability_id.split(".", 1)[0] != kind:
        _warn(f"capability {capability_id!r}: kind {kind!r} disagrees with id prefix; " "skipping")
        return None

    unknown = set(frontmatter) - _CAPABILITY_KNOWN_KEYS
    if unknown:
        _warn(f"capability {capability_id!r}: unknown keys {sorted(unknown)} ignored")

    docs_raw = frontmatter.get("docs", "")
    docs = str(docs_raw) if docs_raw is not None else ""

    try:
        capability = Capability(
            id=capability_id,
            kind=kind,
            path=path.resolve(),
            provides=_coerce_str_list(
                frontmatter.get("provides"),
                context=f"capability {capability_id!r}: provides",
            ),
            env_vars=_coerce_str_list(
                frontmatter.get("env_vars"),
                context=f"capability {capability_id!r}: env_vars",
            ),
            docker=_coerce_docker(frontmatter.get("docker"), capability_id=capability_id),
            probe=_optional_str(frontmatter.get("probe")),
            bootstrap_step=_optional_str(frontmatter.get("bootstrap_step")),
            emit_files=_coerce_emit_files(
                frontmatter.get("emit_files"), capability_id=capability_id
            ),
            deploy_configs=_coerce_deploy_configs(
                frontmatter.get("deploy_configs"), capability_id=capability_id
            ),
            docs=docs,
            body=body.rstrip() + ("\n" if body.strip() else ""),
        )
    except ValueError as exc:
        _warn(f"capability {capability_id!r}: {exc}; skipping")
        return None
    return capability


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def load_capabilities(deployments_path: Path) -> dict[str, Capability]:
    """Walk ``{deployments_path}/docs/capabilities/**/*.md`` → ``{id: Capability}``.

    Skips ``README.md`` files (the catalog README + per-kind READMEs) and
    silently returns an empty dict when the catalog directory doesn't exist
    (deployments source lacks ``docs/capabilities/``). Malformed files log a
    warning and are dropped — never raise.
    """
    root = deployments_path.joinpath(*CAPABILITIES_SUBDIR)
    if not root.is_dir():
        _info(
            f"no capability catalog at {root} — recipes with 'capabilities:' "
            "will resolve to empty (upgrade your deployments source)"
        )
        return {}

    catalog: dict[str, Capability] = {}
    for entry in sorted(root.rglob("*.md")):
        if entry.name.startswith("."):
            continue
        if entry.stem.lower() in _NON_RECIPE_STEMS:
            continue
        if not entry.is_file():
            continue
        capability = _parse_capability_file(entry, root=root)
        if capability is None:
            continue
        if capability.id in catalog:
            _warn(
                f"duplicate capability id {capability.id!r} at {entry} — "
                f"keeping first ({catalog[capability.id].path})"
            )
            continue
        catalog[capability.id] = capability
    return catalog


def resolve(
    recipe: Recipe,
    catalog: dict[str, Capability],
    *,
    add_capabilities: list[str] | None = None,
    remove_capabilities: set[str] | None = None,
) -> ResolvedStack:
    """Resolve ``recipe.capabilities`` against ``catalog``.

    Order is preserved from ``recipe.capabilities``. Unknown ids land in
    :attr:`ResolvedStack.unresolved`; duplicates are deduped (first wins).

    ``add_capabilities`` are appended after the recipe's declared ids (so
    recipe order wins for the overlap). ``remove_capabilities`` are dropped
    before resolution — they never reach ``unresolved`` either. Both
    layered on top so the REPL can offer "swap obs.langsmith → obs.langfuse"
    without forking the recipe.
    """
    removals = remove_capabilities or set()
    seen_ids: set[str] = set()
    resolved: list[Capability] = []
    unresolved: list[str] = []
    effective_ids: list[str] = list(recipe.capabilities)
    if add_capabilities:
        for cap_id in add_capabilities:
            if cap_id not in effective_ids:
                effective_ids.append(cap_id)
    for cap_id in effective_ids:
        if cap_id in removals:
            continue
        if cap_id in seen_ids:
            _warn(
                f"recipe {recipe.slug!r}: capability {cap_id!r} declared twice; "
                "second occurrence ignored"
            )
            continue
        seen_ids.add(cap_id)
        capability = catalog.get(cap_id)
        if capability is None:
            unresolved.append(cap_id)
            continue
        resolved.append(capability)
    return ResolvedStack(capabilities=resolved, unresolved=unresolved)


__all__ = [
    "CAPABILITIES_SUBDIR",
    "Capability",
    "CapabilityKind",
    "DeployConfig",
    "DockerFragment",
    "EmitFile",
    "ResolvedStack",
    "load_capabilities",
    "resolve",
]
