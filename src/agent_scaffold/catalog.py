"""Catalog loader — agent-scaffold's single integration point with the ecosystem.

The agent-deployments repo publishes a top-level ``catalog.yaml`` that lists
every recipe, capability, framework, and cross-cutting doc — plus an embedded
copy of agent-blueprints' pattern index. This module fetches that catalog at
runtime, validates it, and exposes typed views the rest of scaffold consumes.

The dependency direction this enforces is **blueprints → deployments → scaffold**:

- scaffold hardcodes exactly one fact about this ecosystem: ``DEFAULT_CATALOG_URL``
  (overridable via ``--catalog-url`` or ``$AGENT_SCAFFOLD_CATALOG_URL``).
- Everything else — repo identifiers, branch names, doc paths, alias maps,
  cross-cutting category maps, framework-language gating, the blueprint URL
  pattern — comes from the catalog.
- agent-blueprints content is fetched at runtime too, but via the repo URL
  the catalog declares (``catalog.blueprints.repo`` + ``branch``). Scaffold
  never knows the secondary repo is called "blueprints" — that's an
  agent-deployments-declared label.

vX scope (this module's debut):
    The legacy in-process ``ALIAS_TABLE`` / ``CROSS_CUTTING`` /
    ``FRAMEWORK_LANGUAGE`` / ``FRAMEWORK_DOC_TO_ID`` / ``_BLUEPRINT_URL_RE``
    constants in :mod:`agent_scaffold.context` are still present. Pipeline
    code passes the loaded catalog through ``assemble(..., catalog=...)``;
    when present, catalog data wins and the legacy constants are bypassed.
    vX+1 deletes the constants and the embedded fallback's "this is also a
    safety net" framing.

Resolution / fallback order (load_catalog):
    1. Fresh fetch from the resolved URL (explicit kwarg → env → default).
    2. Cached copy at ``{cache_dir}/catalog/<url-hash>.yaml`` if the fetch
       fails or HTTP returns 304 against a stored ETag.
    3. Embedded JSON copy baked into the wheel
       (``_embedded_catalog.json``). Last-resort offline fallback.

If all three fail, raises :class:`CatalogUnavailable` with a clear error so
the CLI can surface the issue to the user.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.error
import urllib.request
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Hardcoded constants — these are the ONLY ecosystem-specific facts scaffold
# carries in code. Everything else loads from the catalog.
# ---------------------------------------------------------------------------

DEFAULT_CATALOG_URL = (
    "https://raw.githubusercontent.com/jagguvarma15/agent-deployments/main/catalog.yaml"
)
"""Default URL scaffold uses when no override is set. Overridable via
``--catalog-url`` flag or ``$AGENT_SCAFFOLD_CATALOG_URL``. Third-party
publishers can host a compatible catalog elsewhere and point a forked
scaffold at it — nothing else in scaffold is repo-specific."""

SCAFFOLD_CATALOG_SCHEMA_VERSION_MAX = 1
"""Maximum ``schema_version`` this scaffold release knows how to parse.
Higher versions are refused with an "upgrade agent-scaffold" message,
matching the pattern in :mod:`agent_scaffold.manifest` for the per-project
manifest schema."""

NETWORK_TIMEOUT_SECONDS = 8.0
"""Per-request HTTP timeout. Short enough to fail fast when offline; long
enough that a slow link doesn't false-positive."""

_EMBEDDED_CATALOG_RESOURCE = "_embedded_catalog.json"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CatalogError(Exception):
    """Base class for catalog loading / validation failures."""


class CatalogUnavailable(CatalogError):
    """Fetch failed and no cached or embedded fallback was usable.

    Carries the underlying fetch error in ``__cause__`` for diagnostics.
    CLI handlers convert this into a user-facing message with the recommended
    remediation (point at a working ``--catalog-url`` or restore network).
    """


class CatalogSchemaError(CatalogError):
    """Catalog parsed but failed schema validation."""


class CatalogVersionTooHigh(CatalogSchemaError):
    """Catalog ``schema_version`` exceeds this scaffold release's max.

    ``got`` and ``max_supported`` are carried as structured attributes so
    error handlers can surface a precise upgrade instruction without
    re-parsing the message text.
    """

    def __init__(self, got: int, max_supported: int) -> None:
        super().__init__(
            f"catalog schema_version={got} is newer than this agent-scaffold "
            f"release supports (max={max_supported}). Upgrade agent-scaffold."
        )
        self.got = got
        self.max_supported = max_supported


# Process-level dedupe set: load_catalog is called several times per command
# (stack options, dashboards, REPL renders); without this, offline-fallback
# warnings print once per call. Each unique message still surfaces once.
_WARN_SEEN: set[str] = set()


def _warn_once(msg: str) -> None:
    if msg in _WARN_SEEN:
        return
    _WARN_SEEN.add(msg)
    print(f"agent-scaffold: warning: {msg}", file=sys.stderr)


def _reset_warn_dedupe() -> None:
    """Test seam — clears the warning dedupe set between test runs."""
    _WARN_SEEN.clear()


# ---------------------------------------------------------------------------
# Pydantic models — mirror catalog.yaml shape. Every model uses
# ``extra="ignore"`` so additive fields in a future catalog version degrade
# gracefully on older scaffold builds (the deployments forward-compat contract).
# This is load-bearing: ``load_catalog`` has no embedded fallback for a schema
# *validation* error, so a stricter ``extra="forbid"`` would brick the tool for
# every user the moment a producer ships an additive field.
# ---------------------------------------------------------------------------


_MODEL_CONFIG = ConfigDict(extra="ignore", frozen=False)


class BlueprintsPointer(BaseModel):
    """The deployments-declared dependency on the secondary (blueprints) repo.

    Scaffold reads ``repo`` + ``branch`` + ``url_pattern`` + ``directory_entry``
    to know how to fetch blueprint content the deployments docs link to. The
    secondary-repo name (``"agent-blueprints"``) is intentionally not
    hardcoded anywhere in scaffold; the catalog supplies it.
    """

    model_config = _MODEL_CONFIG
    repo: str
    branch: str
    catalog_path: str = "patterns-catalog.yaml"
    url_pattern: str = "https://github.com/{repo}/(?:tree|blob|raw)/{branch}/{path}"
    directory_entry: str = "overview.md"


class TierFiles(BaseModel):
    """Optional tier-file map for a pattern (overview/design/implementation/etc.).

    Pass-through from the blueprints catalog. Scaffold uses this for the
    blueprint-URL link rewriter — given a directory link like
    ``patterns/react``, scaffold consults ``tier_files['overview']`` (or
    falls back to ``BlueprintsPointer.directory_entry``) to find the canonical
    entry file.
    """

    model_config = _MODEL_CONFIG
    overview: str | None = None
    design: str | None = None
    implementation: str | None = None
    evolution: str | None = None
    observability: str | None = None
    # The catalog's tier map key uses kebab-case; Pydantic doesn't auto-translate.
    cost_and_latency: str | None = Field(default=None, alias="cost-and-latency")


class PatternEntry(BaseModel):
    """One entry in the catalog's embedded ``patterns[]`` (or ``workflows[]``) block."""

    model_config = _MODEL_CONFIG
    id: str
    name: str
    category: Literal["agent", "workflow"]
    complexity: str | None = None
    description: str | None = None
    dir: str
    tier_files: TierFiles | None = None
    evolvesFrom: list[str] = Field(default_factory=list)
    evolvesInto: list[str] = Field(default_factory=list)
    composableWith: list[str] = Field(default_factory=list)
    requires: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    costTier: str | None = None
    latencyTier: str | None = None


class CompositionEdge(BaseModel):
    """One entry in the catalog's embedded ``compositions[]`` block."""

    model_config = _MODEL_CONFIG
    a: str
    b: str
    kind: Literal["natural", "useful", "complex", "redundant", "anti"]
    rationale: str | None = None


class MCPServerRef(BaseModel):
    """One entry in a recipe's ``mcp_servers[]`` frontmatter block.

    Declares an MCP server the generated agent connects to. ``capability``
    points at a ``kind: mcp`` capability id in the catalog (e.g.
    ``mcp.tavily``); ``transport`` selects between ``stdio`` (in-process
    spawn) and ``streamable_http`` (remote endpoint). ``env`` carries
    per-server environment variable hints surfaced by the scaffold's
    credential-wiring step.
    """

    model_config = _MODEL_CONFIG
    id: str
    capability: str
    transport: Literal["stdio", "streamable_http"] = "stdio"
    env: dict[str, str] = Field(default_factory=dict)


class SkillRef(BaseModel):
    """One entry in a recipe's ``skills[]`` frontmatter block.

    A skill is a file-based, agent-discovered procedural module (per
    Anthropic's skill convention) — kebab-case ``id``, repo-relative
    ``path`` to its ``SKILL.md``, and an optional list of ``triggers``
    (lowercase keywords / phrases that hint when the skill applies).
    """

    model_config = _MODEL_CONFIG
    id: str
    path: str
    triggers: list[str] = Field(default_factory=list)


class EnvContractEntry(BaseModel):
    """One entry in a recipe's auto-derived ``env_contract``.

    The deployments catalog builder aggregates every env var a recipe's
    capabilities declare (plus recipe-level overrides) into this list, with
    ``source_capability`` recording which capability wants it and ``default``
    carrying a recipe-pinned fallback (e.g. ``APP_PORT: 8000``). Entries with
    a default are satisfiable without user input.
    """

    model_config = _MODEL_CONFIG
    name: str
    source_capability: str | None = None
    default: Any = None


class ManifestDoc(BaseModel):
    """One doc in a recipe's ``context_manifest.docs`` — a load_list projection.

    ``when`` is the symbolic predicate the consumer evaluates locally; ``est_tokens``
    is present only for docs that resolved to a local file at generation time."""

    model_config = _MODEL_CONFIG
    path: str
    required: bool = True
    cache_tier: Literal["hot", "warm", "dynamic"] | None = None
    when: str | None = None
    est_tokens: int | None = None


class ContextManifest(BaseModel):
    """``catalog.recipes[].context_manifest`` — the closed, pre-costed context
    set the deployments generator resolved for a recipe. A consumer that honors
    it loads exactly these docs (+ capability closure) and skips speculative
    discovery (prose-keyword scans, transitive link walks)."""

    model_config = _MODEL_CONFIG
    docs: list[ManifestDoc] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    est_total_tokens: int | None = None


class RecipeEntry(BaseModel):
    """One entry in catalog.recipes[]. Lifts the recipe's frontmatter verbatim.

    Pass-through fields preserve whatever the deployments-side generator
    chose to expose. Scaffold's downstream code (discovery, context.assemble,
    capability resolver) reads selected fields; unknown fields are tolerated
    so additive frontmatter changes don't require a scaffold release.
    """

    model_config = _MODEL_CONFIG
    slug: str
    path: str
    title: str
    status: str | None = None
    languages: list[str] = Field(default_factory=list)
    topology: str | None = None
    complexity: str | None = None
    agent_pattern: str | None = None
    required_files: list[str] = Field(default_factory=list)
    recipe_dependencies: dict[str, dict[str, str]] = Field(default_factory=dict)
    external_services: list[Any] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    bootstrap_config: dict[str, Any] | None = None
    roles: list[Any] = Field(default_factory=list)
    load_list: list[dict[str, Any]] = Field(default_factory=list)
    env_contract: list[EnvContractEntry] = Field(default_factory=list)
    # Additive optional fields — recipes carry these to declare advanced
    # 2026-SOTA dependencies. All default to empty so older recipes parse
    # unchanged and older scaffold builds (parsing the same catalog with an
    # earlier model) ignore unknown keys via ``extra="ignore"``.
    mcp_servers: list[MCPServerRef] = Field(default_factory=list)
    skills: list[SkillRef] = Field(default_factory=list)
    guardrails: list[str] = Field(default_factory=list)
    sandbox: str | None = None
    durable_workflow: str | None = None
    tier: str | None = None
    """Author-declared generation tier (``T0``–``T4``). Seeds a curated
    capability set at generation time (see :mod:`agent_scaffold.tiers`);
    ``None`` → no tier."""
    # Port-typed registry projections (generator-derived). ``bindings`` is the
    # port → adapter map; ``context_manifest`` is the closed, pre-costed context
    # set assemble() consumes to skip speculative discovery.
    bindings: dict[str, Any] = Field(default_factory=dict)
    context_manifest: ContextManifest | None = None
    runtime_modes: dict[str, Any] = Field(default_factory=dict)


class CapabilityCard(BaseModel):
    """A capability's discovery card. ``name``/``description`` are required —
    they're what the wizard/UI shows, and the deployments generator hard-enforces
    them (so a published card always carries them); this is the consumer-side
    mirror of that guarantee. ``extra="ignore"`` (not ``forbid``): cards are
    discovery metadata that evolves additively, and a forbidden extra here would
    brick the whole catalog load (no embedded fallback for schema errors)."""

    model_config = _MODEL_CONFIG
    name: str
    description: str
    capabilities_provided: list[str] = Field(default_factory=list)
    required_credentials: list[str] = Field(default_factory=list)


class VerificationEntry(BaseModel):
    """``capabilities[].verification`` — the adapter's pragmatic trust floor.
    ``tier`` is ``T1`` (pinned + reviewed) / ``T2`` (+ CI conformance) / ``T3+``.
    ``delivery`` says how the adapter is served: ``managed`` (cloud hosted,
    needs credentials) or ``self-hosted`` (docker/local) — a free string so new
    producer values degrade instead of bricking the load. ``verified_in`` lists
    the recipe slugs the adapter is proven in."""

    model_config = _MODEL_CONFIG
    tier: str | None = None
    delivery: str | None = None
    verified_in: list[str] = Field(default_factory=list)


class CapabilityEntry(BaseModel):
    """One entry in catalog.capabilities[].

    The scaffold models the full set of catalog-published capability keys so they
    parse into typed fields rather than being dropped, but stays ``extra="ignore"``
    (and ``kind`` stays a free ``str``) to honor the deployments forward-compat
    contract: additive capability fields + new kinds must degrade gracefully on
    older consumers, never hard-fail the catalog load. Bad kinds are still caught
    — non-fatally — by the per-file loader in
    :func:`agent_scaffold.capabilities.load_capabilities`, which also hydrates the
    full spec (docker fragment, emit_files, body) the index doesn't carry."""

    model_config = _MODEL_CONFIG
    id: str
    kind: str
    path: str
    env_vars: list[str] = Field(default_factory=list)
    docker_service: str | None = None
    bootstrap_step: str | None = None
    probe: str | None = None
    # Catalog-published discovery / wiring metadata, modeled so it parses into
    # typed fields; not all are consumed by generation today.
    layer: str | None = None
    requires: list[str] = Field(default_factory=list)
    bootstrap_inputs: dict[str, Any] = Field(default_factory=dict)
    card: CapabilityCard | None = None
    cost_tier: str | None = None
    est_tokens: int | None = None
    provisioning_time: str | None = None
    when_to_load: str | None = None
    tags: list[str] = Field(default_factory=list)
    context_summary: str | None = None
    """Generator-derived compact summary (name + kind + description + env vars +
    docker service + bootstrap + provides flags). A consumer can inject this
    instead of the full markdown body to cut context tokens."""
    # Port-typed registry fields (additive). ``implements.port`` mirrors ``kind``
    # (the deployments invariant); ``verification`` is the adapter's trust floor.
    implements: dict[str, Any] = Field(default_factory=dict)
    verification: VerificationEntry | None = None


class FrameworkEntry(BaseModel):
    """One entry in catalog.frameworks[]. Used to derive the language gating
    map that today lives as the ``FRAMEWORK_LANGUAGE`` constant in context.py."""

    model_config = _MODEL_CONFIG
    id: str
    language: str
    path: str
    package: str | None = None
    versions: dict[str, Any] | None = None
    extra_packages: list[dict[str, Any]] = Field(default_factory=list)


class TierEntry(BaseModel):
    """One entry in the catalog's ``tiers[]`` block — a generation-tier preset.

    A tier expands to a curated set of capability ids seeded into resolution
    (see :mod:`agent_scaffold.tiers`). ``extra="ignore"`` + free-string fields
    keep it forward-compatible: a producer can publish extra tier metadata or
    new tier names without bricking older scaffold builds (there is no embedded
    fallback for a schema *validation* error)."""

    model_config = _MODEL_CONFIG
    name: str
    title: str = ""
    description: str = ""
    extends: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    overlays: list[str] = Field(default_factory=list)


class PortEntry(BaseModel):
    """One entry in catalog.ports[] — an abstract selection axis adapters bind to.

    ``protocol`` (IR ports) or ``concern`` (cross-cutting) is present per port;
    ``cardinality`` is ``one`` / ``optional`` / ``many``; ``default`` is the
    adapter id auto-selected when the port is unbound (``None`` if no single
    default)."""

    model_config = _MODEL_CONFIG
    id: str
    protocol: str | None = None
    concern: str | None = None
    required: bool = False
    cardinality: str = "one"
    default: str | None = None
    interface_version: str | None = None
    kinds: list[str] = Field(default_factory=list)
    adapter_home: str | None = None


class CompatibilityEdge(BaseModel):
    """One entry in catalog.compatibility[] — a derived feature-model edge.

    ``relation`` is ``requires`` / ``excludes`` / ``conflicts`` / ``substitutes``;
    ``via`` is optional provenance (e.g. ``port:eval``)."""

    model_config = _MODEL_CONFIG
    a: str
    b: str
    relation: str
    via: str | None = None


class Catalog(BaseModel):
    """The full deployments catalog. Top-level entrypoint."""

    model_config = _MODEL_CONFIG
    schema_version: int
    generator_version: str | None = None
    blueprints: BlueprintsPointer
    patterns: list[PatternEntry] = Field(default_factory=list)
    workflows: list[PatternEntry] = Field(default_factory=list)
    compositions: list[CompositionEdge] = Field(default_factory=list)
    recipes: list[RecipeEntry] = Field(default_factory=list)
    capabilities: list[CapabilityEntry] = Field(default_factory=list)
    ports: list[PortEntry] = Field(default_factory=list)
    compatibility: list[CompatibilityEdge] = Field(default_factory=list)
    frameworks: list[FrameworkEntry] = Field(default_factory=list)
    tiers: list[TierEntry] = Field(default_factory=list)
    stack: list[str] = Field(default_factory=list)
    cross_cutting_docs: list[str] = Field(default_factory=list)
    pattern_docs: list[str] = Field(default_factory=list)
    aliases: dict[str, str] = Field(default_factory=dict)

    @field_validator("stack", "cross_cutting_docs", "pattern_docs", mode="before")
    @classmethod
    def _coerce_doc_index(cls, value: Any) -> Any:
        """Accept both path-index shapes the catalog has published.

        Older catalogs list plain path strings; newer ones (generator 1.3+)
        publish ``{path, tags, when_to_load}`` mappings. We only need the
        paths — the richer fields are advisory until the scaffold consumes
        them — so normalize to ``list[str]`` and drop entries with no usable
        path rather than failing the whole catalog load.
        """
        if not isinstance(value, list):
            return value
        normalized: list[str] = []
        for entry in value:
            if isinstance(entry, str):
                normalized.append(entry)
            elif isinstance(entry, dict) and isinstance(entry.get("path"), str):
                normalized.append(entry["path"])
        return normalized

    cross_cutting: dict[str, str] = Field(default_factory=dict)
    non_recipe_stems: list[str] = Field(default_factory=list)
    min_alias_length: int = 3

    def model_post_init(self, __context: Any) -> None:
        # Enforce min_alias_length at load time so an over-eager catalog
        # publisher can't accidentally register a one-letter alias that
        # would match almost every recipe body.
        if self.min_alias_length > 1:
            for table_name in ("aliases", "cross_cutting"):
                table = getattr(self, table_name)
                kept = {k: v for k, v in table.items() if len(k) >= self.min_alias_length}
                if len(kept) != len(table):
                    dropped = sorted(set(table) - set(kept))
                    _warn_once(
                        f"catalog {table_name}: dropped "
                        f"{len(dropped)} entries shorter than min_alias_length="
                        f"{self.min_alias_length}: {dropped}"
                    )
                setattr(self, table_name, kept)


# ---------------------------------------------------------------------------
# Load + fetch
# ---------------------------------------------------------------------------


def _cache_key(url: str) -> str:
    """Return a stable filename component for caching a catalog by URL."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def _cache_paths(cache_dir: Path, url: str) -> tuple[Path, Path]:
    """Return ``(catalog_path, etag_path)`` for this URL's cache slot."""
    key = _cache_key(url)
    base = cache_dir / "catalog"
    return base / f"{key}.yaml", base / f"{key}.etag"


def _read_cached(cache_dir: Path, url: str) -> str | None:
    path, _ = _cache_paths(cache_dir, url)
    if path.is_file():
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return None
    return None


def _write_cached(cache_dir: Path, url: str, body: str, etag: str | None) -> None:
    catalog_path, etag_path = _cache_paths(cache_dir, url)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(body, encoding="utf-8")
    if etag:
        etag_path.write_text(etag, encoding="utf-8")


def _read_etag(cache_dir: Path, url: str) -> str | None:
    _, etag_path = _cache_paths(cache_dir, url)
    if etag_path.is_file():
        try:
            return etag_path.read_text(encoding="utf-8").strip() or None
        except OSError:
            return None
    return None


def _read_embedded() -> str | None:
    """Read the embedded fallback catalog (baked into the wheel as JSON).

    JSON not YAML so this fallback path doesn't need PyYAML at import time
    (though we always have PyYAML at runtime; the JSON choice is mostly
    about keeping the embedded payload small and parser-trivial).
    """
    try:
        with (
            resources.files("agent_scaffold")
            .joinpath(_EMBEDDED_CATALOG_RESOURCE)
            .open("r", encoding="utf-8") as f
        ):
            return f.read()
    except (FileNotFoundError, OSError):
        return None


def _fetch(url: str, cache_dir: Path) -> tuple[str, str | None]:
    """Fetch the catalog body from ``url``, returning ``(body, etag)``.

    Supports HTTP(S), ``file://``, and bare local paths. ``If-None-Match``
    is sent when a prior ETag is cached; on 304 we transparently return the
    cached body.
    """
    if url.startswith(("file://", "/", "./")):
        path = url[7:] if url.startswith("file://") else url
        return Path(path).read_text(encoding="utf-8"), None

    headers = {"Accept": "text/yaml, application/yaml, text/plain, */*"}
    prior_etag = _read_etag(cache_dir, url)
    if prior_etag:
        headers["If-None-Match"] = prior_etag

    # http(s) only — file:// and local paths returned above. noqa S310.
    req = urllib.request.Request(url, headers=headers)  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=NETWORK_TIMEOUT_SECONDS) as resp:  # noqa: S310
            body = resp.read().decode("utf-8")
            etag = resp.headers.get("ETag")
            return body, etag
    except urllib.error.HTTPError as exc:
        if exc.code == 304:
            cached = _read_cached(cache_dir, url)
            if cached is None:
                raise CatalogUnavailable(
                    f"received HTTP 304 from {url} but no cached copy is available"
                ) from exc
            return cached, prior_etag
        raise


def load_catalog_for_config(cfg: Any) -> Catalog:
    """Convenience wrapper: load the catalog using URL + cache_dir from a Config.

    Plucked into a separate helper because every assemble() call site that
    cares about the catalog uses these two fields the same way. Avoids the
    boilerplate ``load_catalog(url=cfg.catalog_url, cache_dir=cfg.cache_dir)``
    repeated across cli / cli_update / repl. ``cfg`` is typed as ``Any`` to
    avoid a circular import with :mod:`agent_scaffold.config`.
    """
    return load_catalog(url=cfg.catalog_url, cache_dir=cfg.cache_dir)


def load_catalog(
    *,
    url: str | None = None,
    cache_dir: Path,
    env: dict[str, str] | None = None,
) -> Catalog:
    """Resolve, fetch, validate, and return the catalog.

    Resolution order for the URL: explicit ``url`` arg → ``$AGENT_SCAFFOLD_CATALOG_URL``
    → :data:`DEFAULT_CATALOG_URL`.

    On fetch failure: fall back to the on-disk cache → fall back to the
    embedded JSON → raise :class:`CatalogUnavailable`.
    """
    import os

    env_map = os.environ if env is None else env
    resolved_url = url or env_map.get("AGENT_SCAFFOLD_CATALOG_URL") or DEFAULT_CATALOG_URL

    body: str | None = None
    parse_format: Literal["yaml", "json"] = "yaml"
    fetch_error: Exception | None = None

    try:
        body, etag = _fetch(resolved_url, cache_dir)
        _write_cached(cache_dir, resolved_url, body, etag)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        fetch_error = exc
        # Try cached.
        cached = _read_cached(cache_dir, resolved_url)
        if cached is not None:
            body = cached
            _warn_once(
                f"using cached catalog at {resolved_url} "
                f"(fetch failed: {type(exc).__name__})"
            )

    if body is None:
        # Try embedded fallback. Parse as JSON (the embedded file is JSON).
        embedded = _read_embedded()
        if embedded is not None:
            body = embedded
            parse_format = "json"
            _warn_once(
                f"using embedded catalog fallback "
                f"({type(fetch_error).__name__ if fetch_error else 'no network'})"
            )

    if body is None:
        raise CatalogUnavailable(
            f"could not load catalog from {resolved_url} and no cached or embedded "
            f"fallback is available"
        ) from fetch_error

    try:
        raw = json.loads(body) if parse_format == "json" else yaml.safe_load(body)
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise CatalogSchemaError(
            f"catalog body is not valid {parse_format.upper()}: {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise CatalogSchemaError(
            f"catalog body did not parse as a mapping (got {type(raw).__name__})"
        )

    schema_version = raw.get("schema_version")
    if isinstance(schema_version, int) and schema_version > SCAFFOLD_CATALOG_SCHEMA_VERSION_MAX:
        raise CatalogVersionTooHigh(schema_version, SCAFFOLD_CATALOG_SCHEMA_VERSION_MAX)

    try:
        return Catalog.model_validate(raw)
    except Exception as exc:
        raise CatalogSchemaError(f"catalog validation failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Derived views — what existing scaffold code reads from the catalog
# ---------------------------------------------------------------------------


def alias_lookup(catalog: Catalog, text: str) -> list[tuple[str, str]]:
    """Return ``(alias_key, doc_path)`` pairs whose alias appears in ``text``.

    Word-boundary regex match lifted verbatim from
    :func:`agent_scaffold.context._alias_matches` so behavior is identical to
    the legacy hardcoded path; only the data source changes.
    """
    lowered = text.lower()
    hits: list[tuple[str, str]] = []
    for alias, path in catalog.aliases.items():
        pattern = r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])"
        if re.search(pattern, lowered):
            hits.append((alias, path))
    return hits


def cross_cutting_lookup(catalog: Catalog, text: str) -> list[tuple[str, str]]:
    """Return ``(category, doc_path)`` pairs whose keyword appears in ``text``."""
    lowered = text.lower()
    hits: list[tuple[str, str]] = []
    for category, path in catalog.cross_cutting.items():
        pattern = r"(?<![a-z0-9])" + re.escape(category) + r"(?![a-z0-9])"
        if re.search(pattern, lowered):
            hits.append((category, path))
    return hits


def framework_doc_paths(catalog: Catalog) -> dict[str, dict[str, str]]:
    """Return a ``{doc_path: {"id": ..., "language": ...}}`` map.

    Replaces the ``FRAMEWORK_LANGUAGE`` + ``FRAMEWORK_DOC_TO_ID`` constants
    in :mod:`agent_scaffold.context`. Used by the language- and framework-
    gating checks in ``assemble()``.
    """
    out: dict[str, dict[str, str]] = {}
    for fw in catalog.frameworks:
        out[fw.path] = {"id": fw.id, "language": fw.language}
    return out


def build_secondary_url_re(catalog: Catalog) -> re.Pattern[str]:
    """Compile the regex used to recognize secondary-repo URLs in recipe bodies.

    The pattern template comes from ``catalog.blueprints.url_pattern`` with
    ``{repo}`` and ``{branch}`` substituted, ``{path}`` left as a named
    capture group for the link-rewriter to extract.

    Today the deployments catalog publishes
    ``"https://github.com/{repo}/(?:tree|blob|raw)/{branch}/{path}"`` —
    matching the legacy ``_BLUEPRINT_URL_RE`` exactly. Catalogs from other
    publishers could declare a different pattern.
    """
    pattern_template = catalog.blueprints.url_pattern
    # Escape the static repo + branch components, but allow the regex
    # alternation `(?:tree|blob|raw)` to survive intact — it's part of the
    # template by convention.
    pattern = pattern_template.replace("{repo}", re.escape(catalog.blueprints.repo)).replace(
        "{branch}", re.escape(catalog.blueprints.branch)
    )
    # {path} is the named capture group for the rewriter.
    pattern = pattern.replace("{path}", r"(?P<path>[^?#\s]+)")
    return re.compile("^" + pattern)
