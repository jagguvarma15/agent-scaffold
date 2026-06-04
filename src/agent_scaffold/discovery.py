"""Discover agent recipes inside an agent-deployments repo.

A recipe is any markdown file under ``{deployments_path}/docs/recipes/`` with
an H1 title. Optional YAML frontmatter at the top may provide ``status`` and
``languages``; otherwise sane defaults are used.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

DEFAULT_LANGUAGES = ("python", "typescript")
DEFAULT_STATUS = "unknown"

# Markdown files that share the recipes/ directory but are documentation about
# the directory itself, not recipes. They tend to have valid H1s ("Recipes",
# "Recipe frontmatter schema") so the no-H1 filter doesn't catch them — they
# have to be excluded by name. Compared case-insensitively against the stem.
_NON_RECIPE_STEMS = frozenset({"readme", "schema", "index", "changelog", "contributing", "license"})

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


class DiscoveryError(Exception):
    """Raised when recipes cannot be discovered."""


class ExternalService(BaseModel):
    """A service the recipe declares as an external dependency.

    Q3 surfaces these in `doctor --recipe` and the M4 plan panel. Q6/Q7 (the
    `up` orchestrator) consumes the same schema to know what to provision.
    All fields except ``id`` are optional so recipe authors can declare just
    enough to identify the service, then layer on probe / docker / migrations
    metadata as the recipe matures.
    """

    id: str
    required: bool = True
    env_vars: list[str] = Field(default_factory=list)
    default_local: str | None = None
    docker_service: str | None = None
    probe: str | None = None
    migrations: str | None = None
    explain: str | None = None
    mock_available: bool = False


_CAPABILITY_ID_RE = re.compile(r"^[a-z_]+\.[a-z0-9_-]+$")


ComplexityTier = Literal["basic", "mid", "complex"]


class LoadListEntry(BaseModel):
    """One entry in a recipe's ``load_list:`` frontmatter (D6).

    Tells the context loader which docs to include, with optional per-language
    / per-capability predicates. See ``agent-deployments/docs/recipes/SCHEMA.md``
    for the field-level spec.
    """

    model_config = {"frozen": True}

    path: str
    """Relative path from the recipe (e.g. ``../patterns/react.md``)."""

    required: bool
    """``True``: must be loaded regardless of context budget. ``False``: may be
    dropped first when the budget tightens."""

    when: str | None = None
    """Optional predicate over the resolver scope ``{language, framework,
    capabilities, topology}``. Empty / absent means "always applicable".
    Syntax: see :func:`agent_scaffold.context.evaluate_load_list_predicate`."""


class Recipe(BaseModel):
    slug: str
    title: str
    status: str = DEFAULT_STATUS
    path: Path
    languages: list[str] = Field(default_factory=lambda: list(DEFAULT_LANGUAGES))
    required_files: list[str] = Field(default_factory=list)
    recipe_dependencies: dict[str, dict[str, str]] = Field(default_factory=dict)
    topology: str | None = None
    roles: list[Any] = Field(default_factory=list)
    external_services: list[ExternalService] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    load_list: list[LoadListEntry] = Field(default_factory=list)
    """Structured companion to the prose ``### Load list`` section (D6). When
    present, the context loader pre-populates required entries (whose ``when``
    passes) before walking aliases / cross-cutting / transitive links."""
    """Capability ids declared by the recipe. Resolved against
    ``docs/capabilities/`` by :mod:`agent_scaffold.capabilities`."""
    complexity: ComplexityTier | None = None
    """Author-declared complexity tier. When ``None``, :func:`infer_complexity`
    derives a tier from ``topology`` and the capability id list."""
    agent_pattern: str | None = None
    """Free-form architectural pattern label (e.g. ``react`` / ``rag`` /
    ``planner-executor`` / ``supervisor`` / ``parallel`` / ``event-driven``).
    Surfaced in the recipe picker as a one-line hint; not consumed by codegen."""


_COMPLEX_CAPABILITY_KINDS: frozenset[str] = frozenset({"queue", "frontend", "host"})
_MID_TOPOLOGIES: frozenset[str] = frozenset(
    {"chain", "multi-agent-flat", "multi-agent-hierarchical", "parallel"}
)


def infer_complexity(recipe: Recipe) -> ComplexityTier:
    """Derive a complexity tier for ``recipe``.

    Explicit ``complexity:`` in frontmatter always wins. Otherwise:

    - ``complex`` when the recipe declares any capability whose kind is in
      ``{queue, frontend, host}`` — these signal a full production stack.
    - ``mid`` when ``topology`` names a multi-step or multi-agent shape, or
      the recipe pulls in more than four capabilities.
    - ``basic`` otherwise.

    Kind is inferred from the capability id prefix (``<kind>.<name>``), so
    the catalog need not be loaded to call this — it runs cheaply at picker
    render time.
    """
    if recipe.complexity in {"basic", "mid", "complex"}:
        return recipe.complexity
    kinds = {cap_id.split(".", 1)[0] for cap_id in recipe.capabilities}
    if kinds & _COMPLEX_CAPABILITY_KINDS:
        return "complex"
    if recipe.topology in _MID_TOPOLOGIES:
        return "mid"
    if len(recipe.capabilities) > 4:
        return "mid"
    return "basic"


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


def _first_h1(text: str) -> str | None:
    match = _H1_RE.search(text)
    if not match:
        return None
    return match.group(1).strip()


def _coerce_languages(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v).lower() for v in value]
    if isinstance(value, str):
        return [value.lower()]
    return list(DEFAULT_LANGUAGES)


def _coerce_str_list(value: Any, *, context: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value]
    _warn(f"{context}: expected list of strings, got {type(value).__name__}; ignoring")
    return []


def _sanitize_required_paths(entries: list[str], *, recipe_name: str) -> list[str]:
    """Apply the same path-safety rules used by validate_paths."""
    cleaned: list[str] = []
    for raw in entries:
        if not raw or raw != raw.strip():
            _warn(f"{recipe_name}: empty/whitespace required_files entry {raw!r}; dropping")
            continue
        if raw.startswith(("/", "\\")):
            _warn(f"{recipe_name}: absolute required_files path {raw!r}; dropping")
            continue
        normalized = raw.replace("\\", "/")
        if any(part == ".." for part in normalized.split("/")):
            _warn(f"{recipe_name}: required_files path contains '..': {raw!r}; dropping")
            continue
        cleaned.append(raw)
    return cleaned


def _coerce_recipe_dependencies(value: Any, recipe_name: str) -> dict[str, dict[str, str]]:
    """Coerce frontmatter ``recipe_dependencies`` into ``{lang: {pkg: version}}``.

    Per-language entries must be ``dict[str, str]`` mappings; anything else is
    dropped with a warning. Package version values are coerced via ``str``.
    Language keys are normalized to lowercase.
    """
    if not isinstance(value, dict):
        _warn(
            f"{recipe_name}: recipe_dependencies must be a mapping of language "
            f"to {{package: version}}; got {type(value).__name__}; ignoring"
        )
        return {}
    result: dict[str, dict[str, str]] = {}
    for lang_key, deps in value.items():
        if not isinstance(lang_key, str):
            _warn(
                f"{recipe_name}: skipping malformed recipe_dependencies entry: "
                f"language key {lang_key!r} is not a string"
            )
            continue
        lang = lang_key.lower()
        if not isinstance(deps, dict):
            _warn(
                f"{recipe_name}: skipping malformed recipe_dependencies for "
                f"language {lang!r}: expected mapping of package to version, "
                f"got {type(deps).__name__}"
            )
            continue
        lang_entries: dict[str, str] = {}
        for pkg, version in deps.items():
            if not isinstance(pkg, str):
                _warn(
                    f"{recipe_name}: skipping malformed recipe_dependencies "
                    f"package name {pkg!r} for language {lang!r}; not a string"
                )
                continue
            lang_entries[pkg] = str(version)
        if lang_entries:
            result[lang] = lang_entries
    return result


_EXTERNAL_SERVICE_KNOWN_KEYS = frozenset(
    {
        "id",
        "required",
        "env_vars",
        "default_local",
        "docker_service",
        "probe",
        "migrations",
        "explain",
        "mock_available",
    }
)


def _coerce_external_services(value: Any, recipe_name: str) -> list[ExternalService]:
    """Parse the ``external_services`` frontmatter into typed entries.

    Per-entry rules:
    - Must be a mapping with a non-empty string ``id``; otherwise the entry is
      dropped with a warning.
    - ``env_vars`` is coerced to ``list[str]``.
    - Unknown nested keys log a warning but the rest of the entry still loads
      (forward-compatibility so future fields don't break older scaffold builds).
    """
    if value is None:
        return []
    if not isinstance(value, list):
        _warn(
            f"{recipe_name}: external_services must be a list of mappings; "
            f"got {type(value).__name__}; ignoring"
        )
        return []
    out: list[ExternalService] = []
    for idx, raw in enumerate(value):
        if not isinstance(raw, dict):
            _warn(
                f"{recipe_name}: external_services[{idx}]: expected mapping, "
                f"got {type(raw).__name__}; dropping"
            )
            continue
        svc_id = raw.get("id")
        if not isinstance(svc_id, str) or not svc_id.strip():
            _warn(f"{recipe_name}: external_services[{idx}]: missing/empty 'id'; dropping")
            continue
        unknown = set(raw) - _EXTERNAL_SERVICE_KNOWN_KEYS
        if unknown:
            _warn(
                f"{recipe_name}: external_services[{svc_id!r}]: unknown keys "
                f"{sorted(unknown)} ignored"
            )
        env_vars = _coerce_str_list(
            raw.get("env_vars"),
            context=f"{recipe_name}: external_services[{svc_id!r}].env_vars",
        )
        try:
            svc = ExternalService(
                id=svc_id.strip(),
                required=bool(raw.get("required", True)),
                env_vars=env_vars,
                default_local=_optional_str(raw.get("default_local")),
                docker_service=_optional_str(raw.get("docker_service")),
                probe=_optional_str(raw.get("probe")),
                migrations=_optional_str(raw.get("migrations")),
                explain=_optional_str(raw.get("explain")),
                mock_available=bool(raw.get("mock_available", False)),
            )
        except ValueError as exc:
            _warn(f"{recipe_name}: external_services[{svc_id!r}]: {exc}; dropping")
            continue
        out.append(svc)
    return out


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_complexity(value: Any, recipe_name: str) -> ComplexityTier | None:
    """Parse the optional ``complexity:`` frontmatter into a tier label.

    Unknown values warn once and fall through to ``None`` so :func:`infer_complexity`
    can derive a tier from topology + capability shape instead.
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"basic", "mid", "complex"}:
        return text  # type: ignore[return-value]
    _warn(
        f"{recipe_name}: complexity {text!r} unknown — expected basic|mid|complex; "
        "falling back to inference"
    )
    return None


def _coerce_capabilities(value: Any, recipe_name: str) -> list[str]:
    """Parse the recipe ``capabilities:`` frontmatter into a deduped id list.

    Each id must match ``<kind>.<name>`` (lowercase, dotted). Invalid ids log
    a warning and are dropped; the rest are returned in declaration order.
    """
    raw = _coerce_str_list(value, context=f"{recipe_name}: capabilities")
    seen: set[str] = set()
    out: list[str] = []
    for entry in raw:
        cap_id = entry.strip()
        if not cap_id:
            _warn(f"{recipe_name}: capabilities entry is empty; dropping")
            continue
        if not _CAPABILITY_ID_RE.match(cap_id):
            _warn(
                f"{recipe_name}: capability id {cap_id!r} must match "
                f"^<kind>.<name>$ (lowercase, dotted); dropping"
            )
            continue
        if cap_id in seen:
            _warn(f"{recipe_name}: capability {cap_id!r} declared twice; second ignored")
            continue
        seen.add(cap_id)
        out.append(cap_id)
    return out


# Process-level dedupe set: bootstrap steps re-call `discover_recipes` ~5 times
# per orchestrator run; without this, the same malformed-frontmatter warning
# fires ~150 times in one trial. Authors still see the warning once per process.
_WARN_SEEN: set[str] = set()


def _warn(msg: str) -> None:
    if msg in _WARN_SEEN:
        return
    _WARN_SEEN.add(msg)
    print(f"agent-scaffold: warning: {msg}", file=sys.stderr)


def _reset_warn_dedupe() -> None:
    """Test seam — clears the warning dedupe set between test runs."""
    _WARN_SEEN.clear()


def discover_recipes(deployments_path: Path) -> list[Recipe]:
    """Scan ``{deployments_path}/docs/recipes/*.md`` and return all valid recipes."""
    recipes_dir = deployments_path / "docs" / "recipes"
    if not recipes_dir.is_dir():
        raise DiscoveryError(f"No recipes found at {deployments_path}/docs/recipes")

    recipes: list[Recipe] = []
    for entry in sorted(recipes_dir.iterdir()):
        if entry.name.startswith("."):
            continue
        if not entry.is_file() or entry.suffix.lower() != ".md":
            continue
        if entry.stem.lower() in _NON_RECIPE_STEMS:
            continue

        try:
            text = entry.read_text(encoding="utf-8")
        except OSError as exc:
            _warn(f"could not read {entry}: {exc}")
            continue

        frontmatter, body = _parse_frontmatter(text)
        title = _first_h1(body) or _first_h1(text)
        if title is None:
            _warn(f"skipping {entry.name}: no H1 title")
            continue

        status = str(frontmatter.get("status", DEFAULT_STATUS))
        languages = _coerce_languages(frontmatter.get("languages", DEFAULT_LANGUAGES))
        slug = entry.stem
        required_files = _sanitize_required_paths(
            _coerce_str_list(
                frontmatter.get("required_files"),
                context=f"{entry.name}: required_files",
            ),
            recipe_name=entry.name,
        )
        recipe_dependencies = _coerce_recipe_dependencies(
            frontmatter.get("recipe_dependencies") or {},
            entry.name,
        )

        topology_raw = frontmatter.get("topology")
        topology = str(topology_raw).strip() if isinstance(topology_raw, str) else None
        roles_raw = frontmatter.get("roles")
        roles_list = roles_raw if isinstance(roles_raw, list) else []
        external_services = _coerce_external_services(
            frontmatter.get("external_services"), entry.name
        )
        capabilities = _coerce_capabilities(frontmatter.get("capabilities"), entry.name)
        complexity = _coerce_complexity(frontmatter.get("complexity"), entry.name)
        agent_pattern = _optional_str(frontmatter.get("agent_pattern"))

        recipes.append(
            Recipe(
                slug=slug,
                title=title,
                status=status,
                path=entry.resolve(),
                languages=languages,
                required_files=required_files,
                recipe_dependencies=recipe_dependencies,
                topology=topology,
                roles=roles_list,
                external_services=external_services,
                capabilities=capabilities,
                complexity=complexity,
                agent_pattern=agent_pattern,
            )
        )

    recipes.sort(key=lambda r: r.slug)
    return recipes
