"""Assemble the LLM context bundle for a chosen recipe.

Reads the recipe markdown, then walks any references it contains:

1. Explicit relative markdown links (e.g. ``[text](../patterns/react.md)``).
2. Lowercase alias mentions in prose (``pattern: ReAct``, ``framework: LangGraph``).
3. Cross-cutting concerns when the recipe mentions auth/logging/rate
   limiting/testing.

Framework docs are filtered to the chosen language so a Python project does
not pull in TypeScript framework guides.
"""

from __future__ import annotations

import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from agent_scaffold.capabilities import Capability, ResolvedStack
from agent_scaffold.discovery import CacheTier, Recipe, default_cache_tier
from agent_scaffold.topology import coerce_topology

if TYPE_CHECKING:
    from agent_scaffold.catalog import Catalog

CHARS_PER_TOKEN = 4
TOKEN_WARN_THRESHOLD = 80_000

DEFAULT_MAX_CONTEXT_TOKENS = 60_000
DEFAULT_MAX_LINK_DEPTH = 2
DEFAULT_MAX_TOKENS_PER_DOC = 8_000

# Priority tiers — lower number = higher priority. Tier 1 is the recipe itself
# (always kept). Tier 7 is deep transitive content (drops first).
#
# ``_TIER_CAPABILITY`` sits between Composes and Explicit links: resolved
# capability bodies are explicit recipe declarations (like Composes) but the
# existing essentials-budget check (``<= _TIER_COMPOSES``) is intentionally
# not relaxed — large capability sets can still be dropped to fit a tight
# ``--max-context-tokens`` cap.
_TIER_RECIPE = 1
_TIER_COMPOSES = 2
_TIER_CAPABILITY = 3
_TIER_EXPLICIT_LINK = 4
_TIER_ALIAS = 5
_TIER_CROSS_CUTTING = 6
_TIER_TRANSITIVE = 7

_TIER_LABELS: dict[int, str] = {
    _TIER_RECIPE: "Recipe",
    _TIER_COMPOSES: "Composes / Load as Context",
    _TIER_CAPABILITY: "Capabilities",
    _TIER_EXPLICIT_LINK: "Explicit links",
    _TIER_ALIAS: "Aliased",
    _TIER_CROSS_CUTTING: "Cross-cutting",
    _TIER_TRANSITIVE: "Transitive",
}

_TRUNCATION_MARKER = "\n\n[truncated for context budget]\n"

_SECTION_HEADER_RE = re.compile(r"^##+\s+(.+?)\s*$", re.MULTILINE)
_COMPOSES_HEADER_KEYWORDS = ("composes", "load as context", "load-as-context")

_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


# ---------------------------------------------------------------------------
# Catalog view — vX bridge between hardcoded maps and the new
# :mod:`agent_scaffold.catalog`-driven flow.
# ---------------------------------------------------------------------------
#
# Every helper that consults aliases / cross-cutting / framework gating /
# blueprint URL rewriting takes an optional ``view`` parameter that defaults
# to :data:`_LEGACY_VIEW` (the existing module constants). When
# ``assemble()`` is called with ``catalog=<Catalog>``, it builds a view from
# the catalog and threads it through; legacy callers that don't pass catalog
# see no behavior change.
#
# vX+1 deletes the legacy constants, drops the ``view`` parameter, and reads
# directly from the catalog object threaded through. This file's diff will
# shrink significantly at that point.


@dataclass(frozen=True)
class _CatalogView:
    """Bundle of catalog-derivable data the assemble helpers consult.

    Frozen so the legacy singleton can be shared safely. ``view`` is built
    once per assemble() invocation when a Catalog is supplied; otherwise
    callers fall through to :data:`_LEGACY_VIEW`.
    """

    aliases: dict[str, str]
    cross_cutting: dict[str, str]
    # Map ``rel_doc_path -> {"id": framework_id, "language": "python"|"typescript"}``.
    framework_paths: dict[str, dict[str, str]]
    blueprint_url_re: re.Pattern[str]
    blueprint_directory_entry: str


def _view_from_catalog(catalog: Catalog) -> _CatalogView:
    """Build a view from a loaded Catalog. Late-import to avoid the
    catalog ↔ context cycle at module load time.

    The catalog publishes paths as **repo-root-relative** (``docs/X/Y.md``).
    The assemble helpers expect **docs-relative** paths (``X/Y.md``) so they
    can build ``docs_root / rel_doc`` without double-prefixing. Strip the
    leading ``docs/`` here at the boundary — it's the cheapest place to
    bridge the two conventions and keeps every existing consumer's logic
    unchanged.
    """
    from agent_scaffold.catalog import build_secondary_url_re

    def _strip(p: str) -> str:
        return p[5:] if p.startswith("docs/") else p

    return _CatalogView(
        aliases={k: _strip(v) for k, v in catalog.aliases.items()},
        cross_cutting={k: _strip(v) for k, v in catalog.cross_cutting.items()},
        framework_paths={
            _strip(fw.path): {"id": fw.id, "language": fw.language} for fw in catalog.frameworks
        },
        blueprint_url_re=build_secondary_url_re(catalog),
        blueprint_directory_entry=catalog.blueprints.directory_entry,
    )


class TierStats(BaseModel):
    tier: int
    label: str
    docs: int
    tokens: int


class ContextSummary(BaseModel):
    total_tokens: int
    cap: int
    tiers: list[TierStats]
    dropped: list[str]
    truncated: list[str]

    def render(self) -> str:
        lines = [
            f"Context: {sum(t.docs for t in self.tiers)} docs, ~{self.total_tokens:,} tokens (cap {self.cap:,})"
        ]
        for tier in self.tiers:
            if tier.docs == 0:
                continue
            lines.append(f"  {tier.label}: {tier.docs} docs, {tier.tokens:,} tokens")
        if self.dropped:
            lines.append(f"  Dropped to fit budget: {len(self.dropped)} doc(s)")
            for name in self.dropped:
                lines.append(f"    - {name}")
        if self.truncated:
            lines.append(f"  Truncated: {len(self.truncated)} doc(s)")
            for name in self.truncated:
                lines.append(f"    - {name}")
        return "\n".join(lines)


class ContextSegment(BaseModel):
    """One cache-tier slice of the assembled context.

    The generator turns each segment into its own prompt block with the
    matching Anthropic ``cache_control``: ``hot`` → 1h TTL (stable across
    runs: patterns, frameworks, stack docs), ``warm`` → 5m TTL (recipe body,
    capabilities — stable within a session). Rare ``dynamic``-tier docs fold
    into the warm segment; the truly per-run content is the user-template
    tail, which the generator already leaves uncached.
    """

    cache_tier: CacheTier
    text: str


class AssembledContext(BaseModel):
    recipe_path: Path
    referenced_paths: list[Path]
    body: str
    token_estimate: int
    summary: ContextSummary | None = None
    # Cache-tier slices of the same content as ``body``, populated only for
    # recipes with a structured load_list. ``body`` stays the single joined
    # string — it remains the response-cache fingerprint and the fallback
    # prompt for recipes without segments.
    segments: list[ContextSegment] = []


class ContextBudgetError(RuntimeError):
    """Raised when the recipe + Tier 1/2 alone exceed the configured cap.

    ``essentials_tokens`` and ``current_cap`` are carried as structured fields
    so the wizard / REPL can decide whether bumping to a higher preset cap
    would fit without re-parsing the human message.
    """

    def __init__(self, message: str, *, essentials_tokens: int, current_cap: int) -> None:
        super().__init__(message)
        self.essentials_tokens = essentials_tokens
        self.current_cap = current_cap


def _warn(msg: str) -> None:
    print(f"agent-scaffold: warning: {msg}", file=sys.stderr)


def _docs_root(deployments_path: Path) -> Path:
    return deployments_path / "docs"


# Match GitHub URLs that point into the agent-blueprints repo on main. We
# rewrite these to local paths in the fetched blueprints tree so the link
# walker can descend into them — otherwise the http(s) prefix would cause
# `_resolve_relative` to drop them and the LLM would never see the pattern
# content the deployments docs explicitly point to. The actual regex is
# compiled at runtime from the catalog's ``blueprints.url_pattern`` field;
# see :func:`agent_scaffold.catalog.build_secondary_url_re`.


def _rewrite_blueprint_url(
    link: str, blueprints_root: Path | None, *, view: _CatalogView
) -> Path | None:
    """If ``link`` is an agent-blueprints GitHub URL, return its local path.

    - ``tree/main/<dir>``      → ``<blueprints_root>/<dir>/<directory_entry>``
      (default ``overview.md``; catalog can override via
      ``blueprints.directory_entry``.)
    - ``blob/main/<path.md>``  → ``<blueprints_root>/<path.md>``
    - Trailing slash is stripped.

    Returns ``None`` when the URL doesn't match, or ``blueprints_root`` is
    ``None`` (offline / skipped), or the rewritten file doesn't exist on
    disk. Callers fall through to existing http-drop behavior.

    The URL regex and the directory-entry filename come from ``view`` —
    legacy callers get the hardcoded blueprint repo pattern; catalog-aware
    callers get whatever the deployments catalog declared.
    """
    if blueprints_root is None:
        return None
    match = view.blueprint_url_re.match(link)
    if match is None:
        return None
    rel = match.group("path").rstrip("/")
    if not rel:
        return None
    candidate = blueprints_root / rel
    if candidate.is_dir():
        candidate = candidate / view.blueprint_directory_entry
    elif not rel.lower().endswith(".md"):
        # blob/main/<something> that isn't markdown — skip.
        return None
    return candidate if candidate.is_file() else None


def _resolve_relative(
    link: str,
    current: Path,
    *,
    blueprints_root: Path | None = None,
    view: _CatalogView,
) -> Path | None:
    """Resolve a markdown link to a local file path, or ``None`` to skip.

    Order: blueprints-URL rewrite first (so a fetched blueprints tree wins
    over the http-drop), then relative-link resolution.
    """
    rewritten = _rewrite_blueprint_url(link, blueprints_root, view=view)
    if rewritten is not None:
        return rewritten
    if link.startswith(("http://", "https://", "mailto:", "#")):
        return None
    cleaned = link.split("#", 1)[0].strip()
    if not cleaned or not cleaned.lower().endswith(".md"):
        return None
    candidate = (current.parent / cleaned).resolve()
    return candidate


def _alias_matches(text: str, *, view: _CatalogView) -> list[str]:
    """Return alias keys (lowercased) that appear in ``text``.

    Iterates over ``view.aliases`` (the catalog-derived alias map).
    """
    lowered = text.lower()
    hits: list[str] = []
    for alias in view.aliases:
        # Use word-ish boundaries: alias must be surrounded by non-alnum chars.
        pattern = r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])"
        if re.search(pattern, lowered):
            hits.append(alias)
    return hits


def _cross_cutting_matches(text: str, *, view: _CatalogView) -> list[str]:
    """Return cross-cutting category keys that appear in ``text``."""
    lowered = text.lower()
    hits: list[str] = []
    for category in view.cross_cutting:
        pattern = r"(?<![a-z0-9])" + re.escape(category) + r"(?![a-z0-9])"
        if re.search(pattern, lowered):
            hits.append(category)
    return hits


def _is_wrong_language_framework(rel_doc_path: str, language: str, *, view: _CatalogView) -> bool:
    entry = view.framework_paths.get(rel_doc_path)
    if entry is None or "language" not in entry:
        return False
    return entry["language"] != language.lower()


# Predicate language for recipe `load_list[].when` (D6). Intentionally tiny:
# `<lang|framework|topology> == 'value'` for scalar equality, and
# `capabilities contains 'cap.id'` for membership. Anything else falls through
# to "always True" with a warning so unknown predicates don't accidentally
# drop required docs.
_LOAD_LIST_PRED_EQ_RE = re.compile(
    r"^\s*(language|framework|topology)\s*==\s*['\"]([^'\"]+)['\"]\s*$"
)
_LOAD_LIST_PRED_CONTAINS_RE = re.compile(r"^\s*capabilities\s+contains\s+['\"]([^'\"]+)['\"]\s*$")


_RECIPE_FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n?", re.DOTALL)


def _strip_frontmatter(text: str) -> str:
    """Remove a leading YAML frontmatter block, if present.

    Used inside ``assemble`` to keep prose-based matchers (aliases, cross-
    cutting categories) from accidentally matching keywords that live inside
    the YAML header. Frontmatter is already parsed separately by discovery —
    re-scanning it for prose hits creates double-counting bugs.
    """
    match = _RECIPE_FRONTMATTER_RE.match(text)
    return text[match.end() :] if match is not None else text


def evaluate_load_list_predicate(
    predicate: str | None,
    *,
    language: str,
    framework: str,
    capabilities: list[str],
    topology: str | None,
) -> bool:
    """Evaluate one ``load_list[].when`` predicate against the resolver scope.

    Supports two forms:

    - ``<language|framework|topology> == 'value'`` — scalar equality
    - ``capabilities contains 'cap.id'`` — membership

    An empty / absent predicate is always True. Unknown syntax is also True
    (with a warning) so a malformed predicate never accidentally drops a
    required doc — fail-open is the safer default for context loading.
    """
    if not predicate or not predicate.strip():
        return True
    p = predicate.strip()

    m = _LOAD_LIST_PRED_EQ_RE.match(p)
    if m is not None:
        attr, value = m.group(1), m.group(2)
        scope = {"language": language, "framework": framework, "topology": topology}
        return scope.get(attr) == value

    m = _LOAD_LIST_PRED_CONTAINS_RE.match(p)
    if m is not None:
        return m.group(1) in (capabilities or [])

    _warn(f"unknown load_list predicate {predicate!r}; treating as always-true")
    return True


def _is_other_framework(rel_doc_path: str, selected_framework: str, *, view: _CatalogView) -> bool:
    """True iff the doc is a framework guide for a DIFFERENT framework than selected.

    Used by the alias-tier and transitive walks to avoid loading the wrong
    framework guide (e.g. a Pydantic AI recipe shouldn't transitively pull
    LangGraph). Returns False when the selected framework is unset / "none"
    or doesn't match any known framework id — those signals mean "don't
    filter by framework, fall back to language-only gating".

    Explicit composes / recipe-author links bypass this filter at the call
    site (they go through ``_consider`` directly without this check).
    """
    if not selected_framework or selected_framework.lower() in {"none", ""}:
        return False
    entry = view.framework_paths.get(rel_doc_path)
    if entry is None or "id" not in entry:
        return False
    return entry["id"] != selected_framework


def _format_marker(rel_path: str) -> str:
    return f"<!-- ===== referenced: {rel_path} ===== -->"


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def _composes_link_set(
    recipe_text: str,
    recipe_path: Path,
    *,
    blueprints_root: Path | None = None,
    view: _CatalogView,
) -> set[Path]:
    """Return the set of resolved paths for links inside ``Composes`` /
    ``Load as Context`` sections."""
    paths: set[Path] = set()
    matches = list(_SECTION_HEADER_RE.finditer(recipe_text))
    for idx, header in enumerate(matches):
        title = header.group(1).strip().lower()
        if not any(keyword in title for keyword in _COMPOSES_HEADER_KEYWORDS):
            continue
        start = header.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(recipe_text)
        section = recipe_text[start:end]
        for link_match in _LINK_RE.finditer(section):
            resolved = _resolve_relative(
                link_match.group(1), recipe_path, blueprints_root=blueprints_root, view=view
            )
            if resolved is not None:
                paths.add(resolved)
    return paths


def _truncate(text: str, max_tokens: int) -> tuple[str, bool]:
    """Truncate ``text`` so its token estimate fits ``max_tokens``."""
    if max_tokens <= 0:
        return text, False
    max_chars = max_tokens * CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text, False
    keep = max(0, max_chars - len(_TRUNCATION_MARKER))
    return text[:keep].rstrip() + _TRUNCATION_MARKER, True


def _format_capability_body(capability: Capability, summary: str | None = None) -> str:
    """Render a single capability into a ``## Capability:`` block.

    When ``summary`` (the catalog's generator-derived ``context_summary``) is
    given, the block is just the heading + that compact summary: the consumer
    trades the full markdown body — and the duplicated metadata header, which the
    summary already carries — for a few lines, cutting context tokens. Without a
    summary, falls back to the metadata header (kind, env vars, docker service)
    + the full markdown body parsed by :mod:`agent_scaffold.capabilities`.
    """
    if summary and summary.strip():
        return f"## Capability: {capability.id}\n\n{summary.strip()}\n"
    parts: list[str] = [f"## Capability: {capability.id}", ""]
    meta: list[str] = [f"- kind: `{capability.kind}`"]
    if capability.env_vars:
        meta.append(f"- env vars: {', '.join(f'`{v}`' for v in capability.env_vars)}")
    if capability.docker is not None:
        meta.append(
            f"- docker service: `{capability.docker.service}` (image: `{capability.docker.image}`)"
        )
    if capability.bootstrap_step:
        meta.append(f"- bootstrap step: `{capability.bootstrap_step}`")
    if capability.deploy_configs:
        targets = ", ".join(f"`{c.target}`" for c in capability.deploy_configs)
        meta.append(f"- deploy targets: {targets}")
    parts.extend(meta)
    parts.append("")
    body = capability.body.strip() or capability.docs.strip()
    if body:
        parts.append(body)
    return "\n".join(parts).rstrip() + "\n"


def assemble_capability_tier(stack: ResolvedStack, budget: int) -> tuple[str, list[Path], int]:
    """Render the capability tier in isolation (helper for tests + callers).

    Returns ``(body, included_paths, consumed_tokens)``. Iterates capabilities
    in declaration order and stops when the next capability would exceed
    ``budget``. Dropped capabilities are not signalled here — callers that
    need that info should use :func:`assemble` directly.
    """
    pieces: list[str] = []
    included: list[Path] = []
    consumed = 0
    for capability in stack.capabilities:
        rendered = _format_capability_body(capability)
        tokens = _estimate_tokens(rendered)
        if budget and consumed + tokens > budget:
            break
        pieces.append(rendered)
        included.append(capability.path)
        consumed += tokens
    return ("\n".join(pieces), included, consumed)


def assemble(
    recipe: Recipe,
    language: str,
    framework: str,
    deployments_path: Path,
    *,
    catalog: Catalog,
    blueprints_path: Path | None = None,
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    max_link_depth: int = DEFAULT_MAX_LINK_DEPTH,
    max_tokens_per_doc: int = DEFAULT_MAX_TOKENS_PER_DOC,
    resolved_stack: ResolvedStack | None = None,
) -> AssembledContext:
    """Build the assembled context for ``recipe`` in ``language``.

    Three caps shape the output:

    - ``max_context_tokens``: hard total; lowest-tier docs are dropped first.
    - ``max_link_depth``: how many hops the transitive-link walker takes.
    - ``max_tokens_per_doc``: per-doc cap; longer docs get truncated.

    ``catalog`` is required: alias / cross-cutting / framework gating /
    blueprint URL data is read from it. Callers obtain the catalog via
    :func:`agent_scaffold.catalog.load_catalog_for_config`.

    When ``blueprints_path`` is provided, ``github.com/.../<blueprints-repo>/...``
    links in deployments docs are rewritten to local files in that tree so the
    LLM actually sees the canonical pattern content the docs reference. The
    URL pattern + entry-file convention come from the catalog's
    ``blueprints`` block.
    """
    view = _view_from_catalog(catalog)
    # Catalog-published context-window levers (additive; absent → today's behavior).
    # ``context_summary`` lets the capability tier ship a compact summary instead
    # of the full body; ``context_manifest`` (when it carries a closed doc set)
    # licenses skipping the speculative transitive walk below.
    summary_by_id = {c.id: c.context_summary for c in catalog.capabilities if c.context_summary}
    recipe_manifest = next(
        (r.context_manifest for r in catalog.recipes if r.slug == recipe.slug), None
    )
    manifest_closed = recipe_manifest is not None and bool(recipe_manifest.docs)
    docs_root = _docs_root(deployments_path).resolve()
    blueprints_root = blueprints_path.resolve() if blueprints_path is not None else None
    recipe_path = recipe.path.resolve()

    try:
        recipe_text = recipe_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FileNotFoundError(f"Could not read recipe at {recipe_path}: {exc}") from exc

    # Strip the YAML frontmatter from the text we feed to prose matchers (alias
    # mentions, cross-cutting categories). Without this strip, a recipe whose
    # frontmatter has e.g. ``capabilities contains 'multi-tenancy'`` inside a
    # ``load_list[].when`` predicate would inadvertently trigger the
    # cross-cutting alias for "multi-tenancy" and load the doc even when the
    # predicate's runtime value is False.
    recipe_body = _strip_frontmatter(recipe_text)

    composes_targets = _composes_link_set(
        recipe_body, recipe_path, blueprints_root=blueprints_root, view=view
    )
    # Discovered (resolved_path, tier, label). First-seen wins for tier.
    discovered: dict[Path, tuple[int, str]] = {}
    # Cache tier per doc, authored on the load_list entry (or its path-based
    # default). Docs discovered by other walks fall back to the display-path
    # default at segment-build time.
    authored_cache_tiers: dict[Path, CacheTier] = {}

    # D6 load_list pre-population: ``required: true`` entries (whose ``when``
    # passes) get included at the Composes tier so they're protected from
    # budget pressure. ``required: false`` entries get included at the Cross-
    # cutting tier so they're early to drop. The recipe author's intent here
    # wins over alias-tier filtering — these are explicit declarations.
    #
    # Capability scope: prefer the runtime-resolved stack when available (it
    # reflects user overrides like ``add_capabilities``); fall back to the
    # recipe's own ``capabilities:`` declaration so callers that don't resolve
    # a stack (e.g. simple plan rendering) still get correct predicate eval.
    if resolved_stack is not None:
        capability_ids = [cap.id for cap in resolved_stack.capabilities]
    else:
        capability_ids = list(recipe.capabilities)
    # Normalize the topology to its canonical hyphenated value before predicate
    # evaluation so ``topology == 'multi-agent-flat'`` matches the same value the
    # enum / report path uses — otherwise a recipe declaring an alias / underscore
    # form (``multi_agent_flat``) would coerce to MULTI for the report yet fail
    # the predicate on the raw string (the split-brain this fixes). ``None`` when
    # the topology is absent or unrecognized.
    coerced_topology = coerce_topology(recipe.topology)
    predicate_topology = coerced_topology.value if coerced_topology is not None else None
    # When the catalog publishes a context_manifest, it IS the resolved menu —
    # the author load_list PLUS the recipe's chosen pattern levels and the selected
    # adapters' stack docs, deduplicated by the generator. Drive the load from it
    # so the full menu (not just the raw load_list) reaches the prompt; recipes
    # without a manifest keep the load_list path. Both entry types expose
    # ``path`` / ``required`` / ``when`` / ``cache_tier``.
    manifest_entries: list[Any]
    if recipe_manifest is not None and recipe_manifest.docs:
        manifest_entries = recipe_manifest.docs
    else:
        manifest_entries = recipe.load_list
    for load_entry in manifest_entries:
        if not evaluate_load_list_predicate(
            load_entry.when,
            language=language,
            framework=framework,
            capabilities=capability_ids,
            topology=predicate_topology,
        ):
            continue
        load_resolved = _resolve_relative(
            load_entry.path, recipe_path, blueprints_root=blueprints_root, view=view
        )
        if load_resolved is None:
            _warn(
                f"load_list: could not resolve {load_entry.path!r} from "
                f"{recipe_path.name}; skipping"
            )
            continue
        load_resolved_abs = load_resolved.resolve()
        if not load_resolved_abs.is_file():
            _warn(
                f"load_list: referenced file not found, skipping: "
                f"{load_entry.path} -> {load_resolved_abs}"
            )
            continue
        load_tier = _TIER_COMPOSES if load_entry.required else _TIER_CROSS_CUTTING
        load_label = f"load_list:{load_entry.path}"
        if load_entry.required:
            load_label += " (required)"
        # First-seen wins for tier — but load_list runs first, so it sets the
        # floor. Later walks can't downgrade these (they only upgrade).
        if load_resolved_abs not in discovered or load_tier < discovered[load_resolved_abs][0]:
            discovered[load_resolved_abs] = (load_tier, load_label)
        authored_cache_tiers[load_resolved_abs] = load_entry.cache_tier or default_cache_tier(
            load_entry.path
        )

    def _consider(resolved: Path | None, tier: int, label: str) -> None:
        if resolved is None:
            return
        resolved = resolved.resolve()
        # Accept references rooted in docs/ OR in the fetched blueprints tree.
        try:
            resolved.relative_to(docs_root)
        except ValueError:
            if blueprints_root is None:
                _warn(f"reference outside docs/, skipping: {label}")
                return
            try:
                resolved.relative_to(blueprints_root)
            except ValueError:
                _warn(f"reference outside docs/ and blueprints/, skipping: {label}")
                return
        try:
            rel = resolved.relative_to(docs_root).as_posix()
        except ValueError:
            rel = ""  # not a docs/ path; can't apply language gating
        if rel and _is_wrong_language_framework(rel, language, view=view):
            return
        if not resolved.is_file():
            _warn(f"referenced file not found, skipping: {label}")
            return
        # First-seen tier wins (don't downgrade a Tier 2 doc to Tier 6 later).
        if resolved not in discovered or tier < discovered[resolved][0]:
            discovered[resolved] = (tier, label)

    # Tier 2/3: explicit relative links in the recipe body.
    for match in _LINK_RE.finditer(recipe_body):
        link = match.group(1)
        resolved = _resolve_relative(link, recipe_path, blueprints_root=blueprints_root, view=view)
        if resolved is None:
            continue
        tier = _TIER_COMPOSES if resolved.resolve() in composes_targets else _TIER_EXPLICIT_LINK
        _consider(resolved, tier, link)

    # Tier 4/5 prose heuristics run only for recipes WITHOUT a structured
    # load_list. A load_list is the author's explicit declaration of what to
    # load — prose scanning on top of it re-adds exactly the noise the author
    # curated away (a stray "redis" in a design-rationale paragraph pulling in
    # the whole stack doc). Explicit Composes links and the transitive walk
    # still apply either way.
    if not recipe.load_list and not manifest_closed:
        # Tier 4: alias mentions in prose. Framework-doc aliases (e.g.
        # "LangGraph" mentioned in a paragraph) skip when they don't match the
        # user's selected framework — recipes that genuinely want both
        # framework docs must list them in `## Composes` so they go through
        # the explicit-link path above.
        for alias in _alias_matches(recipe_body, view=view):
            rel_doc = view.aliases[alias]
            if _is_wrong_language_framework(rel_doc, language, view=view):
                continue
            if _is_other_framework(rel_doc, framework, view=view):
                continue
            _consider(docs_root / rel_doc, _TIER_ALIAS, f"alias:{alias}")

        # Tier 5: cross-cutting categories.
        for category in _cross_cutting_matches(recipe_body, view=view):
            rel_doc = view.cross_cutting[category]
            _consider(docs_root / rel_doc, _TIER_CROSS_CUTTING, f"cross-cutting:{category}")

    # Tier 6: transitive walk, depth-capped.
    # Skip the speculative transitive walk when the catalog hands us a closed
    # doc set for this recipe (``context_manifest.docs``): the manifest is the
    # authoritative context, so chasing links would re-add the noise it curated
    # away. Recipes without a manifest keep today's discovery.
    if max_link_depth >= 1 and not manifest_closed:
        # Start with everything we've discovered so far.
        frontier = [(p, 1) for p in list(discovered.keys())]
        while frontier:
            current, depth = frontier.pop(0)
            if depth > max_link_depth:
                continue
            try:
                text = current.read_text(encoding="utf-8")
            except OSError:
                continue
            for match in _LINK_RE.finditer(text):
                link = match.group(1)
                resolved = _resolve_relative(
                    link, current, blueprints_root=blueprints_root, view=view
                )
                if resolved is None:
                    continue
                resolved_abs = resolved.resolve()
                # Accept references in docs/ or in blueprints/.
                in_docs = True
                try:
                    rel = resolved_abs.relative_to(docs_root).as_posix()
                except ValueError:
                    in_docs = False
                    rel = ""
                    if blueprints_root is None:
                        continue
                    try:
                        resolved_abs.relative_to(blueprints_root)
                    except ValueError:
                        continue
                if in_docs and _is_wrong_language_framework(rel, language, view=view):
                    continue
                # Framework-filter the transitive walk: don't follow a chain
                # from (say) react.md → langgraph.md when the user picked
                # pydantic-ai. Recipe-author intent (explicit composes) is
                # already captured in `discovered` via the earlier tier-2 pass
                # so this skip can't drop a recipe-required doc.
                if in_docs and _is_other_framework(rel, framework, view=view):
                    continue
                if not resolved_abs.is_file():
                    continue
                fresh = resolved_abs not in discovered
                if fresh:
                    discovered[resolved_abs] = (_TIER_TRANSITIVE, link)
                if fresh and depth + 1 <= max_link_depth:
                    frontier.append((resolved_abs, depth + 1))

    # Budgeted assembly: keep recipe + Tier 2/3/... until cap is reached.
    # Ship the recipe body WITHOUT its YAML frontmatter. The header (load_list,
    # runtime_modes, smoke_test, required_files, recipe_dependencies, …) is
    # machine-contract metadata already parsed and re-rendered structurally
    # downstream (the capabilities block, required-files block, and role block),
    # so shipping the raw YAML to the model is noise that competes for budget
    # with useful docs. ``recipe_body`` was frontmatter-stripped at the top of
    # assemble().
    recipe_doc = recipe_body.rstrip()
    recipe_tokens = _estimate_tokens(recipe_doc)

    # Read + truncate every discovered doc up front; we need their sizes. The
    # reads run on a small thread pool so I/O overlaps across the (often dozens
    # of) linked docs; order is preserved by mapping over the discovered items
    # positionally, and a read failure drops that doc with a warning.
    doc_entries: list[tuple[Path, int, str, str, int, bool]] = []
    # tuple: (path, tier, label, text, tokens, truncated)

    def _read_doc(
        item: tuple[Path, tuple[int, str]],
    ) -> tuple[Path, int, str, str, int, bool] | None:
        path, (tier, label) = item
        try:
            raw = path.read_text(encoding="utf-8").rstrip()
        except OSError as exc:
            _warn(f"could not read {path}: {exc}")
            return None
        text, was_truncated = _truncate(raw, max_tokens_per_doc)
        return (path, tier, label, text, _estimate_tokens(text), was_truncated)

    discovered_items = list(discovered.items())
    if discovered_items:
        with ThreadPoolExecutor(max_workers=min(8, len(discovered_items))) as pool:
            for entry in pool.map(_read_doc, discovered_items):
                if entry is not None:
                    doc_entries.append(entry)

    # Capability tier: each resolved capability becomes a synthetic doc entry
    # at tier 3 so it participates in the same budget pass as the link tiers.
    # The body is the formatted ``## Capability:`` block (see
    # ``_format_capability_body``); the path is the capability file so summary
    # rendering can show it. Order matches the recipe's declaration order.
    if resolved_stack is not None:
        for capability in resolved_stack.capabilities:
            cap_text = _format_capability_body(capability, summary=summary_by_id.get(capability.id))
            cap_text_truncated, was_truncated = _truncate(cap_text, max_tokens_per_doc)
            doc_entries.append(
                (
                    capability.path,
                    _TIER_CAPABILITY,
                    f"capability:{capability.id}",
                    cap_text_truncated,
                    _estimate_tokens(cap_text_truncated),
                    was_truncated,
                )
            )

    # Sort by (tier, original discovery order). Stable sort preserves insertion order within a tier.
    doc_entries.sort(key=lambda e: e[1])

    # Hard-fail mode: recipe + Tier 1/2 alone exceed the cap.
    essentials_tokens = recipe_tokens + sum(
        entry[4] for entry in doc_entries if entry[1] <= _TIER_COMPOSES
    )
    if essentials_tokens > max_context_tokens:
        raise ContextBudgetError(
            f"recipe + Composes/Load-as-Context docs are ~{essentials_tokens:,} tokens, "
            f"exceeding --max-context-tokens={max_context_tokens:,}. "
            "Raise the cap or remove links from the recipe's Composes section.",
            essentials_tokens=essentials_tokens,
            current_cap=max_context_tokens,
        )

    def _display_rel(path: Path) -> str:
        """Display path for markers / summaries — blueprints get a 'blueprints/' prefix."""
        try:
            return path.relative_to(docs_root).as_posix()
        except ValueError:
            if blueprints_root is not None:
                try:
                    return "blueprints/" + path.relative_to(blueprints_root).as_posix()
                except ValueError:
                    pass
            return path.as_posix()

    # Greedy fill from highest priority down.
    pieces: list[str] = [recipe_doc]
    kept: list[tuple[Path, int, str, str, int, bool]] = []
    dropped: list[str] = []
    running_tokens = recipe_tokens
    for entry in doc_entries:
        path, tier, label, text, tokens, was_truncated = entry
        if running_tokens + tokens > max_context_tokens:
            dropped.append(_display_rel(path))
            continue
        kept.append(entry)
        running_tokens += tokens
        pieces.append("")
        pieces.append(_format_marker(_display_rel(path)))
        pieces.append(text)

    body = "\n".join(pieces).rstrip() + "\n"
    token_estimate = _estimate_tokens(body)

    # Build summary: per-tier counts.
    tier_buckets: dict[int, list[tuple[Path, int]]] = {}
    tier_buckets.setdefault(_TIER_RECIPE, []).append((recipe_path, recipe_tokens))
    for path, tier, _label, _text, tokens, _was in kept:
        tier_buckets.setdefault(tier, []).append((path, tokens))
    tier_stats = [
        TierStats(
            tier=tier,
            label=_TIER_LABELS.get(tier, f"Tier {tier}"),
            docs=len(items),
            tokens=sum(t for _, t in items),
        )
        for tier, items in sorted(tier_buckets.items())
    ]
    truncated_paths = [_display_rel(path) for path, _t, _l, _x, _tok, was in kept if was]
    summary = ContextSummary(
        total_tokens=token_estimate,
        cap=max_context_tokens,
        tiers=tier_stats,
        dropped=dropped,
        truncated=truncated_paths,
    )

    if token_estimate > TOKEN_WARN_THRESHOLD:
        _warn(
            f"assembled context is ~{token_estimate} tokens, above the "
            f"{TOKEN_WARN_THRESHOLD}-token soft limit"
        )

    # Cache-tier segments: same content as ``body``, grouped hot-first so the
    # generator can place per-tier cache breakpoints (hot = 1h TTL, stable
    # across runs; warm = 5m). Built whenever a curated doc set drives the load
    # — a recipe load_list or a catalog context_manifest — since that curation
    # is what makes the hot prefix deterministic enough to cache. Gating on the
    # same condition that chose the load source keeps manifest-only recipes
    # (no load_list) from silently degrading to single-block caching.
    segments: list[ContextSegment] = []
    if recipe.load_list or (recipe_manifest is not None and recipe_manifest.docs):
        hot_pieces: list[str] = []
        warm_pieces: list[str] = [recipe_doc]
        for path, _tier, _label, text, _tokens, _was in kept:
            rel = _display_rel(path)
            doc_cache_tier = authored_cache_tiers.get(path) or default_cache_tier(rel)
            target = hot_pieces if doc_cache_tier == "hot" else warm_pieces
            target.append("")
            target.append(_format_marker(rel))
            target.append(text)
        hot_text = "\n".join(hot_pieces).strip()
        if hot_text:
            segments.append(ContextSegment(cache_tier="hot", text=hot_text + "\n"))
        segments.append(
            ContextSegment(cache_tier="warm", text="\n".join(warm_pieces).rstrip() + "\n")
        )

    return AssembledContext(
        recipe_path=recipe_path,
        referenced_paths=[entry[0] for entry in kept],
        body=body,
        token_estimate=token_estimate,
        summary=summary,
        segments=segments,
    )
