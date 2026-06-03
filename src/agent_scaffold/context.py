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
from pathlib import Path

from pydantic import BaseModel

from agent_scaffold.capabilities import Capability, ResolvedStack
from agent_scaffold.discovery import Recipe

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

# Alias table: lowercase token -> path relative to docs/.
# Framework aliases are tagged with their language so we can filter them.
ALIAS_TABLE: dict[str, str] = {
    "react": "patterns/react.md",
    "rag": "patterns/rag.md",
    "routing": "patterns/routing-tool-use.md",
    "tool use": "patterns/routing-tool-use.md",
    "tool-use": "patterns/routing-tool-use.md",
    "routing-tool-use": "patterns/routing-tool-use.md",
    "prompt chaining": "patterns/prompt-chaining.md",
    "plan, execute, reflect": "patterns/plan-execute-reflect.md",
    "multi-agent": "patterns/multi-agent-flat.md",
    "multi-agent-flat": "patterns/multi-agent-flat.md",
    "multi-agent-hierarchical": "patterns/multi-agent-hierarchical.md",
    "memory": "patterns/memory.md",
    "parallel calls": "patterns/parallel-calls.md",
    "event driven": "patterns/event-driven.md",
    "event-driven": "patterns/event-driven.md",
    "langgraph": "frameworks/langgraph.md",
    "pydantic ai": "frameworks/pydantic-ai.md",
    "pydantic-ai": "frameworks/pydantic-ai.md",
    "crewai": "frameworks/crewai.md",
    "vercel ai sdk": "frameworks/vercel-ai-sdk.md",
    "vercel-ai-sdk": "frameworks/vercel-ai-sdk.md",
    "mastra": "frameworks/mastra.md",
    "qdrant": "stack/vector-qdrant.md",
    "vector-qdrant": "stack/vector-qdrant.md",
    "postgres": "stack/relational-postgres.md",
    "redis": "stack/cache-redis.md",
    "langfuse": "stack/tracing-langfuse.md",
    "fastapi": "stack/api-fastapi.md",
    "hono": "stack/api-hono.md",
    "anthropic": "stack/llm-claude.md",
    "claude": "stack/llm-claude.md",
}

# Subset of ALIAS_TABLE values that are framework docs, keyed by their target
# language. Used to filter out the wrong-language framework guides.
FRAMEWORK_LANGUAGE: dict[str, str] = {
    "frameworks/langgraph.md": "python",
    "frameworks/pydantic-ai.md": "python",
    "frameworks/crewai.md": "python",
    "frameworks/vercel-ai-sdk.md": "typescript",
    "frameworks/mastra.md": "typescript",
}

# Cross-cutting category -> filename (relative to docs/).
CROSS_CUTTING: dict[str, str] = {
    "auth": "cross-cutting/authorization-rbac.md",
    "auth-jwt": "cross-cutting/auth-jwt.md",
    "authorization": "cross-cutting/authorization-rbac.md",
    "authorization-rbac": "cross-cutting/authorization-rbac.md",
    "audit": "cross-cutting/audit-logging.md",
    "audit-logging": "cross-cutting/audit-logging.md",
    "pii": "cross-cutting/pii-gdpr.md",
    "pii-gdpr": "cross-cutting/pii-gdpr.md",
    "gdpr": "cross-cutting/pii-gdpr.md",
    "logging": "cross-cutting/logging-structured.md",
    "rate limiting": "cross-cutting/rate-limiting.md",
    "rate-limiting": "cross-cutting/rate-limiting.md",
    "testing": "cross-cutting/testing-strategy.md",
    "idempotency": "cross-cutting/idempotency.md",
    "resilience": "cross-cutting/resilience.md",
    "distributed-locking": "cross-cutting/distributed-locking.md",
    "distributed locking": "cross-cutting/distributed-locking.md",
    "health": "cross-cutting/health-graceful-shutdown.md",
    "graceful shutdown": "cross-cutting/health-graceful-shutdown.md",
    "security-hardening": "cross-cutting/security-hardening.md",
    "schema-evolution": "cross-cutting/schema-evolution.md",
    "schema evolution": "cross-cutting/schema-evolution.md",
    "validation-strategy": "cross-cutting/validation-strategy.md",
    "caching-strategies": "cross-cutting/caching-strategies.md",
    "multi-tenancy": "cross-cutting/multi-tenancy.md",
    "multi tenancy": "cross-cutting/multi-tenancy.md",
}

_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


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
        if self.truncated:
            lines.append(f"  Truncated: {len(self.truncated)} doc(s)")
        return "\n".join(lines)


class AssembledContext(BaseModel):
    recipe_path: Path
    referenced_paths: list[Path]
    body: str
    token_estimate: int
    summary: ContextSummary | None = None


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
# content the deployments docs explicitly point to.
_BLUEPRINT_URL_RE = re.compile(
    r"^https?://github\.com/jagguvarma15/agent-blueprints/"
    r"(?:tree|blob|raw)/main/(?P<path>[^?#\s]+)"
)


def _rewrite_blueprint_url(link: str, blueprints_root: Path | None) -> Path | None:
    """If ``link`` is an agent-blueprints GitHub URL, return its local path.

    - ``tree/main/<dir>``      → ``<blueprints_root>/<dir>/overview.md``
      (blueprints uses overview.md as every pattern's canonical entry point,
      so a "see the event-driven pattern" link resolves to its overview.)
    - ``blob/main/<path.md>``  → ``<blueprints_root>/<path.md>``
    - Trailing slash is stripped.

    Returns ``None`` when the URL doesn't match, or ``blueprints_root`` is
    ``None`` (offline / skipped), or the rewritten file doesn't exist on
    disk. Callers fall through to existing http-drop behavior.
    """
    if blueprints_root is None:
        return None
    match = _BLUEPRINT_URL_RE.match(link)
    if match is None:
        return None
    rel = match.group("path").rstrip("/")
    if not rel:
        return None
    candidate = blueprints_root / rel
    if candidate.is_dir():
        candidate = candidate / "overview.md"
    elif not rel.lower().endswith(".md"):
        # blob/main/<something> that isn't markdown — skip.
        return None
    return candidate if candidate.is_file() else None


def _resolve_relative(
    link: str, current: Path, *, blueprints_root: Path | None = None
) -> Path | None:
    """Resolve a markdown link to a local file path, or ``None`` to skip.

    Order: blueprints-URL rewrite first (so a fetched blueprints tree wins
    over the http-drop), then relative-link resolution.
    """
    rewritten = _rewrite_blueprint_url(link, blueprints_root)
    if rewritten is not None:
        return rewritten
    if link.startswith(("http://", "https://", "mailto:", "#")):
        return None
    cleaned = link.split("#", 1)[0].strip()
    if not cleaned or not cleaned.lower().endswith(".md"):
        return None
    candidate = (current.parent / cleaned).resolve()
    return candidate


def _alias_matches(text: str) -> list[str]:
    """Return alias keys (lowercased) that appear in ``text``."""
    lowered = text.lower()
    hits: list[str] = []
    for alias in ALIAS_TABLE:
        # Use word-ish boundaries: alias must be surrounded by non-alnum chars.
        pattern = r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])"
        if re.search(pattern, lowered):
            hits.append(alias)
    return hits


def _cross_cutting_matches(text: str) -> list[str]:
    """Return cross-cutting category keys that appear in ``text``."""
    lowered = text.lower()
    hits: list[str] = []
    for category in CROSS_CUTTING:
        pattern = r"(?<![a-z0-9])" + re.escape(category) + r"(?![a-z0-9])"
        if re.search(pattern, lowered):
            hits.append(category)
    return hits


def _is_wrong_language_framework(rel_doc_path: str, language: str) -> bool:
    target = FRAMEWORK_LANGUAGE.get(rel_doc_path)
    if target is None:
        return False
    return target != language.lower()


def _format_marker(rel_path: str) -> str:
    return f"<!-- ===== referenced: {rel_path} ===== -->"


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def _composes_link_set(
    recipe_text: str, recipe_path: Path, *, blueprints_root: Path | None = None
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
                link_match.group(1), recipe_path, blueprints_root=blueprints_root
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


def _format_capability_body(capability: Capability) -> str:
    """Render a single capability into a ``## Capability:`` block.

    Body is the markdown body parsed by :mod:`agent_scaffold.capabilities`.
    A short metadata header (kind, env vars, docker service) is prepended so
    the LLM sees the structural contract even when the body is sparse.
    """
    parts: list[str] = [f"## Capability: {capability.id}", ""]
    meta: list[str] = [f"- kind: `{capability.kind}`"]
    if capability.env_vars:
        meta.append(f"- env vars: {', '.join(f'`{v}`' for v in capability.env_vars)}")
    if capability.docker is not None:
        meta.append(
            f"- docker service: `{capability.docker.service}` "
            f"(image: `{capability.docker.image}`)"
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
    framework: str,  # noqa: ARG001 - retained in API for future per-framework gating
    deployments_path: Path,
    *,
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

    When ``blueprints_path`` is provided, ``https://github.com/.../agent-blueprints/...``
    links in deployments docs are rewritten to local files in that tree so the
    LLM actually sees the canonical pattern content the docs reference.
    """
    docs_root = _docs_root(deployments_path).resolve()
    blueprints_root = blueprints_path.resolve() if blueprints_path is not None else None
    recipe_path = recipe.path.resolve()

    try:
        recipe_text = recipe_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FileNotFoundError(f"Could not read recipe at {recipe_path}: {exc}") from exc

    composes_targets = _composes_link_set(recipe_text, recipe_path, blueprints_root=blueprints_root)
    # Discovered (resolved_path, tier, label). First-seen wins for tier.
    discovered: dict[Path, tuple[int, str]] = {}

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
        if rel and _is_wrong_language_framework(rel, language):
            return
        if not resolved.is_file():
            _warn(f"referenced file not found, skipping: {label}")
            return
        # First-seen tier wins (don't downgrade a Tier 2 doc to Tier 6 later).
        if resolved not in discovered or tier < discovered[resolved][0]:
            discovered[resolved] = (tier, label)

    # Tier 2/3: explicit relative links in the recipe body.
    for match in _LINK_RE.finditer(recipe_text):
        link = match.group(1)
        resolved = _resolve_relative(link, recipe_path, blueprints_root=blueprints_root)
        if resolved is None:
            continue
        tier = _TIER_COMPOSES if resolved.resolve() in composes_targets else _TIER_EXPLICIT_LINK
        _consider(resolved, tier, link)

    # Tier 4: alias mentions in prose.
    for alias in _alias_matches(recipe_text):
        rel_doc = ALIAS_TABLE[alias]
        if _is_wrong_language_framework(rel_doc, language):
            continue
        _consider(docs_root / rel_doc, _TIER_ALIAS, f"alias:{alias}")

    # Tier 5: cross-cutting categories.
    for category in _cross_cutting_matches(recipe_text):
        rel_doc = CROSS_CUTTING[category]
        _consider(docs_root / rel_doc, _TIER_CROSS_CUTTING, f"cross-cutting:{category}")

    # Tier 6: transitive walk, depth-capped.
    if max_link_depth >= 1:
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
                resolved = _resolve_relative(link, current, blueprints_root=blueprints_root)
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
                if in_docs and _is_wrong_language_framework(rel, language):
                    continue
                if not resolved_abs.is_file():
                    continue
                fresh = resolved_abs not in discovered
                if fresh:
                    discovered[resolved_abs] = (_TIER_TRANSITIVE, link)
                if fresh and depth + 1 <= max_link_depth:
                    frontier.append((resolved_abs, depth + 1))

    # Budgeted assembly: keep recipe + Tier 2/3/... until cap is reached.
    recipe_text_clean = recipe_text.rstrip()
    recipe_tokens = _estimate_tokens(recipe_text_clean)

    # Read + truncate every discovered doc up front; we need their sizes.
    doc_entries: list[tuple[Path, int, str, str, int, bool]] = []
    # tuple: (path, tier, label, text, tokens, truncated)
    for path, (tier, label) in discovered.items():
        try:
            raw = path.read_text(encoding="utf-8").rstrip()
        except OSError as exc:
            _warn(f"could not read {path}: {exc}")
            continue
        text, was_truncated = _truncate(raw, max_tokens_per_doc)
        doc_entries.append((path, tier, label, text, _estimate_tokens(text), was_truncated))

    # Capability tier: each resolved capability becomes a synthetic doc entry
    # at tier 3 so it participates in the same budget pass as the link tiers.
    # The body is the formatted ``## Capability:`` block (see
    # ``_format_capability_body``); the path is the capability file so summary
    # rendering can show it. Order matches the recipe's declaration order.
    if resolved_stack is not None:
        for capability in resolved_stack.capabilities:
            cap_text = _format_capability_body(capability)
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
    pieces: list[str] = [recipe_text_clean]
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

    return AssembledContext(
        recipe_path=recipe_path,
        referenced_paths=[entry[0] for entry in kept],
        body=body,
        token_estimate=token_estimate,
        summary=summary,
    )
