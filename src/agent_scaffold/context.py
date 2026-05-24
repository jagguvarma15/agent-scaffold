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

from agent_scaffold.discovery import Recipe

CHARS_PER_TOKEN = 4
TOKEN_WARN_THRESHOLD = 80_000

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


class AssembledContext(BaseModel):
    recipe_path: Path
    referenced_paths: list[Path]
    body: str
    token_estimate: int


def _warn(msg: str) -> None:
    print(f"agent-scaffold: warning: {msg}", file=sys.stderr)


def _docs_root(deployments_path: Path) -> Path:
    return deployments_path / "docs"


def _resolve_relative(link: str, current: Path) -> Path | None:
    """Resolve a relative markdown link against the current file's directory."""
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


def assemble(
    recipe: Recipe,
    language: str,
    framework: str,  # noqa: ARG001 - retained in API for future per-framework gating
    deployments_path: Path,
) -> AssembledContext:
    """Build the assembled context for ``recipe`` in ``language``."""
    docs_root = _docs_root(deployments_path).resolve()
    recipe_path = recipe.path.resolve()

    try:
        recipe_text = recipe_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FileNotFoundError(f"Could not read recipe at {recipe_path}: {exc}") from exc

    visited: set[Path] = {recipe_path}
    pieces: list[str] = [recipe_text.rstrip()]
    referenced_ordered: list[Path] = []

    def _absorb(target: Path, label: str) -> None:
        target = target.resolve()
        if target in visited:
            return
        if not target.is_file():
            _warn(f"referenced file not found, skipping: {label}")
            return
        # Only resolve files that live inside docs/ to keep the bundle scoped.
        try:
            target.relative_to(docs_root)
        except ValueError:
            _warn(f"reference outside docs/, skipping: {label}")
            return
        visited.add(target)
        try:
            text = target.read_text(encoding="utf-8")
        except OSError as exc:
            _warn(f"could not read {target}: {exc}")
            return
        rel = target.relative_to(docs_root).as_posix()
        pieces.append("")
        pieces.append(_format_marker(rel))
        pieces.append(text.rstrip())
        referenced_ordered.append(target)

    # 1. Relative markdown links found in the recipe body.
    for match in _LINK_RE.finditer(recipe_text):
        link = match.group(1)
        resolved = _resolve_relative(link, recipe_path)
        if resolved is None:
            continue
        try:
            rel = resolved.relative_to(docs_root).as_posix()
        except ValueError:
            _warn(f"link outside docs/, skipping: {link}")
            continue
        if _is_wrong_language_framework(rel, language):
            continue
        _absorb(resolved, link)

    # 2. Alias mentions in the prose.
    for alias in _alias_matches(recipe_text):
        rel_doc = ALIAS_TABLE[alias]
        if _is_wrong_language_framework(rel_doc, language):
            continue
        _absorb(docs_root / rel_doc, f"alias:{alias}")

    # 3. Cross-cutting concerns by category.
    for category in _cross_cutting_matches(recipe_text):
        rel_doc = CROSS_CUTTING[category]
        _absorb(docs_root / rel_doc, f"cross-cutting:{category}")

    # 4. Transitive: walk relative links inside each newly-loaded reference.
    queue = list(referenced_ordered)
    while queue:
        current = queue.pop(0)
        try:
            text = current.read_text(encoding="utf-8")
        except OSError:
            continue
        for match in _LINK_RE.finditer(text):
            link = match.group(1)
            resolved = _resolve_relative(link, current)
            if resolved is None:
                continue
            try:
                rel = resolved.relative_to(docs_root).as_posix()
            except ValueError:
                continue
            if _is_wrong_language_framework(rel, language):
                continue
            before = len(visited)
            _absorb(resolved, link)
            if len(visited) > before:
                queue.append(resolved.resolve())

    body = "\n".join(pieces).rstrip() + "\n"
    token_estimate = _estimate_tokens(body)
    if token_estimate > TOKEN_WARN_THRESHOLD:
        _warn(
            f"assembled context is ~{token_estimate} tokens, above the "
            f"{TOKEN_WARN_THRESHOLD}-token soft limit"
        )

    return AssembledContext(
        recipe_path=recipe_path,
        referenced_paths=referenced_ordered,
        body=body,
        token_estimate=token_estimate,
    )
