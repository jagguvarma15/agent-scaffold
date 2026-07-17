"""Fuzzy matching for the REPL — one engine, rapidfuzz.

Powers unknown-command / unknown-capability suggestions, tab completion,
and the ``/stack`` + ``/recipe`` filters. rapidfuzz's ``WRatio`` scorer
handles both typos (``sanbox`` -> ``sandbox``) and partial tokens
(``e2b`` -> ``sandbox.e2b``) in one call, where the stdlib difflib only
caught the former.

Scoring parameters live here so every call site stays consistent.
"""

from __future__ import annotations

from collections.abc import Iterable

from rapidfuzz import fuzz, process

# Suggestion cutoff (0-100). Above the difflib 0.4-0.6 ratios the old sites
# used — WRatio is more generous on partial matches, so a higher floor keeps
# "did you mean" from suggesting unrelated ids.
_SUGGEST_CUTOFF = 70.0

# Filter cutoff for "/stack qdr"-style narrowing. Lower than the suggestion
# floor: a filter should surface anything plausibly related, and the user
# already committed to filtering by typing a query.
_FILTER_CUTOFF = 60.0


def suggest(query: str, candidates: Iterable[str], *, limit: int = 3) -> list[str]:
    """Return up to ``limit`` candidates closest to ``query``, best first.

    Empty when nothing clears the suggestion cutoff — callers fall back to a
    plain "unknown X" message, preserving the difflib behavior of staying
    silent rather than suggesting noise.
    """
    pool = list(candidates)
    if not query or not pool:
        return []
    matches = process.extract(
        query, pool, scorer=fuzz.WRatio, score_cutoff=_SUGGEST_CUTOFF, limit=limit
    )
    return [name for name, _score, _idx in matches]


def filter_matches(query: str, candidates: Iterable[str]) -> list[str]:
    """Candidates matching ``query`` for interactive narrowing, best first.

    A substring hit always qualifies (typing ``qdr`` should surface
    ``vector_db.qdrant`` regardless of the fuzzy score); everything else is
    ranked by ``WRatio`` above the filter cutoff.
    """
    pool = list(candidates)
    if not query:
        return pool
    q = query.lower()
    substring = [c for c in pool if q in c.lower()]
    substring_set = set(substring)
    scored = process.extract(
        query,
        [c for c in pool if c not in substring_set],
        scorer=fuzz.WRatio,
        score_cutoff=_FILTER_CUTOFF,
        limit=None,
    )
    return substring + [name for name, _score, _idx in scored]


def completions(prefix: str, candidates: Iterable[str]) -> list[str]:
    """Tab-completion ranking: exact-prefix hits first (alphabetical), then
    fuzzy matches. An empty prefix returns everything alphabetically."""
    pool = sorted(candidates)
    if not prefix:
        return pool
    p = prefix.lower()
    prefix_hits = [c for c in pool if c.lower().startswith(p)]
    prefix_set = set(prefix_hits)
    fuzzy = [c for c in filter_matches(prefix, pool) if c not in prefix_set]
    return prefix_hits + fuzzy


__all__ = ["completions", "filter_matches", "suggest"]
