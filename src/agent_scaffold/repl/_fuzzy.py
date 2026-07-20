"""Fuzzy matching for the REPL — rapidfuzz when available, difflib otherwise.

Powers unknown-command / unknown-capability suggestions, tab completion,
and the ``/stack`` + ``/recipe`` filters. rapidfuzz's ``WRatio`` scorer
handles both typos (``sanbox`` -> ``sandbox``) and partial tokens
(``e2b`` -> ``sandbox.e2b``) in one call, where the stdlib difflib only
caught the former.

rapidfuzz is a declared dependency, but a stale environment can be missing
it — an editable install whose metadata predates the dependency, or a
partial upgrade — and that raised ``ModuleNotFoundError`` at REPL launch.
A missing nicety must not take the shell down: when the import fails, a
difflib-backed scorer approximates the same contract (substring hits,
token-level partial matches, the same cutoffs) at lower match quality,
and a warning points at the reinstall that restores rapidfuzz.

Scoring parameters live here so every call site stays consistent.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from difflib import SequenceMatcher

log = logging.getLogger(__name__)

try:
    from rapidfuzz import fuzz, process

    _HAVE_RAPIDFUZZ = True
except ImportError:
    _HAVE_RAPIDFUZZ = False
    log.warning(
        "rapidfuzz is not installed — fuzzy matching degrades to a stdlib "
        "fallback. Reinstall dependencies (`uv sync`, or `pip install -e .` "
        "for an editable install) to restore full-quality suggestions."
    )

# Suggestion cutoff (0-100). Above the difflib 0.4-0.6 ratios the old sites
# used — WRatio is more generous on partial matches, so a higher floor keeps
# "did you mean" from suggesting unrelated ids.
_SUGGEST_CUTOFF = 70.0

# Filter cutoff for "/stack qdr"-style narrowing. Lower than the suggestion
# floor: a filter should surface anything plausibly related, and the user
# already committed to filtering by typing a query.
_FILTER_CUTOFF = 60.0

# Separators that delimit tokens inside capability ids and commands
# (``vector_db.qdrant`` -> ``vector_db``, ``qdrant``). The fallback scores
# the best token as well as the whole string so partial-token matches keep
# working without rapidfuzz.
_TOKEN_SPLIT = re.compile(r"[._\-/ ]+")


def _fallback_score(query: str, candidate: str) -> float:
    """Approximate ``fuzz.WRatio``'s 0-100 scale with stdlib difflib."""
    q = query.lower()
    c = candidate.lower()
    if q in c:
        # A substring hit scores near-perfect, shorter candidates first —
        # mirrors WRatio's partial-match generosity.
        return 100.0 - min(len(c) - len(q), 25)
    whole = SequenceMatcher(None, q, c).ratio()
    tokens = [t for t in _TOKEN_SPLIT.split(c) if t]
    best_token = max((SequenceMatcher(None, q, t).ratio() for t in tokens), default=0.0)
    # A token-only match is slightly discounted so whole-string closeness
    # wins ties.
    return 100.0 * max(whole, 0.95 * best_token)


def _fallback_extract(query: str, pool: list[str], cutoff: float) -> list[str]:
    """Candidates clearing ``cutoff``, best first, deterministic on ties."""
    scored = [(c, _fallback_score(query, c)) for c in pool]
    kept = [(c, s) for c, s in scored if s >= cutoff]
    kept.sort(key=lambda item: (-item[1], item[0]))
    return [c for c, _s in kept]


def suggest(query: str, candidates: Iterable[str], *, limit: int = 3) -> list[str]:
    """Return up to ``limit`` candidates closest to ``query``, best first.

    Empty when nothing clears the suggestion cutoff — callers fall back to a
    plain "unknown X" message, preserving the difflib behavior of staying
    silent rather than suggesting noise.
    """
    pool = list(candidates)
    if not query or not pool:
        return []
    if not _HAVE_RAPIDFUZZ:
        return _fallback_extract(query, pool, _SUGGEST_CUTOFF)[:limit]
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
    rest = [c for c in pool if c not in substring_set]
    if not _HAVE_RAPIDFUZZ:
        return substring + _fallback_extract(query, rest, _FILTER_CUTOFF)
    scored = process.extract(
        query,
        rest,
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
