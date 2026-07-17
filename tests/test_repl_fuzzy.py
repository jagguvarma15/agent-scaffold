"""Tests for the REPL fuzzy-matching engine (rapidfuzz-backed)."""

from __future__ import annotations

from agent_scaffold.repl._fuzzy import completions, filter_matches, suggest

_IDS = [
    "cache.redis",
    "vector_db.qdrant",
    "vector_db.chroma",
    "sandbox.e2b",
    "obs.langfuse",
    "obs.langsmith",
]


def test_suggest_handles_typos() -> None:
    assert suggest("sanbox.e2b", _IDS) == ["sandbox.e2b"]
    assert suggest("qdrnt", _IDS)[0] == "vector_db.qdrant"


def test_suggest_handles_partial_tokens() -> None:
    # A bare token difflib would miss — WRatio catches the substring.
    assert "sandbox.e2b" in suggest("e2b", _IDS)


def test_suggest_empty_on_no_match_or_empty_query() -> None:
    assert suggest("zzzznope", _IDS) == []
    assert suggest("", _IDS) == []
    assert suggest("redis", []) == []


def test_suggest_respects_limit() -> None:
    assert len(suggest("vector", _IDS, limit=1)) <= 1


def test_filter_substring_always_qualifies() -> None:
    out = filter_matches("qdr", _IDS)
    assert out[0] == "vector_db.qdrant"


def test_filter_empty_query_returns_all() -> None:
    assert filter_matches("", _IDS) == _IDS


def test_filter_ranks_substring_before_fuzzy() -> None:
    out = filter_matches("lang", _IDS)
    # Both langfuse and langsmith substring-match "lang".
    assert set(out[:2]) == {"obs.langfuse", "obs.langsmith"}


def test_completions_prefix_first_then_fuzzy() -> None:
    out = completions("la", ["obs.langfuse", "obs.langsmith", "cache.redis"])
    assert out[:2] == ["obs.langfuse", "obs.langsmith"]
    assert "cache.redis" not in out


def test_completions_empty_prefix_returns_sorted() -> None:
    assert completions("", ["b", "a", "c"]) == ["a", "b", "c"]


def test_completions_catches_command_typo() -> None:
    cmds = ["observability", "generate", "layer", "stack"]
    assert "observability" in completions("observ", cmds)
    assert completions("genrate", cmds)[0] == "generate"
