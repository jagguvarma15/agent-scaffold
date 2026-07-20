"""Tests for the REPL fuzzy-matching engine (rapidfuzz-backed, difflib fallback)."""

from __future__ import annotations

import builtins
import importlib
import logging
import sys
from collections.abc import Iterator
from types import ModuleType

import pytest

import agent_scaffold.repl._fuzzy as fuzzy_mod
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


def _reload_without_rapidfuzz(mp: pytest.MonkeyPatch) -> ModuleType:
    """Reload ``_fuzzy`` with the rapidfuzz import blocked."""
    real_import = builtins.__import__

    def _blocked(name: str, *args: object, **kwargs: object) -> ModuleType:
        if name.split(".")[0] == "rapidfuzz":
            raise ImportError("rapidfuzz blocked for this test")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    mp.setattr(builtins, "__import__", _blocked)
    mp.delitem(sys.modules, "rapidfuzz", raising=False)
    return importlib.reload(fuzzy_mod)


@pytest.fixture()
def fallback_fuzzy() -> Iterator[ModuleType]:
    """The ``_fuzzy`` module as it loads in an env missing rapidfuzz.

    Restores the real rapidfuzz-backed module afterwards, so test order
    never leaks the degraded engine into other tests.
    """
    mp = pytest.MonkeyPatch()
    try:
        yield _reload_without_rapidfuzz(mp)
    finally:
        mp.undo()
        importlib.reload(fuzzy_mod)


def test_missing_rapidfuzz_does_not_crash_import(fallback_fuzzy: ModuleType) -> None:
    # The original failure mode: ModuleNotFoundError at REPL launch. The
    # module must import and stay functional on the fallback engine.
    assert fallback_fuzzy._HAVE_RAPIDFUZZ is False


def test_fallback_suggest_handles_typos(fallback_fuzzy: ModuleType) -> None:
    assert fallback_fuzzy.suggest("sanbox.e2b", _IDS) == ["sandbox.e2b"]
    assert fallback_fuzzy.suggest("qdrnt", _IDS)[0] == "vector_db.qdrant"


def test_fallback_suggest_handles_partial_tokens(fallback_fuzzy: ModuleType) -> None:
    assert "sandbox.e2b" in fallback_fuzzy.suggest("e2b", _IDS)


def test_fallback_suggest_stays_silent_on_noise(fallback_fuzzy: ModuleType) -> None:
    assert fallback_fuzzy.suggest("zzzznope", _IDS) == []
    assert fallback_fuzzy.suggest("", _IDS) == []
    assert fallback_fuzzy.suggest("redis", []) == []


def test_fallback_suggest_respects_limit(fallback_fuzzy: ModuleType) -> None:
    assert len(fallback_fuzzy.suggest("vector", _IDS, limit=1)) <= 1


def test_fallback_filter_and_completions(fallback_fuzzy: ModuleType) -> None:
    assert fallback_fuzzy.filter_matches("qdr", _IDS)[0] == "vector_db.qdrant"
    assert fallback_fuzzy.filter_matches("", _IDS) == _IDS
    cmds = ["observability", "generate", "layer", "stack"]
    assert "observability" in fallback_fuzzy.completions("observ", cmds)
    assert fallback_fuzzy.completions("genrate", cmds)[0] == "generate"


def test_fallback_warns_at_import(caplog: pytest.LogCaptureFixture) -> None:
    mp = pytest.MonkeyPatch()
    try:
        with caplog.at_level(logging.WARNING, logger="agent_scaffold.repl._fuzzy"):
            _reload_without_rapidfuzz(mp)
        assert any("rapidfuzz is not installed" in r.message for r in caplog.records)
    finally:
        mp.undo()
        importlib.reload(fuzzy_mod)
