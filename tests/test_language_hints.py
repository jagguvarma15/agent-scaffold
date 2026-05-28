"""Tests for ``agent_scaffold.language_hints`` — the leaf YAML loader.

Both CLI and REPL consume this module. We verify the contract here so
neither caller needs to (the CLI wraps the error in typer.BadParameter,
the REPL relies on the wizard pre-validating before calling).
"""

from __future__ import annotations

import pytest

from agent_scaffold.language_hints import (
    UnknownLanguageError,
    available_languages,
    load_language_hints,
)


def test_available_languages_returns_sorted_yaml_slugs() -> None:
    langs = available_languages()
    # The shipped languages package contains at least python + typescript.
    assert "python" in langs
    assert "typescript" in langs
    assert langs == sorted(langs)


def test_load_language_hints_returns_dict_with_manifest_key() -> None:
    """Real bundled python.yaml round-trips through safe_load."""
    hints = load_language_hints("python")
    assert isinstance(hints, dict)
    assert "manifest" in hints  # the contract validator requires this


def test_load_language_hints_raises_for_unknown() -> None:
    with pytest.raises(UnknownLanguageError):
        load_language_hints("klingon")
