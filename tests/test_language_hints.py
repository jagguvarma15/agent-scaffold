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
    reconcile_entry_point,
)

_PY_HINTS = {
    "project_layout": "src",
    "entry_point": "src/{project_name}/main.py",
    "manifest": "pyproject.toml",
    "smoke_check": "uv run python -c 'from {project_name}.main import agent; print(\"ok\")'",
}


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


def test_reconcile_rewrites_to_recipe_app_layout() -> None:
    """An ``app/main.py`` required file overrides the default ``src/`` entry."""
    required = ["Dockerfile", "app/main.py", "app/agent/researcher.py"]
    out = reconcile_entry_point(_PY_HINTS, required)
    assert out["entry_point"] == "app/main.py"
    assert out["project_layout"] == "app"
    assert out["smoke_check"] == ("uv run python -c 'from app.main import agent; print(\"ok\")'")


def test_reconcile_does_not_mutate_input() -> None:
    required = ["app/main.py"]
    reconcile_entry_point(_PY_HINTS, required)
    assert _PY_HINTS["entry_point"] == "src/{project_name}/main.py"


def test_reconcile_noop_for_src_layout_recipe() -> None:
    """A recipe that declares no matching entry leaves the default untouched."""
    required = ["Dockerfile", "tests/unit/test_orchestrator.py"]
    out = reconcile_entry_point(_PY_HINTS, required)
    assert out["entry_point"] == "src/{project_name}/main.py"
    assert out["project_layout"] == "src"
    assert out is _PY_HINTS  # returned unchanged


def test_reconcile_noop_when_basename_differs() -> None:
    """A required file whose basename differs from the entry never reconciles."""
    ts_hints = {"project_layout": "src", "entry_point": "src/index.ts"}
    out = reconcile_entry_point(ts_hints, ["src/server.ts"])
    assert out["entry_point"] == "src/index.ts"


def test_typescript_entry_point_is_canonical_index_ts() -> None:
    """The TS layout default must match project-layout.md + every recipe body
    (``src/index.ts``); ``src/main.ts`` silently breaks the required-files
    contract since no recipe declares an override."""
    hints = load_language_hints("typescript")
    assert hints["entry_point"] == "src/index.ts"
    assert "src/index.ts" in hints["smoke_check"]


def test_python_entry_point_default_is_src_layout() -> None:
    """Python keeps its language default; recipes override to app/ via reconcile."""
    hints = load_language_hints("python")
    assert hints["entry_point"] == "src/{project_name}/main.py"


def test_reconcile_real_python_hints_with_research_assistant() -> None:
    hints = load_language_hints("python")
    out = reconcile_entry_point(
        hints, ["app/main.py", "app/agent/researcher.py", "app/tools/web_search.py"]
    )
    assert out["entry_point"] == "app/main.py"
    assert out["project_layout"] == "app"
