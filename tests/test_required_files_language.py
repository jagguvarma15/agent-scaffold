"""Language-aware filtering of recipe ``required_files``.

A recipe's frontmatter carries ONE ``required_files`` list regardless of the
picked target language, so entries like ``app/main.py`` used to be enforced
on TypeScript runs too. A live run showed the full failure shape: the model
emitted a complete, valid TypeScript project, validation rejected it for the
missing Python paths, and the repair round — told to add ``app/main.py`` —
regenerated the entire project in Python (which then failed for the missing
``package.json``). :func:`required_files_for_language` drops entries wearing
another language's extension before they reach the prompt, the contract
check, the repair loop, or the on-disk verify.
"""

from __future__ import annotations

import pytest

from agent_scaffold.contract import ContractParseError, GenerationResult, validate_required_files
from agent_scaffold.discovery import required_files_for_language

# The research-assistant recipe's list at the time of the failing run: three
# language-neutral paths plus six Python-only paths.
_RECIPE_REQUIRED = [
    "Dockerfile",
    "docker-compose.yml",
    ".github/workflows/ci.yml",
    "app/main.py",
    "app/agent/researcher.py",
    "app/tools/web_search.py",
    "tests/unit/test_schemas.py",
    "tests/integration/test_research.py",
    "tests/eval/test_react_behavior.py",
]

# The file set the model actually emitted in the rejected TypeScript attempt
# (failures/20260721T114334.json) — a complete project, including adapted
# ``.ts`` variants of the recipe's ``.py`` test paths.
_EMITTED_TS_PATHS = [
    "package.json",
    "tsconfig.json",
    "src/config.ts",
    "src/schemas.ts",
    "src/tools/web-search.ts",
    "src/agent/researcher.ts",
    "src/api/research.ts",
    "src/index.ts",
    "tests/unit/test_schemas.ts",
    "tests/integration/test_research.ts",
    "tests/eval/test_react_behavior.ts",
    ".env.example",
    ".gitignore",
    "Dockerfile",
    "docker-compose.yml",
    ".github/workflows/ci.yml",
    "README.md",
]

_TS_HINTS = {"language": "typescript", "manifest": "package.json", "entry_point": "src/index.ts"}


def test_python_paths_drop_for_typescript() -> None:
    filtered = required_files_for_language(_RECIPE_REQUIRED, "typescript")
    assert filtered == ["Dockerfile", "docker-compose.yml", ".github/workflows/ci.yml"]


def test_typescript_paths_drop_for_python() -> None:
    required = ["Dockerfile", "src/index.ts", "src/app.tsx", "lib/util.js", "app/main.py"]
    assert required_files_for_language(required, "python") == ["Dockerfile", "app/main.py"]


def test_python_run_keeps_its_own_paths() -> None:
    assert required_files_for_language(_RECIPE_REQUIRED, "python") == _RECIPE_REQUIRED


def test_no_language_returns_the_list_unchanged() -> None:
    assert required_files_for_language(_RECIPE_REQUIRED, None) == _RECIPE_REQUIRED


def test_unknown_language_keeps_only_neutral_paths() -> None:
    filtered = required_files_for_language(_RECIPE_REQUIRED, "go")
    assert filtered == ["Dockerfile", "docker-compose.yml", ".github/workflows/ci.yml"]


def _ts_result() -> GenerationResult:
    return GenerationResult(
        project_name="research-assistant",
        language="typescript",
        files=[{"path": p, "content": "x"} for p in _EMITTED_TS_PATHS],
        smoke_check="npm test",
    )


def test_valid_typescript_project_passes_with_filtered_requirements() -> None:
    """The rejected live attempt passes once the Python-only paths are gone."""
    filtered = required_files_for_language(_RECIPE_REQUIRED, "typescript")
    validate_required_files(_ts_result(), _TS_HINTS, filtered)


def test_raw_recipe_list_would_still_reject_it() -> None:
    """Documents the pre-fix behavior the filter exists to prevent."""
    with pytest.raises(ContractParseError) as excinfo:
        validate_required_files(_ts_result(), _TS_HINTS, _RECIPE_REQUIRED)
    assert "app/main.py" in str(excinfo.value)
