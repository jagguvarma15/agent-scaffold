"""Tests for agent_scaffold.contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_scaffold.contract import (
    ContractParseError,
    GeneratedFile,
    GenerationResult,
    parse,
    validate_paths,
    validate_required_files,
)


def test_parse_valid_json(mock_responses_path: Path) -> None:
    raw = (mock_responses_path / "valid_python.json").read_text(encoding="utf-8")
    result = parse(raw)
    assert result.project_name == "demo_agent"
    assert result.language == "python"
    assert any(f.path == "pyproject.toml" for f in result.files)


def test_parse_strips_fences(mock_responses_path: Path) -> None:
    raw = (mock_responses_path / "valid_python_fenced.txt").read_text(encoding="utf-8")
    result = parse(raw)
    assert result.project_name == "demo_agent"


def test_parse_malformed_raises(mock_responses_path: Path) -> None:
    raw = (mock_responses_path / "malformed.txt").read_text(encoding="utf-8")
    with pytest.raises(ContractParseError) as excinfo:
        parse(raw)
    assert excinfo.value.raw == raw
    assert "invalid JSON" in excinfo.value.reason
    # Failure tier lets the pipeline render a structured warning instead of
    # parsing the prose reason.
    assert excinfo.value.tier == "json"
    assert excinfo.value.field is None


def test_parse_schema_violation_raises() -> None:
    raw = '{"project_name": "x"}'
    with pytest.raises(ContractParseError) as excinfo:
        parse(raw)
    assert excinfo.value.raw == raw
    assert excinfo.value.tier == "schema"
    # field carries the first ValidationError location dotted-path so users
    # see where to look (e.g. "files", "smoke_check").
    assert excinfo.value.field is not None


def test_validate_paths_rejects_dotdot(tmp_path: Path) -> None:
    result = GenerationResult(
        project_name="x",
        language="python",
        files=[GeneratedFile(path="../escape.txt", content="boom")],
        smoke_check="echo",
    )
    with pytest.raises(ContractParseError, match=r"\.\.") as excinfo:
        validate_paths(result, tmp_path)
    # path-tier failures carry the offending path so users see exactly which
    # file the LLM tried to write outside the project root.
    assert excinfo.value.tier == "path"
    assert excinfo.value.field == "../escape.txt"


def test_validate_paths_rejects_absolute(tmp_path: Path) -> None:
    result = GenerationResult(
        project_name="x",
        language="python",
        files=[GeneratedFile(path="/etc/passwd", content="x")],
        smoke_check="echo",
    )
    with pytest.raises(ContractParseError, match="absolute"):
        validate_paths(result, tmp_path)


def test_validate_paths_rejects_duplicate(tmp_path: Path) -> None:
    result = GenerationResult(
        project_name="x",
        language="python",
        files=[
            GeneratedFile(path="a.txt", content="1"),
            GeneratedFile(path="a.txt", content="2"),
        ],
        smoke_check="echo",
    )
    with pytest.raises(ContractParseError, match="duplicate"):
        validate_paths(result, tmp_path)


def test_validate_paths_rejects_hyphenated_python_module_dir(tmp_path: Path) -> None:
    """The LLM occasionally writes both ``src/foo-bar/`` and ``src/foo_bar/``
    in the same project — only the underscored one is a real Python package.
    The check rejects the hyphenated dir so the repair loop re-prompts the
    model to use the canonical name."""
    result = GenerationResult(
        project_name="restaurant_rebooking",
        language="python",
        files=[
            GeneratedFile(path="src/restaurant_rebooking/__init__.py", content=""),
            GeneratedFile(path="src/restaurant-rebooking/main.py", content="boom"),
        ],
        smoke_check="echo",
    )
    with pytest.raises(ContractParseError, match=r"hyphenated form"):
        validate_paths(result, tmp_path, canonical_module_name="restaurant_rebooking")


def test_validate_paths_allows_correct_underscored_module(tmp_path: Path) -> None:
    """Sanity: when the dir uses the canonical underscored name, no error."""
    result = GenerationResult(
        project_name="restaurant_rebooking",
        language="python",
        files=[
            GeneratedFile(path="src/restaurant_rebooking/__init__.py", content=""),
            GeneratedFile(path="src/restaurant_rebooking/main.py", content="..."),
        ],
        smoke_check="echo",
    )
    validate_paths(result, tmp_path, canonical_module_name="restaurant_rebooking")


def test_validate_paths_no_module_check_when_name_has_no_underscore(
    tmp_path: Path,
) -> None:
    """If the project name has no underscores (e.g., 'demo'), there's no
    hyphenated form to forbid — every src/<dir>/ is acceptable."""
    result = GenerationResult(
        project_name="demo",
        language="python",
        files=[GeneratedFile(path="src/demo/main.py", content="...")],
        smoke_check="echo",
    )
    validate_paths(result, tmp_path, canonical_module_name="demo")


def test_validate_paths_accepts_clean(tmp_path: Path) -> None:
    result = GenerationResult(
        project_name="x",
        language="python",
        files=[GeneratedFile(path="src/main.py", content="x")],
        smoke_check="echo",
    )
    validate_paths(result, tmp_path)  # should not raise


def test_validate_required_files_missing_manifest() -> None:
    hints = {"manifest": "pyproject.toml", "entry_point": "src/{project_name}/main.py"}
    result = GenerationResult(
        project_name="demo",
        language="python",
        files=[
            GeneratedFile(path="src/demo/main.py", content="x"),
            GeneratedFile(path="README.md", content="x"),
            GeneratedFile(path=".env.example", content="x"),
        ],
        smoke_check="echo",
    )
    with pytest.raises(ContractParseError, match="manifest") as excinfo:
        validate_required_files(result, hints)
    # required-files tier carries the missing filename so users see the gap.
    assert excinfo.value.tier == "required-files"
    assert excinfo.value.field == "pyproject.toml"


def test_validate_required_files_missing_entry() -> None:
    hints = {"manifest": "pyproject.toml", "entry_point": "src/{project_name}/main.py"}
    result = GenerationResult(
        project_name="demo",
        language="python",
        files=[
            GeneratedFile(path="pyproject.toml", content="x"),
            GeneratedFile(path="README.md", content="x"),
            GeneratedFile(path=".env.example", content="x"),
        ],
        smoke_check="echo",
    )
    with pytest.raises(ContractParseError, match="entry point"):
        validate_required_files(result, hints)


def test_validate_required_files_passes() -> None:
    hints = {"manifest": "pyproject.toml", "entry_point": "src/{project_name}/main.py"}
    result = GenerationResult(
        project_name="demo",
        language="python",
        files=[
            GeneratedFile(path="pyproject.toml", content="x"),
            GeneratedFile(path="src/demo/main.py", content="x"),
            GeneratedFile(path="README.md", content="x"),
            GeneratedFile(path=".env.example", content="x"),
        ],
        smoke_check="echo",
    )
    validate_required_files(result, hints)


def _base_result() -> GenerationResult:
    return GenerationResult(
        project_name="demo",
        language="python",
        files=[
            GeneratedFile(path="pyproject.toml", content="x"),
            GeneratedFile(path="src/demo/main.py", content="x"),
            GeneratedFile(path="README.md", content="x"),
            GeneratedFile(path=".env.example", content="x"),
        ],
        smoke_check="echo",
    )


def test_validate_required_files_extra_missing_raises() -> None:
    hints = {"manifest": "pyproject.toml", "entry_point": "src/{project_name}/main.py"}
    with pytest.raises(ContractParseError, match=r"missing required file\(s\): Dockerfile"):
        validate_required_files(_base_result(), hints, ["Dockerfile"])


def test_validate_required_files_extra_present_passes() -> None:
    hints = {"manifest": "pyproject.toml", "entry_point": "src/{project_name}/main.py"}
    result = _base_result()
    result.files.append(GeneratedFile(path="Dockerfile", content="FROM scratch"))
    validate_required_files(result, hints, ["Dockerfile"])


def test_validate_required_files_reports_all_missing_in_one_error() -> None:
    # The core fix: a recipe missing several files surfaces ALL of them in one
    # error, so the single repair round can add them together (instead of
    # discovering one missing file per failed generation — the app/ layout bug).
    hints = {"manifest": "pyproject.toml", "entry_point": "src/{project_name}/main.py"}
    result = _base_result()  # has manifest, src/demo/main.py, README, .env.example
    required = ["app/main.py", "app/agent/researcher.py", "app/tools/web_search.py"]
    with pytest.raises(ContractParseError) as excinfo:
        validate_required_files(result, hints, required)
    reason = excinfo.value.reason
    for path in required:
        assert path in reason  # every gap named at once, not just the first
    assert excinfo.value.field == "app/main.py"  # first missing, for back-compat


def test_validate_required_files_recipe_entry_overrides_default_layout() -> None:
    # A recipe declaring its own app/ entry must not also require the generic
    # language-default src/<pkg>/main.py — the model only ever sees the recipe's
    # required_files, never the validation-only entry_point hint.
    hints = {"manifest": "pyproject.toml", "entry_point": "src/{project_name}/main.py"}
    result = GenerationResult(
        project_name="research_assistant",
        language="python",
        files=[
            GeneratedFile(path="pyproject.toml", content="x"),
            GeneratedFile(path="README.md", content="x"),
            GeneratedFile(path=".env.example", content="x"),
            GeneratedFile(path="app/main.py", content="x"),
        ],
        smoke_check="echo",
    )
    # No src/research_assistant/main.py — but the recipe declares app/main.py.
    validate_required_files(result, hints, ["app/main.py"])


def test_validate_required_files_default_entry_enforced_without_recipe_entry() -> None:
    # A recipe that declares no entry of its own still gets the language default.
    hints = {"manifest": "pyproject.toml", "entry_point": "src/{project_name}/main.py"}
    result = GenerationResult(
        project_name="demo",
        language="python",
        files=[
            GeneratedFile(path="pyproject.toml", content="x"),
            GeneratedFile(path="README.md", content="x"),
            GeneratedFile(path=".env.example", content="x"),
            GeneratedFile(path="Dockerfile", content="x"),
        ],
        smoke_check="echo",
    )
    with pytest.raises(ContractParseError, match="entry point"):
        validate_required_files(result, hints, ["Dockerfile"])


# ---------------------------------------------------------------------------
# Structured-output schema
# ---------------------------------------------------------------------------


def test_generation_result_schema_matches_model() -> None:
    """The hand-authored schema stays field-for-field in sync with the model."""
    from agent_scaffold.contract import GENERATION_RESULT_SCHEMA, GenerationResult

    schema_fields = set(GENERATION_RESULT_SCHEMA["properties"])
    model_fields = set(GenerationResult.model_fields)
    assert schema_fields == model_fields
    # The structured-outputs subset mandates every key in required.
    assert set(GENERATION_RESULT_SCHEMA["required"]) == schema_fields
    file_schema = GENERATION_RESULT_SCHEMA["properties"]["files"]["items"]
    from agent_scaffold.contract import GeneratedFile

    assert set(file_schema["properties"]) == set(GeneratedFile.model_fields)


def test_generation_result_schema_fits_the_supported_subset() -> None:
    """Walk the schema asserting the structured-outputs subset invariants:
    every object closes additionalProperties and requires every key; no
    recursion via $ref; no string-length or numeric constraints."""
    from agent_scaffold.contract import GENERATION_RESULT_SCHEMA

    banned = {"$ref", "minLength", "maxLength", "minimum", "maximum", "multipleOf", "pattern"}

    def walk(node: object) -> None:
        if not isinstance(node, dict):
            return
        assert not banned.intersection(node), f"banned keyword in {node}"
        if node.get("type") == "object":
            assert node.get("additionalProperties") is False
            assert set(node.get("required", [])) == set(node.get("properties", {}))
        if node.get("type") == "array":
            assert node.get("minItems", 0) in (0, 1)
        for value in node.values():
            if isinstance(value, dict):
                walk(value)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

    walk(GENERATION_RESULT_SCHEMA)


def test_validate_paths_allows_hyphenated_dirs_for_typescript(tmp_path: Path) -> None:
    """Hyphenated directories are idiomatic in npm projects — the Python
    module-dir check must never fire for a TypeScript run, even when the
    user named the project with an underscore."""
    result = GenerationResult(
        project_name="my_agent",
        language="typescript",
        files=[
            GeneratedFile(path="src/my-agent/index.ts", content="export {};"),
            GeneratedFile(path="package.json", content="{}"),
        ],
        smoke_check="pnpm test",
    )
    validate_paths(result, tmp_path, canonical_module_name="my_agent")  # no raise


def test_normalize_package_manager_pins_unpinned_typescript_package_json() -> None:
    """corepack resolves an unpinned manager to its latest release at image
    build time — pnpm 11 on a node:20 base failed every `RUN pnpm install`
    with ERR_UNKNOWN_BUILTIN_MODULE. The scaffold injects the known-good pin
    deterministically instead of prompting for it."""
    from agent_scaffold.contract import normalize_package_manager

    result = GenerationResult(
        project_name="demo",
        language="typescript",
        files=[GeneratedFile(path="package.json", content='{"name": "demo"}')],
        smoke_check="pnpm test",
    )
    hints = {"language": "typescript", "package_manager_pin": "pnpm@10.33.0"}
    fixed = normalize_package_manager(result, hints)
    data = json.loads(fixed.files[0].content)
    assert data["packageManager"] == "pnpm@10.33.0"


def test_normalize_package_manager_respects_an_existing_pin() -> None:
    from agent_scaffold.contract import normalize_package_manager

    result = GenerationResult(
        project_name="demo",
        language="typescript",
        files=[
            GeneratedFile(path="package.json", content='{"packageManager": "pnpm@9.0.0"}'),
        ],
        smoke_check="pnpm test",
    )
    hints = {"language": "typescript", "package_manager_pin": "pnpm@10.33.0"}
    fixed = normalize_package_manager(result, hints)
    assert json.loads(fixed.files[0].content)["packageManager"] == "pnpm@9.0.0"


def test_normalize_package_manager_noops_outside_typescript() -> None:
    from agent_scaffold.contract import normalize_package_manager

    result = GenerationResult(
        project_name="demo",
        language="python",
        files=[GeneratedFile(path="package.json", content="{}")],
        smoke_check="pytest",
    )
    fixed = normalize_package_manager(result, {"language": "python"})
    assert fixed.files[0].content == "{}"


def test_normalize_package_manager_survives_unparseable_package_json() -> None:
    from agent_scaffold.contract import normalize_package_manager

    result = GenerationResult(
        project_name="demo",
        language="typescript",
        files=[GeneratedFile(path="package.json", content="not json{")],
        smoke_check="pnpm test",
    )
    hints = {"language": "typescript", "package_manager_pin": "pnpm@10.33.0"}
    assert normalize_package_manager(result, hints).files[0].content == "not json{"


def test_typescript_language_hints_declare_the_pin() -> None:
    from agent_scaffold.language_hints import load_language_hints

    pin = load_language_hints("typescript").get("package_manager_pin", "")
    assert pin.startswith("pnpm@")
