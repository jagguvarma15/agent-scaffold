"""Tests for agent_forge.contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_forge.contract import (
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


def test_parse_schema_violation_raises() -> None:
    raw = '{"project_name": "x"}'
    with pytest.raises(ContractParseError) as excinfo:
        parse(raw)
    assert excinfo.value.raw == raw


def test_validate_paths_rejects_dotdot(tmp_path: Path) -> None:
    result = GenerationResult(
        project_name="x",
        language="python",
        files=[GeneratedFile(path="../escape.txt", content="boom")],
        smoke_check="echo",
    )
    with pytest.raises(ContractParseError, match=r"\.\."):
        validate_paths(result, tmp_path)


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
    with pytest.raises(ContractParseError, match="manifest"):
        validate_required_files(result, hints)


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
