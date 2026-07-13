"""Tests for the assert_model_ids backstop against hallucinated model ids."""

from __future__ import annotations

import pytest

from agent_scaffold.contract import (
    ContractParseError,
    GeneratedFile,
    GenerationResult,
    assert_model_ids,
)


def _result(files: list[GeneratedFile]) -> GenerationResult:
    return GenerationResult(
        project_name="x",
        language="python",
        files=files,
        smoke_check="echo",
    )


def test_valid_model_ids_pass() -> None:
    result = _result(
        [
            GeneratedFile(path="app/settings.py", content='model = "claude-sonnet-4-6"'),
            GeneratedFile(
                path="README.md", content="Uses Claude Sonnet 4.6 via claude-sonnet-4-6."
            ),
        ]
    )
    assert_model_ids(result)  # does not raise


def test_fabricated_date_suffix_raises_model_id_tier() -> None:
    result = _result(
        [
            GeneratedFile(
                path="docker-compose.yml",
                content="RESEARCH_MODEL: claude-sonnet-4-6-20250514",
            )
        ]
    )
    with pytest.raises(ContractParseError) as excinfo:
        assert_model_ids(result)
    assert excinfo.value.tier == "model-id"
    # The repair prompt needs the offending file and id verbatim, plus at
    # least one valid replacement suggestion.
    assert "docker-compose.yml" in excinfo.value.reason
    assert "claude-sonnet-4-6-20250514" in excinfo.value.reason
    assert "claude-sonnet-4-6" in excinfo.value.reason


def test_multiple_files_all_reported() -> None:
    result = _result(
        [
            GeneratedFile(path="a.py", content="claude-sonnet-9"),
            GeneratedFile(path="b.env", content="MODEL=claude-opus-9-20990101"),
            GeneratedFile(path="ok.py", content="claude-haiku-4-5"),
        ]
    )
    with pytest.raises(ContractParseError) as excinfo:
        assert_model_ids(result)
    assert "a.py" in excinfo.value.reason
    assert "b.env" in excinfo.value.reason
    assert "ok.py" not in excinfo.value.reason


def test_prose_and_non_model_tokens_ignored() -> None:
    result = _result(
        [
            GeneratedFile(
                path="README.md",
                content="Built with claude-code tooling. Talk to Claude Sonnet in chat.",
            )
        ]
    )
    assert_model_ids(result)  # does not raise
