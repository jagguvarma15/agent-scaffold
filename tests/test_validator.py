"""Tests for agent_scaffold.validator event emission.

The validator's subprocess plumbing is exercised end-to-end by test_cli_e2e;
these tests just lock down the progress-event contract so the rich display can
trust the order/shape of events it gets fed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_scaffold.progress import ProgressEvent
from agent_scaffold.validator import ValidationTier, validate, verify_required_files_on_disk


def test_validate_emits_bash_started_then_done_for_static_tier(
    tmp_path: Path,
) -> None:
    events: list[ProgressEvent] = []
    results = validate(
        tmp_path,
        hints={"language": "python"},
        smoke_check="",
        tiers=[ValidationTier.static],
        on_event=events.append,
    )
    assert results, "expected one ValidationResult"
    kinds = [e.kind for e in events]
    # At least one bash_started followed by bash_done.
    assert "bash_started" in kinds
    assert "bash_done" in kinds
    assert kinds.index("bash_started") < kinds.index("bash_done")
    # The cmd in the started/done payloads should be the ruff invocation.
    started = next(e for e in events if e.kind == "bash_started")
    assert started.payload["cmd"][0] == "ruff"


def test_validate_unsupported_language_skips_without_events(tmp_path: Path) -> None:
    events: list[ProgressEvent] = []
    results = validate(
        tmp_path,
        hints={"language": "rust"},
        smoke_check="",
        tiers=[ValidationTier.static],
        on_event=events.append,
    )
    # Unsupported language: no command runs, so no bash events.
    assert events == []
    assert results[0].passed is True
    assert "no static check" in results[0].output


def test_verify_required_files_on_disk_returns_missing(tmp_path: Path) -> None:
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("x = 1\n")
    missing = verify_required_files_on_disk(
        tmp_path,
        ["Dockerfile", "docker-compose.yml", "src/main.py", "src/missing.py"],
    )
    assert missing == ["docker-compose.yml", "src/missing.py"]


def test_verify_required_files_on_disk_empty_when_all_present(tmp_path: Path) -> None:
    (tmp_path / "a.txt").touch()
    assert verify_required_files_on_disk(tmp_path, ["a.txt"]) == []


def test_verify_required_files_on_disk_rejects_directory_match(tmp_path: Path) -> None:
    """A directory at the required path counts as missing — required_files is for files."""
    (tmp_path / "Dockerfile").mkdir()
    assert verify_required_files_on_disk(tmp_path, ["Dockerfile"]) == ["Dockerfile"]


def test_validate_smoke_tier_emits_bash_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[ProgressEvent] = []
    validate(
        tmp_path,
        hints={"language": "python"},
        smoke_check="true",
        tiers=[ValidationTier.smoke],
        on_event=events.append,
    )
    kinds = [e.kind for e in events]
    assert kinds == ["bash_started", "bash_done"]
    assert events[1].payload["exit_code"] == 0
