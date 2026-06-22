"""Tests for agent_scaffold.validator event emission.

The validator's subprocess plumbing is exercised end-to-end by test_cli_e2e;
these tests just lock down the progress-event contract so the rich display can
trust the order/shape of events it gets fed.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from agent_scaffold.progress import ProgressEvent
from agent_scaffold.validator import (
    ValidationTier,
    _compile_command,
    _run,
    _run_shell,
    tier_command,
    validate,
    verify_required_files_on_disk,
)

_HAS_UV = shutil.which("uv") is not None


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


def test_run_streams_each_output_line_as_bash_line(tmp_path: Path) -> None:
    events: list[ProgressEvent] = []
    passed, output = _run(
        ["sh", "-c", "echo one; echo two >&2; echo three"],
        tmp_path,
        on_event=events.append,
    )
    assert passed
    lines = [e for e in events if e.kind == "bash_line"]
    by_line = {e.payload["line"]: e.payload["stream"] for e in lines}
    assert by_line == {"one": "stdout", "two": "stderr", "three": "stdout"}
    # stdout ordering is deterministic (single pipe); cross-stream interleave
    # depends on scheduler timing, so assert per-stream order only.
    stdout_lines = [e.payload["line"] for e in lines if e.payload["stream"] == "stdout"]
    assert stdout_lines == ["one", "three"]
    # Full output is captured for the repair loop — every line present.
    assert {"one", "two", "three"} <= set(output.splitlines())
    # Event order: started → lines → done.
    kinds = [e.kind for e in events]
    assert kinds[0] == "bash_started"
    assert kinds[-1] == "bash_done"


def test_run_shell_streams_lines_and_reports_exit(tmp_path: Path) -> None:
    events: list[ProgressEvent] = []
    passed, output = _run_shell(
        "echo hello; exit 3",
        tmp_path,
        on_event=events.append,
    )
    assert not passed
    assert "hello" in output
    line = next(e for e in events if e.kind == "bash_line")
    assert line.payload["cmd"] == "echo hello; exit 3"  # original cmd, not the sh wrapper
    done = next(e for e in events if e.kind == "bash_done")
    assert done.payload["exit_code"] == 3


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


# ---------------------------------------------------------------------------
# Compile tier
# ---------------------------------------------------------------------------


def test_compile_command_uses_declared_existing_package_roots(tmp_path: Path) -> None:
    """Only the recipe's declared roots that exist on disk are compiled."""
    (tmp_path / "app").mkdir()
    (tmp_path / "src").mkdir()  # exists but not referenced by the hints
    cmd = _compile_command(
        "python",
        tmp_path,
        {"project_layout": "app", "entry_point": "app/main.py"},
    )
    assert cmd is not None
    assert cmd[:6] == ["uv", "run", "python", "-m", "compileall", "-q"]
    # `app` is declared + present; `src` is present but undeclared → not added.
    assert cmd[6:] == ["app"]


def test_compile_command_falls_back_to_top_level_minus_venv(tmp_path: Path) -> None:
    """No declared root resolves → enumerate top-level files/dirs, skip .venv."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / ".venv").mkdir()
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    cmd = _compile_command(
        "python",
        tmp_path,
        {"project_layout": "src", "entry_point": "src/x/main.py"},  # neither exists
    )
    assert cmd is not None
    targets = cmd[6:]
    assert "pkg" in targets
    assert "main.py" in targets
    assert ".venv" not in targets
    assert "node_modules" not in targets


def test_compile_command_is_noop_for_typescript(tmp_path: Path) -> None:
    assert _compile_command("typescript", tmp_path, {}) is None
    assert _compile_command("rust", tmp_path, {}) is None


def test_tier_command_renders_compile_command(tmp_path: Path) -> None:
    (tmp_path / "app").mkdir()
    rendered = tier_command(
        ValidationTier.compile,
        "python",
        dest=tmp_path,
        hints={"project_layout": "app", "entry_point": "app/main.py"},
    )
    assert rendered == "uv run python -m compileall -q app"
    # Without a dest the compile command can't be derived; fall back to a label.
    assert tier_command(ValidationTier.compile, "python") == "python -m compileall"


def test_validate_compile_tier_noop_for_typescript(tmp_path: Path) -> None:
    results = validate(
        tmp_path,
        hints={"language": "typescript"},
        smoke_check="",
        tiers=[ValidationTier.compile],
    )
    assert results[0].tier is ValidationTier.compile
    assert results[0].passed is True
    assert "no compile check" in results[0].output


@pytest.mark.skipif(not _HAS_UV, reason="compile tier shells out to `uv run`")
def test_validate_compile_tier_passes_on_valid_python(tmp_path: Path) -> None:
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "main.py").write_text("import os\n\nagent = object()\n", encoding="utf-8")
    results = validate(
        tmp_path,
        hints={"language": "python", "project_layout": "app", "entry_point": "app/main.py"},
        smoke_check="",
        tiers=[ValidationTier.compile],
    )
    assert results[0].tier is ValidationTier.compile
    assert results[0].passed is True


@pytest.mark.skipif(not _HAS_UV, reason="compile tier shells out to `uv run`")
def test_validate_compile_tier_fails_on_syntax_error(tmp_path: Path) -> None:
    (tmp_path / "app").mkdir()
    # An incomplete import statement is a SyntaxError that ruff's E999 and the
    # compile tier both reject, but which can hide in a file the linter skips.
    (tmp_path / "app" / "main.py").write_text("from os import\n", encoding="utf-8")
    results = validate(
        tmp_path,
        hints={"language": "python", "project_layout": "app", "entry_point": "app/main.py"},
        smoke_check="",
        tiers=[ValidationTier.compile],
    )
    assert results[0].passed is False
    assert "main.py" in results[0].output
