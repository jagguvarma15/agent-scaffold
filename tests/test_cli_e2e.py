"""End-to-end test: ``agent-scaffold new`` with a mocked Anthropic client."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from agent_scaffold import cli, generator
from agent_scaffold.cli import app


class _Block:
    def __init__(self, text: str) -> None:
        self.text = text


class _Response:
    def __init__(self, text: str) -> None:
        self.content = [_Block(text)]


class _Messages:
    def __init__(self, payload: str) -> None:
        self._payload = payload
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return _Response(self._payload)


class _Client:
    def __init__(self, payload: str) -> None:
        self.messages = _Messages(payload)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_new_non_interactive_generates_project(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_deployments_path: Path,
    mock_responses_path: Path,
) -> None:
    payload = (mock_responses_path / "valid_python.json").read_text(encoding="utf-8")
    fake = _Client(payload)
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: fake)

    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("AGENT_SCAFFOLD_DEPLOYMENTS_PATH", str(mock_deployments_path))
    monkeypatch.setenv("AGENT_SCAFFOLD_CACHE_DIR", str(cache_dir))

    dest = tmp_path / "out" / "demo_agent"

    result = runner.invoke(
        app,
        [
            "new",
            "--non-interactive",
            "--recipe",
            "customer-support-triage",
            "--language",
            "python",
            "--framework",
            "langgraph",
            "--project-name",
            "demo_agent",
            "--dest",
            str(dest),
            "--write-mode",
            "overwrite",
            "--skip-validation",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (dest / "pyproject.toml").is_file()
    assert (dest / "src/demo_agent/main.py").is_file()
    assert (dest / "README.md").is_file()
    assert (dest / ".env.example").is_file()
    assert (dest / "scripts/smoke.sh").is_file()

    # Mocked Anthropic client was called once with cached system prompt.
    assert len(fake.messages.calls) == 1
    sys_block = fake.messages.calls[0]["system"][0]
    assert sys_block["cache_control"] == {"type": "ephemeral"}

    # Generated Python source passes its own ruff check.
    if shutil.which("ruff") is not None:
        proc = subprocess.run(
            ["ruff", "check", str(dest / "src")],
            check=False,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr


def test_new_rejects_response_missing_recipe_required_file(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_deployments_path: Path,
    mock_responses_path: Path,
) -> None:
    # The valid_python.json mock response does NOT include a Dockerfile. The
    # with-required-files recipe demands one, so the contract layer must reject.
    payload = (mock_responses_path / "valid_python.json").read_text(encoding="utf-8")
    fake = _Client(payload)
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: fake)

    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("AGENT_SCAFFOLD_DEPLOYMENTS_PATH", str(mock_deployments_path))
    monkeypatch.setenv("AGENT_SCAFFOLD_CACHE_DIR", str(cache_dir))

    dest = tmp_path / "out" / "demo_agent"

    result = runner.invoke(
        app,
        [
            "new",
            "--non-interactive",
            "--recipe",
            "with-required-files",
            "--language",
            "python",
            "--project-name",
            "demo_agent",
            "--dest",
            str(dest),
            "--write-mode",
            "overwrite",
            "--skip-validation",
        ],
    )
    assert result.exit_code == 1, result.output
    assert "Dockerfile" in result.output


def test_new_effort_high_applies_preset(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_deployments_path: Path,
    mock_responses_path: Path,
) -> None:
    payload = (mock_responses_path / "valid_python.json").read_text(encoding="utf-8")
    fake = _Client(payload)
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: fake)

    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("AGENT_SCAFFOLD_DEPLOYMENTS_PATH", str(mock_deployments_path))
    monkeypatch.setenv("AGENT_SCAFFOLD_CACHE_DIR", str(cache_dir))

    dest = tmp_path / "out" / "demo_agent"
    result = runner.invoke(
        app,
        [
            "new",
            "--non-interactive",
            "--recipe",
            "customer-support-triage",
            "--language",
            "python",
            "--framework",
            "langgraph",
            "--project-name",
            "demo_agent",
            "--dest",
            str(dest),
            "--write-mode",
            "overwrite",
            "--skip-validation",
            "--effort",
            "high",
            "--no-cache",
        ],
    )
    assert result.exit_code == 0, result.output
    call = fake.messages.calls[0]
    assert call["model"] == "claude-opus-4-7"
    assert call["max_tokens"] == 64000
    assert call["thinking"] == {"type": "enabled", "budget_tokens": 16000}
    assert "Production requirements (strict mode)" in call["system"][0]["text"]


def test_new_explicit_model_overrides_effort_preset(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_deployments_path: Path,
    mock_responses_path: Path,
) -> None:
    payload = (mock_responses_path / "valid_python.json").read_text(encoding="utf-8")
    fake = _Client(payload)
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: fake)

    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("AGENT_SCAFFOLD_DEPLOYMENTS_PATH", str(mock_deployments_path))
    monkeypatch.setenv("AGENT_SCAFFOLD_CACHE_DIR", str(cache_dir))

    dest = tmp_path / "out" / "demo_agent"
    result = runner.invoke(
        app,
        [
            "new",
            "--non-interactive",
            "--recipe",
            "customer-support-triage",
            "--language",
            "python",
            "--framework",
            "langgraph",
            "--project-name",
            "demo_agent",
            "--dest",
            str(dest),
            "--write-mode",
            "overwrite",
            "--skip-validation",
            "--effort",
            "low",
            "--model",
            "claude-sonnet-4-6",
            "--no-cache",
        ],
    )
    assert result.exit_code == 0, result.output
    # Explicit --model wins over the low-effort preset's haiku.
    assert fake.messages.calls[0]["model"] == "claude-sonnet-4-6"


def test_new_repair_then_failure_saves_raw(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_deployments_path: Path,
) -> None:
    bad_payload = "not valid json at all"
    fake = _Client(bad_payload)
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: fake)

    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("AGENT_SCAFFOLD_DEPLOYMENTS_PATH", str(mock_deployments_path))
    monkeypatch.setenv("AGENT_SCAFFOLD_CACHE_DIR", str(cache_dir))

    dest = tmp_path / "out" / "demo_agent"

    result = runner.invoke(
        app,
        [
            "new",
            "--non-interactive",
            "--recipe",
            "customer-support-triage",
            "--language",
            "python",
            "--framework",
            "langgraph",
            "--project-name",
            "demo_agent",
            "--dest",
            str(dest),
            "--write-mode",
            "overwrite",
            "--skip-validation",
        ],
    )
    assert result.exit_code == 1
    failures_dir = cache_dir / "failures"
    assert failures_dir.is_dir()
    failure_files = list(failures_dir.iterdir())
    assert failure_files, "expected at least one raw failure file"
    # Output should at least name one of the saved failure files.
    output_no_breaks = result.output.replace("\n", "").replace(" ", "")
    assert any(p.name in output_no_breaks for p in failure_files)
    assert "repair" in result.output.lower()


def test_version_flag(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "agent-scaffold" in result.output


def test_config_command(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_deployments_path: Path,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret-token")
    monkeypatch.setenv("AGENT_SCAFFOLD_DEPLOYMENTS_PATH", str(mock_deployments_path))
    monkeypatch.setenv("AGENT_SCAFFOLD_CACHE_DIR", str(tmp_path / "cache"))
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0, result.output
    assert "secret-token" not in result.output
    assert "***" in result.output


_ = cli  # pragma: no cover - keep the import live for monkeypatching.
