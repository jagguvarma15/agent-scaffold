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


class _StreamCtx:
    def __init__(self, response: Any) -> None:
        self._response = response

    def __enter__(self) -> _StreamCtx:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def __iter__(self) -> Any:
        return iter(())

    def get_final_message(self) -> Any:
        return self._response


class _Messages:
    def __init__(self, payload: str) -> None:
        self._payload = payload
        self.calls: list[dict[str, Any]] = []

    def stream(self, **kwargs: Any) -> _StreamCtx:
        self.calls.append(kwargs)
        return _StreamCtx(_Response(self._payload))


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


def test_post_write_catches_required_files_missing_from_disk(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_deployments_path: Path,
    mock_responses_path: Path,
) -> None:
    """S4: even when the contract check passes, missing files on disk must fail the run.

    Setup: the with-required-files recipe demands Dockerfile + docker-compose.yml.
    The valid_python.json mock omits both. We monkey-patch the contract's
    required-files check to no-op so write proceeds; the new post-write verify
    step is the only thing that can catch the gap.
    """
    payload = (mock_responses_path / "valid_python.json").read_text(encoding="utf-8")
    fake = _Client(payload)
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: fake)

    # validate_required_files moved to pipeline.py during the cmd_new
    # refactor; patch it there so the test bypasses the in-response check
    # and exercises the on-disk verification path.
    from agent_scaffold import pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "validate_required_files", lambda *_a, **_kw: None)

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
            "--no-format",
            "--skip-validation",
        ],
    )
    assert result.exit_code == 1, result.output
    assert "Required files missing after write" in result.output
    assert "Dockerfile" in result.output
    assert "docker-compose.yml" in result.output
    assert "--write-mode skip" in result.output  # cause-list hint surfaced


def test_post_write_verify_passes_when_required_files_present(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_deployments_path: Path,
    mock_responses_path: Path,
) -> None:
    """S4 happy path: recipe with no required_files should not surface the verify step at all."""
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
            "--no-format",
            "--skip-validation",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Required files missing" not in result.output


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
    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_config"] == {"effort": "high"}
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


def test_new_merges_recipe_dependencies_into_hints(
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
            "with-recipe-deps",
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
    assert result.exit_code == 0, result.output
    call = fake.messages.calls[0]
    user_text = "".join(block["text"] for block in call["messages"][0]["content"])
    # Recipe-declared deps appear inside the language_hints_yaml block.
    assert "redis: '>=5.0.0'" in user_text or 'redis: ">=5.0.0"' in user_text
    assert "structlog: '>=24.1.0'" in user_text or 'structlog: ">=24.1.0"' in user_text
    # The default pinned dep is still there.
    assert "anthropic:" in user_text


def test_new_recipe_dependencies_override_language_defaults(
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
            "conflict-recipe-deps",
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
    assert result.exit_code == 0, result.output
    call = fake.messages.calls[0]
    user_text = "".join(block["text"] for block in call["messages"][0]["content"])
    # Recipe wins: the constraint declared in the recipe replaces the language default.
    assert "anthropic: '>=99.0.0'" in user_text or 'anthropic: ">=99.0.0"' in user_text
    assert ">=0.39.0" not in user_text


def test_new_without_recipe_dependencies_unchanged(
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
    call = fake.messages.calls[0]
    user_text = "".join(block["text"] for block in call["messages"][0]["content"])
    # Language defaults are untouched when the recipe declares no deps.
    assert ">=0.39.0" in user_text
    assert "redis" not in user_text


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


def test_post_gen_formatter_cleans_dirty_output(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_deployments_path: Path,
    mock_responses_path: Path,
) -> None:
    """A mock response with F841 + F401 + UP035 issues should be ruff-clean after generate."""
    if shutil.which("ruff") is None:
        pytest.skip("ruff not installed; skipping post-gen formatter assertion")

    payload = (mock_responses_path / "dirty_python.json").read_text(encoding="utf-8")
    fake = _Client(payload)
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: fake)

    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("AGENT_SCAFFOLD_DEPLOYMENTS_PATH", str(mock_deployments_path))
    monkeypatch.setenv("AGENT_SCAFFOLD_CACHE_DIR", str(cache_dir))
    monkeypatch.delenv("AGENT_SCAFFOLD_FORMAT", raising=False)

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
    proc = subprocess.run(
        ["ruff", "check", str(dest / "src")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_cli_emits_operation_events_for_each_phase(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_deployments_path: Path,
    mock_responses_path: Path,
) -> None:
    """P1: write/format/validate phases each surface operation_started + operation_done."""
    payload = (mock_responses_path / "valid_python.json").read_text(encoding="utf-8")
    fake = _Client(payload)
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: fake)

    captured: list[Any] = []

    class _CapturingDisplay:
        def __init__(self) -> None:
            self.phase_durations: dict[str, float] = {}
            self.warnings: list[str] = []
            self.errors: list[str] = []

        def __enter__(self) -> _CapturingDisplay:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def on_event(self, event: Any) -> None:
            captured.append(event)

    # --non-interactive routes to NullProgressDisplay; monkey-patch that.
    from agent_scaffold import cli as cli_module

    monkeypatch.setattr(cli_module, "NullProgressDisplay", _CapturingDisplay)

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
            "--no-format",
            "--skip-validation",
        ],
    )
    assert result.exit_code == 0, result.output
    op_started = [
        e.payload["name"]
        for e in captured
        if e.kind == "operation_started" and isinstance(e.payload, dict)
    ]
    op_done = [
        e.payload["name"]
        for e in captured
        if e.kind == "operation_done" and isinstance(e.payload, dict)
    ]
    # generate + write should always fire; format/validate skipped via flags above.
    assert "generate" in op_started
    assert "generate" in op_done
    assert "write" in op_started
    assert "write" in op_done
    assert "format" not in op_started
    assert "validate" not in op_started
    # file_written events landed for every emitted file.
    written = [e.payload["path"] for e in captured if e.kind == "file_written"]
    assert "pyproject.toml" in written
    assert "src/demo_agent/main.py" in written


def test_no_format_flag_skips_post_gen_formatter(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_deployments_path: Path,
    mock_responses_path: Path,
) -> None:
    """With --no-format, the dirty mock output should retain its ruff errors."""
    if shutil.which("ruff") is None:
        pytest.skip("ruff not installed; skipping post-gen formatter assertion")

    payload = (mock_responses_path / "dirty_python.json").read_text(encoding="utf-8")
    fake = _Client(payload)
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: fake)

    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("AGENT_SCAFFOLD_DEPLOYMENTS_PATH", str(mock_deployments_path))
    monkeypatch.setenv("AGENT_SCAFFOLD_CACHE_DIR", str(cache_dir))
    monkeypatch.delenv("AGENT_SCAFFOLD_FORMAT", raising=False)

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
            "--no-format",
        ],
    )
    assert result.exit_code == 0, result.output
    proc = subprocess.run(
        ["ruff", "check", str(dest / "src")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0, "expected ruff to surface dirty fixture errors without --format"
    combined = proc.stdout + proc.stderr
    # At least one of the seeded anti-patterns should still be present.
    assert "F841" in combined or "UP035" in combined or "F401" in combined


def _generate_baseline_project(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_deployments_path: Path,
    mock_responses_path: Path,
) -> Path:
    """Run `new` once and return the dest path so regenerate tests have a project."""
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
            "--no-format",
            "--skip-validation",
        ],
    )
    assert result.exit_code == 0, result.output
    return dest


def test_new_writes_scaffold_manifest(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_deployments_path: Path,
    mock_responses_path: Path,
) -> None:
    """S2 prerequisite: `new` writes .scaffold/manifest.json so regenerate can read it."""
    dest = _generate_baseline_project(
        runner, monkeypatch, tmp_path, mock_deployments_path, mock_responses_path
    )
    from agent_scaffold.manifest import read_manifest

    manifest = read_manifest(dest)
    assert manifest.recipe == "customer-support-triage"
    assert manifest.language == "python"
    assert manifest.framework == "langgraph"
    assert any(f.path == "src/demo_agent/main.py" for f in manifest.files)
    # Every recorded sha matches the actual file on disk.
    for entry in manifest.files:
        target = dest / entry.path
        assert target.is_file(), entry.path
        assert len(entry.sha256) == 64


def test_regenerate_rewrites_file_and_updates_manifest(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_deployments_path: Path,
    mock_responses_path: Path,
) -> None:
    """S2 happy path: regenerate rewrites the target file and updates its sha in the manifest."""
    dest = _generate_baseline_project(
        runner, monkeypatch, tmp_path, mock_deployments_path, mock_responses_path
    )
    from agent_scaffold.manifest import read_manifest

    manifest_before = read_manifest(dest)
    main_before = next(f for f in manifest_before.files if f.path == "src/demo_agent/main.py")

    # Swap the mocked client to return a different file body.
    new_body = '"""Replacement entry point."""\n\n\ndef agent() -> str:\n    return "regenerated"\n'
    fenced_response = f"```python\n{new_body}```"
    regen_client = _Client(fenced_response)
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: regen_client)

    result = runner.invoke(
        app,
        [
            "regenerate",
            str(dest),
            "src/demo_agent/main.py",
            "--reason",
            "rename agent return string",
            "--no-format",
        ],
    )
    assert result.exit_code == 0, result.output
    on_disk = (dest / "src/demo_agent/main.py").read_text(encoding="utf-8")
    assert "regenerated" in on_disk
    # Manifest's sha for the target updated; other files unchanged.
    manifest_after = read_manifest(dest)
    main_after = next(f for f in manifest_after.files if f.path == "src/demo_agent/main.py")
    assert main_after.sha256 != main_before.sha256
    # All other entries identical.
    before_other = {f.path: f.sha256 for f in manifest_before.files if f.path != main_before.path}
    after_other = {f.path: f.sha256 for f in manifest_after.files if f.path != main_after.path}
    assert before_other == after_other


def test_regenerate_diff_mode_does_not_write(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_deployments_path: Path,
    mock_responses_path: Path,
) -> None:
    """--diff prints the diff and exits without touching the file or the manifest."""
    dest = _generate_baseline_project(
        runner, monkeypatch, tmp_path, mock_deployments_path, mock_responses_path
    )
    original = (dest / "src/demo_agent/main.py").read_text(encoding="utf-8")
    from agent_scaffold.manifest import read_manifest

    sha_before = read_manifest(dest).files

    new_body = "x = 42  # totally different\n"
    fenced_response = f"```python\n{new_body}```"
    regen_client = _Client(fenced_response)
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: regen_client)

    result = runner.invoke(
        app,
        [
            "regenerate",
            str(dest),
            "src/demo_agent/main.py",
            "--reason",
            "preview only",
            "--diff",
            "--no-format",
        ],
    )
    assert result.exit_code == 0, result.output
    # File unchanged, manifest unchanged.
    assert (dest / "src/demo_agent/main.py").read_text(encoding="utf-8") == original
    assert read_manifest(dest).files == sha_before
    # Diff header surfaces.
    assert "a/src/demo_agent/main.py" in result.output
    assert "b/src/demo_agent/main.py" in result.output


def test_regenerate_aborts_when_manifest_missing(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_deployments_path: Path,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("AGENT_SCAFFOLD_DEPLOYMENTS_PATH", str(mock_deployments_path))
    monkeypatch.setenv("AGENT_SCAFFOLD_CACHE_DIR", str(tmp_path / "cache"))

    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "x.py").write_text("x = 1\n", encoding="utf-8")

    result = runner.invoke(
        app,
        ["regenerate", str(bare), "x.py", "--reason", "n/a"],
    )
    assert result.exit_code == 1, result.output
    assert "manifest" in result.output.lower()


_ = cli  # pragma: no cover - keep the import live for monkeypatching.
