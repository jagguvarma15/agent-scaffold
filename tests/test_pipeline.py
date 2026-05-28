"""Tests for ``agent_scaffold.pipeline`` — the post-plan orchestration.

cmd_new used to inline the same body; this module locks in the contract that
the lifted function is callable directly (so the upcoming REPL can use it)
and that recoverable failures raise :class:`PipelineError` with a phase
label instead of dumping a stack trace or calling ``sys.exit``.
"""

from __future__ import annotations

import importlib.resources as resources
from pathlib import Path
from typing import Any

import pytest
import yaml

from agent_scaffold import generator
from agent_scaffold.config import load_config
from agent_scaffold.context import assemble
from agent_scaffold.discovery import discover_recipes
from agent_scaffold.pipeline import (
    PipelineError,
    PipelineInputs,
    RunReport,
    run_generation,
)
from agent_scaffold.progress import NullProgressDisplay
from agent_scaffold.topology import Topology
from agent_scaffold.writer import WriteMode


def _load_python_hints() -> dict:
    """Load the real python.yaml language hints — the lighter test stub
    didn't carry the `manifest` key that contract validation requires."""
    text = (
        resources.files("agent_scaffold.languages")
        .joinpath("python.yaml")
        .read_text(encoding="utf-8")
    )
    return yaml.safe_load(text)


# ---------------------------------------------------------------------------
# Fake Anthropic client (mirrors tests/test_cli_e2e.py — keep in sync)
# ---------------------------------------------------------------------------


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

    def stream(self, **kwargs: Any) -> _StreamCtx:
        return _StreamCtx(_Response(self._payload))


class _Client:
    def __init__(self, payload: str) -> None:
        self.messages = _Messages(payload)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_inputs(
    tmp_path: Path,
    mock_deployments_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    no_cache: bool = True,
) -> PipelineInputs:
    """Assemble a real PipelineInputs against the mock_deployments fixture."""
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("AGENT_SCAFFOLD_DEPLOYMENTS_PATH", str(mock_deployments_path))
    monkeypatch.setenv("AGENT_SCAFFOLD_CACHE_DIR", str(cache_dir))

    cfg = load_config()
    recipes = discover_recipes(mock_deployments_path)
    recipe = next(r for r in recipes if r.slug == "customer-support-triage")
    ctx = assemble(recipe, "python", "langgraph", mock_deployments_path)
    return PipelineInputs(
        cfg=cfg,
        recipe=recipe,
        language="python",
        framework="langgraph",
        project_name="demo_agent",
        raw_project_name="demo-agent",
        dest=tmp_path / "out" / "demo_agent",
        deployments=mock_deployments_path,
        ctx=ctx,
        hints=_load_python_hints(),
        topology=Topology.SINGLE,
        roles=[],
        write_mode=WriteMode.abort,
        strict=False,
        format_output=False,
        skip_validation=True,
        no_cache=no_cache,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_run_generation_writes_files_and_returns_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_deployments_path: Path,
    mock_responses_path: Path,
) -> None:
    payload = (mock_responses_path / "valid_python.json").read_text(encoding="utf-8")
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: _Client(payload))
    inputs = _build_inputs(tmp_path, mock_deployments_path, monkeypatch)

    report = run_generation(inputs, display=NullProgressDisplay())

    assert isinstance(report, RunReport)
    assert report.result is not None
    assert report.report is not None
    assert len(report.report.written) > 0
    assert report.cached is False
    assert (inputs.dest / ".scaffold" / "manifest.json").exists()


# ---------------------------------------------------------------------------
# Failure path — PipelineError carries the phase + hint
# ---------------------------------------------------------------------------


def test_run_generation_raises_pipeline_error_when_write_collides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_deployments_path: Path,
    mock_responses_path: Path,
) -> None:
    """Write into a non-empty dest with write_mode=abort → PipelineError(phase=write)."""
    payload = (mock_responses_path / "valid_python.json").read_text(encoding="utf-8")
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: _Client(payload))
    inputs = _build_inputs(tmp_path, mock_deployments_path, monkeypatch)
    # Pre-create the destination with a colliding file so write_project aborts.
    inputs.dest.mkdir(parents=True)
    (inputs.dest / "Dockerfile").write_text("pre-existing\n", encoding="utf-8")

    with pytest.raises(PipelineError) as excinfo:
        run_generation(inputs, display=NullProgressDisplay())

    assert excinfo.value.phase == "write"
    assert excinfo.value.message  # non-empty message


def test_pipeline_error_preserves_message_and_phase() -> None:
    err = PipelineError("boom", phase="verify", hint="try --write-mode overwrite")
    assert str(err) == "boom"
    assert err.message == "boom"
    assert err.phase == "verify"
    assert err.hint == "try --write-mode overwrite"
