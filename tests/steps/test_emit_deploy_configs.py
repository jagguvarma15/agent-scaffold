"""Tests for ``agent_scaffold.steps.emit_deploy_configs``."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from agent_scaffold.capabilities import Capability, EmitFile, ResolvedStack
from agent_scaffold.orchestrator import StepContext, StepStatus
from agent_scaffold.steps.emit_deploy_configs import EmitDeployConfigsStep


def _vercel_capability(deployments: Path) -> Capability:
    cap_dir = deployments / "host"
    cap_dir.mkdir(parents=True, exist_ok=True)
    cap_file = cap_dir / "vercel.md"
    cap_file.write_text("# vercel", encoding="utf-8")
    template = cap_dir / "templates" / "vercel.json"
    template.parent.mkdir(parents=True, exist_ok=True)
    template.write_text(
        '{"name": "${NEXT_PUBLIC_AGENT_NAME}", "version": 2}',
        encoding="utf-8",
    )
    return Capability(
        id="host.vercel",
        kind="host",
        path=cap_file,
        emit_files=[EmitFile(source="templates/vercel.json", dest="vercel.json")],
    )


def test_detect_skipped_without_host_capability(
    ctx_factory: Callable[..., StepContext],
) -> None:
    result = EmitDeployConfigsStep().detect(ctx_factory())
    assert result.status is StepStatus.SKIPPED


def test_apply_writes_template_with_env_substitution(
    ctx_factory: Callable[..., StepContext],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    deployments = tmp_path / "deployments"
    cap = _vercel_capability(deployments)
    monkeypatch.setenv("NEXT_PUBLIC_AGENT_NAME", "rebooking-agent")
    stack = ResolvedStack(capabilities=[cap])
    result = EmitDeployConfigsStep().apply(
        ctx_factory(resolved_stack=stack, project_dir=project_dir)
    )
    assert result.status is StepStatus.DONE
    rendered = (project_dir / "vercel.json").read_text(encoding="utf-8")
    assert '"name": "rebooking-agent"' in rendered


def test_apply_leaves_unresolved_placeholder_in_template(
    ctx_factory: Callable[..., StepContext],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    cap = _vercel_capability(tmp_path / "deployments")
    monkeypatch.delenv("NEXT_PUBLIC_AGENT_NAME", raising=False)
    stack = ResolvedStack(capabilities=[cap])
    EmitDeployConfigsStep().apply(ctx_factory(resolved_stack=stack, project_dir=project_dir))
    rendered = (project_dir / "vercel.json").read_text(encoding="utf-8")
    assert "${NEXT_PUBLIC_AGENT_NAME}" in rendered  # placeholder preserved


def test_apply_skips_when_dest_exists(
    ctx_factory: Callable[..., StepContext],
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    cap = _vercel_capability(tmp_path / "deployments")
    (project_dir / "vercel.json").write_text('{"model output": true}', encoding="utf-8")
    stack = ResolvedStack(capabilities=[cap])
    result = EmitDeployConfigsStep().apply(
        ctx_factory(resolved_stack=stack, project_dir=project_dir)
    )
    # Step still DONE (no error), but file wasn't overwritten.
    assert result.status is StepStatus.DONE
    body = (project_dir / "vercel.json").read_text(encoding="utf-8")
    assert "model output" in body
    assert "skipped" in result.detail


def test_apply_skips_when_source_missing(
    ctx_factory: Callable[..., StepContext], tmp_path: Path
) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    cap = Capability(
        id="host.vercel",
        kind="host",
        path=tmp_path / "host" / "vercel.md",
        emit_files=[EmitFile(source="templates/missing.json", dest="vercel.json")],
    )
    cap.path.parent.mkdir(parents=True, exist_ok=True)
    cap.path.write_text("# x", encoding="utf-8")
    stack = ResolvedStack(capabilities=[cap])
    result = EmitDeployConfigsStep().apply(
        ctx_factory(resolved_stack=stack, project_dir=project_dir)
    )
    assert result.status is StepStatus.DONE
    assert not (project_dir / "vercel.json").exists()
    assert "skipped 1" in result.detail


def test_fingerprint_stable_for_same_inputs(
    ctx_factory: Callable[..., StepContext], tmp_path: Path
) -> None:
    cap = _vercel_capability(tmp_path / "deployments")
    stack = ResolvedStack(capabilities=[cap])
    step = EmitDeployConfigsStep()
    a = step.fingerprint(ctx_factory(resolved_stack=stack, project_dir=tmp_path))
    b = step.fingerprint(ctx_factory(resolved_stack=stack, project_dir=tmp_path))
    assert a == b
