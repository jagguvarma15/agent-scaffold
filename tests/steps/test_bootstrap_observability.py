"""Tests for ``agent_scaffold.steps.bootstrap_observability``."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from agent_scaffold.capabilities import Capability, ResolvedStack
from agent_scaffold.orchestrator import StepContext, StepState, StepStatus
from agent_scaffold.steps import bootstrap_observability as bo
from agent_scaffold.steps.bootstrap_observability import BootstrapObservabilityStep


def _cap(tmp_path: Path) -> Capability:
    return Capability(
        id="obs.grafana-stack",
        kind="obs",
        path=tmp_path / "grafana.md",
    )


def _stack(tmp_path: Path) -> ResolvedStack:
    return ResolvedStack(capabilities=[_cap(tmp_path)])


def _ctx_with_docker_up_done(
    ctx_factory: Callable[..., StepContext], tmp_path: Path
) -> StepContext:
    """Build a ctx whose ``docker_up`` step is recorded as DONE so the
    bootstrap_observability dependency guard doesn't short-circuit."""
    ctx = ctx_factory(resolved_stack=_stack(tmp_path), project_dir=tmp_path)
    ctx.state.steps["docker_up"] = StepState(status=StepStatus.DONE)
    return ctx


def test_detect_skipped_without_capability(
    ctx_factory: Callable[..., StepContext],
) -> None:
    result = BootstrapObservabilityStep().detect(ctx_factory())
    assert result.status is StepStatus.SKIPPED


def test_apply_provisions_datasources_and_dashboards(
    ctx_factory: Callable[..., StepContext],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Drop a dashboard JSON the step should upload.
    dashboards_dir = tmp_path / "ops" / "grafana" / "dashboards"
    dashboards_dir.mkdir(parents=True)
    (dashboards_dir / "agent.json").write_text(json.dumps({"uid": "agent"}), encoding="utf-8")

    posts: list[tuple[str, str, dict[str, Any]]] = []  # (method, url, payload)

    def fake_http_request(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None = None,
        timeout: float = 10.0,
    ) -> tuple[int, bytes]:
        if url.endswith("/api/health"):
            return 200, json.dumps({"database": "ok"}).encode()
        payload = json.loads(body) if body else {}
        posts.append((method, url, payload))
        return 200, b'{"ok": true}'

    monkeypatch.setattr(bo, "_http_request", fake_http_request)
    monkeypatch.setenv("GRAFANA_URL", "http://localhost:3002")
    monkeypatch.setenv("GRAFANA_ADMIN_PASSWORD", "admin")

    result = BootstrapObservabilityStep().apply(
        _ctx_with_docker_up_done(ctx_factory, tmp_path)
    )
    assert result.status is StepStatus.DONE
    # 2 datasources + 1 dashboard
    posted_urls = [url for _m, url, _p in posts]
    assert any("datasources" in url for url in posted_urls)
    assert any("dashboards/db" in url for url in posted_urls)
    assert "datasources: 2 added" in result.detail
    assert "1 added" in result.detail.split("dashboards:")[1]


def test_apply_failed_when_health_never_returns(
    ctx_factory: Callable[..., StepContext],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bo, "_wait_for_health", lambda *a, **kw: False)
    result = BootstrapObservabilityStep().apply(
        _ctx_with_docker_up_done(ctx_factory, tmp_path)
    )
    assert result.status is StepStatus.FAILED
    assert "/api/health" in (result.error or "")


def test_datasources_skipped_on_409(
    ctx_factory: Callable[..., StepContext],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bo, "_wait_for_health", lambda *a, **kw: True)

    def fake_http_request(
        method: str, url: str, headers: dict[str, str], **kw: Any
    ) -> tuple[int, bytes]:
        if "/api/datasources" in url:
            return 409, b"data source with the same name already exists"
        return 200, b"{}"

    monkeypatch.setattr(bo, "_http_request", fake_http_request)
    result = BootstrapObservabilityStep().apply(
        _ctx_with_docker_up_done(ctx_factory, tmp_path)
    )
    assert result.status is StepStatus.DONE
    assert "datasources: 0 added" in result.detail


def test_fingerprint_changes_with_dashboards(
    ctx_factory: Callable[..., StepContext], tmp_path: Path
) -> None:
    step = BootstrapObservabilityStep()
    a = step.fingerprint(ctx_factory(resolved_stack=_stack(tmp_path), project_dir=tmp_path))
    dashes = tmp_path / "ops" / "grafana" / "dashboards"
    dashes.mkdir(parents=True)
    (dashes / "x.json").write_text("{}", encoding="utf-8")
    b = step.fingerprint(ctx_factory(resolved_stack=_stack(tmp_path), project_dir=tmp_path))
    assert a != b


def test_detect_skipped_when_docker_up_skipped(
    ctx_factory: Callable[..., StepContext], tmp_path: Path
) -> None:
    """If docker_up was SKIPPED (no docker_service declared), Grafana never
    started — observability must skip too instead of polling a phantom port."""
    ctx = ctx_factory(resolved_stack=_stack(tmp_path), project_dir=tmp_path)
    ctx.state.steps["docker_up"] = StepState(status=StepStatus.SKIPPED)

    result = BootstrapObservabilityStep().detect(ctx)
    assert result.status is StepStatus.SKIPPED
    assert "docker_up didn't run" in result.reason


def test_apply_skipped_when_docker_up_skipped(
    ctx_factory: Callable[..., StepContext],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defense-in-depth: even if a caller bypasses detect() and calls apply()
    directly, the same guard catches it before _wait_for_health spins."""

    def fail_if_called(*_a: Any, **_kw: Any) -> bool:
        raise AssertionError("_wait_for_health must not be called when docker_up skipped")

    monkeypatch.setattr(bo, "_wait_for_health", fail_if_called)
    ctx = ctx_factory(resolved_stack=_stack(tmp_path), project_dir=tmp_path)
    ctx.state.steps["docker_up"] = StepState(status=StepStatus.SKIPPED)

    result = BootstrapObservabilityStep().apply(ctx)
    assert result.status is StepStatus.SKIPPED
    assert "docker_up didn't run" in (result.detail or "")
