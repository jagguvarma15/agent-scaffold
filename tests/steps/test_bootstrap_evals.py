"""Tests for ``agent_scaffold.steps.bootstrap_evals``."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from agent_scaffold.eval._common import EvalCase, EvalResult
from agent_scaffold.manifest import Manifest, write_manifest
from agent_scaffold.orchestrator import StepContext, StepStatus
from agent_scaffold.steps.bootstrap_evals import BootstrapEvalsStep

# ---------------------------------------------------------------------------
# Lightweight stand-ins for ResolvedStack / Capability
# ---------------------------------------------------------------------------


class _FakeCap:
    def __init__(self, cap_id: str) -> None:
        self.id = cap_id


class _FakeStack:
    def __init__(self, ids: list[str]) -> None:
        self.capabilities = [_FakeCap(i) for i in ids]


def _make_manifest(
    tmp_path: Path, *, baseline: str | None = None, capabilities: list[str] | None = None
) -> Manifest:
    answers = {"eval_baseline": baseline} if baseline is not None else {}
    manifest = Manifest(
        recipe="restaurant-rebooking",
        language="python",
        framework="langgraph",
        model="claude-test",
        generated_at="2026-05-30T00:00:00+00:00",
        answers=answers,
        capabilities=capabilities or [],
    )
    write_manifest(tmp_path, manifest)
    return manifest


def _seed_eval_config(project_dir: Path, content: str = "providers: []\n") -> None:
    evals = project_dir / "evals"
    evals.mkdir(exist_ok=True)
    (evals / "promptfooconfig.yaml").write_text(content, encoding="utf-8")
    (evals / "cases.yaml").write_text("- description: x\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# detect()
# ---------------------------------------------------------------------------


def test_detect_skipped_without_eval_capability(
    tmp_path: Path, ctx_factory: Callable[..., StepContext]
) -> None:
    ctx = ctx_factory(project_dir=tmp_path, resolved_stack=_FakeStack([]))
    result = BootstrapEvalsStep().detect(ctx)
    assert result.status is StepStatus.SKIPPED
    assert "eval" in result.reason.lower()


def test_detect_skipped_when_only_non_eval_capabilities(
    tmp_path: Path, ctx_factory: Callable[..., StepContext]
) -> None:
    ctx = ctx_factory(
        project_dir=tmp_path,
        resolved_stack=_FakeStack(["obs.langfuse", "vector_db.qdrant"]),
    )
    result = BootstrapEvalsStep().detect(ctx)
    assert result.status is StepStatus.SKIPPED


def test_detect_pending_when_baseline_unset(
    tmp_path: Path,
    ctx_factory: Callable[..., StepContext],
    manifest_factory: Callable[..., Manifest],
) -> None:
    manifest = manifest_factory()
    # manifest_factory's default answers={} → baseline unset.
    ctx = ctx_factory(
        project_dir=tmp_path,
        manifest=manifest,
        resolved_stack=_FakeStack(["eval.promptfoo"]),
    )
    result = BootstrapEvalsStep().detect(ctx)
    assert result.status is StepStatus.PENDING
    assert "eval.promptfoo" in result.reason


def test_detect_done_when_baseline_already_set(
    tmp_path: Path,
    ctx_factory: Callable[..., StepContext],
) -> None:
    manifest = _make_manifest(tmp_path, baseline="0.94")
    ctx = ctx_factory(
        project_dir=tmp_path,
        manifest=manifest,
        resolved_stack=_FakeStack(["eval.promptfoo"]),
    )
    result = BootstrapEvalsStep().detect(ctx)
    assert result.status is StepStatus.DONE


# ---------------------------------------------------------------------------
# apply()
# ---------------------------------------------------------------------------


def _patch_plugin(monkeypatch: pytest.MonkeyPatch, *, result: EvalResult) -> dict[str, Any]:
    """Patch ``get_plugin`` to return a stub whose ``run`` returns ``result``."""
    calls: dict[str, Any] = {"run_calls": 0, "baseline": None, "project_dir": None}

    class _StubPlugin:
        name = "promptfoo"
        cli_binary = "npx"
        install_hint = "n/a"
        config_file = "evals/promptfooconfig.yaml"

        def run(self, project_dir: Path, baseline_total: float | None) -> EvalResult:
            calls["run_calls"] += 1
            calls["project_dir"] = project_dir
            calls["baseline"] = baseline_total
            return result

    # Patch the lazy registry indirectly: replace get_plugin entirely.
    monkeypatch.setattr("agent_scaffold.eval.get_plugin", lambda _t: _StubPlugin())
    return calls


def test_apply_writes_baseline_to_manifest_on_success(
    tmp_path: Path,
    ctx_factory: Callable[..., StepContext],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _make_manifest(tmp_path, capabilities=["eval.promptfoo"])
    fake_result = EvalResult(
        target="promptfoo",
        cases=[EvalCase(name="x", score=0.94, passed=True)],
        total=0.94,
    )
    calls = _patch_plugin(monkeypatch, result=fake_result)

    ctx = ctx_factory(
        project_dir=tmp_path,
        manifest=manifest,
        resolved_stack=_FakeStack(["eval.promptfoo"]),
    )
    result = BootstrapEvalsStep().apply(ctx)

    assert result.status is StepStatus.DONE
    assert calls["run_calls"] == 1
    assert calls["baseline"] is None  # first run; no comparison

    # The baseline was persisted to .scaffold/manifest.json
    from agent_scaffold.manifest import read_manifest

    persisted = read_manifest(tmp_path)
    assert persisted.answers.get("eval_baseline") == "0.9400"


def test_apply_skipped_when_plugin_skipped(
    tmp_path: Path,
    ctx_factory: Callable[..., StepContext],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _make_manifest(tmp_path)
    fake_result = EvalResult(
        target="promptfoo",
        skipped=True,
        skip_reason="npx not on PATH",
    )
    _patch_plugin(monkeypatch, result=fake_result)

    ctx = ctx_factory(
        project_dir=tmp_path,
        manifest=manifest,
        resolved_stack=_FakeStack(["eval.promptfoo"]),
    )
    result = BootstrapEvalsStep().apply(ctx)
    assert result.status is StepStatus.SKIPPED
    assert "npx" in (result.detail or "")


def test_apply_failed_when_plugin_errors(
    tmp_path: Path,
    ctx_factory: Callable[..., StepContext],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _make_manifest(tmp_path)
    fake_result = EvalResult(
        target="promptfoo",
        error="last-run.json not written",
    )
    _patch_plugin(monkeypatch, result=fake_result)

    ctx = ctx_factory(
        project_dir=tmp_path,
        manifest=manifest,
        resolved_stack=_FakeStack(["eval.promptfoo"]),
    )
    result = BootstrapEvalsStep().apply(ctx)
    assert result.status is StepStatus.FAILED
    assert "last-run.json" in (result.error or "")


def test_apply_failed_when_zero_cases_returned(
    tmp_path: Path,
    ctx_factory: Callable[..., StepContext],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty cases list usually means the runner crashed mid-way; surface it."""
    manifest = _make_manifest(tmp_path)
    _patch_plugin(monkeypatch, result=EvalResult(target="promptfoo", cases=[], total=0.0))

    ctx = ctx_factory(
        project_dir=tmp_path,
        manifest=manifest,
        resolved_stack=_FakeStack(["eval.promptfoo"]),
    )
    result = BootstrapEvalsStep().apply(ctx)
    assert result.status is StepStatus.FAILED


def test_apply_skipped_when_no_plugin_for_target(
    tmp_path: Path,
    ctx_factory: Callable[..., StepContext],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _make_manifest(tmp_path)

    def boom(_target: str) -> Any:
        raise KeyError(_target)

    monkeypatch.setattr("agent_scaffold.eval.get_plugin", boom)
    ctx = ctx_factory(
        project_dir=tmp_path,
        manifest=manifest,
        resolved_stack=_FakeStack(["eval.deepeval"]),  # unknown plugin
    )
    result = BootstrapEvalsStep().apply(ctx)
    assert result.status is StepStatus.SKIPPED


# ---------------------------------------------------------------------------
# fingerprint()
# ---------------------------------------------------------------------------


def test_fingerprint_changes_when_eval_config_changes(
    tmp_path: Path,
    ctx_factory: Callable[..., StepContext],
) -> None:
    _seed_eval_config(tmp_path, content="providers: [a]\n")
    ctx = ctx_factory(
        project_dir=tmp_path,
        resolved_stack=_FakeStack(["eval.promptfoo"]),
    )
    step = BootstrapEvalsStep()
    fp_a = step.fingerprint(ctx)

    (tmp_path / "evals" / "promptfooconfig.yaml").write_text("providers: [b]\n", encoding="utf-8")
    fp_b = step.fingerprint(ctx)
    assert fp_a != fp_b


def test_fingerprint_stable_when_inputs_unchanged(
    tmp_path: Path,
    ctx_factory: Callable[..., StepContext],
) -> None:
    _seed_eval_config(tmp_path)
    ctx = ctx_factory(
        project_dir=tmp_path,
        resolved_stack=_FakeStack(["eval.promptfoo"]),
    )
    step = BootstrapEvalsStep()
    assert step.fingerprint(ctx) == step.fingerprint(ctx)


def test_fingerprint_when_no_config_present(
    tmp_path: Path,
    ctx_factory: Callable[..., StepContext],
) -> None:
    """Fingerprint should still be deterministic when no eval files exist yet."""
    ctx = ctx_factory(
        project_dir=tmp_path,
        resolved_stack=_FakeStack(["eval.promptfoo"]),
    )
    step = BootstrapEvalsStep()
    fp = step.fingerprint(ctx)
    assert fp.startswith("sha256:")


# ---------------------------------------------------------------------------
# Manifest round-trip
# ---------------------------------------------------------------------------


def test_manifest_baseline_round_trip(tmp_path: Path) -> None:
    """update_manifest_answer should persist + read back the baseline."""
    from agent_scaffold.manifest import read_manifest, update_manifest_answer

    _make_manifest(tmp_path)
    update_manifest_answer(tmp_path, "eval_baseline", 0.875)
    m = read_manifest(tmp_path)
    assert m.answers.get("eval_baseline") == "0.875"


# ---------------------------------------------------------------------------
# Registered in ALL_STEP_CLASSES + default plan
# ---------------------------------------------------------------------------


def test_bootstrap_evals_in_all_step_classes() -> None:
    from agent_scaffold.steps import ALL_STEP_CLASSES
    from agent_scaffold.steps import BootstrapEvalsStep as _Cls

    assert _Cls in ALL_STEP_CLASSES


def test_bootstrap_evals_in_default_plan_after_smoke_test(
    tmp_path: Path, manifest_factory: Callable[..., Manifest]
) -> None:
    from agent_scaffold.steps import default_steps_for

    steps = default_steps_for(manifest_factory(), None)
    step_ids = [s.id for s in steps]
    assert "bootstrap_evals" in step_ids
    assert step_ids.index("bootstrap_evals") > step_ids.index("smoke_test")
    assert step_ids.index("bootstrap_evals") < step_ids.index("emit_deploy_configs")
