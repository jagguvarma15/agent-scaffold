"""Tests for ``agent_scaffold.pipeline`` — the post-plan orchestration.

cmd_new used to inline the same body; this module locks in the contract that
the lifted function is callable directly (so the upcoming REPL can use it)
and that recoverable failures raise :class:`PipelineError` with a phase
label instead of dumping a stack trace or calling ``sys.exit``.
"""

from __future__ import annotations

import importlib.resources as resources
from functools import partial
from pathlib import Path
from typing import Any

import pytest
import yaml

from agent_scaffold import generator
from agent_scaffold.catalog import Catalog
from agent_scaffold.config import load_config
from agent_scaffold.context import assemble as _real_assemble
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

# Pre-bind a test catalog so each assemble() call stays unchanged. Catalog
# became a required kwarg in vX+1.
_TEST_CATALOG_PATH = Path(__file__).parent / "fixtures" / "catalog_minimal.yaml"
_TEST_CATALOG: Catalog = Catalog.model_validate(
    yaml.safe_load(_TEST_CATALOG_PATH.read_text(encoding="utf-8"))
)
assemble = partial(_real_assemble, catalog=_TEST_CATALOG)


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
        self.calls: list[dict[str, Any]] = []

    def stream(self, **kwargs: Any) -> _StreamCtx:
        self.calls.append(kwargs)
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
    # The entry-point + smoke contract is persisted (the SoT run reads back).
    from agent_scaffold.manifest import read_manifest

    manifest = read_manifest(inputs.dest)
    assert manifest.entry_point == "src/demo_agent/main.py"
    assert manifest.smoke_check is not None
    assert "from demo_agent.main import agent" in manifest.smoke_check


def test_run_generation_persists_app_layout_entry_point(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_deployments_path: Path,
    mock_responses_path: Path,
) -> None:
    """A recipe whose required_files names ``app/main.py`` persists
    ``manifest.entry_point == 'app/main.py'`` end-to-end — the app-layout
    override path (the production layout), not the src default."""
    from agent_scaffold.manifest import read_manifest

    # An app-layout response: rename the generated src package to app/.
    payload = (mock_responses_path / "valid_python.json").read_text(encoding="utf-8")
    payload = payload.replace("src/demo_agent/", "app/")
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: _Client(payload))

    base = _build_inputs(tmp_path, mock_deployments_path, monkeypatch)
    recipe = base.recipe.model_copy(update={"required_files": ["app/main.py"]})
    inputs = PipelineInputs(
        **{**{k: getattr(base, k) for k in base.__dataclass_fields__}, "recipe": recipe}
    )
    run_generation(inputs, display=NullProgressDisplay())

    manifest = read_manifest(inputs.dest)
    assert manifest.entry_point == "app/main.py"
    assert manifest.smoke_check is not None
    assert "from app.main import agent" in manifest.smoke_check


def test_run_generation_resets_runtime_step_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_deployments_path: Path,
    mock_responses_path: Path,
) -> None:
    """Regenerating over a provisioned destination resets the runtime steps.

    DONE markers for docker_up / launch_* / smoke_test describe the previous
    project once the files are rewritten — trusting them lets containers
    built from the old code keep serving. install_deps is content-
    fingerprinted and stays untouched."""
    from agent_scaffold.orchestrator import (
        OrchestratorState,
        StepState,
        StepStatus,
        read_state,
        write_state,
    )

    payload = (mock_responses_path / "valid_python.json").read_text(encoding="utf-8")
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: _Client(payload))
    base = _build_inputs(tmp_path, mock_deployments_path, monkeypatch)
    inputs = PipelineInputs(
        **{
            **{k: getattr(base, k) for k in base.__dataclass_fields__},
            "write_mode": WriteMode.overwrite,
        }
    )
    inputs.dest.mkdir(parents=True)
    write_state(
        inputs.dest,
        OrchestratorState(
            steps={
                "docker_up": StepState(status=StepStatus.DONE, fingerprint="sha256:old"),
                "launch_backend": StepState(status=StepStatus.DONE),
                "install_deps": StepState(status=StepStatus.DONE, fingerprint="sha256:deps"),
            }
        ),
    )

    run_generation(inputs, display=NullProgressDisplay())

    state = read_state(inputs.dest)
    assert state.steps["docker_up"].status == StepStatus.PENDING
    assert state.steps["launch_backend"].status == StepStatus.PENDING
    assert state.steps["install_deps"].status == StepStatus.DONE
    assert state.steps["install_deps"].fingerprint == "sha256:deps"


def test_fresh_generation_creates_no_state_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_deployments_path: Path,
    mock_responses_path: Path,
) -> None:
    """The runtime-state reset is a no-op on a fresh destination."""
    payload = (mock_responses_path / "valid_python.json").read_text(encoding="utf-8")
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: _Client(payload))
    inputs = _build_inputs(tmp_path, mock_deployments_path, monkeypatch)

    run_generation(inputs, display=NullProgressDisplay())

    assert not (inputs.dest / ".scaffold" / "state.json").exists()


def test_manifest_files_include_capability_template_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_deployments_path: Path,
    mock_responses_path: Path,
) -> None:
    """Capability template files (frontend/, evals/, ...) belong to the
    project and must appear in manifest.files alongside model output —
    otherwise update/regenerate can't see them."""
    from agent_scaffold.capabilities import load_capabilities, resolve
    from agent_scaffold.manifest import read_manifest

    payload = (mock_responses_path / "valid_python.json").read_text(encoding="utf-8")
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: _Client(payload))
    base = _build_inputs(tmp_path, mock_deployments_path, monkeypatch)
    stack = resolve(
        base.recipe,
        load_capabilities(mock_deployments_path),
        add_capabilities=["frontend.nextjs-tiny"],
    )
    inputs = PipelineInputs(
        **{
            **{k: getattr(base, k) for k in base.__dataclass_fields__},
            "resolved_stack": stack,
        }
    )

    run_generation(inputs, display=NullProgressDisplay())

    manifest = read_manifest(inputs.dest)
    paths = {f.path for f in manifest.files}
    assert "frontend/package.json" in paths
    assert any(p.startswith("src/demo_agent/") for p in paths)


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


def test_run_generation_threads_refinements_into_llm_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_deployments_path: Path,
    mock_responses_path: Path,
) -> None:
    """End-to-end: refinement fields on PipelineInputs must appear in the
    LLM user message. Without this the REPL's refinement feature is a
    silent no-op — the LLM never sees what the user asked for.
    """
    payload = (mock_responses_path / "valid_python.json").read_text(encoding="utf-8")
    client = _Client(payload)
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: client)

    base = _build_inputs(tmp_path, mock_deployments_path, monkeypatch)
    inputs = PipelineInputs(
        **{
            **{k: getattr(base, k) for k in base.__dataclass_fields__},
            "extra_dependencies": {"python": {"psycopg": "^3.2"}},
            "extra_steps": ["wire prometheus exporter"],
            "removed_steps": {"docker_up"},
            "removed_roles": {"evaluator"},
            "refinement_notes": ["Prefer async/await throughout."],
        }
    )

    run_generation(inputs, display=NullProgressDisplay())

    assert client.messages.calls, "expected the Anthropic client to be called"
    user_content = client.messages.calls[0]["messages"][0]["content"]
    rendered = "".join(
        block["text"] for block in user_content if isinstance(block.get("text"), str)
    )
    assert "# User refinements" in rendered
    assert "psycopg" in rendered and "^3.2" in rendered
    assert "wire prometheus exporter" in rendered
    assert "docker_up" in rendered
    assert "evaluator" in rendered
    assert "Prefer async/await throughout." in rendered


def test_pipeline_error_preserves_message_and_phase() -> None:
    err = PipelineError("boom", phase="verify", hint="try --write-mode overwrite")
    assert str(err) == "boom"
    assert err.message == "boom"
    assert err.phase == "verify"
    assert err.hint == "try --write-mode overwrite"


def test_format_contract_failure_includes_tier_and_field() -> None:
    """The pipeline's warning + error messages should expose the structured
    tier (and optional field) rather than only the prose reason."""
    from agent_scaffold.contract import ContractParseError
    from agent_scaffold.pipeline import _format_contract_failure

    label = _format_contract_failure(
        ContractParseError(
            raw="(files)",
            reason="missing required manifest file: pyproject.toml",
            tier="required-files",
            field="pyproject.toml",
        )
    )
    assert "tier=required-files" in label
    assert "pyproject.toml" in label


def test_format_contract_failure_omits_field_when_absent() -> None:
    from agent_scaffold.contract import ContractParseError
    from agent_scaffold.pipeline import _format_contract_failure

    label = _format_contract_failure(
        ContractParseError(raw="x", reason="invalid JSON: ...", tier="json")
    )
    assert label == "tier=json"


# --- --deep-validate tier selection -----------------------------------------


def test_validation_tiers_default_excludes_docker_and_smoke(
    tmp_path: Path, mock_deployments_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_scaffold.pipeline import _validation_tiers
    from agent_scaffold.validator import ValidationTier

    inputs = _build_inputs(tmp_path, mock_deployments_path, monkeypatch)
    assert inputs.deep_validate is False
    tiers = _validation_tiers(inputs)
    assert tiers == [ValidationTier.static, ValidationTier.build, ValidationTier.compile]
    assert ValidationTier.docker_up not in tiers
    assert ValidationTier.smoke not in tiers


def test_validation_tiers_deep_includes_docker_and_smoke(
    tmp_path: Path, mock_deployments_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    from agent_scaffold.pipeline import _validation_tiers
    from agent_scaffold.validator import ValidationTier

    base = _build_inputs(tmp_path, mock_deployments_path, monkeypatch)
    deep = replace(base, deep_validate=True)
    tiers = _validation_tiers(deep)
    # Fast tiers still come first (cheapest → most expensive); runtime tiers appended.
    assert tiers[:3] == [ValidationTier.static, ValidationTier.build, ValidationTier.compile]
    assert tiers[3:] == [ValidationTier.docker_up, ValidationTier.smoke]


# ---------------------------------------------------------------------------
# Merge-into-existing-path round-trip — full run_generation, not the helpers
# ---------------------------------------------------------------------------


def _first_python_file(payload: str) -> str:
    import json

    data = json.loads(payload)
    return next(f["path"] for f in data["files"] if f["path"].endswith(".py"))


def _payload_with_edit(payload: str, target: str, transform: Any) -> str:
    """Return ``payload`` with ``target``'s content replaced by ``transform(content)``."""
    import json

    data = json.loads(payload)
    for f in data["files"]:
        if f["path"] == target:
            f["content"] = transform(f["content"])
    return json.dumps(data)


def test_merge_round_trip_preserves_user_edit_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_deployments_path: Path,
    mock_responses_path: Path,
) -> None:
    """Generate → snapshot base → user edits a file → regenerate in merge mode.

    Drives the real pipeline twice (model call stubbed). A non-overlapping
    template change and the user's edit must BOTH survive the 3-way merge.
    """
    from dataclasses import replace

    from agent_scaffold.manifest import read_manifest

    payload1 = (mock_responses_path / "valid_python.json").read_text(encoding="utf-8")
    target = _first_python_file(payload1)

    # 1) First generation into an empty dest → writes files + manifest + snapshot.
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: _Client(payload1))
    inputs1 = _build_inputs(tmp_path, mock_deployments_path, monkeypatch)
    run_generation(inputs1, display=NullProgressDisplay())
    dest = inputs1.dest
    assert list(
        (dest / ".scaffold" / "template-snapshots").glob("*.tgz")
    ), "first generation must write a snapshot base for a later merge"

    # 2) The user appends a distinctive line to the file on disk (their edit).
    target_file = dest / target
    target_file.write_text(
        target_file.read_text(encoding="utf-8") + "\n# USER-EDIT-KEEP-ME\n", encoding="utf-8"
    )

    # 3) Regenerate: the "template" changed a DIFFERENT region (a prepended header).
    payload2 = _payload_with_edit(payload1, target, lambda c: "# TEMPLATE-CHANGE-HEADER\n" + c)
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: _Client(payload2))
    run_generation(replace(inputs1, write_mode=WriteMode.merge), display=NullProgressDisplay())

    merged = target_file.read_text(encoding="utf-8")
    assert "# USER-EDIT-KEEP-ME" in merged, "merge dropped the user's edit"
    assert "# TEMPLATE-CHANGE-HEADER" in merged, "merge skipped the template change"
    # Clean merge → no leftover resume point, manifest still intact.
    assert not (dest / ".scaffold" / "update.in-progress.json").exists()
    assert read_manifest(dest).recipe == inputs1.recipe.slug


def test_merge_round_trip_conflict_leaves_markers_and_resume_point(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_deployments_path: Path,
    mock_responses_path: Path,
) -> None:
    """When the user and the template both rewrite the SAME first line, the
    merge writes conflict markers, drops a resume point, and surfaces a
    PipelineError pointing at `update --continue` — all through run_generation."""
    from dataclasses import replace

    payload1 = (mock_responses_path / "valid_python.json").read_text(encoding="utf-8")
    target = _first_python_file(payload1)

    monkeypatch.setattr(generator, "_make_client", lambda _cfg: _Client(payload1))
    inputs1 = _build_inputs(tmp_path, mock_deployments_path, monkeypatch)
    run_generation(inputs1, display=NullProgressDisplay())
    dest = inputs1.dest

    # User rewrites the first line on disk...
    target_file = dest / target
    ours = target_file.read_text(encoding="utf-8").split("\n")
    ours[0] = "# OURS-FIRST-LINE"
    target_file.write_text("\n".join(ours), encoding="utf-8")

    # ...and the template rewrites the SAME first line → an overlapping conflict.
    def _rewrite_first(content: str) -> str:
        lines = content.split("\n")
        lines[0] = "# THEIRS-FIRST-LINE"
        return "\n".join(lines)

    payload2 = _payload_with_edit(payload1, target, _rewrite_first)
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: _Client(payload2))

    with pytest.raises(PipelineError, match="conflict"):
        run_generation(replace(inputs1, write_mode=WriteMode.merge), display=NullProgressDisplay())

    merged = target_file.read_text(encoding="utf-8")
    assert "<<<<<<<" in merged and ">>>>>>>" in merged, "expected git-style conflict markers"
    assert (dest / ".scaffold" / "update.in-progress.json").is_file(), "expected a resume point"
