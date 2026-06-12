"""Tests for the bounded validate→repair loop and its building blocks."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import pytest
import yaml

from agent_scaffold import generator, pipeline
from agent_scaffold.catalog import Catalog
from agent_scaffold.config import load_config
from agent_scaffold.context import assemble as _real_assemble
from agent_scaffold.contract import (
    ContractParseError,
    GeneratedFile,
    GenerationResult,
    parse_file_patch,
)
from agent_scaffold.discovery import discover_recipes
from agent_scaffold.generator import UsageInfo
from agent_scaffold.pipeline import (
    MAX_REPAIR_ROUNDS,
    PipelineError,
    PipelineInputs,
    _implicated_files,
    _merge_patch,
    run_generation,
)
from agent_scaffold.progress import NullProgressDisplay
from agent_scaffold.topology import Topology
from agent_scaffold.validator import ValidationResult, ValidationTier
from agent_scaffold.writer import WriteMode

_TEST_CATALOG_PATH = Path(__file__).parent / "fixtures" / "catalog_minimal.yaml"
_TEST_CATALOG: Catalog = Catalog.model_validate(
    yaml.safe_load(_TEST_CATALOG_PATH.read_text(encoding="utf-8"))
)
assemble = partial(_real_assemble, catalog=_TEST_CATALOG)


# ---------------------------------------------------------------------------
# parse_file_patch
# ---------------------------------------------------------------------------


def _patch_json(path: str, content: str = "x = 1\n") -> str:
    import json

    return json.dumps({"files": [{"path": path, "content": content}]})


def test_parse_file_patch_accepts_known_path(tmp_path: Path) -> None:
    files = parse_file_patch(
        _patch_json("src/app/main.py"),
        tmp_path,
        allowed_paths={"src/app/main.py"},
    )
    assert [f.path for f in files] == ["src/app/main.py"]


def test_parse_file_patch_accepts_new_file_in_existing_dir(tmp_path: Path) -> None:
    files = parse_file_patch(
        _patch_json("src/app/helper.py"),
        tmp_path,
        allowed_paths={"src/app/main.py"},
    )
    assert files[0].path == "src/app/helper.py"


def test_parse_file_patch_accepts_ancestor_dir_and_root(tmp_path: Path) -> None:
    # "src" is an ancestor of an allowed path; project root is always fine.
    for path in ("src/conftest.py", "ruff.toml"):
        files = parse_file_patch(
            _patch_json(path),
            tmp_path,
            allowed_paths={"src/app/main.py"},
        )
        assert files[0].path == path


def test_parse_file_patch_rejects_new_directory_tree(tmp_path: Path) -> None:
    with pytest.raises(ContractParseError) as excinfo:
        parse_file_patch(
            _patch_json("infra/terraform/main.tf"),
            tmp_path,
            allowed_paths={"src/app/main.py"},
        )
    assert excinfo.value.tier == "path"
    assert "infra/terraform/main.tf" in (excinfo.value.field or "")


def test_parse_file_patch_rejects_traversal(tmp_path: Path) -> None:
    with pytest.raises(ContractParseError) as excinfo:
        parse_file_patch(
            _patch_json("../evil.py"),
            tmp_path,
            allowed_paths={"src/app/main.py"},
        )
    assert excinfo.value.tier == "path"


def test_parse_file_patch_invalid_json_and_empty_files(tmp_path: Path) -> None:
    with pytest.raises(ContractParseError) as excinfo:
        parse_file_patch("not json at all", tmp_path, allowed_paths=set())
    assert excinfo.value.tier == "json"

    with pytest.raises(ContractParseError) as excinfo:
        parse_file_patch('{"files": []}', tmp_path, allowed_paths=set())
    assert excinfo.value.tier == "schema"


# ---------------------------------------------------------------------------
# _implicated_files + _merge_patch
# ---------------------------------------------------------------------------


def test_implicated_files_exact_and_suffix_match(tmp_path: Path) -> None:
    (tmp_path / "src" / "app").mkdir(parents=True)
    (tmp_path / "src" / "app" / "main.py").write_text("import os\n", encoding="utf-8")
    (tmp_path / "src" / "app" / "util.py").write_text("x = 1\n", encoding="utf-8")
    known = {"src/app/main.py", "src/app/util.py"}

    # Exact path (ruff style) + bare-filename suffix (tsc style).
    output = "src/app/main.py:1:8: F401 `os` imported but unused\nerror in util.py: TS2304"
    files = _implicated_files(output, tmp_path, known, [])

    assert set(files) == {"src/app/main.py", "src/app/util.py"}
    assert files["src/app/main.py"] == "import os\n"


def test_implicated_files_falls_back_to_required(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    files = _implicated_files(
        "resolver error: no solution found",  # no path in output
        tmp_path,
        {"pyproject.toml"},
        ["pyproject.toml", "missing.md"],
    )
    assert set(files) == {"pyproject.toml"}


def test_merge_patch_replaces_and_appends_in_order() -> None:
    result = GenerationResult(
        project_name="demo",
        language="python",
        smoke_check="true",
        files=[
            GeneratedFile(path="a.py", content="old a"),
            GeneratedFile(path="b.py", content="b"),
        ],
    )
    merged = _merge_patch(
        result,
        [
            GeneratedFile(path="a.py", content="new a"),
            GeneratedFile(path="c.py", content="c"),
        ],
    )
    assert [(f.path, f.content) for f in merged.files] == [
        ("a.py", "new a"),
        ("b.py", "b"),
        ("c.py", "c"),
    ]


# ---------------------------------------------------------------------------
# Run-cumulative usage
# ---------------------------------------------------------------------------


def test_run_usage_accumulates_across_calls() -> None:
    generator.reset_run_usage()
    generator._accumulate_run_usage(UsageInfo(input_tokens=10, output_tokens=5))
    generator._accumulate_run_usage(
        UsageInfo(input_tokens=3, output_tokens=2, cache_read_input_tokens=7)
    )
    usage = generator.get_run_usage()
    assert usage.input_tokens == 13
    assert usage.output_tokens == 7
    assert usage.cache_read_input_tokens == 7
    generator.reset_run_usage()
    assert generator.get_run_usage().input_tokens == 0


def test_validation_repair_prompt_renders_all_placeholders() -> None:
    text = generator._render_validation_repair_prompt(
        recipe_body="RECIPE BODY HERE",
        language_hints={"language": "python"},
        project_file_list=["b.py", "a.py"],
        failing_command="ruff check .",
        validation_output="src/app.py:1:1 F401",
        implicated_files={"src/app.py": "import os\n"},
        language="python",
    )
    assert "RECIPE BODY HERE" in text
    assert "`ruff check .`" in text
    assert "- a.py\n- b.py" in text  # sorted file list
    assert "F401" in text
    assert "import os" in text
    for placeholder in (
        "{recipe_body}",
        "{language_hints_yaml}",
        "{project_file_list}",
        "{failing_command}",
        "{validation_output}",
        "{implicated_files_block}",
    ):
        assert placeholder not in text


# ---------------------------------------------------------------------------
# Loop orchestration inside run_generation (validators + repair scripted)
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


def _build_inputs(
    tmp_path: Path,
    mock_deployments_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> PipelineInputs:
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("AGENT_SCAFFOLD_DEPLOYMENTS_PATH", str(mock_deployments_path))
    monkeypatch.setenv("AGENT_SCAFFOLD_CACHE_DIR", str(cache_dir))
    cfg = load_config()
    recipes = discover_recipes(mock_deployments_path)
    recipe = next(r for r in recipes if r.slug == "customer-support-triage")
    ctx = assemble(recipe, "python", "langgraph", mock_deployments_path)
    import importlib.resources as resources

    hints = yaml.safe_load(
        resources.files("agent_scaffold.languages")
        .joinpath("python.yaml")
        .read_text(encoding="utf-8")
    )
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
        hints=hints,
        topology=Topology.SINGLE,
        roles=[],
        write_mode=WriteMode.abort,
        strict=False,
        format_output=False,
        skip_validation=False,
        no_cache=True,
    )


def _scripted_validate(script: list[bool]) -> Any:
    """Each call pops the next pass/fail flag; exhausted script keeps failing."""
    calls: list[list[ValidationTier]] = []

    def fake_validate(
        dest: Path,
        hints: dict[str, Any],
        smoke_check: str,
        tiers: list[ValidationTier],
        continue_on_failure: bool = False,
        on_event: Any = None,
    ) -> list[ValidationResult]:
        calls.append(tiers)
        passed = script.pop(0) if script else False
        return [
            ValidationResult(
                tier=ValidationTier.static,
                passed=passed,
                output="" if passed else "src/demo_agent/main.py:1:1: F401 unused import",
            )
        ]

    fake_validate.calls = calls  # type: ignore[attr-defined]
    return fake_validate


def test_repair_loop_recovers_and_updates_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_deployments_path: Path,
    mock_responses_path: Path,
) -> None:
    payload = (mock_responses_path / "valid_python.json").read_text(encoding="utf-8")
    patch_payload = (mock_responses_path / "patch_response.json").read_text(encoding="utf-8")
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: _Client(payload))

    repair_calls: list[dict[str, Any]] = []

    def fake_repair_validation(**kwargs: Any) -> str:
        repair_calls.append(kwargs)
        return patch_payload

    monkeypatch.setattr(pipeline, "repair_validation", fake_repair_validation)
    fake_validate = _scripted_validate([False, True])  # fail once, pass after patch
    monkeypatch.setattr(pipeline, "run_validate", fake_validate)

    inputs = _build_inputs(tmp_path, mock_deployments_path, monkeypatch)
    report = run_generation(inputs, display=NullProgressDisplay())

    assert report.result is not None
    assert len(repair_calls) == 1
    # Patched content reached both disk and the in-memory result (→ manifest).
    on_disk = (inputs.dest / "src/demo_agent/main.py").read_text(encoding="utf-8")
    assert "repaired" in on_disk
    in_result = next(f for f in report.result.files if f.path == "src/demo_agent/main.py")
    assert "repaired" in in_result.content
    assert (inputs.dest / ".scaffold" / "manifest.json").exists()
    # Repair prompt received the failing output + the implicated file body.
    kwargs = repair_calls[0]
    assert "F401" in kwargs["validation_output"]
    assert "src/demo_agent/main.py" in kwargs["implicated_files"]
    assert all(r.passed for r in report.validation_results)


def test_repair_loop_exhausts_rounds_then_raises_with_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_deployments_path: Path,
    mock_responses_path: Path,
) -> None:
    payload = (mock_responses_path / "valid_python.json").read_text(encoding="utf-8")
    patch_payload = (mock_responses_path / "patch_response.json").read_text(encoding="utf-8")
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: _Client(payload))

    repair_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        pipeline,
        "repair_validation",
        lambda **kwargs: repair_calls.append(kwargs) or patch_payload,
    )
    monkeypatch.setattr(pipeline, "run_validate", _scripted_validate([]))  # always fails

    inputs = _build_inputs(tmp_path, mock_deployments_path, monkeypatch)
    with pytest.raises(PipelineError) as excinfo:
        run_generation(inputs, display=NullProgressDisplay())

    assert excinfo.value.phase == "validate"
    assert f"{MAX_REPAIR_ROUNDS} repair round(s)" in excinfo.value.message
    assert len(repair_calls) == MAX_REPAIR_ROUNDS
    # Project + manifest survive for manual recovery.
    assert (inputs.dest / "pyproject.toml").exists()
    assert (inputs.dest / ".scaffold" / "manifest.json").exists()


def test_repair_loop_stops_on_repair_error_without_crashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_deployments_path: Path,
    mock_responses_path: Path,
) -> None:
    payload = (mock_responses_path / "valid_python.json").read_text(encoding="utf-8")
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: _Client(payload))

    def broken_repair(**_kwargs: Any) -> str:
        raise RuntimeError("api down")

    monkeypatch.setattr(pipeline, "repair_validation", broken_repair)
    monkeypatch.setattr(pipeline, "run_validate", _scripted_validate([]))

    inputs = _build_inputs(tmp_path, mock_deployments_path, monkeypatch)
    with pytest.raises(PipelineError) as excinfo:
        run_generation(inputs, display=NullProgressDisplay())

    # The repair failure degrades to the validation PipelineError, not a crash.
    assert excinfo.value.phase == "validate"
    assert (inputs.dest / "pyproject.toml").exists()


def test_repair_smoke_failure_patches_files_from_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_deployments_path: Path,
    mock_responses_path: Path,
) -> None:
    """Post-`up` smoke repair: one round, manifest-driven paths, patch lands."""
    from agent_scaffold.config import load_config
    from agent_scaffold.discovery import discover_recipes
    from agent_scaffold.manifest import Manifest, ManifestFile

    project = tmp_path / "proj"
    (project / "src" / "demo_agent").mkdir(parents=True)
    target = project / "src" / "demo_agent" / "main.py"
    target.write_text("broken = True\n", encoding="utf-8")

    manifest = Manifest(
        recipe="customer-support-triage",
        language="python",
        framework="langgraph",
        model="claude-test",
        generated_at="2026-06-12T00:00:00+00:00",
        files=[ManifestFile(path="src/demo_agent/main.py", lines=1, sha256="x")],
        answers={"project_name": "demo_agent"},
    )
    recipe = next(
        r for r in discover_recipes(mock_deployments_path) if r.slug == "customer-support-triage"
    )
    patch_payload = (mock_responses_path / "patch_response.json").read_text(encoding="utf-8")
    repair_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        pipeline,
        "repair_validation",
        lambda **kwargs: repair_calls.append(kwargs) or patch_payload,
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    patched = pipeline.repair_smoke_failure(
        project_dir=project,
        manifest=manifest,
        recipe=recipe,
        cfg=load_config(),
        failure_output="smoke failed: src/demo_agent/main.py raised ValueError",
    )

    assert patched == 1
    assert "repaired" in target.read_text(encoding="utf-8")
    kwargs = repair_calls[0]
    assert kwargs["failing_command"] == "smoke test (post-provisioning)"
    assert "src/demo_agent/main.py" in kwargs["implicated_files"]
    assert kwargs["language"] == "python"


def test_skip_validation_bypasses_repair_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_deployments_path: Path,
    mock_responses_path: Path,
) -> None:
    payload = (mock_responses_path / "valid_python.json").read_text(encoding="utf-8")
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: _Client(payload))
    monkeypatch.setattr(
        pipeline,
        "repair_validation",
        lambda **_k: pytest.fail("repair must not run with skip_validation"),
    )

    base = _build_inputs(tmp_path, mock_deployments_path, monkeypatch)
    inputs = PipelineInputs(
        **{
            **{k: getattr(base, k) for k in base.__dataclass_fields__},
            "skip_validation": True,
        }
    )
    report = run_generation(inputs, display=NullProgressDisplay())
    assert report.result is not None
    assert report.validation_results == []
