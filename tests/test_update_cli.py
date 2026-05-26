"""Tests for the ``agent-scaffold update`` subcommand.

We stub ``_regenerate_for_update`` so the tests never hit the real Anthropic
API. The merge logic itself is unit-tested in ``tests/test_merge.py``; here
we exercise the CLI plumbing: change-detection, plan rendering, conflict
handling, ``--continue`` semantics, and manifest history.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent_scaffold.cli import app
from agent_scaffold.manifest import Manifest, write_manifest
from agent_scaffold.template_snapshot import save_generation_snapshot


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def deployments_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Tiny deployments tree + AGENT_SCAFFOLD_* env vars to wire up config loading."""
    deployments = tmp_path / "deployments"
    (deployments / "docs" / "recipes").mkdir(parents=True)
    (deployments / "docs" / "recipes" / "demo.md").write_text("# demo\n", encoding="utf-8")
    monkeypatch.setenv("AGENT_SCAFFOLD_DEPLOYMENTS_PATH", str(deployments))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    return deployments


def _generated_project(
    project_dir: Path,
    files: dict[str, str],
    *,
    template_sha: str,
    snapshot: bool = True,
) -> Manifest:
    """Write files to disk, build a v2 manifest, snapshot the generation."""
    for rel, content in files.items():
        target = project_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    from agent_scaffold.manifest import build_file_entries

    manifest = Manifest(
        recipe="demo",
        language="python",
        framework="none",
        model="claude-test",
        generated_at="2026-05-26T00:00:00+00:00",
        files=build_file_entries(project_dir, list(files)),
        template_snapshot_sha=template_sha,
        answers={"project_name": "demo"},
    )
    write_manifest(project_dir, manifest)
    if snapshot:
        save_generation_snapshot(project_dir, template_sha, files)
    return manifest


def _stub_regenerate(monkeypatch: pytest.MonkeyPatch, returns: dict[str, str] | None) -> None:
    """Replace the LLM-calling path with a deterministic fake."""
    from agent_scaffold import cli as cli_mod

    monkeypatch.setattr(cli_mod, "_regenerate_for_update", lambda *a, **kw: returns)


def _stub_template_sha(monkeypatch: pytest.MonkeyPatch, *, returns: str) -> None:
    from agent_scaffold import cli as cli_mod

    monkeypatch.setattr(cli_mod, "compute_template_sha", lambda _p: returns)


def test_missing_manifest_emits_friendly_error(
    runner: CliRunner, deployments_fixture: Path, tmp_path: Path
) -> None:
    project = tmp_path / "no-manifest"
    project.mkdir()
    result = runner.invoke(app, ["update", str(project)])
    assert result.exit_code == 1
    assert "manifest" in result.output.lower()


def test_update_short_circuits_when_template_unchanged(
    runner: CliRunner,
    deployments_fixture: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    sha = "a" * 64
    _generated_project(project, {"src/main.py": "ours\n"}, template_sha=sha)
    _stub_template_sha(monkeypatch, returns=sha)
    # Should never hit regenerate — the sha-match check is upstream.
    _stub_regenerate(
        monkeypatch, returns=None
    )  # placeholder; if called, test will catch the None handling
    result = runner.invoke(app, ["update", str(project), "--yes"])
    assert result.exit_code == 0, result.output
    assert "Nothing to update" in result.output


def test_update_dry_run_does_not_write(
    runner: CliRunner,
    deployments_fixture: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    old_sha = "a" * 64
    new_sha = "b" * 64
    _generated_project(project, {"src/main.py": "v1\n"}, template_sha=old_sha)
    _stub_template_sha(monkeypatch, returns=new_sha)
    _stub_regenerate(monkeypatch, returns={"src/main.py": "v2\n", "src/new.py": "added\n"})
    result = runner.invoke(app, ["update", str(project), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert (project / "src/main.py").read_text(encoding="utf-8") == "v1\n", "must not write"
    assert not (project / "src/new.py").exists(), "must not add files"
    assert "Update plan" in result.output


def test_update_happy_path_writes_added_and_modified_files(
    runner: CliRunner,
    deployments_fixture: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    old_sha = "a" * 64
    new_sha = "b" * 64
    _generated_project(project, {"src/main.py": "import os\n\nprint('hi')\n"}, template_sha=old_sha)
    _stub_template_sha(monkeypatch, returns=new_sha)
    _stub_regenerate(
        monkeypatch,
        returns={
            "src/main.py": "import os\nimport sys\n\nprint('hi')\n",  # template added an import
            "src/new.py": "# added by template\n",
        },
    )
    result = runner.invoke(app, ["update", str(project), "--yes"])
    assert result.exit_code == 0, result.output
    assert (project / "src/new.py").is_file()
    main = (project / "src/main.py").read_text(encoding="utf-8")
    assert "import sys" in main
    # in_progress.json must not exist on a clean run.
    assert not (project / ".scaffold" / "update.in-progress.json").is_file()


def test_update_with_conflict_writes_markers_and_in_progress_file(
    runner: CliRunner,
    deployments_fixture: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    old_sha = "a" * 64
    new_sha = "b" * 64
    # Snapshot base = "BASE"; user edited to USER; template now wants TEMPLATE.
    _generated_project(project, {"f.py": "BASE\n"}, template_sha=old_sha)
    (project / "f.py").write_text("USER\n", encoding="utf-8")  # simulate user edit
    _stub_template_sha(monkeypatch, returns=new_sha)
    _stub_regenerate(monkeypatch, returns={"f.py": "TEMPLATE\n"})
    result = runner.invoke(app, ["update", str(project), "--yes"])
    assert result.exit_code == 2, result.output  # conflict exit code
    text = (project / "f.py").read_text(encoding="utf-8")
    assert "<<<<<<< user" in text
    assert ">>>>>>> template" in text
    in_progress = project / ".scaffold" / "update.in-progress.json"
    assert in_progress.is_file()
    payload = json.loads(in_progress.read_text(encoding="utf-8"))
    assert "f.py" in payload["conflicts"]


def test_continue_refuses_when_markers_remain(
    runner: CliRunner,
    deployments_fixture: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    old_sha = "a" * 64
    new_sha = "b" * 64
    _generated_project(project, {"f.py": "BASE\n"}, template_sha=old_sha)
    (project / "f.py").write_text("USER\n", encoding="utf-8")
    _stub_template_sha(monkeypatch, returns=new_sha)
    _stub_regenerate(monkeypatch, returns={"f.py": "TEMPLATE\n"})
    # First run leaves markers + in_progress.json
    assert runner.invoke(app, ["update", str(project), "--yes"]).exit_code == 2
    # Continue without resolving → must refuse
    result = runner.invoke(app, ["update", str(project), "--continue"])
    assert result.exit_code == 1
    assert "still present" in result.output.lower()


def test_continue_succeeds_after_manual_resolution_and_appends_history(
    runner: CliRunner,
    deployments_fixture: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    old_sha = "a" * 64
    new_sha = "b" * 64
    _generated_project(project, {"f.py": "BASE\n"}, template_sha=old_sha)
    (project / "f.py").write_text("USER\n", encoding="utf-8")
    _stub_template_sha(monkeypatch, returns=new_sha)
    _stub_regenerate(monkeypatch, returns={"f.py": "TEMPLATE\n"})
    assert runner.invoke(app, ["update", str(project), "--yes"]).exit_code == 2
    # Manually "resolve" — pick the template's side, strip markers.
    (project / "f.py").write_text("TEMPLATE\n", encoding="utf-8")
    result = runner.invoke(app, ["update", str(project), "--continue"])
    assert result.exit_code == 0, result.output
    # in_progress cleared
    assert not (project / ".scaffold" / "update.in-progress.json").is_file()
    # Manifest history appended
    from agent_scaffold.manifest import read_manifest

    manifest = read_manifest(project)
    assert manifest.template_snapshot_sha == new_sha
    assert len(manifest.update_history) == 1
    assert manifest.update_history[0].to_template_sha == new_sha


def test_update_with_prior_in_progress_refuses_until_continue(
    runner: CliRunner,
    deployments_fixture: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / ".scaffold").mkdir()
    (project / ".scaffold" / "update.in-progress.json").write_text("{}", encoding="utf-8")
    _generated_project(project, {"f.py": "x\n"}, template_sha="a" * 64)
    _stub_template_sha(monkeypatch, returns="b" * 64)
    result = runner.invoke(app, ["update", str(project), "--yes"])
    assert result.exit_code == 1
    assert "in progress" in result.output.lower()


def test_update_bootstraps_when_no_prior_snapshot(
    runner: CliRunner,
    deployments_fixture: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v1-upgraded projects have no snapshot — bootstrap by recording one."""
    project = tmp_path / "proj"
    project.mkdir()
    new_sha = "b" * 64
    # Generate WITHOUT snapshotting — simulates a manifest from before Q8.
    _generated_project(project, {"f.py": "x\n"}, template_sha="a" * 64, snapshot=False)
    _stub_template_sha(monkeypatch, returns=new_sha)
    result = runner.invoke(app, ["update", str(project), "--yes"])
    assert result.exit_code == 0
    assert "bootstrapping" in result.output.lower()
    from agent_scaffold.manifest import read_manifest

    assert read_manifest(project).template_snapshot_sha == new_sha
