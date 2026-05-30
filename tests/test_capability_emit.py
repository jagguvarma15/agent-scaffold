"""Tests for ``agent_scaffold.capability_emit``."""

from __future__ import annotations

from pathlib import Path

from agent_scaffold.capabilities import Capability, EmitFile, ResolvedStack, load_capabilities
from agent_scaffold.capability_emit import copy_capability_templates
from agent_scaffold.writer import WriteMode


def _capabilities_root(mock_deployments_path: Path) -> Path:
    return mock_deployments_path / "docs" / "capabilities"


def _stack(*ids: str, mock_deployments_path: Path) -> ResolvedStack:
    catalog = load_capabilities(mock_deployments_path)
    return ResolvedStack(capabilities=[catalog[i] for i in ids])


def test_glob_emit_preserves_relative_tree(mock_deployments_path: Path, tmp_path: Path) -> None:
    stack = _stack("frontend.nextjs-tiny", mock_deployments_path=mock_deployments_path)
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    result = copy_capability_templates(
        stack=stack,
        capabilities_root=_capabilities_root(mock_deployments_path),
        project_dir=project_dir,
        write_mode=WriteMode.skip,
    )

    assert (project_dir / "frontend" / "package.json").is_file()
    assert (project_dir / "frontend" / "app" / "page.tsx").is_file()
    assert (project_dir / "frontend" / "app" / "api" / "agent" / "route.ts").is_file()
    assert len(result.written) == 3
    assert result.overwritten == []
    assert result.skipped_unsafe == []


def test_single_file_emit(mock_deployments_path: Path, tmp_path: Path) -> None:
    stack = _stack("host.vercel-single", mock_deployments_path=mock_deployments_path)
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    result = copy_capability_templates(
        stack=stack,
        capabilities_root=_capabilities_root(mock_deployments_path),
        project_dir=project_dir,
        write_mode=WriteMode.skip,
    )

    target = project_dir / "vercel.json"
    assert target.read_text() == '{"version": 2, "name": "fixture"}\n'
    assert result.written == [target.resolve()]


def test_model_paths_win_on_collision(mock_deployments_path: Path, tmp_path: Path) -> None:
    stack = _stack("frontend.nextjs-tiny", mock_deployments_path=mock_deployments_path)
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    # Pretend the LLM emitted frontend/package.json — it must NOT be overwritten.
    (project_dir / "frontend").mkdir()
    (project_dir / "frontend" / "package.json").write_text("MODEL OUTPUT")

    result = copy_capability_templates(
        stack=stack,
        capabilities_root=_capabilities_root(mock_deployments_path),
        project_dir=project_dir,
        write_mode=WriteMode.overwrite,
        model_paths={"frontend/package.json"},
    )

    # The model's content is preserved.
    assert (project_dir / "frontend" / "package.json").read_text() == "MODEL OUTPUT"
    # The capability dest landed in skipped_existing, not overwritten.
    skipped_names = {p.name for p in result.skipped_existing}
    assert "package.json" in skipped_names
    # The other two template files were still copied.
    assert (project_dir / "frontend" / "app" / "page.tsx").is_file()


def test_write_mode_skip_preserves_existing_file(
    mock_deployments_path: Path, tmp_path: Path
) -> None:
    stack = _stack("host.vercel-single", mock_deployments_path=mock_deployments_path)
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "vercel.json").write_text("HAND EDITED")

    result = copy_capability_templates(
        stack=stack,
        capabilities_root=_capabilities_root(mock_deployments_path),
        project_dir=project_dir,
        write_mode=WriteMode.skip,
    )
    assert (project_dir / "vercel.json").read_text() == "HAND EDITED"
    assert (project_dir / "vercel.json").resolve() in result.skipped_existing
    assert result.written == []


def test_write_mode_overwrite_replaces_existing_file(
    mock_deployments_path: Path, tmp_path: Path
) -> None:
    stack = _stack("host.vercel-single", mock_deployments_path=mock_deployments_path)
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "vercel.json").write_text("HAND EDITED")

    result = copy_capability_templates(
        stack=stack,
        capabilities_root=_capabilities_root(mock_deployments_path),
        project_dir=project_dir,
        write_mode=WriteMode.overwrite,
    )
    # Overwrite mode replaces the existing file with the template content.
    assert (project_dir / "vercel.json").read_text().startswith('{"version": 2')
    assert (project_dir / "vercel.json").resolve() in result.overwritten


def test_unsafe_dest_paths_are_refused(tmp_path: Path) -> None:
    # Hand-build a capability with traversal in its dest to confirm the
    # path-safety guard catches it (the loader rejects these at parse time
    # too, but this is the defence-in-depth layer).
    cap_dir = tmp_path / "cap_root" / "host" / "evil"
    cap_dir.mkdir(parents=True)
    cap_file = cap_dir / "evil.md"
    cap_file.write_text("# evil")
    (cap_dir / "evil.txt").write_text("payload")
    cap = Capability(
        id="host.evil",
        kind="host",
        path=cap_file,
        emit_files=[EmitFile(source="evil.txt", dest="../escape.txt")],
    )
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    result = copy_capability_templates(
        stack=ResolvedStack(capabilities=[cap]),
        capabilities_root=tmp_path / "cap_root",
        project_dir=project_dir,
        write_mode=WriteMode.overwrite,
    )
    assert result.written == []
    assert "../escape.txt" in result.skipped_unsafe[0]
    # Sanity: the escape file was NOT created above project_dir.
    assert not (tmp_path / "escape.txt").exists()


def test_missing_source_reported(tmp_path: Path) -> None:
    cap_dir = tmp_path / "cap_root" / "host" / "ghost"
    cap_dir.mkdir(parents=True)
    cap_file = cap_dir / "ghost.md"
    cap_file.write_text("# ghost")
    cap = Capability(
        id="host.ghost",
        kind="host",
        path=cap_file,
        emit_files=[EmitFile(source="not-real.txt", dest="anywhere.txt")],
    )
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    result = copy_capability_templates(
        stack=ResolvedStack(capabilities=[cap]),
        capabilities_root=tmp_path / "cap_root",
        project_dir=project_dir,
        write_mode=WriteMode.overwrite,
    )
    assert result.missing_source == ["host.ghost:not-real.txt"]
    assert result.written == []


def test_no_emit_files_no_ops(mock_deployments_path: Path, tmp_path: Path) -> None:
    # Capability with no emit_files (e.g. cache.redis fixture).
    stack = _stack("cache.redis", mock_deployments_path=mock_deployments_path)
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    result = copy_capability_templates(
        stack=stack,
        capabilities_root=_capabilities_root(mock_deployments_path),
        project_dir=project_dir,
        write_mode=WriteMode.overwrite,
    )
    assert result.total_actions() == 0
    assert list(project_dir.iterdir()) == []
