"""Tests for the vendored-blueprints resolution path in ``sources.py``.

Verifies that ``resolve_blueprints`` prefers ``<deployments_path>/vendored/
blueprints/`` when present, falls through to legacy fetch behavior when not,
and that ``override`` always wins over the vendored shortcut.
"""

from __future__ import annotations

from pathlib import Path

from agent_scaffold.sources import (
    VENDORED_BLUEPRINTS_SUBPATH,
    ResolvedSource,
    resolve_blueprints,
)


def _make_deployments_tree(root: Path, *, vendored: bool) -> Path:
    """Create a fake deployments checkout with or without vendored content."""
    docs = root / "docs" / "recipes"
    docs.mkdir(parents=True)
    if vendored:
        v = (
            root
            / VENDORED_BLUEPRINTS_SUBPATH[0]
            / VENDORED_BLUEPRINTS_SUBPATH[1]
            / "patterns"
            / "react"
        )
        v.mkdir(parents=True)
        (v / "overview.md").write_text("# ReAct\n", encoding="utf-8")
    return root


def test_vendored_path_preferred_when_present(tmp_path: Path) -> None:
    """A deployments tree with vendored/blueprints/ wins over fetch."""
    deployments = _make_deployments_tree(tmp_path / "deps", vendored=True)
    cache = tmp_path / "cache"

    result = resolve_blueprints(
        override=None,
        mode="auto",
        cache_dir=cache,
        env={},  # no env override
        deployments_path=deployments,
    )

    assert isinstance(result, ResolvedSource)
    assert result.kind == "vendored"
    expected = deployments / "vendored" / "blueprints"
    assert result.path == expected.resolve()
    assert "vendored" in result.label.lower()
    # No GitHub fetch ever happened — cache dir stays empty.
    assert not (cache / "blueprints").exists()


def test_vendored_skipped_when_dir_missing(tmp_path: Path) -> None:
    """No vendored dir → fall through to mode=skip (no network in tests)."""
    deployments = _make_deployments_tree(tmp_path / "deps", vendored=False)

    result = resolve_blueprints(
        override=None,
        mode="skip",
        cache_dir=tmp_path / "cache",
        env={},
        deployments_path=deployments,
    )

    assert result.kind == "skipped"
    assert result.path is None


def test_vendored_skipped_when_dir_empty(tmp_path: Path) -> None:
    """Empty vendored/blueprints/ dir → fall through (not yet populated)."""
    deployments = tmp_path / "deps"
    empty = deployments / "vendored" / "blueprints"
    empty.mkdir(parents=True)

    result = resolve_blueprints(
        override=None,
        mode="skip",
        cache_dir=tmp_path / "cache",
        env={},
        deployments_path=deployments,
    )

    assert result.kind == "skipped"


def test_override_wins_over_vendored(tmp_path: Path) -> None:
    """An explicit ``--blueprints-path`` ignores the vendored shortcut."""
    deployments = _make_deployments_tree(tmp_path / "deps", vendored=True)
    override = tmp_path / "explicit-blueprints"
    override.mkdir()

    result = resolve_blueprints(
        override=override,
        mode="auto",
        cache_dir=tmp_path / "cache",
        env={},
        deployments_path=deployments,
    )

    assert result.kind == "explicit-path"
    assert result.path == override.resolve()


def test_deployments_path_none_falls_through(tmp_path: Path) -> None:
    """No deployments_path means we can't shortcut — fall through to skip."""
    result = resolve_blueprints(
        override=None,
        mode="skip",
        cache_dir=tmp_path / "cache",
        env={},
        deployments_path=None,
    )

    assert result.kind == "skipped"


def test_resolve_blueprints_no_kwarg_keeps_legacy_behavior(tmp_path: Path) -> None:
    """Callers that don't pass ``deployments_path`` keep the legacy resolver."""
    result = resolve_blueprints(
        override=None,
        mode="skip",
        cache_dir=tmp_path / "cache",
        env={},
        # deployments_path omitted entirely
    )

    assert result.kind == "skipped"
