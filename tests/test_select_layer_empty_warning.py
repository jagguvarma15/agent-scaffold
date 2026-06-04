"""Tests for the empty-catalog warning in ``_select_layer``.

When the catalog has zero capabilities for the requested kinds, the
picker emits a yellow heads-up instead of silently no-op-ping — a silent
return previously made customize look like it skipped user picks.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from agent_scaffold.config import Config
from agent_scaffold.repl.session import SessionState
from agent_scaffold.repl.shell import _select_layer
from agent_scaffold.sources import DEPLOYMENTS_SPEC, ResolvedSource


@pytest.fixture
def state_with_empty_catalog(tmp_path: Path) -> SessionState:
    """SessionState pointing at a deployments tree with no docs/capabilities/."""
    (tmp_path / "deployments" / "docs").mkdir(parents=True)
    cfg = Config(
        anthropic_api_key="test-key",
        cache_dir=tmp_path / "cache",
        failures_dir=tmp_path / "cache" / "failures",
    )
    src = ResolvedSource(
        spec=DEPLOYMENTS_SPEC,
        path=tmp_path / "deployments",
        label="empty",
        kind="explicit-path",
        commit_sha=None,
    )
    return SessionState(cfg=cfg, deployments=src, blueprints=src)


def test_empty_layer_returns_empty_list(state_with_empty_catalog: SessionState) -> None:
    out = _select_layer(state_with_empty_catalog, ("relational", "cache"), "Memory")
    assert out == []


def test_empty_layer_emits_console_warning(state_with_empty_catalog: SessionState) -> None:
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=120)
    _select_layer(state_with_empty_catalog, ("eval",), "Eval", console=console)
    output = buf.getvalue()
    assert "catalog has no" in output
    assert "eval" in output
    assert "Eval" in output


def test_empty_layer_without_console_stays_silent(
    state_with_empty_catalog: SessionState,
) -> None:
    # Test seam: callers that don't pass a console (e.g. unit tests of the
    # apply path) don't get spurious prints.
    out = _select_layer(state_with_empty_catalog, ("tools",), "Tools")
    assert out == []
