"""Pin the bundled capability catalog.

PyPI installs use the bundled fallback when the GitHub fetch is skipped
(offline, rate-limited, --no-fetch). If the sync script ever drops
``capabilities/`` from its whitelist again, the customize-mode layer
walk goes silent and the wizard appears to skip user selections.

These tests are the safety net.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_scaffold.capabilities import load_capabilities


@pytest.fixture
def bundled_path() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent / "src" / "agent_scaffold" / "_bundled_deployments"


def test_bundled_ships_a_capability_catalog(bundled_path: Path) -> None:
    catalog = load_capabilities(bundled_path)
    assert catalog, (
        "bundled fallback has no capability catalog — customize mode will "
        "silently no-op for PyPI installs. Check scripts/sync_deployments.sh."
    )


def test_bundled_catalog_covers_every_layer_with_caps_today(bundled_path: Path) -> None:
    catalog = load_capabilities(bundled_path)
    by_kind: dict[str, list[str]] = {}
    for cap in catalog.values():
        by_kind.setdefault(cap.kind, []).append(cap.id)
    # Every layer with at least one capability in the upstream agent-deployments
    # repo must surface in the bundled fallback. ``tools`` isn't here yet —
    # those land in the deployments PR for Phase 3.
    required = {"relational", "cache", "vector_db", "obs", "eval", "frontend", "host", "queue"}
    missing = required - set(by_kind)
    assert not missing, f"bundled catalog missing kinds: {sorted(missing)}"


def test_sync_script_whitelist_includes_capabilities() -> None:
    """If the sync script ever drops ``capabilities`` again, this fires.

    The bundled fallback is rebuilt from the sync script's whitelist during
    every release; an omission silently re-breaks customize for PyPI users.
    """
    here = Path(__file__).resolve()
    sync_script = here.parent.parent / "scripts" / "sync_deployments.sh"
    text = sync_script.read_text(encoding="utf-8")
    assert "capabilities" in text, (
        "scripts/sync_deployments.sh does not copy docs/capabilities/ — "
        "PyPI bundled fallback will ship an empty catalog."
    )
