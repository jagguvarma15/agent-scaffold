"""Failure artifacts under ~/.cache/agent-scaffold/failures/ are secret-mode
and redacted: model output can echo credential-shaped strings."""

from __future__ import annotations

import stat
from pathlib import Path

from agent_scaffold.pipeline import _save_failure


def test_save_failure_is_owner_only_and_redacted(tmp_path: Path) -> None:
    raw = '{"files": [], "note": "key sk-ant-api03-abcdefghijklmnopqrstuvwxyz012345"}'
    path = _save_failure(raw, tmp_path / "failures")
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600
    content = path.read_text(encoding="utf-8")
    assert "sk-ant-api03-abcdefghijklmnopqrstuvwxyz012345" not in content
