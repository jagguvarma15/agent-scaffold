"""Tests for ``agent_scaffold.welcome``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent_scaffold._scaffold_dir import SCAFFOLD_DIR
from agent_scaffold.capabilities import Capability, DockerFragment, ResolvedStack
from agent_scaffold.manifest import Manifest
from agent_scaffold.welcome import (
    WelcomeRow,
    _collect_rows,
    _first_host_port,
    _open_browser_safe,
    render_welcome_panel,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _manifest(
    *,
    language: str = "python",
    capabilities: list[str] | None = None,
) -> Manifest:
    return Manifest(
        recipe="restaurant-rebooking",
        language=language,
        framework="langgraph",
        model="claude-test",
        generated_at="2026-05-30T00:00:00+00:00",
        capabilities=capabilities or [],
    )


def _capability(
    cap_id: str,
    *,
    kind: str | None = None,
    ports: list[str] | None = None,
) -> Capability:
    docker = (
        DockerFragment(service=cap_id.split(".", 1)[1], image="example/x:1", ports=ports or [])
        if ports is not None
        else None
    )
    return Capability(
        id=cap_id,
        kind=kind or cap_id.split(".", 1)[0],  # default kind from id prefix
        path=Path(f"/nonexistent/{cap_id}.md"),
        docker=docker,
    )


def _full_stack() -> ResolvedStack:
    return ResolvedStack(
        capabilities=[
            _capability("obs.grafana-stack", ports=["3002:3000"]),
            _capability("obs.langfuse", ports=["3001:3000"]),
            _capability("vector_db.qdrant", ports=["6333:6333", "6334:6334"]),
        ]
    )


def _write_pid_file(project_dir: Path, *, port: int = 3000, pid: int = 4321) -> None:
    pid_file = project_dir / SCAFFOLD_DIR / "frontend.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(
        json.dumps({"pid": pid, "port": port, "started_at": "2026-05-30T00:00:00+00:00"}),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# _collect_rows — order + presence
# ---------------------------------------------------------------------------


def test_empty_stack_yields_only_backend_and_stop(tmp_path: Path) -> None:
    rows = list(_collect_rows(tmp_path, _manifest(), None))
    labels = [r.label for r in rows]
    assert labels == ["Backend", "Stop everything"]
    backend = rows[0]
    assert backend.url == "http://localhost:8000"  # python default


def test_typescript_backend_uses_3000(tmp_path: Path) -> None:
    rows = list(_collect_rows(tmp_path, _manifest(language="typescript"), None))
    backend = next(r for r in rows if r.label == "Backend")
    assert backend.url == "http://localhost:3000"


def test_run_summary_row_when_file_exists(tmp_path: Path) -> None:
    summary = tmp_path / SCAFFOLD_DIR / "run-summary.md"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text("# Run summary\n", encoding="utf-8")
    rows = list(_collect_rows(tmp_path, _manifest(), None))
    labels = [r.label for r in rows]
    assert "Run summary" in labels
    row = next(r for r in rows if r.label == "Run summary")
    assert row.url == str(summary)
    # Stop row stays last.
    assert labels[-1] == "Stop everything"


def test_run_log_row_when_dir_provided(tmp_path: Path) -> None:
    rows = list(
        _collect_rows(tmp_path, _manifest(), None, run_log_dir="/cache/runs/20260612-abc")
    )
    row = next(r for r in rows if r.label == "Run log")
    assert row.url == "/cache/runs/20260612-abc"

    # Absent without a run log dir.
    rows = list(_collect_rows(tmp_path, _manifest(), None))
    assert all(r.label != "Run log" for r in rows)


def test_full_stack_row_order(tmp_path: Path) -> None:
    _write_pid_file(tmp_path, port=3000)
    manifest = _manifest(
        capabilities=[
            "obs.grafana-stack",
            "obs.langfuse",
            "vector_db.qdrant",
            "eval.promptfoo",
        ]
    )
    rows = list(_collect_rows(tmp_path, manifest, _full_stack()))
    assert [r.label for r in rows] == [
        "Frontend",
        "Backend",
        "Grafana",
        "Tempo",
        "Langfuse",
        "Qdrant",
        "Eval",
        "Stop everything",
    ]


def test_frontend_row_reads_port_from_pid_file(tmp_path: Path) -> None:
    _write_pid_file(tmp_path, port=4001)
    rows = list(_collect_rows(tmp_path, _manifest(), None))
    frontend = next(r for r in rows if r.label == "Frontend")
    assert frontend.url == "http://localhost:4001"


def test_missing_pid_file_omits_frontend_row(tmp_path: Path) -> None:
    rows = list(_collect_rows(tmp_path, _manifest(), None))
    assert all(r.label != "Frontend" for r in rows)


def test_malformed_pid_file_omits_frontend_row(tmp_path: Path) -> None:
    pid_file = tmp_path / SCAFFOLD_DIR / "frontend.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text("{garbage", encoding="utf-8")
    rows = list(_collect_rows(tmp_path, _manifest(), None))
    assert all(r.label != "Frontend" for r in rows)


def test_grafana_row_shows_admin_credentials_note(tmp_path: Path) -> None:
    rows = list(_collect_rows(tmp_path, _manifest(), _full_stack()))
    grafana = next(r for r in rows if r.label == "Grafana")
    assert "admin" in grafana.note
    assert "GRAFANA_ADMIN_PASSWORD" in grafana.note


def test_qdrant_uses_dashboard_path(tmp_path: Path) -> None:
    rows = list(_collect_rows(tmp_path, _manifest(), _full_stack()))
    qdrant = next(r for r in rows if r.label == "Qdrant")
    assert qdrant.url.endswith("/dashboard")
    assert "6333" in qdrant.url  # first host port


def test_capability_with_no_docker_block_is_skipped(tmp_path: Path) -> None:
    stack = ResolvedStack(capabilities=[_capability("obs.langfuse", ports=None)])
    rows = list(_collect_rows(tmp_path, _manifest(), stack))
    assert all(r.label != "Langfuse" for r in rows)


def test_eval_row_only_when_manifest_declares_it(tmp_path: Path) -> None:
    no_eval_rows = list(_collect_rows(tmp_path, _manifest(), _full_stack()))
    assert all(r.label != "Eval" for r in no_eval_rows)
    eval_rows = list(
        _collect_rows(tmp_path, _manifest(capabilities=["eval.promptfoo"]), _full_stack())
    )
    assert any(r.label == "Eval" for r in eval_rows)


# ---------------------------------------------------------------------------
# _first_host_port
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ports", "expected"),
    [
        (["3002:3000"], 3002),
        (["3002:3000", "6334:6334"], 3002),
        (["3002:3000/tcp"], 3002),
        (["bad"], None),
        ([], None),
    ],
)
def test_first_host_port_parsing(ports: list[str], expected: int | None) -> None:
    cap = _capability("obs.grafana-stack", ports=ports)
    assert _first_host_port(cap) == expected


def test_first_host_port_skips_malformed_then_finds_good() -> None:
    cap = _capability("obs.grafana-stack", ports=["bad", "3002:3000"])
    assert _first_host_port(cap) == 3002


# ---------------------------------------------------------------------------
# render_welcome_panel — smoke check
# ---------------------------------------------------------------------------


def test_render_welcome_panel_returns_rich_panel(tmp_path: Path) -> None:
    from rich.panel import Panel as RichPanel

    panel = render_welcome_panel(tmp_path, _manifest(), None)
    assert isinstance(panel, RichPanel)


# ---------------------------------------------------------------------------
# _open_browser_safe
# ---------------------------------------------------------------------------


def test_open_browser_safe_returns_false_on_explicit_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BROWSER", "none")
    assert _open_browser_safe("http://localhost:3000") is False


def test_open_browser_safe_swallows_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    import webbrowser

    def boom(*_a: Any, **_kw: Any) -> bool:
        raise RuntimeError("no browser in this sandbox")

    monkeypatch.setattr(webbrowser, "open", boom)
    monkeypatch.delenv("BROWSER", raising=False)
    assert _open_browser_safe("http://localhost:3000") is False


# ---------------------------------------------------------------------------
# WelcomeRow stability
# ---------------------------------------------------------------------------


def test_welcome_row_is_immutable_dataclass() -> None:
    row = WelcomeRow(label="X", url="http://x", note="n")
    with pytest.raises(Exception):  # noqa: B017,PT011 — frozen=True raises FrozenInstanceError
        row.label = "Y"  # type: ignore[misc]
