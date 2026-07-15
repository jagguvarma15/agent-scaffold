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
    assert labels == ["Backend", "Stack health", "Stop everything"]
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
    rows = list(_collect_rows(tmp_path, _manifest(), None, run_log_dir="/cache/runs/20260612-abc"))
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
        "Stack health",
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


def test_containerized_frontend_row_falls_back_to_3000(tmp_path: Path) -> None:
    # Docker mode writes no frontend.pid; a resolved frontend capability still
    # surfaces the canonical :3000 so the reachable chat UI is always listed.
    manifest = _manifest(capabilities=["frontend.minimal-chat"])
    rows = list(_collect_rows(tmp_path, manifest, None))
    frontend = next(r for r in rows if r.label == "Frontend")
    assert frontend.url == "http://localhost:3000"


def test_pid_port_wins_over_capability_fallback(tmp_path: Path) -> None:
    _write_pid_file(tmp_path, port=4001)
    manifest = _manifest(capabilities=["frontend.minimal-chat"])
    rows = list(_collect_rows(tmp_path, manifest, None))
    frontend = next(r for r in rows if r.label == "Frontend")
    assert frontend.url == "http://localhost:4001"  # dev-server PID wins


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

    panel = render_welcome_panel(tmp_path, _manifest(), None, probe=False)
    assert isinstance(panel, RichPanel)


def _render_text(panel: Any) -> str:
    import io

    from rich.console import Console

    buf = io.StringIO()
    Console(file=buf, width=200).print(panel)
    return buf.getvalue()


def test_render_marks_unreachable_urls_not_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When a service URL can't be reached, the row is shown but flagged."""
    monkeypatch.setattr(
        "agent_scaffold.welcome._probe_urls_live",
        lambda urls, **_: {u: False for u in urls},
    )
    panel = render_welcome_panel(tmp_path, _manifest(), _full_stack(), probe=True)
    text = _render_text(panel)
    assert "not running" in text
    assert "○" in text
    assert "http://localhost:8000" in text  # the URL is still shown, just marked


def test_render_marks_reachable_urls_live(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agent_scaffold.welcome._probe_urls_live",
        lambda urls, **_: {u: True for u in urls},
    )
    panel = render_welcome_panel(tmp_path, _manifest(), _full_stack(), probe=True)
    text = _render_text(panel)
    assert "not running" not in text
    assert "●" in text


def test_render_probe_false_does_not_touch_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_a: Any, **_k: Any) -> dict[str, bool]:
        raise AssertionError("liveness probe must not run when probe=False")

    monkeypatch.setattr("agent_scaffold.welcome._probe_urls_live", _boom)
    panel = render_welcome_panel(tmp_path, _manifest(), _full_stack(), probe=False)
    text = _render_text(panel)
    assert "not running" not in text
    assert "○" not in text
    assert "●" not in text


def test_probe_urls_live_detects_open_and_closed_ports() -> None:
    import socket

    from agent_scaffold.welcome import _probe_urls_live

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    open_port = srv.getsockname()[1]

    # A port we bind then close is reliably "nothing listening".
    closed = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    closed.bind(("127.0.0.1", 0))
    closed_port = closed.getsockname()[1]
    closed.close()

    try:
        result = _probe_urls_live(
            [f"http://127.0.0.1:{open_port}", f"http://127.0.0.1:{closed_port}"],
            timeout=0.5,
        )
        assert result[f"http://127.0.0.1:{open_port}"] is True
        assert result[f"http://127.0.0.1:{closed_port}"] is False
    finally:
        srv.close()


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


def test_unconnected_cloud_option_gets_a_connect_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_scaffold.stack_options import MODE_CLOUD, CredentialSpec, StackOption

    option = StackOption(
        id="langsmith",
        title="LangSmith",
        capability_ids=frozenset({"obs.langsmith"}),
        kind="obs",
        mode=MODE_CLOUD,
        credentials=(CredentialSpec(var="LANGCHAIN_API_KEY"),),
        managed_vars=("LANGCHAIN_API_KEY",),
        docker_service=None,
        probe="langsmith_workspace",
        bootstrap_step="bootstrap_langsmith",
        key_page_url=None,
    )
    monkeypatch.setattr("agent_scaffold.stack_options.load_stack_options", lambda _caps: [option])
    monkeypatch.setattr("agent_scaffold.envfile.build_runtime_env", lambda *_a, **_k: {})
    rows = list(_collect_rows(tmp_path, _manifest(capabilities=["obs.langsmith"]), None))
    row = next(r for r in rows if r.label == "LangSmith")
    assert row.url == "agent-scaffold connect langsmith"
    assert "not connected" in row.note

    # A wired credential drops the row.
    monkeypatch.setattr(
        "agent_scaffold.envfile.build_runtime_env",
        lambda *_a, **_k: {"LANGCHAIN_API_KEY": "lsv2_x"},
    )
    rows = list(_collect_rows(tmp_path, _manifest(capabilities=["obs.langsmith"]), None))
    assert all(r.label != "LangSmith" for r in rows)
