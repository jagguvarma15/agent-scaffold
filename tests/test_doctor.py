"""Tests for ``agent_scaffold.doctor`` and the ``doctor`` Typer command."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from agent_scaffold import doctor as doctor_mod
from agent_scaffold.cli import app
from agent_scaffold.doctor import (
    CheckResult,
    CheckStatus,
    DockerCheck,
    DoctorReport,
    PythonCheck,
    RuffCheck,
    UvCheck,
    baseline_checks,
    run_checks,
)


def _completed(
    returncode: int = 0, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["dummy"], returncode=returncode, stdout=stdout, stderr=stderr
    )


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ---------------------------------------------------------------------------
# DoctorReport / run_checks
# ---------------------------------------------------------------------------


class _ConstCheck:
    def __init__(self, result: CheckResult) -> None:
        self._result = result
        self.id = result.id
        self.category = result.category

    def run(self) -> CheckResult:
        return self._result


def _result(
    status: CheckStatus, *, id: str = "tool.x", category: str = "Tools", title: str = "x"
) -> CheckResult:
    return CheckResult(id=id, category=category, status=status, title=title)


def test_report_summary_counts_each_status() -> None:
    report = DoctorReport(
        results=[
            _result(CheckStatus.OK, id="a"),
            _result(CheckStatus.OK, id="b"),
            _result(CheckStatus.WARN, id="c"),
            _result(CheckStatus.FAIL, id="d"),
            _result(CheckStatus.SKIP, id="e"),
        ]
    )
    assert report.summary == {"ok": 2, "warn": 1, "fail": 1, "skip": 1}


def test_report_exit_code_one_when_any_fail() -> None:
    failing = DoctorReport(results=[_result(CheckStatus.OK), _result(CheckStatus.FAIL)])
    passing = DoctorReport(results=[_result(CheckStatus.OK), _result(CheckStatus.WARN)])
    assert failing.exit_code == 1
    assert passing.exit_code == 0


def test_run_checks_invokes_each_check_once() -> None:
    a = _ConstCheck(_result(CheckStatus.OK, id="a"))
    b = _ConstCheck(_result(CheckStatus.FAIL, id="b"))
    report = run_checks([a, b])
    assert [r.id for r in report.results] == ["a", "b"]
    assert report.exit_code == 1


def test_baseline_checks_returns_four_ids() -> None:
    ids = [c.id for c in baseline_checks()]
    assert ids == ["tool.python", "tool.uv", "tool.docker", "tool.ruff"]


# ---------------------------------------------------------------------------
# PythonCheck
# ---------------------------------------------------------------------------


def test_python_check_ok_when_version_meets_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor_mod, "_py_version", lambda: (3, 12, 1))
    result = PythonCheck().run()
    assert result.status == CheckStatus.OK
    assert "python 3.12.1" in result.title
    assert result.explain_topic == "python"


def test_python_check_fail_when_below_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor_mod, "_py_version", lambda: (3, 10, 0))
    result = PythonCheck().run()
    assert result.status == CheckStatus.FAIL
    assert "3.10" in result.title
    assert "pyenv" in result.fix_hint


# ---------------------------------------------------------------------------
# Binary-version checks: shared paths via UvCheck and RuffCheck
# ---------------------------------------------------------------------------


def test_uv_check_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor_mod.shutil, "which", lambda b: "/usr/bin/uv")
    monkeypatch.setattr(doctor_mod, "_run_cmd", lambda cmd: _completed(0, "uv 0.4.20\n"))
    result = UvCheck().run()
    assert result.status == CheckStatus.OK
    assert "uv 0.4.20" in result.title


def test_uv_check_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor_mod.shutil, "which", lambda b: None)
    result = UvCheck().run()
    assert result.status == CheckStatus.FAIL
    assert "missing" in result.title
    assert "astral.sh/uv" in result.fix_hint


def test_uv_check_below_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor_mod.shutil, "which", lambda b: "/usr/bin/uv")
    monkeypatch.setattr(doctor_mod, "_run_cmd", lambda cmd: _completed(0, "uv 0.2.5\n"))
    result = UvCheck().run()
    assert result.status == CheckStatus.FAIL
    assert result.detail == "need >=0.4"


def test_uv_check_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor_mod.shutil, "which", lambda b: "/usr/bin/uv")

    def boom(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=5)

    monkeypatch.setattr(doctor_mod, "_run_cmd", boom)
    result = UvCheck().run()
    assert result.status == CheckStatus.FAIL
    assert "timed out" in result.title


def test_uv_check_parse_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor_mod.shutil, "which", lambda b: "/usr/bin/uv")
    monkeypatch.setattr(doctor_mod, "_run_cmd", lambda cmd: _completed(0, "no version here\n"))
    result = UvCheck().run()
    assert result.status == CheckStatus.FAIL
    assert "unparseable" in result.title


def test_uv_check_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor_mod.shutil, "which", lambda b: "/usr/bin/uv")
    monkeypatch.setattr(doctor_mod, "_run_cmd", lambda cmd: _completed(2, "", "permission denied"))
    result = UvCheck().run()
    assert result.status == CheckStatus.FAIL
    assert "permission denied" in result.detail


def test_ruff_check_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor_mod.shutil, "which", lambda b: "/usr/bin/ruff")
    monkeypatch.setattr(doctor_mod, "_run_cmd", lambda cmd: _completed(0, "ruff 0.6.9\n"))
    result = RuffCheck().run()
    assert result.status == CheckStatus.OK
    assert "ruff 0.6.9" in result.title
    assert result.explain_topic == "ruff"


def test_ruff_check_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor_mod.shutil, "which", lambda b: None)
    result = RuffCheck().run()
    assert result.status == CheckStatus.FAIL
    assert "uv tool install ruff" in result.fix_hint


def test_binary_check_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor_mod.shutil, "which", lambda b: "/usr/bin/uv")

    def boom(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        raise OSError("EACCES")

    monkeypatch.setattr(doctor_mod, "_run_cmd", boom)
    result = UvCheck().run()
    assert result.status == CheckStatus.FAIL
    assert "EACCES" in result.detail


# ---------------------------------------------------------------------------
# DockerCheck — the daemon-not-running case matters most
# ---------------------------------------------------------------------------


def test_docker_check_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor_mod.shutil, "which", lambda b: None)
    result = DockerCheck().run()
    assert result.status == CheckStatus.FAIL
    assert "docker missing" in result.title


def test_docker_check_daemon_not_running(monkeypatch: pytest.MonkeyPatch) -> None:
    """Server template returns empty string + non-zero exit when daemon is down."""
    monkeypatch.setattr(doctor_mod.shutil, "which", lambda b: "/usr/local/bin/docker")
    monkeypatch.setattr(
        doctor_mod,
        "_run_cmd",
        lambda cmd: _completed(1, "", "Cannot connect to the Docker daemon"),
    )
    result = DockerCheck().run()
    assert result.status == CheckStatus.WARN
    assert "daemon not running" in result.detail
    assert "colima start" in result.fix_hint


def test_docker_check_daemon_empty_stdout_zero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Some Docker builds exit 0 with empty server string when daemon is down."""
    monkeypatch.setattr(doctor_mod.shutil, "which", lambda b: "/usr/local/bin/docker")
    monkeypatch.setattr(doctor_mod, "_run_cmd", lambda cmd: _completed(0, "\n"))
    result = DockerCheck().run()
    assert result.status == CheckStatus.WARN


def test_docker_check_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor_mod.shutil, "which", lambda b: "/usr/local/bin/docker")
    monkeypatch.setattr(doctor_mod, "_run_cmd", lambda cmd: _completed(0, "25.0.3\n"))
    result = DockerCheck().run()
    assert result.status == CheckStatus.OK
    assert "25.0.3" in result.title


def test_docker_check_below_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor_mod.shutil, "which", lambda b: "/usr/local/bin/docker")
    monkeypatch.setattr(doctor_mod, "_run_cmd", lambda cmd: _completed(0, "20.10.7\n"))
    result = DockerCheck().run()
    assert result.status == CheckStatus.FAIL
    assert "need >=24.0" in result.detail


def test_docker_check_timeout_treated_as_daemon_down(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor_mod.shutil, "which", lambda b: "/usr/local/bin/docker")

    def boom(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=5)

    monkeypatch.setattr(doctor_mod, "_run_cmd", boom)
    result = DockerCheck().run()
    assert result.status == CheckStatus.WARN
    assert "colima start" in result.fix_hint


def test_docker_check_unparseable_server_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor_mod.shutil, "which", lambda b: "/usr/local/bin/docker")
    monkeypatch.setattr(doctor_mod, "_run_cmd", lambda cmd: _completed(0, "nightly\n"))
    result = DockerCheck().run()
    assert result.status == CheckStatus.FAIL
    assert "unparseable" in result.title


# ---------------------------------------------------------------------------
# Subprocess hygiene: no shell=True, real timeout passed to subprocess.run
# ---------------------------------------------------------------------------


def test_run_cmd_uses_explicit_timeout_and_no_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return _completed(0, "uv 0.5.0\n")

    monkeypatch.setattr(doctor_mod.subprocess, "run", fake_run)
    doctor_mod._run_cmd(["uv", "--version"])
    assert captured["kwargs"]["shell"] is False
    assert captured["kwargs"]["timeout"] == doctor_mod._SUBPROCESS_TIMEOUT
    assert captured["kwargs"]["capture_output"] is True


# ---------------------------------------------------------------------------
# CLI: `agent-scaffold doctor` end-to-end via Typer's CliRunner
# ---------------------------------------------------------------------------


@pytest.fixture
def all_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """All four baseline checks pass — exit code 0 expected."""

    def which(binary: str) -> str | None:
        return f"/usr/bin/{binary}"

    def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        binary = cmd[0]
        if binary == "uv":
            return _completed(0, "uv 0.4.20\n")
        if binary == "ruff":
            return _completed(0, "ruff 0.6.9\n")
        if binary == "docker":
            return _completed(0, "25.0.3\n")
        return _completed(0, "")

    monkeypatch.setattr(doctor_mod.shutil, "which", which)
    monkeypatch.setattr(doctor_mod, "_run_cmd", run_cmd)
    monkeypatch.setattr(doctor_mod, "_py_version", lambda: (3, 12, 1))


def test_cli_doctor_exit_zero_when_all_ok(runner: CliRunner, all_ok: None) -> None:
    res = runner.invoke(app, ["doctor"])
    assert res.exit_code == 0, res.output
    assert "python 3.12.1" in res.output
    assert "Summary:" in res.output


def test_cli_doctor_exit_one_on_any_fail(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(doctor_mod.shutil, "which", lambda b: None)
    res = runner.invoke(app, ["doctor"])
    assert res.exit_code == 1
    assert "missing" in res.output


def test_cli_doctor_json_emits_valid_payload(runner: CliRunner, all_ok: None) -> None:
    res = runner.invoke(app, ["doctor", "--json"])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["schema_version"] == 1
    assert payload["exit_code"] == 0
    assert payload["summary"]["ok"] == 4
    assert {r["id"] for r in payload["results"]} == {
        "tool.python",
        "tool.uv",
        "tool.docker",
        "tool.ruff",
    }
    for r in payload["results"]:
        # No Rich markup in JSON output.
        assert "[" not in r["title"]


def test_cli_doctor_json_includes_exit_code_one_on_fail(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(doctor_mod.shutil, "which", lambda b: None)
    res = runner.invoke(app, ["doctor", "--json"])
    assert res.exit_code == 1
    payload = json.loads(res.output)
    assert payload["exit_code"] == 1
    assert payload["summary"]["fail"] >= 1


def test_cli_doctor_explain_missing_topic_fails_soft(runner: CliRunner) -> None:
    res = runner.invoke(app, ["doctor", "--explain", "this-topic-does-not-exist"])
    assert res.exit_code == 0
    assert "No docs yet" in res.output


def test_cli_doctor_explain_reads_bundled_doc(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_doc = tmp_path / "docker.md"
    fake_doc.write_text("# Docker getting started\n\nInstall Docker Desktop.\n")
    monkeypatch.setattr(
        "agent_scaffold.cli_doctor._resolve_explain_doc",
        lambda topic: fake_doc if topic == "docker" else None,
    )
    monkeypatch.delenv("PAGER", raising=False)
    res = runner.invoke(app, ["doctor", "--explain", "docker"])
    assert res.exit_code == 0
    assert "Docker getting started" in res.output


def test_cli_doctor_explain_via_pager(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When PAGER is set and stdout is a TTY, the doc is fed to the pager."""
    fake_doc = tmp_path / "uv.md"
    fake_doc.write_text("# uv\n")
    monkeypatch.setattr(
        "agent_scaffold.cli_doctor._resolve_explain_doc",
        lambda topic: fake_doc if topic == "uv" else None,
    )
    monkeypatch.setenv("PAGER", "cat")
    # CliRunner pipes stdout, so isatty() is False — exercises the
    # non-TTY fallback path which just prints. That's intentional;
    # we just want to confirm no crash and no error code.
    res = runner.invoke(app, ["doctor", "--explain", "uv"])
    assert res.exit_code == 0
    assert "# uv" in res.output


def test_cli_doctor_unknown_recipe_exits_one(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, all_ok: None
) -> None:
    """`--recipe <slug>` against a missing recipe is a hard error."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-1234")
    res = runner.invoke(app, ["doctor", "--recipe", "no-such-recipe"])
    assert res.exit_code == 1
    assert "Unknown recipe" in res.output


def test_cli_doctor_recipe_adds_auth_and_service_sections(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    all_ok: None,
) -> None:
    """`--recipe <slug>` adds Authentication + Recipe services check sections."""
    from pathlib import Path

    fixture_root = Path(__file__).parent / "fixtures" / "mock_deployments"
    monkeypatch.setenv("AGENT_SCAFFOLD_DEPLOYMENTS_PATH", str(fixture_root))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-1234")

    res = runner.invoke(
        app,
        [
            "doctor",
            "--recipe",
            "with-external-services",
            "--no-probes",
        ],
    )
    # exit code depends on probe outcomes; we just want the new sections present.
    assert "Authentication" in res.output
    assert "Recipe services" in res.output
    # --no-probes means every service row is SKIP.
    assert "probes disabled" in res.output


def test_cli_doctor_recipe_json_includes_services(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    all_ok: None,
) -> None:
    from pathlib import Path

    fixture_root = Path(__file__).parent / "fixtures" / "mock_deployments"
    monkeypatch.setenv("AGENT_SCAFFOLD_DEPLOYMENTS_PATH", str(fixture_root))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-1234")

    res = runner.invoke(
        app,
        ["doctor", "--recipe", "with-external-services", "--no-probes", "--json"],
    )
    # discovery._warn writes to stderr which CliRunner merges into output,
    # and one of the warning strings itself contains `{package: version}`.
    # Anchor on the actual JSON header instead.
    json_start = res.output.index('{\n  "schema_version"')
    payload = json.loads(res.output[json_start:])
    ids = {r["id"] for r in payload["results"]}
    assert "auth.backend" in ids
    assert "auth.anthropic_key" in ids
    assert any(rid.startswith("service.") for rid in ids)


def test_cli_doctor_recipe_timeout_propagates(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    all_ok: None,
) -> None:
    """`--timeout 12` is threaded into the probe call."""
    from pathlib import Path

    from agent_scaffold import probes

    fixture_root = Path(__file__).parent / "fixtures" / "mock_deployments"
    monkeypatch.setenv("AGENT_SCAFFOLD_DEPLOYMENTS_PATH", str(fixture_root))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-1234")

    captured: list[float] = []

    def fake_run_probe(svc: Any, *, timeout: float = 5.0, skip: bool = False) -> Any:
        captured.append(timeout)
        from agent_scaffold.doctor import CheckResult, CheckStatus

        return CheckResult(
            id=f"service.{svc.id}",
            category="Recipe services",
            status=CheckStatus.OK,
            title=f"{svc.id}: stubbed",
        )

    monkeypatch.setattr(probes, "run_probe", fake_run_probe)
    res = runner.invoke(
        app,
        ["doctor", "--recipe", "with-external-services", "--timeout", "12"],
    )
    assert res.exit_code == 0, res.output
    assert captured  # at least one probe was called
    assert all(t == 12.0 for t in captured)
