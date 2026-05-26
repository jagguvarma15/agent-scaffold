"""``smoke_test`` step: run the project's smoke suite.

Selection order:

1. ``scripts/smoke.sh`` if it exists — run via ``bash``.
2. Else ``pytest -m smoke`` — run via ``uv run pytest`` if any smoke-marked
   tests collect; otherwise ``SKIPPED``.

When running pytest we also parse its trailing summary line
(``"=== 12 passed, 1 failed in 4.21s ==="``) and emit a ``StepProgress``
event so the user sees the pass/fail counts as soon as pytest exits.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from agent_scaffold.orchestrator import (
    DetectionResult,
    StepContext,
    StepProgress,
    StepResult,
    StepStatus,
    compute_fingerprint,
)
from agent_scaffold.steps._subprocess import stream_subprocess

_DEFAULT_TIMEOUT = 600.0
_SMOKE_SH = Path("scripts") / "smoke.sh"
# pytest's summary line orders categories by what was non-zero, so we can't
# assume a fixed sequence. Instead, scan for ``<int> <category>`` tokens.
_PYTEST_TOKEN_RE = re.compile(r"(\d+)\s+(passed|failed|errors?|skipped)", re.IGNORECASE)


@dataclass
class SmokeTestStep:
    """Run the project's smoke tests; surface the result count in the panel."""

    id: str = "smoke_test"
    description: str = "Run smoke tests"
    # seed may be SKIPPED if there's no script; that doesn't block smoke.
    depends_on: tuple[str, ...] = ("seed",)
    timeout: float = _DEFAULT_TIMEOUT
    troubleshoot: dict[str, str] = field(
        default_factory=lambda: {
            "ANTHROPIC_API_KEY": (
                "key not set or invalid — re-run `agent-scaffold up --force wire_credentials`"
            ),
            "ConnectionError": (
                "a service is down — `agent-scaffold doctor --recipe <name>` to identify which"
            ),
            "ratelimit": (
                "Anthropic rate limit — wait 60s and retry, or switch to a less-loaded model"
            ),
            "TimeoutError": "agent step exceeded its timeout — check the test's timeout setting",
        }
    )

    # ---- detection ----------------------------------------------------

    def detect(self, ctx: StepContext) -> DetectionResult:
        kind = self._select_kind(ctx)
        if kind is None:
            return DetectionResult(
                StepStatus.SKIPPED,
                reason="no scripts/smoke.sh and no `pytest -m smoke` items collectible",
            )
        return DetectionResult(StepStatus.PENDING, reason=f"will run {kind}")

    # ---- apply --------------------------------------------------------

    def apply(self, ctx: StepContext) -> StepResult:
        kind = self._select_kind(ctx)
        if kind is None:
            return StepResult(StepStatus.SKIPPED, detail="no smoke tests")
        if kind == "shell":
            if shutil.which("bash") is None:
                return StepResult(StepStatus.FAILED, error="`bash` not found on PATH")
            cmd = ["bash", str(_SMOKE_SH)]
        else:
            if shutil.which("uv") is None:
                return StepResult(
                    StepStatus.FAILED, error="`uv` not found on PATH — install_deps first"
                )
            cmd = ["uv", "run", "pytest", "-m", "smoke", "--tb=short", "-q"]

        # Buffer stdout so we can parse pytest's summary line; the streaming
        # callback still gets every line for the live panel.
        stdout_buffer: list[str] = []
        original_cb = ctx.callback

        def _tee(event: object) -> None:
            # We use a wrapper to record stdout lines for parsing.
            from agent_scaffold.orchestrator import StepLog

            if isinstance(event, StepLog) and event.stream == "stdout":
                stdout_buffer.append(event.line)
            if original_cb is not None:
                original_cb(event)  # type: ignore[arg-type]

        ctx.callback = _tee
        try:
            result = stream_subprocess(
                cmd,
                cwd=ctx.project_dir,
                step_id=self.id,
                callback=ctx.callback,
                timeout=self.timeout,
            )
        finally:
            ctx.callback = original_cb

        summary = _parse_pytest_summary("\n".join(stdout_buffer))
        if summary is not None and ctx.callback is not None:
            ctx.callback(StepProgress(step_id=self.id, message=_format_summary(summary)))
        if result.exit_code != 0:
            return StepResult(
                StepStatus.FAILED,
                error=(
                    f"smoke timed out after {result.duration:.0f}s"
                    if result.timed_out
                    else f"smoke failed (exit {result.exit_code})"
                ),
                stderr_tail=result.stderr_tail,
            )
        detail = (
            _format_summary(summary) if summary is not None else f"ok in {result.duration:.1f}s"
        )
        return StepResult(StepStatus.DONE, detail=detail)

    # ---- fingerprint --------------------------------------------------

    def fingerprint(self, ctx: StepContext) -> str:
        src_sha = _sha256_dir(ctx.project_dir / "src")
        test_sha = _sha256_dir(ctx.project_dir / "tests")
        return compute_fingerprint(
            {
                "src_tree_sha": src_sha,
                "test_tree_sha": test_sha,
                "model": ctx.manifest.model,
            }
        )

    # ---- helpers ------------------------------------------------------

    def _select_kind(self, ctx: StepContext) -> str | None:
        if (ctx.project_dir / _SMOKE_SH).is_file():
            return "shell"
        if shutil.which("uv") is None:
            return None
        rc = _pytest_collect_rc(ctx.project_dir)
        if rc == 0:
            return "pytest"
        return None


def _pytest_collect_rc(project_dir: Path) -> int:
    """Run ``pytest -m smoke --collect-only -q`` and return its rc.

    Exit code 0 means at least one smoke-marked item collected (pytest
    documents non-zero for "no tests collected").
    """
    import subprocess

    try:
        proc = subprocess.run(  # noqa: S603
            ["uv", "run", "pytest", "-m", "smoke", "--collect-only", "-q"],
            cwd=str(project_dir),
            check=False,
            capture_output=True,
            text=True,
            timeout=30.0,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return -1
    return int(proc.returncode)


def _parse_pytest_summary(text: str) -> dict[str, int] | None:
    """Return counts keyed by ``passed/failed/errors/skipped`` from the summary line.

    pytest prints categories in whatever order they're non-zero
    (``"3 failed, 2 passed"`` vs ``"5 passed"``), so we scan all ``<n> <name>``
    tokens on lines that look like the summary footer (have ``in <time>`` or
    surrounding ``===``).
    """
    for line in reversed(text.splitlines()):
        if "==" not in line and " in " not in line.lower():
            continue
        tokens = _PYTEST_TOKEN_RE.findall(line)
        if not tokens:
            continue
        out: dict[str, int] = {}
        for count, label in tokens:
            key = label.lower().rstrip("s") + ("s" if label.lower().startswith("error") else "")
            # Normalise ``error/errors`` → ``errors`` to match the brief.
            if key == "error":
                key = "errors"
            out[key] = int(count)
        if out:
            return out
    return None


def _format_summary(summary: dict[str, int]) -> str:
    parts = []
    for key in ("passed", "failed", "errors", "skipped"):
        if summary.get(key):
            parts.append(f"{summary[key]} {key}")
    return ", ".join(parts) or "no result line"


def _sha256_dir(directory: Path) -> str | None:
    if not directory.is_dir():
        return None
    h = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            h.update(path.relative_to(directory).as_posix().encode("utf-8"))
            h.update(b"\0")
            h.update(path.read_bytes())
    return h.hexdigest()


__all__: Sequence[str] = ["SmokeTestStep"]
