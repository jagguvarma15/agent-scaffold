"""Promptfoo eval plugin.

Shells out to ``npx promptfoo eval --config evals/promptfooconfig.yaml
--output evals/last-run.json``. Promptfoo runs the suite to completion, writes
its JSON report to the output path, and exits non-zero only on hard CLI
failure (not on per-case failures — those land in the JSON). This plugin
reads the JSON, normalizes to ``EvalResult``, and lets the CLI render.

Promptfoo's JSON schema has a few subtly different shapes across versions.
We try the canonical ``results.results[].success / .score`` path first and
fall back to top-level ``results[]`` when present. Anything we can't parse
becomes an empty ``cases`` list with ``error`` set — the CLI shows the
error rather than silently passing.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from agent_scaffold.eval._common import (
    EvalCase,
    EvalResult,
    cli_present,
    compute_delta,
)

name = "promptfoo"
cli_binary = "npx"
install_hint = "npm install -g pnpm && pnpm dlx promptfoo --help (or `npm i -g promptfoo`)"
config_file: str | None = "evals/promptfooconfig.yaml"

_DEFAULT_TIMEOUT = 300.0  # 5 min cap — long enough for ~50 LLM cases
_OUTPUT_FILE = "evals/last-run.json"


def run(project_dir: Path, baseline_total: float | None) -> EvalResult:
    cmd = [
        "npx",
        "promptfoo",
        "eval",
        "--config",
        config_file or "evals/promptfooconfig.yaml",
        "--output",
        _OUTPUT_FILE,
    ]

    if not cli_present(cli_binary):
        return EvalResult(
            target=name,
            cmd_run=cmd,
            skipped=True,
            skip_reason=f"{cli_binary} not on PATH — install Node and run `{install_hint}`",
        )

    config_path = project_dir / (config_file or "evals/promptfooconfig.yaml")
    if not config_path.is_file():
        return EvalResult(
            target=name,
            cmd_run=cmd,
            skipped=True,
            skip_reason=f"no {config_file} — recipe ships no eval config",
        )

    try:
        proc = subprocess.run(  # noqa: S603 — list-form, shell=False
            cmd,
            cwd=str(project_dir),
            check=False,
            capture_output=True,
            text=True,
            timeout=_DEFAULT_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return EvalResult(
            target=name,
            cmd_run=cmd,
            error=f"promptfoo eval timed out after {_DEFAULT_TIMEOUT:.0f}s",
        )
    except (OSError, FileNotFoundError) as exc:
        return EvalResult(
            target=name,
            cmd_run=cmd,
            skipped=True,
            skip_reason=f"could not invoke {cli_binary}: {exc}",
        )

    output_path = project_dir / _OUTPUT_FILE
    if not output_path.is_file():
        # Promptfoo crashed before writing JSON; surface stderr tail.
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
        return EvalResult(
            target=name,
            cmd_run=cmd,
            error=(
                "promptfoo did not write " + _OUTPUT_FILE + "; tail: " + " | ".join(tail)
                if tail
                else "promptfoo did not write " + _OUTPUT_FILE
            ),
        )

    try:
        data = json.loads(output_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return EvalResult(
            target=name,
            cmd_run=cmd,
            error=f"{_OUTPUT_FILE} is not valid JSON: {exc}",
        )

    cases = _extract_cases(data)
    total = _weighted_total(cases)
    return EvalResult(
        target=name,
        cases=cases,
        total=total,
        baseline_total=baseline_total,
        delta=compute_delta(total, baseline_total),
        cmd_run=cmd,
    )


def _extract_cases(data: Any) -> list[EvalCase]:
    """Pull a list of per-case ``EvalCase`` from a Promptfoo JSON payload.

    Tries multiple known shapes; returns ``[]`` if none match (the caller
    then computes ``total=0.0`` and the regression check treats it as a
    failed run rather than silently passing).
    """
    # Shape A — current Promptfoo: {"results": {"results": [{"success", "score", ...}]}}
    inner = _get_nested(data, ("results", "results"))
    if isinstance(inner, list):
        return [_case_from_row(row, idx) for idx, row in enumerate(inner)]
    # Shape B — flatter older shape: {"results": [...]}.
    top = data.get("results") if isinstance(data, dict) else None
    if isinstance(top, list):
        return [_case_from_row(row, idx) for idx, row in enumerate(top)]
    return []


def _case_from_row(row: Any, idx: int) -> EvalCase:
    if not isinstance(row, dict):
        return EvalCase(name=f"case-{idx}", score=0.0, passed=False)
    description = (
        row.get("description")
        or _get_nested(row, ("test", "description"))
        or _get_nested(row, ("vars", "name"))
        or f"case-{idx}"
    )
    raw_score = row.get("score")
    if raw_score is None:
        # Promptfoo sometimes only emits ``success`` (bool) — map to 0/1.
        raw_score = 1.0 if row.get("success") else 0.0
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        score = 0.0
    # Clamp into [0, 1]; runaway LLM-judged scores shouldn't blow the total.
    score = max(0.0, min(1.0, score))
    passed = bool(row.get("success", score >= 0.5))
    return EvalCase(name=str(description), score=score, passed=passed)


def _weighted_total(cases: list[EvalCase]) -> float:
    """Equal-weight mean. Promptfoo doesn't expose per-case weights in v1."""
    if not cases:
        return 0.0
    return sum(c.score for c in cases) / len(cases)


def _get_nested(data: Any, path: tuple[str, ...]) -> Any:
    cursor: Any = data
    for key in path:
        if not isinstance(cursor, dict) or key not in cursor:
            return None
        cursor = cursor[key]
    return cursor


__all__ = ["run"]
