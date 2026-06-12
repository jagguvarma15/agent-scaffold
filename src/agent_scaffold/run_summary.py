"""Write ``.scaffold/run-summary.md`` into the generated project.

The generation report panel scrolls away with the terminal; this file is the
durable, human-readable record that travels WITH the project: what was
generated from which recipe and deployments snapshot, whether validation
passed (and how many repair rounds it took), which env vars are still
missing (names only — values live in the shell, the encrypted vault, or
``.env.local``), and how to start the thing.

``agent-scaffold up`` appends/refreshes a Provisioning section after each
run so the file always reflects the latest state.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_scaffold._scaffold_dir import SCAFFOLD_DIR
from agent_scaffold.contract import GenerationResult
from agent_scaffold.discovery import Recipe

RUN_SUMMARY_FILENAME = "run-summary.md"

_PROVISIONING_HEADER = "## Provisioning"
_PROVISIONING_RE = re.compile(rf"\n{re.escape(_PROVISIONING_HEADER)}.*?(?=\n## |\Z)", re.DOTALL)


def run_summary_path(project_dir: Path) -> Path:
    return project_dir / SCAFFOLD_DIR / RUN_SUMMARY_FILENAME


def write_run_summary(
    project_dir: Path,
    *,
    recipe: Recipe,
    language: str,
    framework: str,
    model: str,
    result: GenerationResult,
    template_sha: str | None,
    validation_results: list[Any],
    repair_rounds: int,
    resolved_stack: Any | None,
    run_log_dir: str = "",
) -> Path:
    """Render and write the summary. Returns the path; never raises upward —
    callers treat a summary-write failure as a warning, not a run failure."""
    lines: list[str] = [
        f"# Run summary — {result.project_name}",
        "",
        f"Generated {datetime.now(UTC).isoformat(timespec='seconds')} by agent-scaffold.",
        "",
        "## Selections",
        "",
        f"- Recipe: `{recipe.slug}`" + (f" ({recipe.status})" if recipe.status else ""),
        f"- Language / framework: {language} / {framework or '(none)'}",
        f"- Model: {model}",
    ]
    if template_sha:
        lines.append(f"- Deployments snapshot: `{template_sha[:16]}`")
    lines += [
        "",
        "## Generation",
        "",
        f"- Files: {len(result.files)}",
        _validation_line(validation_results, repair_rounds),
    ]
    if result.known_limitations:
        lines.append("- Known limitations:")
        lines.extend(f"  - {item}" for item in result.known_limitations)

    lines += _environment_section(recipe, resolved_stack, project_dir)
    lines += _start_section(project_dir, result)
    lines += [
        "",
        "## Artifacts",
        "",
        f"- Manifest: `{SCAFFOLD_DIR}/manifest.json`",
    ]
    if run_log_dir:
        lines.append(f"- Run log: `{run_log_dir}`")

    path = run_summary_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _validation_line(validation_results: list[Any], repair_rounds: int) -> str:
    if not validation_results:
        return "- Validation: skipped"
    parts = [f"{r.tier.value} {'✓' if r.passed else '✗ FAILING'}" for r in validation_results]
    suffix = ""
    if repair_rounds:
        suffix = f" (after {repair_rounds} repair round{'s' if repair_rounds != 1 else ''})"
    return f"- Validation: {', '.join(parts)}{suffix}"


def _environment_section(
    recipe: Recipe, resolved_stack: Any | None, project_dir: Path
) -> list[str]:
    """Env var NAMES with set/missing status. Values never appear here."""
    from agent_scaffold.preflight import collect_env_requirements

    requirements = collect_env_requirements(recipe, None, resolved_stack, project_dir)
    if not requirements:
        return []
    lines = ["", "## Environment", ""]
    for req in requirements:
        status = "set" if req.satisfied else ("MISSING" if req.required else "missing (optional)")
        lines.append(f"- `{req.name}` — {status} ({req.source})")
    lines += [
        "",
        "_Names only. Values live in your shell, the encrypted secrets vault"
        " (`agent-scaffold secrets list`), or `.env.local` — never in this file._",
    ]
    return lines


def _start_section(project_dir: Path, result: GenerationResult) -> list[str]:
    lines = ["", "## Start", "", "```bash", f"cd {project_dir}"]
    lines.extend(result.post_install)
    lines.append("agent-scaffold up")
    if result.smoke_check:
        lines.append(result.smoke_check)
    lines.append("```")
    return lines


def append_provisioning_section(
    project_dir: Path,
    step_summary: dict[str, int],
) -> None:
    """Append (or refresh) the Provisioning section after an ``up`` run.

    Idempotent: an existing Provisioning section is replaced so repeated
    ``up`` runs keep exactly one, reflecting the latest state. Best-effort —
    a missing summary file (older project) is a silent no-op.
    """
    path = run_summary_path(project_dir)
    if not path.is_file():
        return
    counts = ", ".join(f"{count} {name}" for name, count in step_summary.items() if count)
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    section = (
        f"\n{_PROVISIONING_HEADER}\n\n"
        f"- Last `agent-scaffold up`: {stamp}\n"
        f"- Steps: {counts or 'none ran'}\n"
    )
    try:
        text = path.read_text(encoding="utf-8")
        text = _PROVISIONING_RE.sub("", text).rstrip() + "\n" + section
        path.write_text(text, encoding="utf-8")
    except OSError:
        return


__all__ = [
    "RUN_SUMMARY_FILENAME",
    "append_provisioning_section",
    "run_summary_path",
    "write_run_summary",
]
