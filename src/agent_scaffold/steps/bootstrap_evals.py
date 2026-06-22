"""``bootstrap_evals`` step: run the eval suite once, store the baseline.

Runs after ``smoke_test``: the smoke check has already proven the project
boots and the basic happy path works, so the eval LLM calls are worth
spending tokens on. Stores the resulting ``total`` in
``manifest.answers["eval_baseline"]`` so ``agent-scaffold eval`` later
compares against a real baseline rather than the empty-string default.

Detection rules:

- No ``eval.*`` capability resolved → ``SKIPPED``.
- ``manifest.answers["eval_baseline"]`` already set → ``DONE``.
- Otherwise ``PENDING``.

The fingerprint hashes the eval config files + the chosen eval target.
Editing ``evals/cases.recipe.yaml`` invalidates the baseline so the next
``up --resume`` re-baselines.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from agent_scaffold.orchestrator import (
    DetectionResult,
    StepContext,
    StepLog,
    StepResult,
    StepStatus,
    compute_fingerprint,
)

_DEFAULT_TIMEOUT = 120.0  # opt-in (run via `agent-scaffold eval`); bounded so it can't hang `up`
_EVAL_CONFIG_PATHS = (
    "evals/promptfooconfig.yaml",
    "evals/cases.yaml",
    "evals/cases.recipe.yaml",
)


def _first_eval_capability(ctx: StepContext) -> str | None:
    """Return the first ``eval.*`` capability id on the resolved stack, or ``None``."""
    stack = getattr(ctx, "resolved_stack", None)
    if stack is None:
        return None
    for cap in getattr(stack, "capabilities", None) or []:
        cap_id = getattr(cap, "id", "")
        if cap_id.startswith("eval."):
            return cap_id
    return None


def _read_baseline(ctx: StepContext) -> str | None:
    """``manifest.answers["eval_baseline"]`` if set; tolerates the older shape."""
    answers = getattr(ctx.manifest, "answers", None) or {}
    raw = answers.get("eval_baseline")
    return raw if raw else None


def _config_sha(project_dir: Path) -> str | None:
    """SHA-256 over the eval config files that exist. ``None`` if all missing."""
    digest = hashlib.sha256()
    any_present = False
    for rel in _EVAL_CONFIG_PATHS:
        path = project_dir / rel
        if not path.is_file():
            continue
        any_present = True
        digest.update(rel.encode("utf-8"))
        digest.update(path.read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest() if any_present else None


@dataclass
class BootstrapEvalsStep:
    """Run the eval suite once during ``up`` and persist the baseline score."""

    id: str = "bootstrap_evals"
    description: str = "Run the eval suite once + store the baseline score"
    depends_on: tuple[str, ...] = ("smoke_test",)
    timeout: float = _DEFAULT_TIMEOUT
    troubleshoot: dict[str, str] = field(
        default_factory=lambda: {
            "npx: command not found": (
                "Node not on PATH — install Node 20+ (e.g. `brew install node`) "
                "then `agent-scaffold up --retry bootstrap_evals`"
            ),
            "ANTHROPIC_API_KEY": (
                "Promptfoo needs ANTHROPIC_API_KEY — set it in .env.local or your shell"
            ),
        }
    )

    # ---- detection ----------------------------------------------------

    def detect(self, ctx: StepContext) -> DetectionResult:
        cap_id = _first_eval_capability(ctx)
        if cap_id is None:
            return DetectionResult(
                StepStatus.SKIPPED, reason="no eval.* capability — recipe ships no evals"
            )
        if _read_baseline(ctx) is not None:
            return DetectionResult(StepStatus.DONE, reason="eval_baseline already set in manifest")
        return DetectionResult(StepStatus.PENDING, reason=f"will baseline {cap_id} on first run")

    # ---- apply --------------------------------------------------------

    def apply(self, ctx: StepContext) -> StepResult:
        from agent_scaffold.eval import get_plugin
        from agent_scaffold.manifest import update_manifest_answer

        cap_id = _first_eval_capability(ctx)
        if cap_id is None:
            return StepResult(StepStatus.SKIPPED, detail="no eval.* capability")

        target = cap_id.split(".", 1)[1] if "." in cap_id else cap_id
        try:
            plugin = get_plugin(target)
        except KeyError:
            return StepResult(
                StepStatus.SKIPPED,
                detail=f"no eval plugin registered for {target!r}",
            )

        result = plugin.run(ctx.project_dir, baseline_total=None)

        if result.skipped:
            return StepResult(StepStatus.SKIPPED, detail=result.skip_reason or f"{target} skipped")
        if result.error is not None:
            return StepResult(
                StepStatus.FAILED,
                error=f"{target}: {result.error}",
            )
        if not result.cases:
            return StepResult(
                StepStatus.FAILED,
                error=f"{target} returned no cases — config or runner bug",
            )

        try:
            update_manifest_answer(ctx.project_dir, "eval_baseline", f"{result.total:.4f}")
        except Exception as exc:  # noqa: BLE001 — manifest write hits the FS
            return StepResult(
                StepStatus.FAILED,
                error=f"could not persist eval_baseline: {exc}",
            )

        ctx.emit(
            StepLog(
                step_id=self.id,
                line=(
                    f"eval baseline: {result.total:.2f} "
                    f"({result.passed_count}/{len(result.cases)} passed)"
                ),
                stream="stdout",
            )
        )
        return StepResult(
            StepStatus.DONE,
            detail=f"baseline {result.total:.2f} stored ({len(result.cases)} cases)",
        )

    # ---- fingerprint --------------------------------------------------

    def fingerprint(self, ctx: StepContext) -> str:
        return compute_fingerprint(
            {
                "capability": _first_eval_capability(ctx),
                "config_sha": _config_sha(ctx.project_dir),
            }
        )


__all__ = ["BootstrapEvalsStep"]
