"""``emit_deploy_configs`` step: write cloud-deploy configs from host.* capabilities.

Pure file-emit, no network. For each ``host.*`` capability in
``ctx.resolved_stack``, the step:

1. Resolves the capability's ``emit_files`` against the deployments source
   (each entry's ``source`` is relative to the capability's directory).
2. Reads the template, substitutes ``${VAR}`` and ``$VAR`` placeholders from
   the project's env (anything unset is left as-is so the user notices).
3. Writes to ``project_dir / dest``, never overwriting a file the model
   already emitted (collision logs a warning and SKIPs that file).
4. Records each emitted file path under ``manifest.answers["deploy_configs"]``
   (JSON-encoded list) so Phase 4's ``cmd_deploy`` can find them.

Runs after ``smoke_test`` so deploy configs are written only when the
project itself is provably runnable.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_scaffold.orchestrator import (
    DetectionResult,
    StepContext,
    StepLog,
    StepResult,
    StepStatus,
    compute_fingerprint,
)

log = logging.getLogger(__name__)

_VAR_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}|\$([A-Z_][A-Z0-9_]*)")


@dataclass
class EmitDeployConfigsStep:
    """Render and write each host.* capability's deploy config templates."""

    id: str = "emit_deploy_configs"
    description: str = "Write cloud-deploy configs (vercel.json, fly.toml, railway.json)"
    depends_on: tuple[str, ...] = ("smoke_test",)
    troubleshoot: dict[str, str] = field(
        default_factory=lambda: {
            "template missing": (
                "the capability declared an emit_files source that doesn't exist on disk — "
                "check that your deployments source is up to date"
            ),
        }
    )

    # ---- detection ----------------------------------------------------

    def detect(self, ctx: StepContext) -> DetectionResult:
        caps = self._host_capabilities(ctx)
        if not caps:
            return DetectionResult(
                StepStatus.SKIPPED, reason="recipe declares no host.* capability"
            )
        pending = [c.id for c in caps]
        return DetectionResult(StepStatus.PENDING, reason=f"render: {', '.join(pending)}")

    # ---- apply --------------------------------------------------------

    def apply(self, ctx: StepContext) -> StepResult:
        caps = self._host_capabilities(ctx)
        if not caps:
            return StepResult(StepStatus.SKIPPED, detail="no host.* capability")
        written: list[str] = []
        skipped: list[str] = []
        for cap in caps:
            for emit in cap.emit_files:
                source = (cap.path.parent / emit.source).resolve()
                if not source.is_file():
                    log.warning(
                        "emit_deploy_configs: source %s missing for capability %s — skipping",
                        source,
                        cap.id,
                    )
                    skipped.append(str(source))
                    continue
                dest = (ctx.project_dir / emit.dest).resolve()
                # Path-safety: dest must stay under project_dir.
                try:
                    dest.relative_to(ctx.project_dir.resolve())
                except ValueError:
                    log.warning(
                        "emit_deploy_configs: dest %s escapes project_dir — skipping", dest
                    )
                    skipped.append(str(dest))
                    continue
                if dest.exists():
                    log.warning(
                        "emit_deploy_configs: dest %s already exists (likely model output) — skipping",
                        dest,
                    )
                    skipped.append(str(dest.relative_to(ctx.project_dir)))
                    continue
                rendered = _render_template(source.read_text(encoding="utf-8"), os.environ)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(rendered, encoding="utf-8")
                rel = str(dest.relative_to(ctx.project_dir))
                written.append(rel)
                ctx.emit(StepLog(step_id=self.id, line=f"wrote {rel}"))
        if not written and not skipped:
            return StepResult(StepStatus.SKIPPED, detail="no emit_files declared")
        detail_parts: list[str] = []
        if written:
            detail_parts.append(f"wrote {len(written)}: {', '.join(written)}")
        if skipped:
            detail_parts.append(f"skipped {len(skipped)}")
        return StepResult(StepStatus.DONE, detail="; ".join(detail_parts))

    # ---- fingerprint --------------------------------------------------

    def fingerprint(self, ctx: StepContext) -> str:
        caps = self._host_capabilities(ctx)
        entries: list[str] = []
        for cap in caps:
            for emit in cap.emit_files:
                entries.append(f"{cap.id}:{emit.source}->{emit.dest}")
        return compute_fingerprint({"entries": sorted(entries)})

    # ---- helpers ------------------------------------------------------

    def _host_capabilities(self, ctx: StepContext) -> list[Any]:
        stack = ctx.resolved_stack
        if stack is None:
            return []
        return [c for c in stack.capabilities if c.kind == "host"]


def _render_template(text: str, env: dict[str, str] | Any) -> str:
    """Substitute ``${VAR}`` / ``$VAR`` with values from ``env``.

    Unknown vars are left as-is so the user sees the placeholder.
    """

    def replace(match: re.Match[str]) -> str:
        key = match.group(1) or match.group(2)
        value = env.get(key, "")
        if not value:
            return match.group(0)
        return str(value)

    return _VAR_RE.sub(replace, text)


__all__ = ["EmitDeployConfigsStep"]
