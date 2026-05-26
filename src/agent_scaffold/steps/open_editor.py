"""``open_editor`` step: drop the user into ``$EDITOR ./README.md`` when ``up`` finishes.

Cosmetic — no side effect inside the project. The whole point is to leave
the developer pointed at the obvious "what next" surface (README.md) when
provisioning is done.

Skipped in ``--yes`` mode: an editor in CI is never what we want.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field

from agent_scaffold.orchestrator import (
    DetectionResult,
    StepContext,
    StepResult,
    StepStatus,
    compute_fingerprint,
)

_FALLBACK_EDITORS: tuple[str, ...] = ("code", "cursor", "nano", "vim")


@dataclass
class OpenEditorStep:
    """Open ``README.md`` in the resolved editor; no-op in non-interactive runs."""

    id: str = "open_editor"
    description: str = "Open README in $EDITOR"
    depends_on: tuple[str, ...] = ()
    # CLI sets this when invoked with --yes so we skip silently in CI.
    yes: bool = False
    # Allows tests to inject a fake $EDITOR resolution.
    editor_override: str | None = None
    troubleshoot: dict[str, str] = field(default_factory=dict)

    # ---- detection ----------------------------------------------------

    def detect(self, ctx: StepContext) -> DetectionResult:
        if self.yes:
            return DetectionResult(
                StepStatus.SKIPPED,
                reason="--yes mode — never opens an editor",
            )
        editor = self._resolve_editor()
        if editor is None:
            return DetectionResult(
                StepStatus.SKIPPED,
                reason="$EDITOR unset and no fallback (code/cursor/nano/vim) on PATH",
            )
        readme = ctx.project_dir / "README.md"
        if not readme.is_file():
            return DetectionResult(
                StepStatus.SKIPPED,
                reason="no README.md in the project — nothing to open",
            )
        return DetectionResult(StepStatus.PENDING, reason=f"will open {readme.name} in {editor}")

    # ---- apply --------------------------------------------------------

    def apply(self, ctx: StepContext) -> StepResult:
        if self.yes:
            return StepResult(StepStatus.SKIPPED, detail="--yes mode")
        editor = self._resolve_editor()
        if editor is None:
            return StepResult(StepStatus.SKIPPED, detail="no editor resolved")
        readme = ctx.project_dir / "README.md"
        if not readme.is_file():
            return StepResult(StepStatus.SKIPPED, detail="no README.md")
        try:
            # ``shlex.split`` so $EDITOR can carry flags (e.g. ``code -n``).
            import shlex

            cmd = [*shlex.split(editor), str(readme)]
            proc = subprocess.run(cmd, check=False, shell=False)  # noqa: S603 — list-form
        except (FileNotFoundError, OSError) as exc:
            return StepResult(
                StepStatus.FAILED,
                error=f"failed to invoke editor: {type(exc).__name__}: {exc}",
            )
        if proc.returncode != 0:
            # Don't fail the run if the editor itself returns non-zero — many
            # GUI editors return immediately and the user's actual edit happens
            # asynchronously.
            return StepResult(
                StepStatus.DONE,
                detail=f"editor exited with {proc.returncode} (treated as ok)",
            )
        return StepResult(StepStatus.DONE, detail=f"opened {readme.name} in {editor.split()[0]}")

    # ---- fingerprint --------------------------------------------------

    def fingerprint(self, ctx: StepContext) -> str:
        return compute_fingerprint(
            {
                "editor": self._resolve_editor() or "",
                "readme_exists": (ctx.project_dir / "README.md").is_file(),
            }
        )

    # ---- helpers ------------------------------------------------------

    def _resolve_editor(self) -> str | None:
        if self.editor_override is not None:
            return self.editor_override
        env_editor = os.environ.get("EDITOR", "").strip() or os.environ.get("VISUAL", "").strip()
        if env_editor:
            # Validate the first token actually exists on PATH so we don't
            # subprocess.run a typo.
            head = env_editor.split()[0]
            if shutil.which(head) is not None:
                return env_editor
        for candidate in _FALLBACK_EDITORS:
            if shutil.which(candidate) is not None:
                return candidate
        return None


__all__: Sequence[str] = ["OpenEditorStep"]
