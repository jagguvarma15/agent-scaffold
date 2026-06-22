"""``open_editor`` step: open ``./README.md`` in a GUI editor when ``up`` finishes.

Cosmetic — no side effect inside the project. The whole point is to leave the
developer pointed at the obvious "what next" surface (README.md) when
provisioning is done.

Two hard rules, both learned the hard way:

* **Never block.** The editor is spawned detached (fire-and-forget); we never
  ``wait()`` for it. Waiting is what froze ``up`` for minutes when ``$EDITOR``
  resolved to a terminal editor that owns the foreground until you quit it —
  and which the provisioning Live panel made impossible to even see or quit.
* **Only GUI editors.** A terminal editor (vim, nano, …) can't run here: the
  Live panel owns the TTY, so it would garble or hang. We skip those with a
  hint instead of launching something unusable. GUI editors (code, cursor, …)
  open a window and let provisioning move on.

Skipped in ``--yes`` mode: an editor in CI is never what we want.
"""

from __future__ import annotations

import os
import shlex
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

# GUI editors we'll auto-launch as a detached window. Terminal editors are
# deliberately absent — see ``_TERMINAL_EDITORS``.
_FALLBACK_EDITORS: tuple[str, ...] = ("code", "cursor", "windsurf", "zed", "subl")

# Editors that need the foreground terminal. We refuse to auto-launch these:
# during provisioning the Live panel owns the TTY, so a terminal editor either
# garbles the display or — with a blocking launch — hangs the whole run until
# you manage to quit it (which you can't, because you can't see it).
_TERMINAL_EDITORS: frozenset[str] = frozenset(
    {
        "vi", "vim", "nvim", "nano", "pico", "emacs", "emacsclient",
        "ed", "micro", "helix", "hx", "kak", "joe", "ne", "vis",
    }
)  # fmt: skip


def _editor_name(editor_cmd: str) -> str:
    """The bare program name from an editor command (``'code -n'`` -> ``'code'``)."""
    tokens = editor_cmd.split()
    head = tokens[0] if tokens else editor_cmd
    name = os.path.basename(head)
    return name[:-4] if name.lower().endswith(".exe") else name


def _is_terminal_editor(editor_cmd: str) -> bool:
    """True if the command is a terminal editor we must not auto-launch here."""
    return _editor_name(editor_cmd).lower() in _TERMINAL_EDITORS


@dataclass
class OpenEditorStep:
    """Open ``README.md`` in the resolved GUI editor; no-op in non-interactive runs."""

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
                reason="$EDITOR unset and no GUI fallback (code/cursor/windsurf/zed/subl) on PATH",
            )
        if _is_terminal_editor(editor):
            return DetectionResult(
                StepStatus.SKIPPED,
                reason=f"{_editor_name(editor)} is a terminal editor — open the README yourself",
            )
        readme = ctx.project_dir / "README.md"
        if not readme.is_file():
            return DetectionResult(
                StepStatus.SKIPPED,
                reason="no README.md in the project — nothing to open",
            )
        return DetectionResult(
            StepStatus.PENDING, reason=f"will open {readme.name} in {_editor_name(editor)}"
        )

    # ---- apply --------------------------------------------------------

    def apply(self, ctx: StepContext) -> StepResult:
        if self.yes:
            return StepResult(StepStatus.SKIPPED, detail="--yes mode")
        editor = self._resolve_editor()
        if editor is None:
            return StepResult(StepStatus.SKIPPED, detail="no editor resolved")
        if _is_terminal_editor(editor):
            # Refusing on purpose: a terminal editor here can't render (the Live
            # panel owns the TTY) and the old blocking launch hung the run.
            return StepResult(
                StepStatus.SKIPPED,
                detail=f"{_editor_name(editor)} is a terminal editor — skipped so it can't block",
            )
        readme = ctx.project_dir / "README.md"
        if not readme.is_file():
            return StepResult(StepStatus.SKIPPED, detail="no README.md")

        # ``shlex.split`` so $EDITOR can carry flags (e.g. ``code -n``).
        cmd = [*shlex.split(editor), str(readme)]
        try:
            # Detached + fire-and-forget: open the window and return immediately.
            # No ``wait()``, no returncode check — blocking here is the bug this
            # step exists to avoid. stdio -> /dev/null and a new session keep the
            # editor fully decoupled from our terminal and the Live panel.
            subprocess.Popen(  # noqa: S603 — list-form, shell=False
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except (FileNotFoundError, OSError) as exc:
            # Best-effort + cosmetic: a launch failure must never fail the run.
            return StepResult(
                StepStatus.SKIPPED,
                detail=f"couldn't launch {_editor_name(editor)} ({type(exc).__name__}) — open the README yourself",
            )
        return StepResult(StepStatus.DONE, detail=f"opened {readme.name} in {_editor_name(editor)}")

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
            # spawn a typo.
            head = env_editor.split()[0]
            if shutil.which(head) is not None:
                return env_editor
        for candidate in _FALLBACK_EDITORS:
            if shutil.which(candidate) is not None:
                return candidate
        return None


__all__: Sequence[str] = ["OpenEditorStep"]
