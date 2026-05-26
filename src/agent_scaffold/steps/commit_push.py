"""``commit_push`` step: opt-in commit of provisioning artifacts + push.

Visible-action safety rules (load-bearing — see acceptance criteria on Q7):

- **Always prompts before commit AND before push**, even with ``--yes``.
  ``--yes`` only suppresses the *plan* confirmation; mutating remote state
  requires ``--yes --confirm-commit-push``.
- **Allowlist for ``git add``** — never ``git add .``. We only stage files
  this provisioning flow itself touches:

    .scaffold/manifest.json
    .scaffold/state.json (only when no FAILED entries)
    .env.example (only when present; useful template)
    .gitignore (only when present and we just ensured an entry)

  ``.env.local`` is never staged.
- **Never ``--no-verify``.** If the pre-commit hook fails, the step fails
  with the hook's output surfaced — fix the underlying issue, don't bypass.
- Push targets ``origin/<current-branch>`` only; non-``origin`` remotes are
  out of scope per Q7.

This step is **default OFF**: ``default_steps_for`` only includes it when
``setup_steps`` opt-in is configured, or when the user passes
``--only commit_push``.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
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

_DEFAULT_TIMEOUT = 60.0
_COMMIT_MESSAGE = "chore: agent-scaffold up — initial provisioning"

# Strict allowlist. ``git add`` will silently no-op on any path that doesn't
# exist, so we filter to real files before invoking it.
_ALLOWED_PATHS: tuple[str, ...] = (
    ".scaffold/manifest.json",
    ".scaffold/state.json",
    ".env.example",
    ".gitignore",
)


@dataclass
class CommitPushStep:
    """Commit provisioning artifacts and push to origin — opt-in only."""

    id: str = "commit_push"
    description: str = "Commit provisioning state and push to origin (opt-in)"
    depends_on: tuple[str, ...] = ("smoke_test",)
    # Wired by the CLI from ``--yes --confirm-commit-push``. False by default
    # means we always prompt — even when the wider run is non-interactive.
    confirm_commit_push: bool = False
    troubleshoot: dict[str, str] = field(
        default_factory=lambda: {
            "fatal: not a git repository": (
                "run `git init` first, or skip this step with `--skip commit_push`"
            ),
            "non-fast-forward": ("remote has new commits — `git pull --rebase` first, then retry"),
            "Permission denied (publickey)": (
                "SSH key not registered with origin host — check `ssh -T git@<host>`"
            ),
            "pre-commit": (
                "pre-commit hook failed — fix the issue and re-run " "(DO NOT pass --no-verify)"
            ),
            "nothing to commit": (
                "no allowed paths changed since the last commit — skip with `--skip commit_push`"
            ),
        }
    )

    # ---- detection ----------------------------------------------------

    def detect(self, ctx: StepContext) -> DetectionResult:
        if not _is_git_repo(ctx.project_dir):
            return DetectionResult(StepStatus.SKIPPED, reason="not a git repository")
        dirty = _dirty_allowlisted_paths(ctx.project_dir)
        if not dirty:
            return DetectionResult(
                StepStatus.SKIPPED,
                reason="no allow-listed paths changed; nothing to commit",
            )
        return DetectionResult(
            StepStatus.PENDING,
            reason=f"would stage {len(dirty)} file(s): {', '.join(dirty)}",
        )

    # ---- apply --------------------------------------------------------

    def apply(self, ctx: StepContext) -> StepResult:
        if not _is_git_repo(ctx.project_dir):
            return StepResult(StepStatus.SKIPPED, detail="not a git repo")
        dirty = _dirty_allowlisted_paths(ctx.project_dir)
        if not dirty:
            return StepResult(StepStatus.SKIPPED, detail="nothing to commit")

        # State-file safety: refuse to commit ``state.json`` if any step is
        # currently FAILED — stderr tails may carry hostnames or paths.
        safe_dirty = self._strip_unsafe_state_json(ctx, dirty)

        ctx.emit(
            StepLog(
                step_id=self.id,
                line=f"would commit: {', '.join(safe_dirty)}",
                stream="stdout",
            )
        )
        if not self._confirm("Commit these files?"):
            return StepResult(StepStatus.SKIPPED, detail="user declined commit")

        rc = _run_git(["add", *safe_dirty], cwd=ctx.project_dir)
        if rc.returncode != 0:
            return StepResult(
                StepStatus.FAILED,
                error=f"git add failed (exit {rc.returncode})",
                stderr_tail=rc.stderr.strip(),
            )
        commit = _run_git(
            ["commit", "-m", _COMMIT_MESSAGE],
            cwd=ctx.project_dir,
        )
        if commit.returncode != 0:
            return StepResult(
                StepStatus.FAILED,
                error=f"git commit failed (exit {commit.returncode})",
                stderr_tail=(commit.stderr or commit.stdout or "").strip(),
            )

        if not _has_origin_remote(ctx.project_dir):
            return StepResult(
                StepStatus.DONE,
                detail=f"committed {len(safe_dirty)} file(s); no `origin` remote to push to",
            )
        branch = _current_branch(ctx.project_dir)
        if not self._confirm(f"Push to origin/{branch}?"):
            return StepResult(
                StepStatus.DONE,
                detail=f"committed {len(safe_dirty)} file(s); push declined",
            )
        push = _run_git(["push", "origin", branch], cwd=ctx.project_dir)
        if push.returncode != 0:
            return StepResult(
                StepStatus.FAILED,
                error=f"git push failed (exit {push.returncode})",
                stderr_tail=(push.stderr or "").strip(),
            )
        return StepResult(
            StepStatus.DONE,
            detail=f"committed + pushed {len(safe_dirty)} file(s) to origin/{branch}",
        )

    # ---- fingerprint --------------------------------------------------

    def fingerprint(self, ctx: StepContext) -> str:
        return compute_fingerprint(
            {
                "head": _current_head(ctx.project_dir),
                "branch": _current_branch(ctx.project_dir),
            }
        )

    # ---- helpers ------------------------------------------------------

    def _confirm(self, prompt: str) -> bool:
        """Always prompt unless ``--yes --confirm-commit-push`` was set."""
        if self.confirm_commit_push:
            return True
        try:
            answer = input(f"{prompt} [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer in ("y", "yes")

    def _strip_unsafe_state_json(self, ctx: StepContext, paths: list[str]) -> list[str]:
        if ".scaffold/state.json" not in paths:
            return paths
        state_path = ctx.project_dir / ".scaffold" / "state.json"
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return [p for p in paths if p != ".scaffold/state.json"]
        steps = payload.get("steps") or {}
        if any((s.get("status") == "failed") for s in steps.values() if isinstance(s, dict)):
            return [p for p in paths if p != ".scaffold/state.json"]
        return paths


def _is_git_repo(project_dir: Path) -> bool:
    return _run_git(["rev-parse", "--git-dir"], cwd=project_dir).returncode == 0


def _dirty_allowlisted_paths(project_dir: Path) -> list[str]:
    """``git status --porcelain`` filtered to the allow-listed paths."""
    proc = _run_git(["status", "--porcelain", "--", *_ALLOWED_PATHS], cwd=project_dir)
    if proc.returncode != 0:
        return []
    found: list[str] = []
    for line in (proc.stdout or "").splitlines():
        if len(line) < 4:
            continue
        rel = line[3:].strip()
        if rel in _ALLOWED_PATHS:
            found.append(rel)
    return found


def _has_origin_remote(project_dir: Path) -> bool:
    proc = _run_git(["remote"], cwd=project_dir)
    if proc.returncode != 0:
        return False
    return "origin" in (proc.stdout or "").split()


def _current_branch(project_dir: Path) -> str:
    proc = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=project_dir)
    return (proc.stdout or "").strip() or "HEAD"


def _current_head(project_dir: Path) -> str:
    proc = _run_git(["rev-parse", "HEAD"], cwd=project_dir)
    return (proc.stdout or "").strip()


def _run_git(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run git with a short timeout — every call is local-only."""
    try:
        return subprocess.run(  # noqa: S603 — list-form, shell=False, git only
            ["git", *args],
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
            timeout=_DEFAULT_TIMEOUT,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        # Return a synthetic completed-process so callers can branch on rc.
        return subprocess.CompletedProcess(
            args=["git", *args],
            returncode=-1,
            stdout="",
            stderr=f"{type(exc).__name__}: {exc}",
        )


__all__: Sequence[str] = ["CommitPushStep"]
