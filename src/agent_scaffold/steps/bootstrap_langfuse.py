"""``bootstrap_langfuse`` step: persist LANGFUSE_* env vars after web-UI provisioning.

Langfuse's first-run flow is interactive — the user visits ``LANGFUSE_HOST``
in a browser, creates the org + project, and copies a public/secret key
pair. Unlike LangSmith there's no programmatic ``create_project`` we can
call without UI-provisioning hacks, so this step's job is narrow:

1. Confirm the keys are reachable in the environment (the
   ``wire_credentials`` step prompts for them and stores via keyring +
   ``.env.local``).
2. Append the canonical ``LANGFUSE_*`` triple to ``.env.local``
   idempotently so the generated app picks them up at runtime.

Skips cleanly when:

- No ``obs.langfuse`` capability on the recipe.
- ``LANGFUSE_PUBLIC_KEY`` or ``LANGFUSE_SECRET_KEY`` isn't in env (a
  ``wire_credentials`` failure has already surfaced; don't double-report).
"""

from __future__ import annotations

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

_ENV_LOCAL = ".env.local"
_DEFAULT_HOST = "http://localhost:3001"


@dataclass
class BootstrapLangfuseStep:
    """Persist LANGFUSE_* env vars after the user provisions a project via UI."""

    id: str = "bootstrap_langfuse"
    description: str = "Write Langfuse tracing env vars to .env.local"
    depends_on: tuple[str, ...] = ("wire_credentials",)
    troubleshoot: dict[str, str] = field(
        default_factory=lambda: {
            "401": "API key rejected — rotate keys at LANGFUSE_HOST web UI",
            "missing": (
                "create the project in the Langfuse UI, then re-run "
                "with --retry wire_credentials"
            ),
        }
    )

    # ---- detection ----------------------------------------------------

    def detect(self, ctx: StepContext) -> DetectionResult:
        if not self._has_capability(ctx):
            return DetectionResult(
                StepStatus.SKIPPED, reason="recipe declares no obs.langfuse capability"
            )
        if not _both_keys_present(ctx):
            return DetectionResult(
                StepStatus.SKIPPED,
                reason=(
                    "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set — "
                    "create project at LANGFUSE_HOST, then --retry wire_credentials"
                ),
            )
        return DetectionResult(
            StepStatus.PENDING, reason="persist LANGFUSE_* env vars to .env.local"
        )

    # ---- apply --------------------------------------------------------

    def apply(self, ctx: StepContext) -> StepResult:
        if not self._has_capability(ctx):
            return StepResult(StepStatus.SKIPPED, detail="no obs.langfuse capability")
        if not _both_keys_present(ctx):
            return StepResult(
                StepStatus.SKIPPED,
                detail="LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set",
            )
        host = ctx.env_get("LANGFUSE_HOST", _DEFAULT_HOST)
        public_key = ctx.env_get("LANGFUSE_PUBLIC_KEY")
        secret_key = ctx.env_get("LANGFUSE_SECRET_KEY")
        written = _write_tracing_env(ctx.project_dir, host, public_key, secret_key)
        ctx.emit(
            StepLog(
                step_id=self.id,
                line=f"langfuse: env wired against {host}",
            )
        )
        return StepResult(
            StepStatus.DONE,
            detail=(
                f"wrote {written} env var(s) to .env.local"
                if written
                else "env vars already in .env.local"
            ),
        )

    # ---- fingerprint --------------------------------------------------

    def fingerprint(self, ctx: StepContext) -> str:
        return compute_fingerprint(
            {
                "has_capability": self._has_capability(ctx),
                "host": ctx.env_get("LANGFUSE_HOST", _DEFAULT_HOST),
            }
        )

    # ---- helpers ------------------------------------------------------

    def _has_capability(self, ctx: StepContext) -> bool:
        stack = ctx.resolved_stack
        if stack is None:
            return False
        return any(c.id == "obs.langfuse" for c in stack.capabilities)


def _both_keys_present(ctx: StepContext) -> bool:
    # ctx.env_get resolves runtime_env (vault-aware) before os.environ, so
    # keys stored by wire_credentials in the project vault are seen here.
    return bool(ctx.env_get("LANGFUSE_PUBLIC_KEY") and ctx.env_get("LANGFUSE_SECRET_KEY"))


def _write_tracing_env(project_dir: Path, host: str, public_key: str, secret_key: str) -> int:
    """Idempotently append three LANGFUSE_* vars to ``.env.local``.

    Returns the number of vars actually written (0 if all were already present).
    Mirrors ``bootstrap_langsmith._write_tracing_env`` — kept independent so
    the two steps don't import each other.
    """
    target = project_dir / _ENV_LOCAL
    existing = target.read_text(encoding="utf-8") if target.is_file() else ""
    desired = {
        "LANGFUSE_HOST": host,
        "LANGFUSE_PUBLIC_KEY": public_key,
        "LANGFUSE_SECRET_KEY": secret_key,
    }
    to_add: list[str] = []
    for key, value in desired.items():
        if not _env_key_present(existing, key):
            to_add.append(f"{key}={value}")
    if not to_add:
        return 0
    prefix = "" if not existing or existing.endswith("\n") else "\n"
    body = existing + prefix + "\n".join(to_add) + "\n"
    target.write_text(body, encoding="utf-8")
    try:
        target.chmod(0o600)
    except OSError:
        pass
    return len(to_add)


def _env_key_present(env_text: str, key: str) -> bool:
    for line in env_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        existing_key, _, _ = stripped.partition("=")
        if existing_key.strip() == key:
            return True
    return False


__all__ = ["BootstrapLangfuseStep"]
