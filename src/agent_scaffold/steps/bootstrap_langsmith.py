"""``bootstrap_langsmith`` step: create LangSmith project + write tracing env.

Runs after ``wire_credentials`` so ``LANGCHAIN_API_KEY`` is present (either
in env or via the project's ``.env.local``). For the ``obs.langsmith``
capability:

1. Resolve the project name from ``manifest.answers["langsmith_project"]``
   (falls back to ``manifest.recipe``).
2. Call ``Client.read_project(project_name=...)``; if that 404s,
   ``create_project()``.
3. Append the three canonical tracing env vars to ``<project>/.env.local``:
   ``LANGCHAIN_TRACING_V2``, ``LANGCHAIN_PROJECT``, ``LANGCHAIN_ENDPOINT``.

Skips cleanly when:

- No ``obs.langsmith`` capability on the recipe.
- ``langsmith`` SDK isn't installed (optional dep via ``[obs]`` extra).
- ``LANGCHAIN_API_KEY`` isn't reachable — the ``wire_credentials`` step
  failure has already surfaced; we don't double-report.
"""

from __future__ import annotations

import os
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

_ENV_LOCAL = ".env.local"
_DEFAULT_ENDPOINT = "https://api.smith.langchain.com"


@dataclass
class BootstrapLangSmithStep:
    """Create the LangSmith project + write tracing env vars."""

    id: str = "bootstrap_langsmith"
    description: str = "Create LangSmith project + write tracing env vars"
    depends_on: tuple[str, ...] = ("wire_credentials",)
    troubleshoot: dict[str, str] = field(
        default_factory=lambda: {
            "401": (
                "API key rejected — rotate LANGCHAIN_API_KEY at "
                "https://smith.langchain.com/settings"
            ),
            "Unauthorized": (
                "API key rejected — rotate LANGCHAIN_API_KEY at "
                "https://smith.langchain.com/settings"
            ),
            "ImportError": ('install the obs extra: pip install "agent-scaffold-cli[obs]"'),
        }
    )

    # ---- detection ----------------------------------------------------

    def detect(self, ctx: StepContext) -> DetectionResult:
        if not self._has_langsmith_capability(ctx):
            return DetectionResult(
                StepStatus.SKIPPED, reason="recipe declares no obs.langsmith capability"
            )
        api_key = os.environ.get("LANGCHAIN_API_KEY", "").strip()
        if not api_key:
            return DetectionResult(
                StepStatus.SKIPPED,
                reason="LANGCHAIN_API_KEY not set; re-run with --retry wire_credentials",
            )
        return DetectionResult(
            StepStatus.PENDING,
            reason=f"ensure project {_project_name(ctx)!r} exists",
        )

    # ---- apply --------------------------------------------------------

    def apply(self, ctx: StepContext) -> StepResult:
        if not self._has_langsmith_capability(ctx):
            return StepResult(StepStatus.SKIPPED, detail="no obs.langsmith capability")
        api_key = os.environ.get("LANGCHAIN_API_KEY", "").strip()
        if not api_key:
            return StepResult(
                StepStatus.SKIPPED,
                detail="LANGCHAIN_API_KEY not set",
            )
        try:
            from langsmith import Client
        except ImportError:
            return StepResult(
                StepStatus.SKIPPED,
                detail='langsmith SDK not installed (pip install "agent-scaffold-cli[obs]")',
            )
        project_name = _project_name(ctx)
        endpoint = os.environ.get("LANGCHAIN_ENDPOINT", _DEFAULT_ENDPOINT)
        try:
            client = Client(api_key=api_key, api_url=endpoint)
        except Exception as exc:  # noqa: BLE001
            return StepResult(
                StepStatus.FAILED,
                error=f"langsmith: Client init failed: {exc}",
            )
        action, error = _ensure_project(client, project_name)
        if error is not None:
            return StepResult(StepStatus.FAILED, error=error)
        ctx.emit(
            StepLog(
                step_id=self.id,
                line=f"langsmith: project {project_name!r} {action}",
            )
        )
        written = _write_tracing_env(ctx.project_dir, project_name, endpoint)
        return StepResult(
            StepStatus.DONE,
            detail=(
                f"project {action}, wrote {written} env var(s) to .env.local"
                if written
                else f"project {action}, env vars already in .env.local"
            ),
        )

    # ---- fingerprint --------------------------------------------------

    def fingerprint(self, ctx: StepContext) -> str:
        return compute_fingerprint(
            {
                "has_capability": self._has_langsmith_capability(ctx),
                "project": _project_name(ctx),
                "endpoint": os.environ.get("LANGCHAIN_ENDPOINT", _DEFAULT_ENDPOINT),
            }
        )

    # ---- helpers ------------------------------------------------------

    def _has_langsmith_capability(self, ctx: StepContext) -> bool:
        stack = ctx.resolved_stack
        if stack is None:
            return False
        return any(c.id == "obs.langsmith" for c in stack.capabilities)


def _project_name(ctx: StepContext) -> str:
    answers = ctx.manifest.answers if ctx.manifest else {}
    return answers.get("langsmith_project") or answers.get("project_name") or ctx.manifest.recipe


def _ensure_project(client: Any, project_name: str) -> tuple[str, str | None]:
    """Return ``(action, error)``. ``action`` is ``"exists"`` or ``"created"``."""
    try:
        client.read_project(project_name=project_name)
        return ("exists", None)
    except Exception as exc:  # noqa: BLE001 — SDK raises a typed not-found we can't import without coupling
        if not _is_not_found(exc):
            return ("error", f"langsmith: read_project failed: {exc}")
    try:
        client.create_project(project_name=project_name)
    except Exception as exc:  # noqa: BLE001
        return ("error", f"langsmith: create_project failed: {exc}")
    return ("created", None)


def _is_not_found(exc: BaseException) -> bool:
    """Heuristic: detect LangSmith's "not found" without importing its private types."""
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return "notfound" in name or "404" in text or "not found" in text


def _write_tracing_env(project_dir: Path, project_name: str, endpoint: str) -> int:
    """Idempotently append three LANGCHAIN_* vars to ``.env.local``.

    Returns the number of vars actually written (0 if all were already present).
    """
    target = project_dir / _ENV_LOCAL
    existing = target.read_text(encoding="utf-8") if target.is_file() else ""
    desired = {
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_PROJECT": project_name,
        "LANGCHAIN_ENDPOINT": endpoint,
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


__all__ = ["BootstrapLangSmithStep"]
