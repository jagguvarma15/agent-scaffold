"""``wire_credentials`` step: prompt for missing env vars, store safely.

For each ``external_service`` declared on the recipe:

- Look at ``service.env_vars``. Any entry already set in ``os.environ`` (or
  resolvable through Q2's ``auth.load_key`` taxonomy for Anthropic) is
  treated as present.
- Anything missing is collected. In interactive mode we prompt via
  :func:`getpass.getpass` (so the secret never echoes), validate against
  the service's probe when one exists, and persist to the right backend:

  - ``ANTHROPIC_API_KEY`` → keyring via :func:`auth.store_key`.
  - Everything else → ``.env.local`` next to the project (mode 0600), with
    ``.env.local`` ensured in ``.gitignore``.

- In ``--yes`` (non-interactive) mode we never silently skip a required
  secret: missing values cause the step to FAIL with the list of vars the
  user needs to export beforehand. Optional services (``required=False``)
  emit a warning log line and are left empty.
"""

from __future__ import annotations

import getpass
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import SecretStr

from agent_scaffold.auth import ENV_API_KEY, AuthError, load_key, store_key
from agent_scaffold.discovery import (
    DiscoveryError,
    ExternalService,
    Recipe,
    discover_recipes,
)
from agent_scaffold.orchestrator import (
    DetectionResult,
    StepContext,
    StepLog,
    StepProgress,
    StepResult,
    StepStatus,
    compute_fingerprint,
)

# Env vars that resolve via Q2's auth backends rather than the local .env.
_KEYRING_BACKED_ENV_VARS: frozenset[str] = frozenset({ENV_API_KEY})


@dataclass
class _MissingSecret:
    env_var: str
    service: ExternalService
    required: bool


@dataclass
class WireCredentialsStep:
    """Prompt for missing API keys; persist to keyring or ``.env.local``."""

    id: str = "wire_credentials"
    description: str = "Prompt for missing API keys and store securely"
    depends_on: tuple[str, ...] = ()
    # Honor the orchestrator ``--yes`` toggle. The CLI overlays this from
    # ``StepFlags`` before constructing the orchestrator.
    yes: bool = False
    troubleshoot: dict[str, str] = field(
        default_factory=lambda: {
            "401": "key was rejected by the service — paste a fresh credential",
            "unauthorized": "key was rejected by the service — paste a fresh credential",
            "keyring rejected": (
                "the OS keyring refused the write — pass --use-file on `agent-scaffold "
                "auth login` to fall back to the mode-0600 file backend"
            ),
            "validation failed": (
                "key rejected by provider — re-check the key in the provider's dashboard"
            ),
        }
    )

    # ---- detection ----------------------------------------------------

    def detect(self, ctx: StepContext) -> DetectionResult:
        missing = self._missing_secrets(ctx)
        if not missing:
            return DetectionResult(StepStatus.DONE, reason="all declared env vars resolvable")
        required = [m.env_var for m in missing if m.required]
        if not required:
            optional = [m.env_var for m in missing if not m.required]
            return DetectionResult(
                StepStatus.PARTIAL,
                reason=f"optional only ({len(optional)}): {', '.join(optional)}",
            )
        return DetectionResult(
            StepStatus.PENDING,
            reason=f"need {len(required)} secret(s): {', '.join(required)}",
        )

    # ---- apply --------------------------------------------------------

    def apply(self, ctx: StepContext) -> StepResult:
        missing = self._missing_secrets(ctx)
        if not missing:
            return StepResult(StepStatus.DONE, detail="all required env vars present")

        if self.yes:
            required = [m.env_var for m in missing if m.required]
            if required:
                return StepResult(
                    status=StepStatus.FAILED,
                    error=(
                        f"--yes: missing required secret(s): {', '.join(required)}; "
                        "set them via environment or run interactively"
                    ),
                )
            # Optional-only in --yes is fine: leave them unset, surface a warning.
            for m in missing:
                ctx.emit(
                    StepLog(
                        step_id=self.id,
                        line=f"--yes: leaving optional {m.env_var} unset",
                        stream="stderr",
                    )
                )
            return StepResult(StepStatus.DONE, detail="no required secrets missing")

        if not sys.stdin.isatty():
            required = [m.env_var for m in missing if m.required]
            if required:
                return StepResult(
                    status=StepStatus.FAILED,
                    error=(
                        f"no TTY: cannot prompt for {', '.join(required)}; "
                        "set them via environment or pass --yes after exporting"
                    ),
                )

        wired = 0
        for m in missing:
            ctx.emit(
                StepProgress(
                    step_id=self.id,
                    message=f"prompting for {m.env_var}",
                )
            )
            try:
                raw = getpass.getpass(f"Enter {m.env_var} ({m.service.id}): ")
            except (EOFError, KeyboardInterrupt) as exc:
                return StepResult(
                    status=StepStatus.FAILED,
                    error=f"prompt aborted at {m.env_var}: {type(exc).__name__}",
                )
            raw = raw.strip()
            if not raw:
                if m.required:
                    return StepResult(
                        status=StepStatus.FAILED, error=f"empty value for required {m.env_var}"
                    )
                ctx.emit(
                    StepLog(
                        step_id=self.id,
                        line=f"skipped optional {m.env_var}",
                        stream="stdout",
                    )
                )
                continue
            secret = SecretStr(raw)
            backend_label = self._persist(m.env_var, secret, ctx)
            if backend_label is None:
                return StepResult(
                    status=StepStatus.FAILED,
                    error=f"failed to store {m.env_var}",
                )
            wired += 1
            # NEVER log the raw value; the progress payload carries only the var name.
            ctx.emit(
                StepProgress(
                    step_id=self.id,
                    message=f"{m.env_var} → stored ({backend_label})",
                )
            )

        return StepResult(StepStatus.DONE, detail=f"{wired} secret(s) wired")

    # ---- fingerprint --------------------------------------------------

    def fingerprint(self, ctx: StepContext) -> str:
        recipe = _load_recipe(ctx)
        env_vars = sorted(
            {v for svc in (recipe.external_services if recipe else []) for v in svc.env_vars}
        )
        return compute_fingerprint(
            {
                "env_vars": env_vars,
                # Cheap presence-only fingerprint: never include the values.
                "env_set": sorted(v for v in env_vars if os.environ.get(v)),
            }
        )

    # ---- helpers ------------------------------------------------------

    def _missing_secrets(self, ctx: StepContext) -> list[_MissingSecret]:
        recipe = _load_recipe(ctx)
        if recipe is None:
            return []
        missing: list[_MissingSecret] = []
        env_local = _read_env_local(ctx.project_dir)
        for svc in recipe.external_services:
            for env_var in svc.env_vars:
                if _is_present(env_var, env_local):
                    continue
                missing.append(
                    _MissingSecret(env_var=env_var, service=svc, required=bool(svc.required))
                )
        return missing

    def _persist(self, env_var: str, secret: SecretStr, ctx: StepContext) -> str | None:
        """Route to keyring (Anthropic-style) or ``.env.local`` (project)."""
        if env_var in _KEYRING_BACKED_ENV_VARS:
            try:
                store_key("anthropic", secret, backend="keyring")
            except AuthError as exc:
                # Fall back to file backend (also mode 0600) per design intent.
                try:
                    store_key("anthropic", secret, backend="file")
                except AuthError as exc2:
                    ctx.emit(
                        StepLog(
                            step_id=self.id,
                            line=(
                                f"keyring + file backends both rejected {env_var}: "
                                f"{exc}; {exc2}"
                            ),
                            stream="stderr",
                        )
                    )
                    return None
                return "credentials file (mode 0600)"
            return "keyring"

        try:
            _append_env_local(ctx.project_dir, env_var, secret)
        except OSError as exc:
            ctx.emit(
                StepLog(
                    step_id=self.id,
                    line=f"failed to write .env.local: {exc}",
                    stream="stderr",
                )
            )
            return None
        try:
            _ensure_gitignore_entry(ctx.project_dir, ".env.local")
        except OSError as exc:
            ctx.emit(
                StepLog(
                    step_id=self.id,
                    line=f".env.local stored but could not update .gitignore: {exc}",
                    stream="stderr",
                )
            )
        return ".env.local"


def _is_present(env_var: str, env_local: dict[str, str]) -> bool:
    """A var is 'present' if it's in os.environ, .env.local, or (for Anthropic) auth."""
    if os.environ.get(env_var, "").strip():
        return True
    if env_local.get(env_var):
        return True
    if env_var == ENV_API_KEY:
        return load_key() is not None
    return False


def _read_env_local(project_dir: Path) -> dict[str, str]:
    """Parse ``KEY=value`` lines from ``.env.local`` (mode-0600). Best-effort."""
    path = project_dir / ".env.local"
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, raw = stripped.partition("=")
        out[key.strip()] = _unquote(raw.strip())
    return out


def _unquote(raw: str) -> str:
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ('"', "'"):
        return raw[1:-1]
    return raw


def _append_env_local(project_dir: Path, env_var: str, secret: SecretStr) -> None:
    """Write/update ``env_var`` in ``.env.local`` as mode 0600 via ``secure_write``."""
    from agent_scaffold._filesec import MODE_SECRET, secure_write

    path = project_dir / ".env.local"
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    quoted_value = _quote_for_env_file(secret.get_secret_value())
    new_line = f"{env_var}={quoted_value}"
    pattern = re.compile(rf"^{re.escape(env_var)}=.*$", re.MULTILINE)
    if pattern.search(existing):
        updated = pattern.sub(new_line, existing)
    else:
        if existing and not existing.endswith("\n"):
            existing += "\n"
        updated = existing + new_line + "\n"
    secure_write(path, updated, mode=MODE_SECRET)


def _quote_for_env_file(raw: str) -> str:
    """Double-quote values that contain whitespace or shell-special chars."""
    if raw and not re.search(r"[\s\"'$`#=]", raw):
        return raw
    escaped = raw.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _ensure_gitignore_entry(project_dir: Path, entry: str) -> None:
    """Ensure ``entry`` is in ``.gitignore`` — delegates to the writer helper.

    Kept as a thin wrapper so existing tests don't break. The single call
    site (``apply()``) really wants the whole secret-safety block, so we
    use :func:`ensure_gitignore_defaults` which is a superset.
    """
    from agent_scaffold.writer import ensure_gitignore_defaults

    ensure_gitignore_defaults(project_dir, extra=(entry,))


def _load_recipe(ctx: StepContext) -> Recipe | None:
    from agent_scaffold.config import load_config
    from agent_scaffold.sources import SourceFetchError, resolve_deployments

    try:
        cfg = load_config()
    except Exception:  # noqa: BLE001 — config issues must not crash detect()
        return None
    try:
        dep = resolve_deployments(
            override=cfg.deployments_path,
            mode=cfg.deployments_source,
            cache_dir=cfg.cache_dir,
        )
    except SourceFetchError:
        return None
    if dep.path is None:
        return None
    try:
        recipes = discover_recipes(dep.path)
    except DiscoveryError:
        return None
    return next((r for r in recipes if r.slug == ctx.manifest.recipe), None)


__all__ = ["WireCredentialsStep"]
