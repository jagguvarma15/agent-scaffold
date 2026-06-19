"""Dependency-light helpers for a project's ``.env.local`` file.

Extracted from ``steps/wire_credentials.py`` so callers that only need to
*check* env-var presence (the pre-flight gate in ``cmd_new``) don't import
the orchestrator chain. The wire_credentials step re-imports these, so both
paths share one definition of "present" and one ``.env.local`` format.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from pydantic import SecretStr

from agent_scaffold.auth import ENV_API_KEY, load_key

ENV_LOCAL_FILENAME = ".env.local"


def read_env_local(project_dir: Path) -> dict[str, str]:
    """Parse ``KEY=value`` lines from ``.env.local`` (mode-0600). Best-effort."""
    path = project_dir / ENV_LOCAL_FILENAME
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
        out[key.strip()] = unquote(raw.strip())
    return out


def unquote(raw: str) -> str:
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ('"', "'"):
        return raw[1:-1]
    return raw


def is_present(env_var: str, env_local: dict[str, str]) -> bool:
    """A var is 'present' if it's in os.environ, .env.local, or (for Anthropic) auth."""
    if os.environ.get(env_var, "").strip():
        return True
    if env_local.get(env_var):
        return True
    if env_var == ENV_API_KEY:
        return load_key() is not None
    return False


def append_env_local(project_dir: Path, env_var: str, secret: SecretStr) -> None:
    """Write/update ``env_var`` in ``.env.local`` as mode 0600 via ``secure_write``."""
    from agent_scaffold._filesec import MODE_SECRET, secure_write

    path = project_dir / ENV_LOCAL_FILENAME
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    quoted_value = quote_for_env_file(secret.get_secret_value())
    new_line = f"{env_var}={quoted_value}"
    pattern = re.compile(rf"^{re.escape(env_var)}=.*$", re.MULTILINE)
    if pattern.search(existing):
        updated = pattern.sub(new_line, existing)
    else:
        if existing and not existing.endswith("\n"):
            existing += "\n"
        updated = existing + new_line + "\n"
    secure_write(path, updated, mode=MODE_SECRET)


def quote_for_env_file(raw: str) -> str:
    """Double-quote values that contain whitespace or shell-special chars."""
    if raw and not re.search(r"[\s\"'$`#=]", raw):
        return raw
    escaped = raw.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def build_runtime_env(project_dir: Path, namespace: str | None) -> dict[str, str]:
    """Resolve the full environment for a project's subprocesses.

    Precedence (highest wins): live shell env > project secrets vault >
    ``.env.local``. The vault is read in ONE batch per call so OS keychain
    consent prompts fire once per run, not once per variable. Values flow
    only into subprocess ``env=`` — never into prompts, logs, or panels
    (those paths go through ``_redact``).
    """
    merged: dict[str, str] = dict(read_env_local(project_dir))
    if namespace:
        from agent_scaffold.auth import load_project_secrets

        for name, secret in load_project_secrets(namespace).items():
            merged[name] = secret.get_secret_value()
    merged.update(os.environ)
    # The Anthropic key commonly lives only in the OS keyring / mode-0600
    # credentials file (set via `scaffold auth login`, e.g. by the installer) —
    # not in the shell env, ``.env.local``, or the project vault. Generated
    # agents build an Anthropic client at import/startup, so resolve the key the
    # same way the CLI does and inject it; otherwise the backend crashes on boot
    # with "Could not resolve authentication method". Anything that already set
    # it (shell / ``.env.local`` / vault) still wins.
    if not merged.get(ENV_API_KEY, "").strip():
        resolved = load_key()
        if resolved is not None:
            merged[ENV_API_KEY] = resolved.get_secret_value()
    return merged


__all__ = [
    "ENV_LOCAL_FILENAME",
    "append_env_local",
    "build_runtime_env",
    "is_present",
    "quote_for_env_file",
    "read_env_local",
    "unquote",
]
