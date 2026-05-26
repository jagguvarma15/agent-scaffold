"""Rule 1 audit: secrets must never arrive via argv.

``/proc/<pid>/cmdline`` is readable by every user on a typical Linux box
(and on macOS via ``ps``-like APIs). A flag like ``--api-key sk-ant-...``
leaks the credential to anyone who can list processes for the lifetime of
the invocation. Always require a separate input channel (env var, file,
``getpass`` paste).

This test walks Typer's registered commands, inspects each parameter's
declared CLI flag names, and fails if a credential-shaped name (``token``,
``api_key``, ``password``, ...) shows up outside the explicit allow-list.
"""

from __future__ import annotations

import inspect
from typing import Any

import typer

from agent_scaffold.cli import app

# Substrings that suggest a parameter could carry a secret.
_SECRET_KEYWORDS = ("token", "key", "secret", "password", "credential", "api_key")

# Allow-list of ``(command_name, parameter_name)`` we've audited and deemed
# safe. Each entry must have a one-line comment explaining why the apparent
# secret-shaped name is actually a non-secret identifier.
_ALLOWED: frozenset[tuple[str, str]] = frozenset(
    {
        # ``auth setup-token <name>`` takes the *name* of the credential to
        # store; the credential value itself is read from stdin or getpass.
        ("auth setup-token", "name"),
        # ``auth login --name`` is the credential identifier (e.g. "anthropic"),
        # not the secret itself.
        ("auth login", "name"),
        # ``auth logout --name`` same as above.
        ("auth logout", "name"),
        # Token-budget knobs on `new` / `regenerate` — integer LLM context
        # limits, not credentials. Names contain "token" but the values are
        # ints fed straight to the API.
        ("new", "max_tokens"),
        ("new", "max_context_tokens"),
        ("new", "max_tokens_per_doc"),
        ("regenerate", "max_tokens"),
        # Backend-selector boolean flags on `auth login` — pick the storage
        # backend, not the secret.
        ("auth login", "use_keyring"),
    }
)


def _iter_subcommands(typer_app: typer.Typer, prefix: str = "") -> list[tuple[str, Any]]:
    """Yield ``(command_path, callback)`` for every Typer command + sub-app."""
    out: list[tuple[str, Any]] = []
    for cmd in typer_app.registered_commands:
        name = cmd.name or (cmd.callback.__name__ if cmd.callback else "<anon>")
        path = f"{prefix} {name}".strip()
        if cmd.callback is not None:
            out.append((path, cmd.callback))
    for sub in typer_app.registered_groups:
        sub_path = f"{prefix} {sub.name}".strip()
        if isinstance(sub.typer_instance, typer.Typer):
            out.extend(_iter_subcommands(sub.typer_instance, prefix=sub_path))
    return out


def test_no_credential_shaped_params_outside_allowlist() -> None:
    violations: list[str] = []
    for command_path, callback in _iter_subcommands(app):
        sig = inspect.signature(callback)
        for param_name in sig.parameters:
            lowered = param_name.lower()
            if not any(needle in lowered for needle in _SECRET_KEYWORDS):
                continue
            if (command_path, param_name) in _ALLOWED:
                continue
            violations.append(f"{command_path}: parameter {param_name!r}")
    assert not violations, (
        "Credential-shaped CLI parameters detected (rule 1 — no secrets in argv). "
        "Either rename, accept via env / getpass / stdin, or add to the allow-list "
        "with a justification comment:\n  " + "\n  ".join(violations)
    )
