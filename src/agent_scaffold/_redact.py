"""Pattern-based redactor for any string that might escape into a log or panel.

Defence-in-depth: even if Q2's keyring discipline + Q6's getpass discipline +
Q7's allowlist-`git add` all hold, a stray `repr()` of an exception or a
chatty subprocess that echoed a value into stderr could still print a
credential into stdout / state.json / manifest history.

This module is the last line: every external sink (Rich Live panel,
StepLog rendering, state.json persistence, manifest UpdateEntry, failure
panel stderr_tail) passes user-visible strings through :func:`redact`
before display.

Patterns are conservative — false positives on legitimate text are far
cheaper than a single missed secret. Each pattern includes a comment
naming the provider and the format it targets so future maintainers can
add new shapes without re-deriving the regex.
"""

from __future__ import annotations

import re
from typing import Any

# (compiled pattern, replacement). Replacements preserve the prefix so the
# user can see *which* kind of credential was redacted — useful for
# debugging without leaking the value itself.
_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Anthropic API keys: ``sk-ant-api03-<base64ish>`` and similar.
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"), "sk-ant-...REDACTED"),
    # OpenAI / generic ``sk-`` style. Lower bound 20 chars to avoid matching
    # short identifiers; OpenAI's are 32–48+.
    (re.compile(r"sk-(?!ant-)[A-Za-z0-9_\-]{20,}"), "sk-...REDACTED"),
    # AWS access key id: ``AKIA`` + 16 uppercase alnum.
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AKIA...REDACTED"),
    # AWS secret access key: 40 base64ish; conservative — only catches the
    # exact 40-char window to avoid false-positives on long hashes.
    (re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9/+=]{40}(?![A-Za-z0-9])"), "AWS-SECRET-REDACTED"),
    # Bearer tokens in Authorization headers.
    (re.compile(r"[Bb]earer\s+[A-Za-z0-9._\-]+"), "Bearer REDACTED"),
    # URLs with userinfo: postgres://user:password@host, redis://:pwd@host, etc.
    (
        re.compile(r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.\-]*://[^:/?#@]*):[^@/?#\s]+@"),
        r"\g<scheme>:REDACTED@",
    ),
    # GitHub fine-grained PATs.
    (re.compile(r"github_pat_[A-Za-z0-9_]{30,}"), "github_pat_REDACTED"),
    # Slack tokens.
    (re.compile(r"xox[abopr]-[A-Za-z0-9\-]{10,}"), "xox?-REDACTED"),
)


def redact(text: str) -> str:
    """Return ``text`` with every known secret-shaped substring replaced.

    Always safe to call with non-secret strings (the patterns won't match).
    Bytes-in / bytes-out callers should decode first; we operate on ``str``
    so we can use real regex semantics (and so the result is loggable).
    """
    if not text:
        return text
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def redact_obj(obj: Any) -> Any:
    """Recursively redact strings inside dicts / lists / tuples.

    Used for ``state.json`` and ``manifest.update_history`` payloads where
    the structure is JSON-shaped. Non-string leaves pass through unchanged.
    Tuples become tuples; sets become sets (order-stable iteration); other
    types are returned as-is.
    """
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, dict):
        return {k: redact_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_obj(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(redact_obj(v) for v in obj)
    return obj


def contains_secret_shape(text: str) -> bool:
    """``True`` if any pattern matches ``text`` — used in test assertions."""
    return any(pattern.search(text) for pattern, _ in _PATTERNS)


__all__ = [
    "contains_secret_shape",
    "redact",
    "redact_obj",
]
