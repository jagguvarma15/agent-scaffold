"""File-system primitives that get secret-handling right.

Three call sites need to write files with explicit modes today:

- :mod:`agent_scaffold.auth` (Q2) writes the credentials INI as mode ``0o600``.
- :mod:`agent_scaffold.steps.wire_credentials` (Q6) writes ``.env.local`` as
  mode ``0o600`` next to the project.
- :mod:`agent_scaffold.manifest` (S2 + Q8) writes ``manifest.json`` as
  ``0o644`` (no secrets, but stable mode for reproducibility).

Each of those used to inline the same three-step dance: ``umask(0o077)``,
``os.open(..., flags, mode)``, ``chmod(mode)``. That's the canonical defence
against a permissive default umask leaking new files as ``0o644`` (or worse,
``0o666``) and exposing keys.

This module centralises the pattern as :func:`secure_write` so every caller
gets it right by construction:

1. ``umask(0o077)`` flips the default to "owner-only".
2. Write to a sibling tempfile (``<path>.tmp.<pid>``) with the requested
   mode passed to :func:`os.open`. ``fsync`` before close so a crash
   doesn't leave a half-written file at the final path.
3. ``os.replace`` is atomic on POSIX, so readers always see either the
   old file or the new one — never half-written.
4. ``chmod`` belt-and-suspenders the mode in case ``umask`` was already
   set non-zero by a wrapper.
5. ``umask`` is restored even if any step raised.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

# Mode for files that contain secrets. Owner read+write only.
MODE_SECRET = 0o600
# Mode for files that don't contain secrets but should still be unambiguous.
MODE_PUBLIC = 0o644


class FileSecurityError(Exception):
    """Raised when a security-relevant filesystem invariant cannot be honoured."""


def secure_write(
    path: Path,
    content: str | bytes,
    *,
    mode: int = MODE_SECRET,
    encoding: str = "utf-8",
) -> Path:
    """Atomically write ``content`` to ``path`` with the requested ``mode``.

    The defaults target secret-bearing files. Pass ``mode=MODE_PUBLIC`` for
    non-secret artifacts (e.g. ``manifest.json``).

    Returns ``path`` for ergonomic chaining.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = content.encode(encoding) if isinstance(content, str) else content
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    old_umask = os.umask(0o077)
    try:
        fd = os.open(tmp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, mode)
        try:
            os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            os.replace(tmp, path)
        except OSError:
            tmp.unlink(missing_ok=True)
            raise
        # Belt + suspenders: a wrapper may have left umask permissive, or the
        # destination's existing inode may have a wider mode. chmod re-asserts.
        os.chmod(path, mode)
    finally:
        os.umask(old_umask)
    return path


def assert_secret_mode(path: Path, *, expected: int = MODE_SECRET) -> None:
    """Raise :class:`FileSecurityError` if ``path``'s mode is wider than expected.

    Used in tests + as a runtime check on already-existing credential files.
    """
    if not path.is_file():
        return
    actual = stat.S_IMODE(path.stat().st_mode)
    if actual != expected:
        raise FileSecurityError(
            f"{path}: mode {oct(actual)} expected {oct(expected)}; "
            f"fix with `chmod {oct(expected)[2:]} {path}`"
        )


__all__ = [
    "MODE_PUBLIC",
    "MODE_SECRET",
    "FileSecurityError",
    "assert_secret_mode",
    "secure_write",
]
