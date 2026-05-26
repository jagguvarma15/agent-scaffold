"""Credential storage for the Anthropic API key.

Three backends in priority order at resolution time:

1. **env** — ``ANTHROPIC_API_KEY`` (highest; CI / ``op run --`` win without ceremony)
2. **keyring** — ``python-keyring`` (macOS Keychain / Windows Credential Manager /
   Linux Secret Service). The plaintext file backend is **refused** so a silent
   fallback can never undermine the storage guarantee.
3. **file** — INI at ``$XDG_CONFIG_HOME/agent-scaffold/credentials`` with
   mode ``0o600`` (group/other have no access).

Tests stub at ``store_key`` / ``load_key`` / ``delete_key`` / the keyring
module — no real OS keychains are touched.
"""

from __future__ import annotations

import configparser
import logging
import os
import stat
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import keyring
import keyring.errors
from pydantic import SecretStr

log = logging.getLogger(__name__)

SERVICE_NAME = "agent-scaffold"
DEFAULT_KEY_NAME = "anthropic"
ENV_API_KEY = "ANTHROPIC_API_KEY"

BackendKind = Literal["keyring", "file", "env"]

# Native backends. Anything else (PlaintextKeyring, EncryptedKeyring,
# Null backend, etc.) is treated as "not safe to store secrets in" and
# triggers the AuthError path in detect_backend().
_NATIVE_BACKEND_NAMES: frozenset[str] = frozenset(
    {
        "Keyring",  # macOS keyring.backends.macOS.Keyring
        "WinVaultKeyring",  # Windows
        "SecretServiceKeyring",  # Linux freedesktop.org
        "KWallet5",  # Linux KDE
        "libsecretKeyring",  # Linux GNOME via libsecret
    }
)


class AuthError(Exception):
    """Raised for any auth-layer failure (refused backend, no creds, etc.)."""


@dataclass(frozen=True)
class StoredCredential:
    name: str
    backend: BackendKind
    masked_value: str
    created: str | None = None


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------


def _classify(backend: object) -> BackendKind | None:
    """Return ``"keyring"`` if the backend is OS-native, else ``None``."""
    cls_name = backend.__class__.__name__
    if cls_name in _NATIVE_BACKEND_NAMES:
        return "keyring"
    # ChainerBackend wraps a priority list; peek inside.
    if cls_name == "ChainerBackend":
        inner: Iterable[object] = getattr(backend, "backends", ())
        for b in inner:
            if b.__class__.__name__ in _NATIVE_BACKEND_NAMES:
                return "keyring"
    return None


def detect_backend() -> BackendKind:
    """Return the active backend kind, or raise on a refused (plaintext) one.

    Only OS-native backends and ``ChainerBackend`` wrappers that include one
    are accepted as ``"keyring"``. ``PlaintextKeyring`` / ``EncryptedKeyring``
    / ``Null`` backends raise ``AuthError`` so the caller is forced to choose
    ``--use-file`` or ``--use-env`` explicitly.
    """
    backend = keyring.get_keyring()
    classified = _classify(backend)
    if classified is not None:
        return classified
    raise AuthError(
        f"Refusing keyring backend: {backend.__class__.__name__}. "
        "Install a native backend (macOS Keychain / Windows Credential Manager / "
        "Linux Secret Service) or pass --use-file for a mode-0600 fallback."
    )


def describe_backend() -> str:
    """Human-readable name for the active backend (used in `auth status`)."""
    backend = keyring.get_keyring()
    cls_name = backend.__class__.__name__
    pretty = {
        "Keyring": "macOS Keychain",
        "WinVaultKeyring": "Windows Credential Manager",
        "SecretServiceKeyring": "Linux Secret Service",
        "KWallet5": "KDE Wallet",
        "libsecretKeyring": "libsecret (Linux)",
    }
    return pretty.get(cls_name, cls_name)


# ---------------------------------------------------------------------------
# Credentials file
# ---------------------------------------------------------------------------


def credentials_path() -> Path:
    """Resolve the INI credentials path. Honors ``XDG_CONFIG_HOME``."""
    base = os.environ.get("XDG_CONFIG_HOME")
    if base:
        root = Path(base).expanduser()
    else:
        root = Path.home() / ".config"
    return root / "agent-scaffold" / "credentials"


def _read_credentials_file() -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    path = credentials_path()
    if path.is_file():
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode != 0o600:
            log.warning(
                "credentials file %s has mode %o (expected 0600); fix with `chmod 600 %s`",
                path,
                mode,
                path,
            )
        parser.read(path, encoding="utf-8")
    return parser


def write_credentials_file(name: str, value: SecretStr) -> None:
    """Write/overwrite ``name`` in the credentials INI as mode 0600."""
    import io

    from agent_scaffold._filesec import MODE_SECRET, secure_write

    parser = _read_credentials_file()
    parser[name] = {
        "api_key": value.get_secret_value(),
        "created": datetime.now(UTC).isoformat(),
    }
    buf = io.StringIO()
    parser.write(buf)
    secure_write(credentials_path(), buf.getvalue(), mode=MODE_SECRET)


def _delete_from_credentials_file(name: str) -> bool:
    import io

    from agent_scaffold._filesec import MODE_SECRET, secure_write

    path = credentials_path()
    if not path.is_file():
        return False
    parser = _read_credentials_file()
    if name not in parser:
        return False
    parser.remove_section(name)
    buf = io.StringIO()
    parser.write(buf)
    secure_write(path, buf.getvalue(), mode=MODE_SECRET)
    return True


def _load_from_credentials_file(name: str) -> SecretStr | None:
    path = credentials_path()
    if not path.is_file():
        return None
    parser = _read_credentials_file()
    if name not in parser:
        return None
    raw = parser[name].get("api_key", "").strip()
    if not raw:
        return None
    return SecretStr(raw)


def _list_credentials_file() -> list[StoredCredential]:
    parser = _read_credentials_file()
    out: list[StoredCredential] = []
    for name in parser.sections():
        raw = parser[name].get("api_key", "").strip()
        if not raw:
            continue
        out.append(
            StoredCredential(
                name=name,
                backend="file",
                masked_value=mask(raw),
                created=parser[name].get("created"),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Store / load / delete / list across backends
# ---------------------------------------------------------------------------


def store_key(
    name: str,
    value: SecretStr,
    backend: BackendKind = "keyring",
) -> StoredCredential:
    """Persist ``value`` under ``name`` in ``backend``.

    ``backend="env"`` is a no-op except for returning the descriptor — callers
    use the returned ``StoredCredential`` to print an export line.
    """
    if backend == "env":
        return StoredCredential(
            name=name,
            backend="env",
            masked_value=mask(value.get_secret_value()),
        )
    if backend == "file":
        write_credentials_file(name, value)
        return StoredCredential(
            name=name,
            backend="file",
            masked_value=mask(value.get_secret_value()),
        )
    # backend == "keyring" — fail loudly on unsafe backends.
    detect_backend()
    try:
        keyring.set_password(SERVICE_NAME, name, value.get_secret_value())
    except keyring.errors.PasswordSetError as exc:
        raise AuthError(f"keyring rejected the write: {exc}") from exc
    return StoredCredential(
        name=name,
        backend="keyring",
        masked_value=mask(value.get_secret_value()),
    )


def load_key(name: str = DEFAULT_KEY_NAME) -> SecretStr | None:
    """Resolve a key. Order: env > keyring > file. ``None`` if nothing matches."""
    env_value = os.environ.get(ENV_API_KEY, "").strip()
    if env_value:
        return SecretStr(env_value)
    try:
        from_kr = keyring.get_password(SERVICE_NAME, name)
    except keyring.errors.KeyringError as exc:
        log.debug("keyring lookup failed for %s/%s: %s", SERVICE_NAME, name, exc)
        from_kr = None
    if from_kr:
        return SecretStr(from_kr)
    return _load_from_credentials_file(name)


def resolve_active(name: str = DEFAULT_KEY_NAME) -> tuple[SecretStr, BackendKind] | None:
    """Like ``load_key`` but also reports which backend supplied the value."""
    env_value = os.environ.get(ENV_API_KEY, "").strip()
    if env_value:
        return SecretStr(env_value), "env"
    try:
        from_kr = keyring.get_password(SERVICE_NAME, name)
    except keyring.errors.KeyringError:
        from_kr = None
    if from_kr:
        return SecretStr(from_kr), "keyring"
    from_file = _load_from_credentials_file(name)
    if from_file is not None:
        return from_file, "file"
    return None


def delete_key(name: str = DEFAULT_KEY_NAME) -> bool:
    """Remove ``name`` from every backend it lives in. ``True`` if anything was removed."""
    removed = False
    try:
        keyring.delete_password(SERVICE_NAME, name)
        removed = True
    except keyring.errors.PasswordDeleteError:
        pass
    except keyring.errors.KeyringError as exc:
        log.debug("keyring delete failed: %s", exc)
    if _delete_from_credentials_file(name):
        removed = True
    return removed


def list_credentials() -> list[StoredCredential]:
    """Inventory across keyring + credentials file. Env vars are not listed."""
    creds: list[StoredCredential] = []
    # We can't enumerate keyring entries (no API for it) — only the names
    # we know about. The default name is the only well-known one; users
    # using multi-key setups will see them in the file backend.
    try:
        v = keyring.get_password(SERVICE_NAME, DEFAULT_KEY_NAME)
        if v:
            creds.append(
                StoredCredential(
                    name=DEFAULT_KEY_NAME,
                    backend="keyring",
                    masked_value=mask(v),
                )
            )
    except keyring.errors.KeyringError:
        pass
    creds.extend(_list_credentials_file())
    return creds


# ---------------------------------------------------------------------------
# Masking and validation
# ---------------------------------------------------------------------------


def mask(key: str) -> str:
    """``sk-ant-api03-abc...4j2k`` → ``sk-ant-...4j2k``. Shows at most 4 tail chars."""
    if not key:
        return ""
    tail = key[-4:] if len(key) >= 8 else key
    if key.startswith("sk-ant-"):
        return f"sk-ant-...{tail}"
    if len(key) <= 8:
        return "***" + tail
    return f"{key[:3]}...{tail}"


def validate_anthropic_key(key: SecretStr) -> tuple[bool, str]:
    """Probe the key with ``models.list()``. Returns ``(ok, message)``.

    Format-sanity-check first (saves a network call on obvious garbage).
    Imports ``anthropic`` lazily so the dependency only loads when we
    actually intend to validate (``auth.py`` itself stays import-cheap).
    """
    raw = key.get_secret_value()
    if not raw.startswith("sk-ant-"):
        return False, "key does not look like an Anthropic key (expected sk-ant-... prefix)"
    try:
        import anthropic
    except ImportError:
        return False, "anthropic SDK is not installed; cannot validate key"
    try:
        client = anthropic.Anthropic(api_key=raw, timeout=10.0)
        page = client.models.list(limit=1)
        count = len(list(page.data))
        return True, f"validated ({count} model(s) visible)"
    except anthropic.AuthenticationError:
        return False, "key rejected by Anthropic API (401)"
    except Exception as exc:  # noqa: BLE001 - any SDK error is a probe failure
        return False, f"probe failed: {type(exc).__name__}: {exc}"


__all__ = [
    "DEFAULT_KEY_NAME",
    "ENV_API_KEY",
    "SERVICE_NAME",
    "AuthError",
    "BackendKind",
    "StoredCredential",
    "credentials_path",
    "delete_key",
    "describe_backend",
    "detect_backend",
    "list_credentials",
    "load_key",
    "mask",
    "resolve_active",
    "store_key",
    "validate_anthropic_key",
    "write_credentials_file",
]
