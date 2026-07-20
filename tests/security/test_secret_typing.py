"""Rule 3 audit: credential-bearing function parameters must be typed ``SecretStr``.

``SecretStr`` defangs the two most common leakage paths: ``repr()`` /
``str()`` of a SecretStr returns ``'**********'``, and accidental logging
prints the masked form. A bare ``str`` makes those leaks trivial.

This test enumerates the canonical Q2 / Q9 entry points and checks via
``inspect.signature`` that each credential parameter is annotated as
``SecretStr`` (or ``SecretStr | None`` for getters).
"""

from __future__ import annotations

from typing import get_type_hints

from pydantic import SecretStr

from agent_scaffold.auth import store_key, validate_anthropic_key, write_credentials_file


def _hints(func: object) -> dict[str, type]:
    try:
        return get_type_hints(func)
    except Exception:  # noqa: BLE001 — annotation chasing can fail under TYPE_CHECKING gates
        return {}


def test_store_key_value_is_secretstr() -> None:
    hints = _hints(store_key)
    assert hints.get("value") is SecretStr, (
        f"store_key.value must be typed SecretStr; got {hints.get('value')!r}"
    )


def test_write_credentials_file_value_is_secretstr() -> None:
    hints = _hints(write_credentials_file)
    assert hints.get("value") is SecretStr, (
        f"write_credentials_file.value must be typed SecretStr; got {hints.get('value')!r}"
    )


def test_validate_anthropic_key_takes_secretstr() -> None:
    hints = _hints(validate_anthropic_key)
    assert hints.get("key") is SecretStr, (
        f"validate_anthropic_key.key must be typed SecretStr; got {hints.get('key')!r}"
    )


def test_wire_credentials_persist_takes_secretstr() -> None:
    """``WireCredentialsStep._persist`` is the project-secret entry point."""
    from agent_scaffold.steps.wire_credentials import WireCredentialsStep

    hints = _hints(WireCredentialsStep._persist)
    assert hints.get("secret") is SecretStr, (
        f"WireCredentialsStep._persist.secret must be typed SecretStr; got {hints.get('secret')!r}"
    )


def test_config_api_key_is_secretstr() -> None:
    """``Config.anthropic_api_key`` lives for the whole process — mask it."""
    from agent_scaffold.config import Config

    field = Config.model_fields["anthropic_api_key"]
    assert field.annotation is SecretStr, (
        f"Config.anthropic_api_key must be typed SecretStr; got {field.annotation!r}"
    )
