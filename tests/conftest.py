"""Shared pytest fixtures for agent-scaffold tests."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from agent_scaffold import capabilities as _capabilities
from agent_scaffold import catalog as _catalog
from agent_scaffold import discovery as _discovery

FIXTURES_DIR = Path(__file__).parent / "fixtures"
MOCK_DEPLOYMENTS = FIXTURES_DIR / "mock_deployments"
MOCK_RESPONSES = FIXTURES_DIR / "mock_responses"


@pytest.fixture(autouse=True)
def _reset_discovery_warn_dedupe() -> Iterator[None]:
    """Reset the process-level warning dedupe sets before each test so that
    capsys-based assertions on warnings remain deterministic across tests."""
    _discovery._reset_warn_dedupe()
    _capabilities._reset_warn_dedupe()
    _catalog._reset_warn_dedupe()
    yield


@pytest.fixture(autouse=True)
def _isolated_secret_backends(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """No test may ever touch the developer's real keychain or credentials file.

    The vault code paths call ``keyring`` directly (not just through
    ``store_key``/``load_key`` seams), so a missed stub would write real
    OS-keychain entries from a unit test — this happened once. Every test now
    gets an in-memory keyring (classified as native by ``detect_backend``)
    and a throwaway ``XDG_CONFIG_HOME`` for the credentials INI. Tests that
    need different backend behavior layer their own monkeypatches on top.
    """
    import keyring
    import keyring.errors

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path_factory.mktemp("xdg-config")))

    store: dict[tuple[str, str], str] = {}

    class Keyring:  # noqa: N801 — name must match auth._NATIVE_BACKEND_NAMES
        pass

    def _get(service: str, name: str) -> str | None:
        return store.get((service, name))

    def _set(service: str, name: str, value: str) -> None:
        store[(service, name)] = value

    def _delete(service: str, name: str) -> None:
        if (service, name) not in store:
            raise keyring.errors.PasswordDeleteError(name)
        del store[(service, name)]

    monkeypatch.setattr(keyring, "get_keyring", lambda: Keyring())
    monkeypatch.setattr(keyring, "get_password", _get)
    monkeypatch.setattr(keyring, "set_password", _set)
    monkeypatch.setattr(keyring, "delete_password", _delete)


@pytest.fixture
def mock_deployments_path() -> Path:
    return MOCK_DEPLOYMENTS


@pytest.fixture
def mock_responses_path() -> Path:
    return MOCK_RESPONSES


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip all AGENT_SCAFFOLD_* and ANTHROPIC_* env vars for isolated tests."""
    for key in list(os.environ):
        if key.startswith("AGENT_SCAFFOLD_") or key.startswith("ANTHROPIC_"):
            monkeypatch.delenv(key, raising=False)
    yield
