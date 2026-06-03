"""Shared pytest fixtures for agent-scaffold tests."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from agent_scaffold import capabilities as _capabilities
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
    yield


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
