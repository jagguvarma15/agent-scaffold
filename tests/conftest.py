"""Shared pytest fixtures for agent-forge tests."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
MOCK_DEPLOYMENTS = FIXTURES_DIR / "mock_deployments"
MOCK_RESPONSES = FIXTURES_DIR / "mock_responses"


@pytest.fixture
def mock_deployments_path() -> Path:
    return MOCK_DEPLOYMENTS


@pytest.fixture
def mock_responses_path() -> Path:
    return MOCK_RESPONSES


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip all AGENT_FORGE_* and ANTHROPIC_* env vars for isolated tests."""
    for key in list(os.environ):
        if key.startswith("AGENT_FORGE_") or key.startswith("ANTHROPIC_"):
            monkeypatch.delenv(key, raising=False)
    yield
