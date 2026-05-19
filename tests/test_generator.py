"""Tests for agent_scaffold.generator (mockable client seam)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anthropic
import httpx
import pytest

from agent_scaffold import generator
from agent_scaffold.config import Config
from agent_scaffold.context import AssembledContext
from agent_scaffold.generator import (
    GenerationRequest,
    _render_repair_message,
    _render_user_message,
    generate,
)

LARGE_BODY = "# Recipe\n\nHello.\n" + ("filler context line.\n" * 600)


def _config(tmp_path: Path) -> Config:
    return Config(
        deployments_path=tmp_path,
        anthropic_api_key="test",
        cache_dir=tmp_path / "cache",
        failures_dir=tmp_path / "cache" / "failures",
    )


def _request(tmp_path: Path, body: str = LARGE_BODY) -> GenerationRequest:
    ctx = AssembledContext(
        recipe_path=tmp_path / "r.md",
        referenced_paths=[],
        body=body,
        token_estimate=10,
    )
    return GenerationRequest(
        project_name="demo_agent",
        target_language="python",
        framework="langgraph",
        assembled_context=ctx,
        language_hints={"language": "python", "manifest": "pyproject.toml"},
    )


def test_user_message_substitutes_all_placeholders(tmp_path: Path) -> None:
    req = _request(tmp_path)
    context_block, tail_block = _render_user_message(req)
    # The cacheable context carries language hints + assembled spec.
    assert "language: python" in context_block
    assert "# Recipe" in context_block
    # Project-specific data lives in the tail so the cache key is stable.
    assert "Name: demo_agent" in tail_block
    assert "Target language: python" in tail_block
    # Literal JSON braces from the template should survive.
    assert '"project_name": string' in tail_block


def test_repair_message_substitutes(tmp_path: Path) -> None:
    rendered = _render_repair_message("RAW_DATA", "VALIDATION_ERROR")
    assert "RAW_DATA" in rendered
    assert "VALIDATION_ERROR" in rendered


class _FakeBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.content = [_FakeBlock(text)]


class _FakeMessages:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = responses
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class _FakeClient:
    def __init__(self, responses: list[Any]) -> None:
        self.messages = _FakeMessages(responses)


def test_generate_returns_text_and_caches_system(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeClient([_FakeResponse("hello world")])
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: fake)
    out = generate(_request(tmp_path), _config(tmp_path))
    assert out == "hello world"
    call = fake.messages.calls[0]
    assert call["model"]
    assert call["max_tokens"] == 32000
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "operating principles" in call["system"][0]["text"].lower()


def test_generate_user_content_is_cached_block_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeClient([_FakeResponse("ok")])
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: fake)
    generate(_request(tmp_path), _config(tmp_path))
    content = fake.messages.calls[0]["messages"][0]["content"]
    assert isinstance(content, list)
    assert len(content) == 2
    assert content[0]["cache_control"] == {"type": "ephemeral"}
    assert "# Recipe" in content[0]["text"]
    assert "Name: demo_agent" not in content[0]["text"]
    assert "Name: demo_agent" in content[1]["text"]
    assert "cache_control" not in content[1]


def test_generate_falls_back_to_single_block_for_tiny_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeClient([_FakeResponse("ok")])
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: fake)
    generate(_request(tmp_path, body="tiny\n"), _config(tmp_path))
    content = fake.messages.calls[0]["messages"][0]["content"]
    assert isinstance(content, list)
    assert len(content) == 1
    assert "cache_control" not in content[0]
    assert "Name: demo_agent" in content[0]["text"]


def _rate_limit_error() -> anthropic.RateLimitError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status_code=429, request=request)
    return anthropic.RateLimitError("rate", response=response, body=None)


def test_generate_retries_on_rate_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient([_rate_limit_error(), _rate_limit_error(), _FakeResponse("ok")])
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: fake)
    sleeps: list[float] = []
    monkeypatch.setattr(generator.time, "sleep", lambda s: sleeps.append(s))
    out = generate(_request(tmp_path), _config(tmp_path))
    assert out == "ok"
    assert sleeps == [1.0, 2.0]
    assert len(fake.messages.calls) == 3


def test_generate_gives_up_after_max_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeClient(
        [_rate_limit_error(), _rate_limit_error(), _rate_limit_error(), _rate_limit_error()]
    )
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: fake)
    monkeypatch.setattr(generator.time, "sleep", lambda _s: None)
    with pytest.raises(anthropic.RateLimitError):
        generate(_request(tmp_path), _config(tmp_path))
    assert len(fake.messages.calls) == 4  # 1 initial + 3 retries


def test_generate_non_retryable_error_raises_immediately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeClient([ValueError("boom")])
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: fake)
    monkeypatch.setattr(generator.time, "sleep", lambda _s: None)
    with pytest.raises(ValueError, match="boom"):
        generate(_request(tmp_path), _config(tmp_path))
    assert len(fake.messages.calls) == 1
