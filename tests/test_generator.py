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


def _config(tmp_path: Path, **overrides: Any) -> Config:
    return Config(
        deployments_path=tmp_path,
        anthropic_api_key="test",
        cache_dir=tmp_path / "cache",
        failures_dir=tmp_path / "cache" / "failures",
        **overrides,
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


class _FakeStream:
    def __init__(self, item: Any) -> None:
        self._item = item

    def __enter__(self) -> "_FakeStream":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def get_final_message(self) -> Any:
        if isinstance(self._item, BaseException):
            raise self._item
        return self._item


class _FakeMessages:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = responses
        self.calls: list[dict[str, Any]] = []

    def stream(self, **kwargs: Any) -> _FakeStream:
        self.calls.append(kwargs)
        item = self._responses.pop(0)
        return _FakeStream(item)


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


def test_generate_includes_legacy_thinking_for_non_adaptive_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeClient([_FakeResponse("ok")])
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: fake)
    cfg = _config(tmp_path, model="claude-sonnet-4-6", thinking_budget=8000)
    generate(_request(tmp_path), cfg)
    call = fake.messages.calls[0]
    assert call["thinking"] == {"type": "enabled", "budget_tokens": 8000}
    assert "output_config" not in call


def test_generate_uses_adaptive_thinking_for_opus_4_7(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeClient([_FakeResponse("ok")])
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: fake)
    cfg = _config(tmp_path, model="claude-opus-4-7", thinking_budget=16000)
    generate(_request(tmp_path), cfg)
    call = fake.messages.calls[0]
    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_config"] == {"effort": "high"}


def test_generate_omits_thinking_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeClient([_FakeResponse("ok")])
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: fake)
    generate(_request(tmp_path), _config(tmp_path))
    call = fake.messages.calls[0]
    assert "thinking" not in call
    assert "output_config" not in call


def test_budget_to_effort_buckets() -> None:
    assert generator._budget_to_effort(2000) == "low"
    assert generator._budget_to_effort(8000) == "medium"
    assert generator._budget_to_effort(16000) == "high"
    assert generator._budget_to_effort(30000) == "xhigh"
    assert generator._budget_to_effort(60000) == "max"


def test_generate_strict_loads_strict_system_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeClient([_FakeResponse("ok")])
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: fake)
    req = _request(tmp_path).model_copy(update={"strict": True})
    generate(req, _config(tmp_path))
    call = fake.messages.calls[0]
    text = call["system"][0]["text"]
    assert "Production requirements (strict mode)" in text


def test_generate_non_strict_does_not_load_strict_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeClient([_FakeResponse("ok")])
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: fake)
    generate(_request(tmp_path), _config(tmp_path))
    text = fake.messages.calls[0]["system"][0]["text"]
    assert "Production requirements (strict mode)" not in text


def test_thinking_response_extracts_only_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Anthropic ThinkingBlocks have no `.text` attribute; _extract_text should
    # walk them safely and concatenate only the text blocks.
    class _ThinkingBlock:
        thinking = "deliberating..."

    class _Response:
        def __init__(self) -> None:
            self.content = [_ThinkingBlock(), _FakeBlock("final answer")]

    fake = _FakeClient([_Response()])
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: fake)
    out = generate(_request(tmp_path), _config(tmp_path, thinking_budget=4000))
    assert out == "final answer"
