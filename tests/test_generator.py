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
    _render_refinement_block,
    _render_repair_message,
    _render_user_message,
    extract_fenced_content,
    generate,
    generate_single_file,
)

# Comfortably above the largest per-model cache minimum (Opus: 4096 tokens ≈
# 16 KB at 4 chars/token), so the default request exercises the cached-block path.
LARGE_BODY = "# Recipe\n\nHello.\n" + ("filler context line.\n" * 1000)


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


def test_user_message_omits_refinement_block_when_empty(tmp_path: Path) -> None:
    """Vanilla cmd_new runs (no REPL refinements) get no extra block."""
    req = _request(tmp_path)
    _context_block, tail_block = _render_user_message(req)
    assert "# User refinements" not in tail_block


def test_role_block_renders_in_tail_when_agent_role_set(tmp_path: Path) -> None:
    """The describe-step persona lands in the per-run tail as the # Agent role block."""
    ctx = AssembledContext(
        recipe_path=tmp_path / "r.md",
        referenced_paths=[],
        body="# Recipe\n\nHello.\n",
        token_estimate=10,
    )
    req = GenerationRequest(
        project_name="demo_agent",
        target_language="python",
        framework="langgraph",
        assembled_context=ctx,
        language_hints={"language": "python", "manifest": "pyproject.toml"},
        agent_role="You are a docs assistant. Cite your sources.",
    )
    context_block, tail_block = _render_user_message(req)
    assert "# Agent role" in tail_block
    assert "You are a docs assistant. Cite your sources." in tail_block
    assert "system prompt" in tail_block  # instructs the model to adopt it
    # Per-run intent → tail, never the cacheable context prefix.
    assert "# Agent role" not in context_block


def test_role_block_omitted_when_agent_role_unset(tmp_path: Path) -> None:
    req = _request(tmp_path)  # no agent_role
    _context_block, tail_block = _render_user_message(req)
    assert "# Agent role" not in tail_block


def test_refinement_block_renders_each_field(tmp_path: Path) -> None:
    """Every refinement accumulator surfaces in the rendered block.

    This is the contract that fixes the silent-no-op bug: each field set on
    GenerationRequest must appear in the user message so the LLM can act on
    it. If a field's content is missing here, the LLM never sees the
    refinement and the bug regresses.
    """
    ctx = AssembledContext(
        recipe_path=tmp_path / "r.md",
        referenced_paths=[],
        body="# Recipe\n\nHello.\n",
        token_estimate=10,
    )
    req = GenerationRequest(
        project_name="demo_agent",
        target_language="python",
        framework="langgraph",
        assembled_context=ctx,
        language_hints={"language": "python", "manifest": "pyproject.toml"},
        extra_dependencies={"python": {"postgres": "^16"}},
        extra_steps=["wire prometheus exporter"],
        removed_steps=["docker_up"],
        removed_roles=["evaluator"],
        refinement_notes=["Prefer async/await throughout."],
    )

    block = _render_refinement_block(req)
    assert "# User refinements" in block
    assert "postgres" in block and "^16" in block and "python" in block
    assert "wire prometheus exporter" in block
    assert "docker_up" in block
    assert "evaluator" in block
    assert "Prefer async/await throughout." in block

    # And the same content must land in the rendered user-message tail —
    # not the cacheable context block — so per-run refinements don't
    # poison the prompt cache.
    _context_block, tail_block = _render_user_message(req)
    assert "# User refinements" in tail_block
    assert "postgres" in tail_block
    assert "# User refinements" not in _context_block


def test_refinement_block_skips_unused_sections(tmp_path: Path) -> None:
    """Sections for empty fields don't appear (no dangling headings)."""
    ctx = AssembledContext(
        recipe_path=tmp_path / "r.md",
        referenced_paths=[],
        body="# Recipe\n",
        token_estimate=5,
    )
    req = GenerationRequest(
        project_name="demo_agent",
        target_language="python",
        framework="langgraph",
        assembled_context=ctx,
        language_hints={"language": "python", "manifest": "pyproject.toml"},
        refinement_notes=["Use uv, not pip."],
    )
    block = _render_refinement_block(req)
    assert "Use uv, not pip." in block
    assert "## Additional dependencies" not in block
    assert "## Skip these steps" not in block
    assert "## Skip these roles" not in block


def test_repair_message_substitutes(tmp_path: Path) -> None:
    rendered = _render_repair_message("RAW_DATA", "VALIDATION_ERROR")
    assert "RAW_DATA" in rendered
    assert "VALIDATION_ERROR" in rendered


class _FakeBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeResponse:
    def __init__(self, text: str, stop_reason: str | None = None) -> None:
        self.content = [_FakeBlock(text)]
        self.stop_reason = stop_reason


class _FakeStream:
    def __init__(self, item: Any, events: list[Any] | None = None) -> None:
        self._item = item
        self._events = events or []

    def __enter__(self) -> _FakeStream:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def __iter__(self) -> Any:
        yield from self._events

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
    # No effort for legacy thinking; output_config still carries the
    # structured-output format.
    assert "effort" not in call["output_config"]
    assert call["output_config"]["format"]["type"] == "json_schema"


def test_generate_uses_adaptive_thinking_for_opus_4_7(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeClient([_FakeResponse("ok")])
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: fake)
    cfg = _config(tmp_path, model="claude-opus-4-7", thinking_budget=16000)
    generate(_request(tmp_path), cfg)
    call = fake.messages.calls[0]
    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_config"]["effort"] == "high"


@pytest.mark.parametrize("model", ["claude-opus-4-8", "claude-sonnet-5"])
def test_generate_uses_adaptive_thinking_for_current_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    # Opus 4.8 and Sonnet 5 are adaptive-only: the legacy budget_tokens shape
    # would be rejected with an HTTP 400, so an effort budget must translate to
    # the adaptive request instead.
    fake = _FakeClient([_FakeResponse("ok")])
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: fake)
    cfg = _config(tmp_path, model=model, thinking_budget=16000)
    generate(_request(tmp_path), cfg)
    call = fake.messages.calls[0]
    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_config"]["effort"] == "high"
    assert "budget_tokens" not in call["thinking"]


def test_generate_omits_thinking_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeClient([_FakeResponse("ok")])
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: fake)
    generate(_request(tmp_path), _config(tmp_path))
    call = fake.messages.calls[0]
    assert "thinking" not in call
    # Thinking off leaves only the structured-output format on output_config.
    assert "effort" not in call["output_config"]
    assert call["output_config"]["format"]["type"] == "json_schema"


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


class _Delta:
    def __init__(self, kind: str, text: str) -> None:
        self.type = kind
        if kind == "thinking_delta":
            self.thinking = text
        else:
            self.text = text


class _Event:
    def __init__(self, kind: str, **payload: Any) -> None:
        self.type = kind
        for k, v in payload.items():
            setattr(self, k, v)


class _Usage:
    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_generate_drains_stream_and_callback_receives_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = [
        _Event("message_start"),
        _Event("content_block_delta", delta=_Delta("thinking_delta", "ponder.")),
        _Event("content_block_delta", delta=_Delta("text_delta", '{"path": "src/main.py"}')),
        _Event("message_delta", usage=_Usage(input_tokens=123, output_tokens=45)),
        _Event("message_stop"),
    ]
    fake_response = _FakeResponse("ok")
    fake = _FakeClient([])
    # Replace stream-creating behavior so we can attach events.
    fake.messages._responses = [fake_response]

    def _stream(**kwargs: Any) -> _FakeStream:
        fake.messages.calls.append(kwargs)
        return _FakeStream(fake.messages._responses.pop(0), events=events)

    fake.messages.stream = _stream  # type: ignore[method-assign]
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: fake)

    received: list[generator.ProgressEvent] = []
    out = generate(_request(tmp_path), _config(tmp_path), progress=received.append)
    assert out == "ok"
    kinds = [e.kind for e in received]
    # B2: stream_started must be the very first event so the display can show
    # a pre-fill hint before any deltas arrive.
    assert kinds[0] == "stream_started"
    payload = received[0].payload
    assert isinstance(payload, dict)
    assert payload["input_tokens_estimate"] > 0
    assert payload["thinking_enabled"] is False
    assert "thinking_delta" in kinds
    assert "text_delta" in kinds
    # message_delta and the final synthetic usage event both push usage updates.
    assert kinds.count("usage") >= 1
    assert "done" in kinds


def test_generate_stream_started_reports_thinking_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_response = _FakeResponse("ok")
    fake = _FakeClient([])
    fake.messages._responses = [fake_response]

    def _stream(**kwargs: Any) -> _FakeStream:
        fake.messages.calls.append(kwargs)
        return _FakeStream(fake.messages._responses.pop(0), events=[_Event("message_stop")])

    fake.messages.stream = _stream  # type: ignore[method-assign]
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: fake)

    received: list[generator.ProgressEvent] = []
    cfg = _config(tmp_path, model="claude-opus-4-7", thinking_budget=16000)
    generate(_request(tmp_path), cfg, progress=received.append)
    assert received[0].kind == "stream_started"
    assert received[0].payload["thinking_enabled"] is True
    assert received[0].payload["model"] == "claude-opus-4-7"


def test_drain_stream_no_callback_still_iterates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A no-callback run must still iterate the stream so heartbeat/abort apply."""
    events = [_Event("message_stop")]
    fake_response = _FakeResponse("ok")
    fake = _FakeClient([])

    def _stream(**kwargs: Any) -> _FakeStream:
        fake.messages.calls.append(kwargs)
        return _FakeStream(fake_response, events=events)

    fake.messages.stream = _stream  # type: ignore[method-assign]
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: fake)
    assert generate(_request(tmp_path), _config(tmp_path)) == "ok"


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


def test_extract_fenced_content_returns_largest_block() -> None:
    text = (
        "Here's an example:\n\n"
        "```python\nexample = 1\n```\n\n"
        "And the real answer:\n\n"
        "```python\n"
        "def real() -> int:\n"
        "    return 42\n"
        "```\n"
    )
    content = extract_fenced_content(text)
    assert "def real" in content
    assert "example = 1" not in content


def test_extract_fenced_content_raises_on_no_block() -> None:
    with pytest.raises(ValueError):
        extract_fenced_content("plain prose, no fences here")


def test_extract_fenced_content_handles_unlabeled_fence() -> None:
    text = "```\nhello\nworld\n```"
    assert extract_fenced_content(text) == "hello\nworld"


def test_generate_single_file_uses_strict_system_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Single-file regen must reuse the strict system prompt so lint guidance carries over."""
    fake = _FakeClient([_FakeResponse("```python\nx = 1\n```")])
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: fake)
    out = generate_single_file(
        config=_config(tmp_path),
        recipe_body="# Recipe",
        target_path="src/demo/main.py",
        current_content="x = 0\n",
        neighbours={"src/demo/other.py": "from demo.main import x\n"},
        reason="bump x to 1",
        language="python",
    )
    assert "```python" in out
    call = fake.messages.calls[0]
    sys_text = call["system"][0]["text"]
    assert "Lint cleanliness" in sys_text  # strict prompt is in effect
    user_text = call["messages"][0]["content"][0]["text"]
    assert "src/demo/main.py" in user_text
    assert "bump x to 1" in user_text
    assert "src/demo/other.py" in user_text


# ---------------------------------------------------------------------------
# Structured outputs
# ---------------------------------------------------------------------------


def test_generate_sends_structured_output_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The generation call carries the contract schema as output_config.format."""
    from agent_scaffold.contract import GENERATION_RESULT_SCHEMA

    fake = _FakeClient([_FakeResponse("ok")])
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: fake)
    generate(_request(tmp_path), _config(tmp_path))
    fmt = fake.messages.calls[0]["output_config"]["format"]
    assert fmt == {"type": "json_schema", "schema": GENERATION_RESULT_SCHEMA}


def test_generate_merges_format_with_adaptive_effort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """format merges into the effort dict adaptive models already send."""
    fake = _FakeClient([_FakeResponse("ok")])
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: fake)
    cfg = _config(tmp_path, model="claude-opus-4-8", thinking_budget=16000)
    generate(_request(tmp_path), cfg)
    oc = fake.messages.calls[0]["output_config"]
    assert oc["effort"] == "high"
    assert oc["format"]["type"] == "json_schema"


def test_generate_legacy_contract_omits_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The escape hatch restores the free-form request shape."""
    fake = _FakeClient([_FakeResponse("ok")])
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: fake)
    cfg = _config(tmp_path).model_copy(update={"legacy_contract": True})
    generate(_request(tmp_path), cfg)
    assert "output_config" not in fake.messages.calls[0]


def test_repair_sends_structured_output_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The repaired re-emission is grammar-constrained too."""
    fake = _FakeClient([_FakeResponse("ok")])
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: fake)
    generator.repair("{bad", "unterminated", _config(tmp_path))
    assert fake.messages.calls[0]["output_config"]["format"]["type"] == "json_schema"


def test_refusal_stop_reason_raises_contract_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_scaffold.contract import ContractParseError

    fake = _FakeClient([_FakeResponse("", stop_reason="refusal")])
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: fake)
    with pytest.raises(ContractParseError) as exc_info:
        generate(_request(tmp_path), _config(tmp_path))
    assert exc_info.value.tier == "refusal"
    # No retry: a refusal is deterministic for the same prompt.
    assert len(fake.messages.calls) == 1


def test_truncation_stop_reason_raises_contract_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_scaffold.contract import ContractParseError

    fake = _FakeClient([_FakeResponse('{"partial', stop_reason="max_tokens")])
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: fake)
    with pytest.raises(ContractParseError) as exc_info:
        generate(_request(tmp_path), _config(tmp_path))
    assert exc_info.value.tier == "truncation"
    assert "max_tokens" in exc_info.value.reason
    assert exc_info.value.raw == '{"partial'


# ---------------------------------------------------------------------------
# Repair-model routing
# ---------------------------------------------------------------------------


def _repair_validation_kwargs(tmp_path: Path) -> dict[str, Any]:
    return {
        "recipe_body": "# Recipe\n",
        "language_hints": {"language": "python", "manifest": "pyproject.toml"},
        "project_file_list": ["pyproject.toml"],
        "failing_command": "uv sync",
        "validation_output": "boom",
        "implicated_files": {},
        "language": "python",
    }


def test_repair_validation_routes_to_the_repair_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The validation-repair call uses repair_model, not the session model."""
    fake = _FakeClient([_FakeResponse("ok")])
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: fake)
    cfg = _config(tmp_path, model="claude-opus-4-8", repair_model="claude-sonnet-5")
    generator.repair_validation(config=cfg, **_repair_validation_kwargs(tmp_path))
    assert fake.messages.calls[0]["model"] == "claude-sonnet-5"


def test_repair_validation_keeps_session_model_when_equal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeClient([_FakeResponse("ok")])
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: fake)
    cfg = _config(tmp_path, model="claude-haiku-4-5", repair_model="claude-haiku-4-5")
    generator.repair_validation(config=cfg, **_repair_validation_kwargs(tmp_path))
    assert fake.messages.calls[0]["model"] == "claude-haiku-4-5"
