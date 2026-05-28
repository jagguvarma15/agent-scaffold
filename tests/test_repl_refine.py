"""Tests for ``agent_scaffold.repl.refine`` — Haiku-based refinement interpreter.

Every test monkeypatches ``_make_haiku_client`` so the suite never touches
the Anthropic API. Covers the happy path (JSON in → StatePatch out), the
markdown-fence tolerance, schema validation (drop bad-type keys, ignore
unknown keys), and the recoverable-failure paths
(:class:`RefinementError` on network / parse / shape errors).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anthropic
import pytest

from agent_scaffold.config import Config
from agent_scaffold.repl.refine import (
    INTERPRET_SYSTEM,
    RefinementError,
    _parse_json,
    _patch_from_dict,
    interpret_refinement,
    serialize_state_for_prompt,
)
from agent_scaffold.repl.session import SessionState, StatePatch
from agent_scaffold.sources import DEPLOYMENTS_SPEC, ResolvedSource

# ---------------------------------------------------------------------------
# Fake Anthropic client
# ---------------------------------------------------------------------------


class _Block:
    def __init__(self, text: str) -> None:
        self.text = text


class _Response:
    def __init__(self, text: str) -> None:
        self.content = [_Block(text)]


class _Messages:
    """Capture the kwargs sent to messages.create so tests can verify them."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _Response:
        self.calls.append(kwargs)
        return _Response(self.response_text)


class _Client:
    def __init__(self, response_text: str) -> None:
        self.messages = _Messages(response_text)


def _install_client(monkeypatch: pytest.MonkeyPatch, response_text: str) -> _Client:
    client = _Client(response_text)
    monkeypatch.setattr("agent_scaffold.repl.refine._make_haiku_client", lambda _cfg: client)
    return client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(
        anthropic_api_key="test-key",
        cache_dir=tmp_path / "cache",
        failures_dir=tmp_path / "cache" / "failures",
    )


@pytest.fixture
def base_state(cfg: Config, tmp_path: Path) -> SessionState:
    src = ResolvedSource(
        spec=DEPLOYMENTS_SPEC,
        path=tmp_path / "deployments",
        label="test",
        kind="explicit-path",
        commit_sha=None,
    )
    return SessionState(cfg=cfg, deployments=src, blueprints=src)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_interpret_returns_patch_from_clean_json(
    monkeypatch: pytest.MonkeyPatch, base_state: SessionState, cfg: Config
) -> None:
    _install_client(monkeypatch, '{"model":"claude-sonnet-4-6","strict":true}')
    patch = interpret_refinement(base_state, "swap to sonnet, be strict", cfg)
    assert patch.model == "claude-sonnet-4-6"
    assert patch.strict is True


def test_interpret_tolerates_markdown_fence_wrap(
    monkeypatch: pytest.MonkeyPatch, base_state: SessionState, cfg: Config
) -> None:
    """Smaller models sometimes wrap JSON in ```json…``` despite being told not to."""
    body = '```json\n{"model":"claude-sonnet-4-6"}\n```'
    _install_client(monkeypatch, body)
    patch = interpret_refinement(base_state, "...", cfg)
    assert patch.model == "claude-sonnet-4-6"


def test_interpret_routes_accumulators_through_dedicated_fields(
    monkeypatch: pytest.MonkeyPatch, base_state: SessionState, cfg: Config
) -> None:
    payload = {
        "add_dependencies": {"python": {"postgres": ">=14", "redis": ">=7"}},
        "remove_steps": ["smoke_test"],
        "remove_roles": ["kafka-consumer"],
        "notes": "use ECS not GKE",
    }
    _install_client(monkeypatch, json.dumps(payload))
    patch = interpret_refinement(base_state, "...", cfg)
    assert patch.add_dependencies == {"python": {"postgres": ">=14", "redis": ">=7"}}
    assert patch.remove_steps == ["smoke_test"]
    assert patch.remove_roles == ["kafka-consumer"]
    assert patch.notes == "use ECS not GKE"


def test_interpret_sends_haiku_model_and_system_prompt(
    monkeypatch: pytest.MonkeyPatch, base_state: SessionState, cfg: Config
) -> None:
    client = _install_client(monkeypatch, '{"strict":true}')
    interpret_refinement(base_state, "be strict", cfg)
    assert client.messages.calls, "should have called messages.create exactly once"
    call = client.messages.calls[0]
    # Hard-coded — refinement is a system tool.
    assert "haiku" in call["model"]
    assert call["system"] == INTERPRET_SYSTEM
    assert call["max_tokens"] == 1024
    # The user message should include the refinement text verbatim.
    user_text = call["messages"][0]["content"]
    assert "be strict" in user_text


# ---------------------------------------------------------------------------
# Empty input — no API call
# ---------------------------------------------------------------------------


def test_interpret_empty_text_returns_empty_patch_no_api_call(
    monkeypatch: pytest.MonkeyPatch, base_state: SessionState, cfg: Config
) -> None:
    def fail(_cfg: Config) -> _Client:
        raise AssertionError("_make_haiku_client should not be called for empty input")

    monkeypatch.setattr("agent_scaffold.repl.refine._make_haiku_client", fail)
    assert interpret_refinement(base_state, "", cfg) == StatePatch()
    assert interpret_refinement(base_state, "   \n", cfg) == StatePatch()


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_interpret_malformed_json_raises_refinement_error(
    monkeypatch: pytest.MonkeyPatch, base_state: SessionState, cfg: Config
) -> None:
    _install_client(monkeypatch, "not json {{{")
    with pytest.raises(RefinementError, match="not valid JSON"):
        interpret_refinement(base_state, "...", cfg)


def test_interpret_non_object_response_raises(
    monkeypatch: pytest.MonkeyPatch, base_state: SessionState, cfg: Config
) -> None:
    _install_client(monkeypatch, "[1, 2, 3]")
    with pytest.raises(RefinementError, match="JSON object"):
        interpret_refinement(base_state, "...", cfg)


def test_interpret_anthropic_error_wraps_as_refinement_error(
    monkeypatch: pytest.MonkeyPatch, base_state: SessionState, cfg: Config
) -> None:
    class _BadMessages:
        def create(self, **kwargs: Any) -> Any:
            raise anthropic.APIConnectionError(request=None)  # type: ignore[arg-type]

    class _BadClient:
        messages = _BadMessages()

    monkeypatch.setattr("agent_scaffold.repl.refine._make_haiku_client", lambda _cfg: _BadClient())
    with pytest.raises(RefinementError, match="Haiku call failed"):
        interpret_refinement(base_state, "...", cfg)


def test_interpret_response_without_content_raises(
    monkeypatch: pytest.MonkeyPatch, base_state: SessionState, cfg: Config
) -> None:
    class _EmptyContent:
        content: list[Any] = []

    class _M:
        def create(self, **kwargs: Any) -> Any:
            return _EmptyContent()

    class _C:
        messages = _M()

    monkeypatch.setattr("agent_scaffold.repl.refine._make_haiku_client", lambda _cfg: _C())
    with pytest.raises(RefinementError, match="empty content"):
        interpret_refinement(base_state, "...", cfg)


# ---------------------------------------------------------------------------
# Schema whitelist — silent drops for hallucinated / wrong-type keys
# ---------------------------------------------------------------------------


def test_patch_from_dict_drops_unknown_keys() -> None:
    patch = _patch_from_dict({"model": "claude-sonnet-4-6", "color": "blue"})
    assert patch.model == "claude-sonnet-4-6"


def test_patch_from_dict_drops_wrong_type_scalars() -> None:
    """Bad types are silently dropped — hallucination shouldn't crash the REPL."""
    patch = _patch_from_dict(
        {
            "model": 42,  # not a string
            "max_tokens": "lots",  # not an int
            "strict": "yes",  # not a bool
            "thinking_budget": True,  # bool is a subclass of int but disallowed
        }
    )
    assert patch.model is None
    assert patch.max_tokens is None
    assert patch.strict is None
    assert patch.thinking_budget is None


def test_patch_from_dict_drops_empty_strings() -> None:
    patch = _patch_from_dict({"model": "", "framework": "langgraph"})
    assert patch.model is None
    assert patch.framework == "langgraph"


def test_patch_from_dict_keeps_well_typed_accumulators() -> None:
    patch = _patch_from_dict(
        {
            "add_dependencies": {"python": {"x": "1"}, "bad": "skip-me"},
            "remove_steps": ["a", 1, "b"],  # ints dropped
        }
    )
    assert patch.add_dependencies == {"python": {"x": "1"}}
    assert patch.remove_steps == ["a", "b"]


def test_patch_from_dict_blank_notes_returns_none() -> None:
    assert _patch_from_dict({"notes": "   "}).notes is None
    assert _patch_from_dict({"notes": "real note"}).notes == "real note"


# ---------------------------------------------------------------------------
# _parse_json fence handling
# ---------------------------------------------------------------------------


def test_parse_json_handles_bare_object() -> None:
    assert _parse_json('{"a": 1}') == {"a": 1}


def test_parse_json_strips_json_fence() -> None:
    assert _parse_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_json_strips_plain_fence() -> None:
    assert _parse_json('```\n{"a": 1}\n```') == {"a": 1}


def test_parse_json_rejects_non_object_at_root() -> None:
    with pytest.raises(RefinementError, match="JSON object"):
        _parse_json("[1,2]")


# ---------------------------------------------------------------------------
# State serialization
# ---------------------------------------------------------------------------


def test_serialize_state_emits_stable_json_with_user_fields(
    base_state: SessionState,
) -> None:
    out = serialize_state_for_prompt(base_state)
    payload = json.loads(out)
    # Required keys are present even when unset (so the LLM sees the schema).
    for key in ("recipe", "language", "framework", "model", "strict", "extra_dependencies"):
        assert key in payload
    # Session-scope inputs (cfg, deployments, blueprints) are NOT leaked into the prompt.
    assert "cfg" not in payload
    assert "deployments" not in payload
    assert "blueprints" not in payload
