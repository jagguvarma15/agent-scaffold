"""Anthropic API integration: send the assembled prompt and return raw text.

The Anthropic client is created via ``_make_client`` so tests can monkeypatch
the seam and supply canned responses.
"""

from __future__ import annotations

import hashlib
import importlib.resources as resources
import logging
import time
from typing import Any, Protocol, cast

import anthropic
import yaml
from pydantic import BaseModel

from agent_forge.config import Config
from agent_forge.context import AssembledContext

logger = logging.getLogger(__name__)

PROMPTS_PACKAGE = "agent_forge.prompts"
SYSTEM_PROMPT_FILE = "system.md"
USER_TEMPLATE_FILE = "user_template.md"
REPAIR_TEMPLATE_FILE = "repair.md"


class GenerationRequest(BaseModel):
    project_name: str
    target_language: str
    framework: str
    assembled_context: AssembledContext
    language_hints: dict[str, Any]


class _MessagesClient(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


class _AnthropicLike(Protocol):
    @property
    def messages(self) -> _MessagesClient: ...


def _make_client(config: Config) -> _AnthropicLike:
    """Construct the Anthropic SDK client. Test seam."""
    return cast(
        _AnthropicLike,
        anthropic.Anthropic(api_key=config.anthropic_api_key),
    )


def _load_prompt(filename: str) -> str:
    return resources.files(PROMPTS_PACKAGE).joinpath(filename).read_text(encoding="utf-8")


def prompts_signature() -> str:
    """Stable hash of the bundled prompt files; used as a cache-key component."""
    h = hashlib.sha256()
    for filename in (SYSTEM_PROMPT_FILE, USER_TEMPLATE_FILE, REPAIR_TEMPLATE_FILE):
        h.update(filename.encode())
        h.update(b"\0")
        h.update(_load_prompt(filename).encode())
        h.update(b"\0")
    return h.hexdigest()[:8]


def _render_user_message(req: GenerationRequest) -> str:
    template = _load_prompt(USER_TEMPLATE_FILE)
    hints_yaml = yaml.safe_dump(req.language_hints, sort_keys=False).strip()
    # Use str.replace because the template contains literal `{` / `}` for the
    # JSON example block.
    rendered = (
        template.replace("{project_name}", req.project_name)
        .replace("{target_language}", req.target_language)
        .replace("{language_hints_yaml}", hints_yaml)
        .replace("{assembled_context}", req.assembled_context.body)
    )
    return rendered


def _render_repair_message(raw_response: str, validation_error: str) -> str:
    template = _load_prompt(REPAIR_TEMPLATE_FILE)
    return template.replace("{validation_error}", validation_error).replace(
        "{raw_response}", raw_response
    )


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, anthropic.RateLimitError):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        status = getattr(exc, "status_code", None)
        return status is not None and 500 <= status < 600
    if isinstance(exc, anthropic.APIConnectionError):
        return True
    return False


def _extract_text(response: Any) -> str:
    """Extract concatenated text from an Anthropic ``Message`` response."""
    parts: list[str] = []
    for block in getattr(response, "content", []) or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "".join(parts)


class UsageInfo(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


def _extract_usage(response: Any) -> UsageInfo:
    """Extract token usage from an Anthropic response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return UsageInfo()
    return UsageInfo(
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
        cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
    )


# Module-level last usage for reporting
_last_usage: UsageInfo = UsageInfo()


def get_last_usage() -> UsageInfo:
    """Return token usage from the most recent API call."""
    return _last_usage


def _call_with_retry(
    client: _AnthropicLike,
    *,
    config: Config,
    system_blocks: list[dict[str, Any]],
    user_message: str,
) -> str:
    global _last_usage
    delays = [1.0, 2.0, 4.0]
    last_exc: BaseException | None = None
    for attempt in range(len(delays) + 1):
        try:
            logger.debug("Calling %s (attempt %d, max_tokens=%d)", config.model, attempt + 1, config.max_tokens)
            t0 = time.time()
            response = client.messages.create(
                model=config.model,
                max_tokens=config.max_tokens,
                system=system_blocks,
                messages=[{"role": "user", "content": user_message}],
            )
            elapsed = time.time() - t0
            _last_usage = _extract_usage(response)
            logger.debug(
                "Response received in %.1fs — input: %d, output: %d tokens",
                elapsed, _last_usage.input_tokens, _last_usage.output_tokens,
            )
            return _extract_text(response)
        except Exception as exc:
            last_exc = exc
            if attempt >= len(delays) or not _is_retryable(exc):
                raise
            logger.debug("Retryable error: %s — retrying in %.0fs", exc, delays[attempt])
            time.sleep(delays[attempt])
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("unreachable")


def _system_blocks() -> list[dict[str, Any]]:
    system_text = _load_prompt(SYSTEM_PROMPT_FILE)
    return [
        {
            "type": "text",
            "text": system_text,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def generate(req: GenerationRequest, config: Config) -> str:
    """Send the assembled prompt to the Anthropic API and return raw text."""
    client = _make_client(config)
    user_message = _render_user_message(req)
    return _call_with_retry(
        client,
        config=config,
        system_blocks=_system_blocks(),
        user_message=user_message,
    )


def repair(raw_response: str, validation_error: str, config: Config) -> str:
    """Ask the model to repair a previous invalid response."""
    client = _make_client(config)
    user_message = _render_repair_message(raw_response, validation_error)
    return _call_with_retry(
        client,
        config=config,
        system_blocks=_system_blocks(),
        user_message=user_message,
    )
