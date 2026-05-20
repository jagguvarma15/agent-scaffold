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

from agent_scaffold.config import Config
from agent_scaffold.context import AssembledContext

logger = logging.getLogger(__name__)

PROMPTS_PACKAGE = "agent_scaffold.prompts"
SYSTEM_PROMPT_FILE = "system.md"
SYSTEM_STRICT_PROMPT_FILE = "system_strict.md"
USER_TEMPLATE_FILE = "user_template.md"
REPAIR_TEMPLATE_FILE = "repair.md"
CACHE_SPLIT_MARKER = "<!-- ===== CACHE SPLIT ===== -->"

# Anthropic ephemeral cache requires a minimum of 1024 tokens on Sonnet/Opus 4.x.
# We approximate tokens as len/4. A degenerate recipe with no references could
# fall under this threshold; we fall back to a single uncached block in that case.
_MIN_CACHE_CHARS = 1024 * 4


class GenerationRequest(BaseModel):
    project_name: str
    target_language: str
    framework: str
    assembled_context: AssembledContext
    language_hints: dict[str, Any]
    extra_required: list[str] = []
    strict: bool = False


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
    for filename in (
        SYSTEM_PROMPT_FILE,
        SYSTEM_STRICT_PROMPT_FILE,
        USER_TEMPLATE_FILE,
        REPAIR_TEMPLATE_FILE,
    ):
        h.update(filename.encode())
        h.update(b"\0")
        h.update(_load_prompt(filename).encode())
        h.update(b"\0")
    return h.hexdigest()[:8]


def _render_extra_required_block(extra_required: list[str]) -> str:
    if not extra_required:
        return ""
    lines = [f"- Recipe-required: {path}" for path in extra_required]
    return "\n" + "\n".join(lines)


def _render_user_message(req: GenerationRequest) -> tuple[str, str]:
    """Render the user message split into (cacheable_context, project_tail).

    The cacheable_context block holds the language hints and assembled spec —
    stable per recipe+language so repeat runs hit the prompt cache. The
    project_tail holds project-specific data (name) and the output-format
    instructions (including any recipe-required files), varying per run.
    """
    template = _load_prompt(USER_TEMPLATE_FILE)
    hints_yaml = yaml.safe_dump(req.language_hints, sort_keys=False).strip()
    extra_block = _render_extra_required_block(req.extra_required)
    rendered = (
        template.replace("{project_name}", req.project_name)
        .replace("{target_language}", req.target_language)
        .replace("{language_hints_yaml}", hints_yaml)
        .replace("{assembled_context}", req.assembled_context.body)
        .replace("{extra_required_block}", extra_block)
    )
    if CACHE_SPLIT_MARKER in rendered:
        context_block, tail_block = rendered.split(CACHE_SPLIT_MARKER, 1)
        return context_block.rstrip() + "\n", tail_block.lstrip()
    return rendered, ""


def _build_user_content(context_block: str, tail_block: str) -> list[dict[str, Any]]:
    """Build a multi-block user content payload, caching the context block.

    Falls back to a single uncached block when the context is too small to
    meet Anthropic's minimum cache size, or when no tail block is present.
    """
    if not tail_block:
        return [{"type": "text", "text": context_block}]
    if len(context_block) < _MIN_CACHE_CHARS:
        return [{"type": "text", "text": context_block + tail_block}]
    return [
        {
            "type": "text",
            "text": context_block,
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": tail_block},
    ]


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
    user_content: list[dict[str, Any]],
) -> str:
    global _last_usage
    delays = [1.0, 2.0, 4.0]
    last_exc: BaseException | None = None
    create_kwargs: dict[str, Any] = {
        "model": config.model,
        "max_tokens": config.max_tokens,
        "system": system_blocks,
        "messages": [{"role": "user", "content": user_content}],
    }
    if config.thinking_budget:
        logger.debug("Extended thinking enabled, budget=%d", config.thinking_budget)
        create_kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": config.thinking_budget,
        }
    for attempt in range(len(delays) + 1):
        try:
            logger.debug(
                "Calling %s (attempt %d, max_tokens=%d)",
                config.model,
                attempt + 1,
                config.max_tokens,
            )
            t0 = time.time()
            response = client.messages.create(**create_kwargs)
            elapsed = time.time() - t0
            _last_usage = _extract_usage(response)
            logger.debug(
                "Response received in %.1fs — input: %d, output: %d tokens",
                elapsed,
                _last_usage.input_tokens,
                _last_usage.output_tokens,
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


def _system_blocks(strict: bool = False) -> list[dict[str, Any]]:
    filename = SYSTEM_STRICT_PROMPT_FILE if strict else SYSTEM_PROMPT_FILE
    system_text = _load_prompt(filename)
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
    context_block, tail_block = _render_user_message(req)
    return _call_with_retry(
        client,
        config=config,
        system_blocks=_system_blocks(req.strict),
        user_content=_build_user_content(context_block, tail_block),
    )


def repair(
    raw_response: str,
    validation_error: str,
    config: Config,
    strict: bool = False,
) -> str:
    """Ask the model to repair a previous invalid response."""
    client = _make_client(config)
    user_message = _render_repair_message(raw_response, validation_error)
    return _call_with_retry(
        client,
        config=config,
        system_blocks=_system_blocks(strict),
        user_content=[{"type": "text", "text": user_message}],
    )
