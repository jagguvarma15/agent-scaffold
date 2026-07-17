"""Anthropic API integration: send the assembled prompt and return raw text.

The Anthropic client is created via ``_make_client`` so tests can monkeypatch
the seam and supply canned responses.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.resources as resources
import logging
import re
import time
from collections.abc import Callable, Iterator
from typing import Any, Protocol, cast

import anthropic
import yaml
from pydantic import BaseModel

from agent_scaffold import models
from agent_scaffold.config import Config
from agent_scaffold.context import AssembledContext
from agent_scaffold.progress import ProgressEvent

logger = logging.getLogger(__name__)

PROMPTS_PACKAGE = "agent_scaffold.prompts"
SYSTEM_PROMPT_FILE = "system.md"
SYSTEM_STRICT_PROMPT_FILE = "system_strict.md"
USER_TEMPLATE_FILE = "user_template.md"
REPAIR_TEMPLATE_FILE = "repair.md"
SINGLE_FILE_TEMPLATE_FILE = "single_file.md"
VALIDATION_REPAIR_TEMPLATE_FILE = "validation_repair.md"
CACHE_SPLIT_MARKER = "<!-- ===== CACHE SPLIT ===== -->"
# Hot/warm boundary inside the context block, present only when the assembled
# context carries cache-tier segments (load_list recipes). Everything before
# it is hot (1h cache TTL — stable across runs), everything after is warm
# (5m — stable within a session).
CACHE_SPLIT_WARM_MARKER = "<!-- ===== CACHE SPLIT WARM ===== -->"

_FENCED_BLOCK_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)\n```", re.DOTALL)

# Tokens are approximated as len/4 (matching context.py). The per-model cache
# minimum comes from models.min_cache_tokens; a block below it gets no
# breakpoint because the API would silently refuse to cache it.
_CHARS_PER_TOKEN = 4


def _min_cache_chars(model: str) -> int:
    return models.min_cache_tokens(model) * _CHARS_PER_TOKEN


def _budget_to_effort(budget: int) -> str:
    """Map the legacy ``thinking_budget`` int to an ``output_config.effort`` tier.

    Anthropic's adaptive API accepts: ``low``, ``medium``, ``high``, ``xhigh``,
    ``max``. The CLI's effort presets currently produce 8000 (→ medium) and
    16000 (→ high); the wider buckets future-proof other values.
    """
    if budget <= 4000:
        return "low"
    if budget <= 10000:
        return "medium"
    if budget <= 20000:
        return "high"
    if budget <= 40000:
        return "xhigh"
    return "max"


def _build_thinking_kwargs(model: str, thinking_budget: int | None) -> dict[str, Any]:
    """Return the ``thinking``/``output_config`` kwargs for ``messages.stream``.

    Returns an empty dict when no thinking is requested. Picks the adaptive or
    legacy enabled shape based on the model name.
    """
    if not thinking_budget:
        return {}
    if models.uses_adaptive_thinking(model):
        return {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": _budget_to_effort(thinking_budget)},
        }
    return {"thinking": {"type": "enabled", "budget_tokens": thinking_budget}}


class GenerationRequest(BaseModel):
    project_name: str
    target_language: str
    framework: str
    assembled_context: AssembledContext
    language_hints: dict[str, Any]
    extra_required: list[str] = []
    strict: bool = False
    # Refinement deltas — populated by the REPL's free-text interpreter so
    # the LLM honours "add postgres / skip docker_up / use sonnet" directives.
    # Empty by default; cmd_new leaves them untouched.
    extra_dependencies: dict[str, dict[str, str]] = {}
    extra_steps: list[str] = []
    removed_steps: list[str] = []
    removed_roles: list[str] = []
    refinement_notes: list[str] = []
    # The agent's role / persona — the user's "describe your agent" text (or the
    # recipe's default). The model must adopt it as the backend's system prompt
    # so the generated /chat agent answers in character. Empty for vanilla runs.
    agent_role: str | None = None
    # Compact summary of the resolved capability stack. Rendered into the
    # user template as a "Resolved capabilities" block so the LLM sees the
    # contract in a single scannable section (the full bodies live in the
    # assembled context under "## Capability:" headers).
    capabilities_brief: list[dict[str, Any]] = []


class _MessageStream(Protocol):
    def get_final_message(self) -> Any: ...
    def __iter__(self) -> Iterator[Any]: ...


class _MessageStreamManager(Protocol):
    def __enter__(self) -> _MessageStream: ...
    def __exit__(self, *args: Any) -> Any: ...


class _MessagesClient(Protocol):
    def stream(self, **kwargs: Any) -> _MessageStreamManager: ...


class _AnthropicLike(Protocol):
    @property
    def messages(self) -> _MessagesClient: ...


# How long without a stream event before we surface a heartbeat / abort.
HEARTBEAT_WARN_SECONDS = 30.0
HEARTBEAT_ABORT_SECONDS = 300.0


class StreamStuckError(RuntimeError):
    """Raised when the Anthropic stream goes silent past HEARTBEAT_ABORT_SECONDS."""


def _make_client(config: Config) -> _AnthropicLike:
    """Construct the Anthropic SDK client. Test seam."""
    return cast(
        _AnthropicLike,
        anthropic.Anthropic(api_key=config.anthropic_api_key),
    )


@functools.lru_cache(maxsize=8)
def _load_prompt(filename: str) -> str:
    """Read a bundled prompt file. Cached because the files ship inside the
    wheel and never change at runtime — without the cache, ``_render_user_message``
    + ``prompts_signature`` re-read the same five files on every generate
    *and* every cache-fingerprint computation."""
    return resources.files(PROMPTS_PACKAGE).joinpath(filename).read_text(encoding="utf-8")


@functools.lru_cache(maxsize=1)
def prompts_signature() -> str:
    """Stable hash of the bundled prompt files; used as a cache-key component.

    Memoized — same rationale as :func:`_load_prompt`. Bundled prompts don't
    change at runtime, so a one-shot computation per process is enough; without
    it we re-hashed all five files on every ``run_generation``.
    """
    h = hashlib.sha256()
    for filename in (
        SYSTEM_PROMPT_FILE,
        SYSTEM_STRICT_PROMPT_FILE,
        USER_TEMPLATE_FILE,
        REPAIR_TEMPLATE_FILE,
        SINGLE_FILE_TEMPLATE_FILE,
        VALIDATION_REPAIR_TEMPLATE_FILE,
    ):
        h.update(filename.encode())
        h.update(b"\0")
        h.update(_load_prompt(filename).encode())
        h.update(b"\0")
    return h.hexdigest()[:8]


def _render_extra_required_block(extra_required: list[str]) -> str:
    if not extra_required:
        return ""
    lines = [f"  - {path}" for path in extra_required]
    # Emphatic + exact: recipes sometimes require a layout (e.g. an ``app/``
    # package) that conflicts with the language's idiomatic ``src/`` convention.
    # Without this the model emits its preferred layout and the required-files
    # contract fails. The re-export-shim hint lets it keep its layout *and*
    # satisfy the required paths.
    return (
        "\n- The recipe REQUIRES a file at each of these EXACT paths — emit every "
        "one verbatim. Do not relocate them (e.g. into `src/`), rename them, or "
        "omit them. If your package layout differs, still create the file at the "
        "required path (a thin module that re-exports from your actual layout is "
        "fine):\n" + "\n".join(lines)
    )


def _render_capabilities_block(req: GenerationRequest) -> str:
    """Render the per-run "Resolved capabilities" summary.

    Lists each resolved capability id alongside its canonical env vars,
    docker service name, and any frontend ``emit_files`` glob — the same
    facts ``_render_user_message`` already exposes elsewhere, but in a
    single scannable block the LLM can lean on while writing
    ``docker-compose.yml`` and ``.env.example``. Empty when no capabilities
    resolved.
    """
    if not req.capabilities_brief:
        return ""
    parts: list[str] = ["# Resolved capabilities", ""]
    for cap in req.capabilities_brief:
        cap_id = cap.get("id", "?")
        kind = cap.get("kind", "?")
        env_vars = cap.get("env_vars") or []
        docker_service = cap.get("docker_service")
        emit_globs = cap.get("emit_globs") or []
        parts.append(f"- **{cap_id}** ({kind})")
        if env_vars:
            joined = ", ".join(f"`{v}`" for v in env_vars)
            parts.append(f"  - env vars: {joined}")
        if docker_service:
            parts.append(f"  - docker service: `{docker_service}`")
        if emit_globs:
            parts.append(
                "  - templates copied by scaffold (do NOT re-emit): "
                + ", ".join(f"`{g}`" for g in emit_globs)
            )
    return "\n" + "\n".join(parts) + "\n"


def _render_role_block(req: GenerationRequest) -> str:
    """Render the "# Agent role" block — the user's persona for the backend.

    Lives in the project_tail (after CACHE SPLIT) like refinements: it's per-run
    user intent, not part of the stable recipe spec. Returns ``""`` when unset so
    vanilla runs (no description, no recipe default) see no extra block.
    """
    if not req.agent_role or not req.agent_role.strip():
        return ""
    return (
        "\n# Agent role\n\n"
        "The user described the agent they want. Adopt this as the backend's "
        "**system prompt** — the generated agent must use it (verbatim or very "
        "close) as its system message so the `POST /chat` endpoint answers in "
        "character. Set the project's display title from it too:\n\n"
        f"{req.agent_role.strip()}\n"
    )


def _render_refinement_block(req: GenerationRequest) -> str:
    """Render the per-run "User refinements" Markdown block.

    Lives in the project_tail (after CACHE SPLIT) so it never poisons the
    prompt-cache key — refinements are per-run by design. Returns ``""``
    when nothing's set so vanilla cmd_new runs see no extra block.

    The leading/trailing newlines exist so the block can be substituted on
    its own line in user_template.md without producing blank-line artifacts
    when empty.
    """
    if not (
        req.extra_dependencies
        or req.extra_steps
        or req.removed_steps
        or req.removed_roles
        or req.refinement_notes
    ):
        return ""
    parts: list[str] = [
        "# User refinements",
        "",
        "The user has refined the spec with the following directives. "
        "Honour them in the generated files; they override the recipe's defaults.",
    ]
    if req.extra_dependencies:
        parts.extend(["", "## Additional dependencies", ""])
        for lang, pkgs in req.extra_dependencies.items():
            for pkg, version in pkgs.items():
                parts.append(f"- {lang}: `{pkg}` = `{version}`")
    if req.extra_steps:
        parts.extend(["", "## Additional setup steps", ""])
        parts.extend(f"- {step}" for step in req.extra_steps)
    if req.removed_steps:
        parts.extend(["", "## Skip these steps", ""])
        parts.extend(f"- {step}" for step in req.removed_steps)
    if req.removed_roles:
        parts.extend(["", "## Skip these roles", ""])
        parts.extend(f"- {role}" for role in req.removed_roles)
    if req.refinement_notes:
        parts.extend(["", "## Additional guidance", ""])
        parts.extend(req.refinement_notes)
    return "\n" + "\n".join(parts) + "\n"


def _render_user_message(req: GenerationRequest) -> tuple[str, str]:
    """Render the user message split into (cacheable_context, project_tail).

    The cacheable_context block holds the language hints and assembled spec —
    stable per recipe+language so repeat runs hit the prompt cache. The
    project_tail holds project-specific data (name, refinements) and the
    output-format instructions (including any recipe-required files), all
    of which vary per run.

    When the assembled context carries cache-tier segments (load_list
    recipes), the hot/warm boundary is marked with
    :data:`CACHE_SPLIT_WARM_MARKER` inside the context block so
    :func:`_build_user_content` can place per-tier breakpoints.
    """
    template = _load_prompt(USER_TEMPLATE_FILE)
    hints_yaml = yaml.safe_dump(req.language_hints, sort_keys=False).strip()
    extra_block = _render_extra_required_block(req.extra_required)
    refinement_block = _render_refinement_block(req)
    capabilities_block = _render_capabilities_block(req)
    rendered = (
        template.replace("{project_name}", req.project_name)
        .replace("{target_language}", req.target_language)
        .replace("{language_hints_yaml}", hints_yaml)
        .replace("{assembled_context}", _context_for_prompt(req.assembled_context))
        .replace("{extra_required_block}", extra_block)
        .replace("{refinement_block}", refinement_block)
        .replace("{capabilities_block}", capabilities_block)
        .replace("{role_block}", _render_role_block(req))
    )
    if CACHE_SPLIT_MARKER in rendered:
        context_block, tail_block = rendered.split(CACHE_SPLIT_MARKER, 1)
        return context_block.rstrip() + "\n", tail_block.lstrip()
    return rendered, ""


def _context_for_prompt(ctx: Any) -> str:
    """The assembled context as it appears in the prompt.

    Segment-aware recipes get the hot docs first, then the warm-tier marker,
    then the warm docs (recipe body + capabilities + the rest) — same content
    as ``ctx.body``, regrouped so the stable hot prefix survives warm-tier
    churn in the prompt cache.
    """
    segments = getattr(ctx, "segments", None) or []
    if not segments:
        return str(ctx.body)
    hot = "\n".join(s.text for s in segments if s.cache_tier == "hot").strip()
    rest = "\n".join(s.text for s in segments if s.cache_tier != "hot").strip()
    if not hot:
        return rest + "\n"
    return hot + f"\n{CACHE_SPLIT_WARM_MARKER}\n" + rest + "\n"


def _hot_cache_control(hot_ttl_1h: bool) -> dict[str, Any]:
    """Cache control for the stable hot prefix.

    A one-shot generation reads nothing back from cache, so the default 5m
    write (plain ephemeral) is the right cost trade — 1h writes bill input at
    2x versus 1.25x for 5m. ``hot_ttl_1h`` opts the prefix back into the 1h
    TTL for sessions that regenerate within the hour (Config.cache_ttl)."""
    if hot_ttl_1h:
        return {"type": "ephemeral", "ttl": "1h"}
    return {"type": "ephemeral"}


def _build_user_content(
    context_block: str, tail_block: str, model: str, *, hot_ttl_1h: bool = False
) -> list[dict[str, Any]]:
    """Build a multi-block user content payload with tiered cache breakpoints.

    Layout when the context carries a hot/warm split (load_list recipes):

        [hints + hot docs   → cache_control ephemeral (5m by default; 1h opt-in)]
        [warm docs          → cache_control ephemeral (5m)]
        [project tail       → uncached]

    With the cached system block that's 3 breakpoints — one spare under
    Anthropic's 4-breakpoint limit. When the hot prefix opts into a 1h TTL it
    still precedes the 5m warm block, as the API requires. Blocks under
    ``model``'s minimum cacheable size collapse into their neighbor rather
    than wasting a breakpoint; recipes without segments keep the legacy
    single cached context block.
    """
    min_cache_chars = _min_cache_chars(model)
    hot_cc = _hot_cache_control(hot_ttl_1h)
    if not tail_block:
        return [{"type": "text", "text": context_block}]

    if CACHE_SPLIT_WARM_MARKER in context_block:
        hot_block, warm_block = context_block.split(CACHE_SPLIT_WARM_MARKER, 1)
        hot_block = hot_block.rstrip() + "\n"
        warm_block = warm_block.lstrip()
        if len(hot_block) < min_cache_chars:
            # Hot too small to cache alone — fold into one warm-cached block.
            context_block = hot_block + warm_block
        elif len(warm_block) < min_cache_chars:
            # Warm too small — one hot-cached block covers everything stable.
            return [
                {
                    "type": "text",
                    "text": hot_block + warm_block,
                    "cache_control": hot_cc,
                },
                {"type": "text", "text": tail_block},
            ]
        else:
            return [
                {
                    "type": "text",
                    "text": hot_block,
                    "cache_control": hot_cc,
                },
                {
                    "type": "text",
                    "text": warm_block,
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": tail_block},
            ]

    if len(context_block) < min_cache_chars:
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

# Run-cumulative usage: a single `new` invocation can make several API calls
# (generate + JSON repair + validation-repair rounds). ``get_last_usage``
# only reflects the most recent call, which silently under-reports cost the
# moment a repair fires — the report panel reads this accumulator instead.
_run_usage: UsageInfo = UsageInfo()


def get_last_usage() -> UsageInfo:
    """Return token usage from the most recent API call."""
    return _last_usage


def reset_run_usage() -> None:
    """Zero the run-cumulative usage counter. Call at the start of a run."""
    global _run_usage
    _run_usage = UsageInfo()


def get_run_usage() -> UsageInfo:
    """Return token usage summed over every API call since the last reset."""
    return _run_usage


def _accumulate_run_usage(usage: UsageInfo) -> None:
    global _run_usage
    _run_usage = UsageInfo(
        input_tokens=_run_usage.input_tokens + usage.input_tokens,
        output_tokens=_run_usage.output_tokens + usage.output_tokens,
        cache_read_input_tokens=(
            _run_usage.cache_read_input_tokens + usage.cache_read_input_tokens
        ),
        cache_creation_input_tokens=(
            _run_usage.cache_creation_input_tokens + usage.cache_creation_input_tokens
        ),
    )


def _event_kind(event: Any) -> str | None:
    """Return the SDK event's type string, or ``None`` if not introspectable."""
    return getattr(event, "type", None)


def _delta_text(delta: Any) -> str:
    """Extract a delta's text payload across the SDK's delta variants."""
    for attr in ("text", "thinking", "partial_json"):
        value = getattr(delta, attr, None)
        if isinstance(value, str):
            return value
    if isinstance(delta, dict):
        for key in ("text", "thinking", "partial_json"):
            value = delta.get(key)
            if isinstance(value, str):
                return value
    return ""


def _usage_payload(event: Any) -> dict[str, int] | None:
    usage = getattr(event, "usage", None)
    if usage is None and isinstance(event, dict):
        usage = event.get("usage")
    if usage is None:
        return None
    return {
        "input_tokens": getattr(usage, "input_tokens", 0)
        or (usage.get("input_tokens", 0) if isinstance(usage, dict) else 0)
        or 0,
        "output_tokens": getattr(usage, "output_tokens", 0)
        or (usage.get("output_tokens", 0) if isinstance(usage, dict) else 0)
        or 0,
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0)
        or (usage.get("cache_read_input_tokens", 0) if isinstance(usage, dict) else 0)
        or 0,
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0)
        or (usage.get("cache_creation_input_tokens", 0) if isinstance(usage, dict) else 0)
        or 0,
    }


def _drain_stream(
    stream: Any,
    callback: Callable[[ProgressEvent], None] | None,
) -> None:
    """Iterate every event on the stream, mapping to ProgressEvents.

    Also enforces the heartbeat/abort sentinels so a stuck stream can't hang
    the CLI forever. Surfaces a heartbeat ProgressEvent at each warn interval.
    """
    last_event_at = time.monotonic()
    next_warn = HEARTBEAT_WARN_SECONDS
    for event in stream:
        now = time.monotonic()
        silence = now - last_event_at
        if silence >= HEARTBEAT_ABORT_SECONDS:
            raise StreamStuckError(
                f"No streaming events for {int(silence)}s; aborting. "
                "Try lowering --max-context-tokens or --thinking."
            )
        if silence >= next_warn and callback is not None:
            callback(ProgressEvent(kind="heartbeat", payload=int(silence)))
            next_warn = silence + HEARTBEAT_WARN_SECONDS
        last_event_at = now

        kind = _event_kind(event)
        if callback is None:
            continue
        if kind == "content_block_delta":
            delta = getattr(event, "delta", None) or (
                event.get("delta") if isinstance(event, dict) else None
            )
            if delta is None:
                continue
            delta_kind = getattr(delta, "type", None) or (
                delta.get("type") if isinstance(delta, dict) else None
            )
            text = _delta_text(delta)
            if delta_kind == "thinking_delta":
                callback(ProgressEvent(kind="thinking_delta", payload=text))
            elif delta_kind in ("text_delta", "input_json_delta"):
                callback(ProgressEvent(kind="text_delta", payload=text))
        elif kind == "message_delta":
            usage = _usage_payload(event)
            if usage is not None:
                callback(ProgressEvent(kind="usage", payload=usage))
        elif kind == "message_stop":
            callback(ProgressEvent(kind="done"))


def _estimate_input_tokens(
    system_blocks: list[dict[str, Any]],
    user_content: list[dict[str, Any]],
) -> int:
    """Cheap chars/4 estimate of the prompt size, used by ``stream_started``.

    The real input_tokens count arrives later via ``message_delta`` usage, but
    we need a rough number up-front to pick a pre-fill wait bucket for the
    progress panel.
    """
    chars = 0
    for block in (*system_blocks, *user_content):
        text = block.get("text") if isinstance(block, dict) else None
        if isinstance(text, str):
            chars += len(text)
    return chars // 4


def _call_with_retry(
    client: _AnthropicLike,
    *,
    config: Config,
    system_blocks: list[dict[str, Any]],
    user_content: list[dict[str, Any]],
    progress: Callable[[ProgressEvent], None] | None = None,
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
    thinking_kwargs = _build_thinking_kwargs(config.model, config.thinking_budget)
    if thinking_kwargs:
        logger.debug(
            "Extended thinking: model=%s, budget=%s, payload=%s",
            config.model,
            config.thinking_budget,
            thinking_kwargs,
        )
        create_kwargs.update(thinking_kwargs)
    input_tokens_estimate = _estimate_input_tokens(system_blocks, user_content)
    thinking_enabled = bool(thinking_kwargs)
    for attempt in range(len(delays) + 1):
        try:
            logger.debug(
                "Streaming from %s (attempt %d, max_tokens=%d)",
                config.model,
                attempt + 1,
                config.max_tokens,
            )
            t0 = time.time()
            if progress is not None:
                progress(
                    ProgressEvent(
                        kind="stream_started",
                        payload={
                            "input_tokens_estimate": input_tokens_estimate,
                            "thinking_enabled": thinking_enabled,
                            "model": config.model,
                        },
                    )
                )
            with client.messages.stream(**create_kwargs) as stream:
                _drain_stream(stream, progress)
                response = stream.get_final_message()
            elapsed = time.time() - t0
            _last_usage = _extract_usage(response)
            _accumulate_run_usage(_last_usage)
            if progress is not None:
                progress(
                    ProgressEvent(
                        kind="usage",
                        payload={
                            "input_tokens": _last_usage.input_tokens,
                            "output_tokens": _last_usage.output_tokens,
                            "cache_read_input_tokens": _last_usage.cache_read_input_tokens,
                            "cache_creation_input_tokens": _last_usage.cache_creation_input_tokens,
                        },
                    )
                )
            logger.debug(
                "Response received in %.1fs — input: %d, output: %d tokens",
                elapsed,
                _last_usage.input_tokens,
                _last_usage.output_tokens,
            )
            return _extract_text(response)
        except Exception as exc:
            last_exc = exc
            if progress is not None:
                progress(ProgressEvent(kind="error", payload=str(exc)))
            if attempt >= len(delays) or not _is_retryable(exc):
                raise
            logger.debug("Retryable error: %s — retrying in %.0fs", exc, delays[attempt])
            time.sleep(delays[attempt])
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("unreachable")


def _system_blocks(strict: bool = False, *, ttl_1h: bool = False) -> list[dict[str, Any]]:
    """System prompt block. ``ttl_1h=True`` opts into the 1h TTL (Config.cache_ttl):
    the API requires 1h cache entries to precede 5m ones in the prompt
    hierarchy, and the system block sits upstream of the hot context block, so
    it takes the same TTL. The default 5m write is the cheaper trade for a
    one-shot generation that reads nothing back."""
    filename = SYSTEM_STRICT_PROMPT_FILE if strict else SYSTEM_PROMPT_FILE
    system_text = _load_prompt(filename)
    return [
        {
            "type": "text",
            "text": system_text,
            "cache_control": _hot_cache_control(ttl_1h),
        }
    ]


def generate(
    req: GenerationRequest,
    config: Config,
    progress: Callable[[ProgressEvent], None] | None = None,
) -> str:
    """Send the assembled prompt to the Anthropic API and return raw text."""
    client = _make_client(config)
    context_block, tail_block = _render_user_message(req)
    # The 1h TTL is opt-in (Config.cache_ttl). It only ever applies when the
    # context is tiered — an untiered prompt has no hot prefix to keep warm.
    hot_ttl_1h = config.cache_ttl == "1h" and CACHE_SPLIT_WARM_MARKER in context_block
    return _call_with_retry(
        client,
        config=config,
        system_blocks=_system_blocks(req.strict, ttl_1h=hot_ttl_1h),
        user_content=_build_user_content(
            context_block, tail_block, config.model, hot_ttl_1h=hot_ttl_1h
        ),
        progress=progress,
    )


def repair(
    raw_response: str,
    validation_error: str,
    config: Config,
    strict: bool = False,
    progress: Callable[[ProgressEvent], None] | None = None,
) -> str:
    """Ask the model to repair a previous invalid response."""
    client = _make_client(config)
    user_message = _render_repair_message(raw_response, validation_error)
    return _call_with_retry(
        client,
        config=config,
        system_blocks=_system_blocks(strict),
        user_content=[{"type": "text", "text": user_message}],
        progress=progress,
    )


_LANGUAGE_FENCE: dict[str, str] = {
    "python": "python",
    "typescript": "typescript",
    "javascript": "javascript",
}


def _language_fence(language: str, target_path: str) -> str:
    """Pick a fence label that matches the file's language for syntax highlighting."""
    suffix = target_path.rsplit(".", 1)[-1] if "." in target_path else ""
    by_suffix = {
        "py": "python",
        "ts": "typescript",
        "tsx": "tsx",
        "js": "javascript",
        "jsx": "jsx",
        "md": "markdown",
        "json": "json",
        "yml": "yaml",
        "yaml": "yaml",
        "toml": "toml",
        "sh": "bash",
    }
    if suffix in by_suffix:
        return by_suffix[suffix]
    return _LANGUAGE_FENCE.get(language, "text")


def _render_neighbours_block(neighbours: dict[str, str], fence: str) -> str:
    if not neighbours:
        return "(no neighbour files detected.)"
    parts: list[str] = []
    for path, content in neighbours.items():
        parts.append(f"### `{path}`\n\n```{fence}\n{content}\n```")
    return "\n\n".join(parts)


def _render_single_file_prompt(
    recipe_body: str,
    target_path: str,
    current_content: str,
    neighbours: dict[str, str],
    reason: str,
    language: str,
) -> str:
    template = _load_prompt(SINGLE_FILE_TEMPLATE_FILE)
    fence = _language_fence(language, target_path)
    return (
        template.replace("{language_fence}", fence)
        .replace("{recipe_body}", recipe_body)
        .replace("{target_path}", target_path)
        .replace("{current_content}", current_content)
        .replace("{neighbours_block}", _render_neighbours_block(neighbours, fence))
        .replace("{reason}", reason.strip() or "(no reason supplied)")
    )


def extract_fenced_content(text: str) -> str:
    """Return the body of the LARGEST fenced code block in ``text``.

    The single-file prompt instructs the model to emit exactly one fenced
    block, but tolerate the model surrounding it with a brief preamble or
    emitting multiple blocks (e.g., an example + the answer). Picking the
    largest block is a defensive heuristic — the replacement file is almost
    always longer than any incidental example snippet.
    """
    matches = list(_FENCED_BLOCK_RE.finditer(text))
    if not matches:
        raise ValueError("no fenced code block found in single-file response")
    return max(matches, key=lambda m: len(m.group(1))).group(1)


def generate_single_file(
    *,
    config: Config,
    recipe_body: str,
    target_path: str,
    current_content: str,
    neighbours: dict[str, str],
    reason: str,
    language: str,
    progress: Callable[[ProgressEvent], None] | None = None,
) -> str:
    """Re-prompt the model for the replacement contents of one file.

    Returns the raw response text; the caller is responsible for extracting
    the fenced block via :func:`extract_fenced_content`.
    """
    client = _make_client(config)
    user_message = _render_single_file_prompt(
        recipe_body=recipe_body,
        target_path=target_path,
        current_content=current_content,
        neighbours=neighbours,
        reason=reason,
        language=language,
    )
    # Reuse the strict system prompt so the model honours its lint-cleanliness
    # + production-requirements guidance even on single-file regen.
    return _call_with_retry(
        client,
        config=config,
        system_blocks=_system_blocks(strict=True),
        user_content=[{"type": "text", "text": user_message}],
        progress=progress,
    )


def _render_validation_repair_prompt(
    *,
    recipe_body: str,
    language_hints: dict[str, Any],
    project_file_list: list[str],
    failing_command: str,
    validation_output: str,
    implicated_files: dict[str, str],
    language: str,
) -> str:
    template = _load_prompt(VALIDATION_REPAIR_TEMPLATE_FILE)
    hints_yaml = yaml.safe_dump(language_hints, sort_keys=False).strip()
    file_list = "\n".join(f"- {path}" for path in sorted(project_file_list)) or "(none)"
    fence = _LANGUAGE_FENCE.get(language, "text")
    return (
        template.replace("{language_hints_yaml}", hints_yaml)
        .replace("{recipe_body}", recipe_body)
        .replace("{project_file_list}", file_list)
        .replace("{failing_command}", failing_command)
        .replace("{validation_output}", validation_output)
        .replace(
            "{implicated_files_block}",
            _render_neighbours_block(implicated_files, fence),
        )
    )


def repair_validation(
    *,
    config: Config,
    recipe_body: str,
    language_hints: dict[str, Any],
    project_file_list: list[str],
    failing_command: str,
    validation_output: str,
    implicated_files: dict[str, str],
    language: str,
    progress: Callable[[ProgressEvent], None] | None = None,
) -> str:
    """Ask the model for targeted file fixes after a validation tier failed.

    The prompt deliberately carries the recipe body only — not the full
    assembled context — mirroring the single-file regenerate flow: repair
    calls stay cheap and the failure output plus implicated file bodies are
    the load-bearing context. Returns the raw response; the caller parses it
    with :func:`agent_scaffold.contract.parse_file_patch`.
    """
    client = _make_client(config)
    user_message = _render_validation_repair_prompt(
        recipe_body=recipe_body,
        language_hints=language_hints,
        project_file_list=project_file_list,
        failing_command=failing_command,
        validation_output=validation_output,
        implicated_files=implicated_files,
        language=language,
    )
    return _call_with_retry(
        client,
        config=config,
        system_blocks=_system_blocks(strict=True),
        user_content=[{"type": "text", "text": user_message}],
        progress=progress,
    )
