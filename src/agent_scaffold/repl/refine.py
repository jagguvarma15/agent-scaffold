r"""LLM-interpreted free-text refinements for the REPL.

The dispatcher in :mod:`agent_scaffold.repl.commands` routes anything that
isn't a slash command or a bare recipe slug here. The user types
something like::

    swap to sonnet, add postgres, and skip the smoke test

and we ship it to Claude Haiku with the current state JSON, asking for a
:class:`~agent_scaffold.repl.session.StatePatch`-shaped JSON object back.
The result is applied via :func:`~agent_scaffold.repl.session.apply_patch`.

Design choices:

- **Haiku-only.** This is a system tool, not user-controllable. ~1k input
  + ~200 output tokens per call, so the per-refinement cost is around
  $0.002 — cheap enough that we don't bother caching.
- **Schema-only output.** The system prompt enumerates the valid patch
  keys and demands JSON with no prose. We tolerate one wrapping markdown
  fence (``\`\`\`json``) since smaller models often add it despite being told not to.
- **Soft failures.** Network errors, malformed JSON, or schema-invalid
  responses raise :class:`RefinementError`; the dispatcher surfaces a
  yellow warning and leaves the state untouched. The user can retry or
  drop down to slash commands.
- **Test seam at the bottom.** ``_make_haiku_client`` is the only network
  call; tests monkeypatch it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

import anthropic

from agent_scaffold.config import Config
from agent_scaffold.discovery import Recipe
from agent_scaffold.repl.session import SessionState, StatePatch

# Hard-coded — refinement is a system tool, not user-tunable. If Haiku 4.5
# is retired this constant gets a one-line update.
_REFINE_MODEL = "claude-haiku-4-5-20251001"
_REFINE_MAX_TOKENS = 1_024

# Subset of SessionState fields the LLM is allowed to patch. recipe /
# project_name / dest stay user-controlled (free-text typos for those
# would be too disruptive); cfg / deployments / blueprints are
# session-scope and never change inside the REPL.
_PATCHABLE_SCALARS: tuple[str, ...] = (
    "model",
    "effort",
    "framework",
    "language",
    "strict",
    "max_tokens",
    "thinking_budget",
    "stack_mode",
)


# Canonical mapping of every refinement key Haiku is allowed to emit to a
# one-line description. Consumed by:
#
# - ``/help refine`` in ``repl/commands.py`` to render the user-facing key
#   table without duplicating the prose.
# - ``tests/test_repl_refine.py::test_refinement_keys_constant_matches_system_prompt``
#   to assert the INTERPRET_SYSTEM enumeration below stays in sync.
#
# Order is the order the table renders in: scalars first, accumulators
# next, then the free-form ``notes`` escape hatch.
REFINEMENT_KEYS: dict[str, str] = {
    "model": "Override model (e.g. claude-sonnet-4-6, claude-haiku-4-5-20251001, claude-opus-4-7).",
    "effort": "Preset bundle: low | medium | high (model + tokens + thinking + strict).",
    "framework": "Framework name (e.g. langgraph, pydantic_ai, vercel_ai_sdk).",
    "language": "Target language: python | typescript.",
    "strict": "Toggle strict generation prompt (true | false).",
    "max_tokens": "Anthropic max_tokens cap for this run (integer).",
    "thinking_budget": "Extended-thinking token budget (integer; null disables).",
    "stack_mode": "Capability stack mode: quick | customize.",
    "add_dependencies": "Extra pins to inject: {language: {package: version}}.",
    "add_steps": "Extra post-write steps to run (e.g. [docker_up, seed]).",
    "remove_steps": "Post-write steps to skip (e.g. [smoke_test]).",
    "remove_roles": "Multi-agent roles to drop.",
    "add_capabilities": "Capability ids to enable (e.g. [obs.langfuse]).",
    "remove_capabilities": "Capability ids to drop (e.g. [obs.langsmith]).",
    "notes": "Free-form guidance appended verbatim to the LLM prompt.",
}

_VALID_STACK_MODES: frozenset[str] = frozenset({"quick", "customize"})

# Strict guidance: enumerate keys, give two examples, and demand JSON-only
# output. Two examples is enough for Haiku to nail the format; three would
# spend tokens without measurably improving quality.
INTERPRET_SYSTEM = """You translate user refinement requests into JSON patches over an agent-scaffold session state.

Return ONLY a JSON object — no prose, no markdown code fence. Valid keys (all optional):

  model            string  — Anthropic model id (e.g. "claude-sonnet-4-6", "claude-haiku-4-5-20251001", "claude-opus-4-7")
  effort           "low" | "medium" | "high"  — preset bundling model + tokens + thinking + strict prompt
  framework        string  — framework name (e.g. "langgraph", "pydantic_ai", "vercel_ai_sdk")
  language         "python" | "typescript"
  strict           boolean — use the strict generation prompt
  max_tokens       integer — Anthropic max_tokens cap for this run
  thinking_budget  integer — extended-thinking token budget; null to disable
  stack_mode       "quick" | "customize"  — recipe defaults vs per-layer customize walk
  add_dependencies {language: {package: version}}  — extra pins to inject into the recipe
  add_steps        [string]  — extra post-write steps to run (e.g. ["docker_up", "seed"])
  remove_steps     [string]  — steps to skip (e.g. ["smoke_test"])
  remove_roles     [string]  — multi-agent roles to drop
  add_capabilities    [string] — capability ids to enable (e.g. ["obs.langfuse"])
  remove_capabilities [string] — capability ids to drop (e.g. ["obs.langsmith"])
  notes            string  — anything that doesn't fit above, appended verbatim as extra LLM guidance

Examples:

User: "swap to sonnet and skip the smoke test"
{"model":"claude-sonnet-4-6","remove_steps":["smoke_test"]}

User: "use the cheapest model, add postgres connection pool"
{"effort":"low","add_dependencies":{"python":{"pgbouncer":">=1.21"}}}

User: "use langfuse instead of langsmith"
{"add_capabilities":["obs.langfuse"],"remove_capabilities":["obs.langsmith"]}

User: "drop observability"
{"remove_capabilities":["obs.langsmith","obs.langfuse"]}

User: "let me pick each layer myself"
{"stack_mode":"customize"}

If a request doesn't map cleanly to a key, capture it in "notes"."""


# The "describe your agent" first step: map a free-text description + the list of
# available recipes into a recipe suggestion plus the seeds that flow downstream
# (agent_role → backend system prompt, agent_title → chat frontend).
DESCRIBE_SYSTEM = """You help a developer scaffold an AI agent. Given a free-text description of the agent they want and a list of available starter recipes, return ONLY a JSON object — no prose, no markdown fence — with these keys:

  suggested_recipe_slug  string|null — the slug of the BEST-matching recipe from the provided list, or null if none fit
  agent_role             string — a concise system prompt (2-4 sentences) the agent should adopt: who it is, what it does, how it behaves. Write it as instructions TO the agent ("You are ...").
  agent_title            string — a short product-style name for the agent (<= 4 words), for the chat UI title
  use_case               string — one short line summarizing what the agent is for

Choose suggested_recipe_slug ONLY from the provided slugs. If the description is vague, still infer a sensible agent_role and agent_title.

Example:
Available recipes:
- docs-rag-qa: Documentation Q&A over a corpus [rag]
- memory-assistant: Personal assistant with memory [react]
User's agent description: "a bot that answers questions about our internal docs with citations"
{"suggested_recipe_slug":"docs-rag-qa","agent_role":"You are a documentation assistant. Answer the user's question using only the retrieved internal documentation, and cite the source of each claim. If the docs do not cover it, say so plainly.","agent_title":"Docs Q&A","use_case":"Answer questions about internal docs with citations"}"""


@dataclass(frozen=True)
class DescriptionResult:
    """What :func:`interpret_description` extracts from the first-step free text.

    Every field is independently optional: a vague description may still yield a
    role/title with no confident recipe, and an API hiccup degrades to all-``None``
    (the wizard just proceeds without a suggestion).
    """

    suggested_recipe_slug: str | None = None
    agent_role: str | None = None
    agent_title: str | None = None
    use_case: str | None = None


class RefinementError(Exception):
    """Raised when the Haiku call or its response can't produce a usable patch.

    The dispatcher catches this and shows a yellow warning so the user can
    retry or fall back to slash commands. State is left unchanged.
    """


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def interpret_refinement(state: SessionState, text: str, cfg: Config) -> StatePatch:
    """Turn a free-text refinement into a typed :class:`StatePatch`.

    Empty / whitespace text is a no-op (returns an empty patch without
    burning an API call). All other paths invoke Haiku once and parse the
    response strictly. Any API or parse failure raises
    :class:`RefinementError` with a one-line summary; callers should treat
    it as recoverable (warn + leave state intact).
    """
    if not text.strip():
        return StatePatch()

    client = _make_haiku_client(cfg)
    # Last line of the "secrets never reach an LLM" guarantee: if the user
    # pastes a credential into free text, strip the secret-shaped substring
    # before the message leaves the machine. State serialization carries
    # names/slugs only, but it includes prior refinement notes — same rule.
    from agent_scaffold._redact import redact

    user_msg = (
        f"Current state:\n{redact(serialize_state_for_prompt(state))}\n\n"
        f"User refinement: {redact(text)}"
    )
    try:
        response = client.messages.create(
            model=_REFINE_MODEL,
            max_tokens=_REFINE_MAX_TOKENS,
            system=INTERPRET_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
    except anthropic.AnthropicError as exc:
        raise RefinementError(f"Haiku call failed: {exc}") from exc

    raw = _extract_text(response)
    payload = _parse_json(raw)
    return _patch_from_dict(payload)


def interpret_description(text: str, recipes: Iterable[Recipe], cfg: Config) -> DescriptionResult:
    """Map the "describe your agent" free text to a suggestion + prompt seeds.

    Returns a :class:`DescriptionResult`: a validated recipe slug (one of
    ``recipes``, else ``None``), an ``agent_role`` system prompt, an
    ``agent_title``, and a one-line ``use_case``. Empty text is a no-op. Any
    Haiku/parse failure raises :class:`RefinementError`; the wizard treats it as
    recoverable — proceed to the picker with no suggestion.
    """
    recipe_list = list(recipes)
    if not text.strip():
        return DescriptionResult()

    from agent_scaffold._redact import redact

    catalog = "\n".join(_render_recipe_line(r) for r in recipe_list)
    user_msg = (
        f"Available recipes:\n{catalog}\n\n" f"User's agent description: {redact(text.strip())}"
    )
    client = _make_haiku_client(cfg)
    try:
        response = client.messages.create(
            model=_REFINE_MODEL,
            max_tokens=_REFINE_MAX_TOKENS,
            system=DESCRIBE_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
    except anthropic.AnthropicError as exc:
        raise RefinementError(f"Haiku call failed: {exc}") from exc

    raw = _extract_text(response)
    payload = _parse_json(raw)
    return _description_from_dict(payload, valid_slugs={r.slug for r in recipe_list})


def _render_recipe_line(recipe: Recipe) -> str:
    """One ``- slug: title [pattern]`` line for the suggestion prompt."""
    pattern = getattr(recipe, "agent_pattern", None)
    suffix = f" [{pattern}]" if pattern else ""
    return f"- {recipe.slug}: {recipe.title}{suffix}"


def serialize_state_for_prompt(state: SessionState) -> str:
    """JSON-render the user-visible mutable fields of ``state``.

    Keep this small and stable — every byte sent to Haiku costs tokens,
    and changing field names breaks the few-shot examples in
    :data:`INTERPRET_SYSTEM`.
    """
    payload: dict[str, Any] = {
        "recipe": getattr(state.recipe, "slug", None),
        "language": state.language,
        "framework": state.framework,
        "project_name": state.project_name,
        "model": state.model,
        "effort": state.effort,
        "max_tokens": state.max_tokens,
        "thinking_budget": state.thinking_budget,
        "strict": state.strict,
        "extra_dependencies": state.extra_dependencies,
        "extra_steps": state.extra_steps,
        "removed_steps": sorted(state.removed_steps),
        "removed_roles": sorted(state.removed_roles),
        "refinement_notes": state.refinement_notes,
    }
    return json.dumps(payload, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def _extract_text(response: Any) -> str:
    """Pull the text out of an Anthropic Message.

    Anthropic's response shape is ``response.content[0].text``; we don't
    assume more than that because tests stub it with a minimal duck-type.
    """
    try:
        blocks = response.content
    except AttributeError as exc:
        raise RefinementError("response object has no .content") from exc
    if not blocks:
        raise RefinementError("response had empty content")
    first = blocks[0]
    text = getattr(first, "text", None)
    if not isinstance(text, str):
        raise RefinementError("response content[0] missing .text")
    return text


def _parse_json(raw: str) -> dict[str, Any]:
    """Parse the LLM's response, tolerating one wrapping markdown fence."""
    body = raw.strip()
    fence = _FENCE_RE.match(body)
    if fence is not None:
        body = fence.group(1).strip()
    try:
        loaded = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RefinementError(f"response was not valid JSON: {exc.msg}") from exc
    if not isinstance(loaded, dict):
        raise RefinementError(f"response root must be a JSON object, got {type(loaded).__name__}")
    return cast(dict[str, Any], loaded)


def _patch_from_dict(data: dict[str, Any]) -> StatePatch:
    """Whitelist the keys the LLM is allowed to set, then build a patch.

    Unknown keys are dropped silently — a hallucinated key shouldn't crash
    the REPL, and the user can see what landed via the delta render.
    Schema-mismatched values (e.g. ``model: 42``) are also dropped.
    """
    scalars: dict[str, Any] = {}
    for key in _PATCHABLE_SCALARS:
        if key not in data:
            continue
        value = data[key]
        if key in {"max_tokens", "thinking_budget"}:
            if isinstance(value, bool) or not isinstance(value, int):
                # bool is a subclass of int — exclude explicitly.
                continue
        elif key == "strict":
            if not isinstance(value, bool):
                continue
        elif key == "stack_mode":
            if not isinstance(value, str) or value not in _VALID_STACK_MODES:
                continue
        else:
            if not isinstance(value, str) or not value:
                continue
        scalars[key] = value

    add_deps = _coerce_add_deps(data.get("add_dependencies"))
    add_steps = _coerce_str_list(data.get("add_steps"))
    remove_steps = _coerce_str_list(data.get("remove_steps"))
    remove_roles = _coerce_str_list(data.get("remove_roles"))
    add_caps = _coerce_str_list(data.get("add_capabilities"))
    remove_caps = _coerce_str_list(data.get("remove_capabilities"))
    notes = data.get("notes")
    if not isinstance(notes, str) or not notes.strip():
        notes_clean: str | None = None
    else:
        notes_clean = notes.strip()

    return StatePatch(
        add_dependencies=add_deps or None,
        add_steps=add_steps or None,
        remove_steps=remove_steps or None,
        remove_roles=remove_roles or None,
        add_capabilities=add_caps or None,
        remove_capabilities=remove_caps or None,
        notes=notes_clean,
        **scalars,
    )


def _coerce_add_deps(value: Any) -> dict[str, dict[str, str]] | None:
    if not isinstance(value, dict):
        return None
    out: dict[str, dict[str, str]] = {}
    for lang, pkgs in value.items():
        if not isinstance(lang, str) or not isinstance(pkgs, dict):
            continue
        clean = {
            str(pkg): str(ver)
            for pkg, ver in pkgs.items()
            if isinstance(pkg, str) and isinstance(ver, str)
        }
        if clean:
            out[lang] = clean
    return out or None


def _coerce_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [v for v in cast(Iterable[Any], value) if isinstance(v, str) and v]


def _clean_str(value: Any) -> str | None:
    """Stripped non-empty string, else ``None`` (drops blanks / wrong types)."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _description_from_dict(data: dict[str, Any], *, valid_slugs: set[str]) -> DescriptionResult:
    """Build a :class:`DescriptionResult`, validating the slug against the catalog."""
    slug = data.get("suggested_recipe_slug")
    slug_clean = slug if isinstance(slug, str) and slug in valid_slugs else None
    return DescriptionResult(
        suggested_recipe_slug=slug_clean,
        agent_role=_clean_str(data.get("agent_role")),
        agent_title=_clean_str(data.get("agent_title")),
        use_case=_clean_str(data.get("use_case")),
    )


# ---------------------------------------------------------------------------
# Test seam — only place that touches the SDK
# ---------------------------------------------------------------------------


def _make_haiku_client(cfg: Config) -> Any:
    """Construct the Anthropic client used for refinement calls.

    Separate from ``generator._make_client`` so the protocol there can
    stay narrowly typed around ``messages.stream``; refinement uses the
    one-shot ``messages.create`` instead.
    """
    return anthropic.Anthropic(api_key=cfg.anthropic_api_key)
