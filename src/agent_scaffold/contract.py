"""Parse and validate the LLM's JSON output.

The Anthropic-side response is supposed to be a JSON object matching
``GenerationResult``. We strip optional fence markers, parse, validate
shape with Pydantic, then validate path safety and required files.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

_FENCE_OPEN_RE = re.compile(r"^```(?:json)?\s*\n", re.IGNORECASE)
_FENCE_CLOSE_RE = re.compile(r"\n```\s*$")


class ContractParseError(Exception):
    """Raised when the LLM response does not satisfy the generation contract."""

    def __init__(self, raw: str, reason: str) -> None:
        super().__init__(reason)
        self.raw = raw
        self.reason = reason


class GeneratedFile(BaseModel):
    path: str
    content: str


class GenerationResult(BaseModel):
    project_name: str
    language: str
    files: list[GeneratedFile] = Field(min_length=1)
    post_install: list[str] = Field(default_factory=list)
    smoke_check: str
    known_limitations: list[str] = Field(default_factory=list)


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    open_match = _FENCE_OPEN_RE.match(stripped)
    if open_match:
        stripped = stripped[open_match.end() :]
        close_match = _FENCE_CLOSE_RE.search(stripped)
        if close_match:
            stripped = stripped[: close_match.start()]
    return stripped.strip()


def parse(raw: str) -> GenerationResult:
    """Parse a raw LLM response into a :class:`GenerationResult`."""
    cleaned = _strip_fences(raw)
    try:
        data: Any = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ContractParseError(
            raw=raw,
            reason=(
                f"invalid JSON: {exc}\n"
                "Hint: The LLM response was not valid JSON. "
                "Re-run the command to retry, or check the saved failure file."
            ),
        ) from exc

    try:
        return GenerationResult.model_validate(data)
    except ValidationError as exc:
        raise ContractParseError(
            raw=raw,
            reason=(
                f"Schema validation failed:\n{exc}\n"
                "Hint: The JSON structure didn't match the expected contract. "
                "The repair flow will attempt to fix this automatically."
            ),
        ) from exc


def validate_paths(result: GenerationResult, dest: Path) -> None:
    """Ensure every emitted path is safe and unique within ``dest``."""
    dest_resolved = dest.resolve()
    seen: set[str] = set()
    for entry in result.files:
        raw_path = entry.path
        if not raw_path or raw_path != raw_path.strip():
            raise ContractParseError(
                raw=raw_path, reason=f"empty or whitespace-padded path: {raw_path!r}"
            )
        if raw_path.startswith(("/", "\\")):
            raise ContractParseError(raw=raw_path, reason=f"absolute path not allowed: {raw_path}")
        normalized = raw_path.replace("\\", "/")
        if any(part == ".." for part in normalized.split("/")):
            raise ContractParseError(raw=raw_path, reason=f"'..' segment not allowed: {raw_path}")
        candidate = (dest_resolved / normalized).resolve()
        try:
            candidate.relative_to(dest_resolved)
        except ValueError as exc:
            raise ContractParseError(
                raw=raw_path, reason=f"path escapes destination: {raw_path}"
            ) from exc
        if normalized in seen:
            raise ContractParseError(raw=raw_path, reason=f"duplicate path: {raw_path}")
        seen.add(normalized)


def validate_required_files(result: GenerationResult, hints: dict[str, Any]) -> None:
    """Ensure manifest, entry point, README, and .env.example are emitted."""
    paths = {f.path.replace("\\", "/") for f in result.files}

    manifest = hints.get("manifest")
    if not manifest:
        raise ContractParseError(raw="(hints)", reason="language hints missing 'manifest'")
    if manifest not in paths:
        raise ContractParseError(
            raw="(files)", reason=f"missing required manifest file: {manifest}"
        )

    entry_template = hints.get("entry_point", "")
    entry_point = entry_template.replace("{project_name}", result.project_name)
    if entry_point and entry_point not in paths:
        raise ContractParseError(
            raw="(files)", reason=f"missing required entry point: {entry_point}"
        )

    for required in ("README.md", ".env.example"):
        if required not in paths:
            raise ContractParseError(raw="(files)", reason=f"missing required file: {required}")
