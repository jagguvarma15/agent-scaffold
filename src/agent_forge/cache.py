"""Simple file-based response caching keyed by the full set of generation inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def cache_key(inputs: dict[str, Any]) -> str:
    """Produce a stable hash for the generation inputs."""
    payload = json.dumps(inputs, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def get_cached(cache_dir: Path, inputs: dict[str, Any]) -> str | None:
    """Return cached raw LLM response text if it exists, else None."""
    key = cache_key(inputs)
    path = cache_dir / "responses" / f"{key}.json"
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return None


def save_cache(cache_dir: Path, inputs: dict[str, Any], response: str) -> None:
    """Save a raw LLM response to cache."""
    key = cache_key(inputs)
    responses_dir = cache_dir / "responses"
    responses_dir.mkdir(parents=True, exist_ok=True)
    path = responses_dir / f"{key}.json"
    path.write_text(response, encoding="utf-8")
