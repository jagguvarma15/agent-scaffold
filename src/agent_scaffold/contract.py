"""Parse and validate the LLM's JSON output.

The Anthropic-side response is supposed to be a JSON object matching
``GenerationResult``. We strip optional fence markers, parse, validate
shape with Pydantic, then validate path safety and required files.

The capability-aware passes (:func:`merge_capability_fragments` and
:func:`check_frontend_collisions`) run after the structural validators
when a resolved capability stack is available. They:

- ensure ``docker-compose.yml`` contains every service declared by every
  capability's ``docker:`` fragment, filling in missing ones from the
  capability data; and
- flag files the model emitted under a frontend capability's
  ``emit_files`` glob (the scaffold's copier handles those paths
  exclusively; the LLM must not author them).
"""

from __future__ import annotations

import fnmatch
import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, Field, ValidationError

if TYPE_CHECKING:
    from agent_scaffold.capabilities import ResolvedStack

log = logging.getLogger(__name__)

_COMPOSE_FILENAMES = ("docker-compose.yml", "docker-compose.yaml", "compose.yaml")

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


def validate_paths(
    result: GenerationResult,
    dest: Path,
    *,
    canonical_module_name: str | None = None,
) -> None:
    """Ensure every emitted path is safe and unique within ``dest``.

    When ``canonical_module_name`` is given (the Python-underscored form of
    the project name), also reject any ``src/<dir>/`` segment that uses the
    hyphenated form of the same name. The LLM occasionally splits the
    project across ``src/foo-bar/`` and ``src/foo_bar/`` — only one can be
    a real Python package, so this is a generation bug. Raising triggers
    the repair loop, which usually self-corrects on retry.
    """
    dest_resolved = dest.resolve()
    seen: set[str] = set()
    hyphenated_form = (
        canonical_module_name.replace("_", "-")
        if canonical_module_name and "_" in canonical_module_name
        else None
    )
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

        if hyphenated_form is not None and canonical_module_name is not None:
            parts = normalized.split("/")
            if "src" in parts:
                idx = parts.index("src")
                if idx + 1 < len(parts) and parts[idx + 1] == hyphenated_form:
                    raise ContractParseError(
                        raw=raw_path,
                        reason=(
                            f"Python module directory uses hyphenated form "
                            f"{hyphenated_form!r} but the canonical module name is "
                            f"{canonical_module_name!r}; rename src/{hyphenated_form}/ "
                            f"to src/{canonical_module_name}/."
                        ),
                    )


def validate_required_files(
    result: GenerationResult,
    hints: dict[str, Any],
    extra_required: list[str] | None = None,
) -> None:
    """Ensure manifest, entry point, README, .env.example, and any
    recipe-specific ``extra_required`` files are emitted.
    """
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

    for required in extra_required or []:
        normalized = required.replace("\\", "/")
        if normalized not in paths:
            raise ContractParseError(
                raw="(files)",
                reason=f"missing recipe-required file: {required}",
            )


# ---------------------------------------------------------------------------
# Capability-aware post-parse passes
# ---------------------------------------------------------------------------


def merge_capability_fragments(
    result: GenerationResult,
    stack: ResolvedStack | None,
) -> GenerationResult:
    """Ensure ``docker-compose.yml`` contains every capability's docker service.

    For each capability with a ``docker:`` fragment, if the service name
    isn't already in the model-emitted compose file, append it from the
    capability's data. Pinned image tags from the capability always win on
    conflict (a stable infra version is more important than the LLM's
    occasional drift to ``:latest``).

    No-op when:

    - ``stack`` is ``None``
    - no capability has a ``docker:`` fragment
    - the model emitted no compose file AND no capability needs one

    Re-emits the merged file with stable key ordering so re-running on the
    same input produces byte-identical output.
    """
    if stack is None:
        return result
    docker_caps = [c for c in stack.capabilities if c.docker is not None]
    if not docker_caps:
        return result

    compose_index, compose_path = _find_compose(result)
    existing_yaml = result.files[compose_index].content if compose_index is not None else ""
    compose_data = _parse_compose_yaml(existing_yaml)
    services = compose_data.setdefault("services", {})
    if not isinstance(services, dict):
        log.warning(
            "merge_capability_fragments: compose.services is %s, replacing with empty dict",
            type(services).__name__,
        )
        services = {}
        compose_data["services"] = services

    added: list[str] = []
    overridden: list[str] = []
    for cap in docker_caps:
        frag = cap.docker
        if frag is None:  # mypy narrowing
            continue
        block = _fragment_to_compose_block(frag)
        if frag.service in services:
            # Reconcile image tag: capability pin wins.
            existing = services[frag.service]
            if isinstance(existing, dict) and existing.get("image") != frag.image:
                log.info(
                    "merge_capability_fragments: pinning %s image to %s " "(LLM emitted %s)",
                    frag.service,
                    frag.image,
                    existing.get("image"),
                )
                existing["image"] = frag.image
                overridden.append(frag.service)
            continue
        services[frag.service] = block
        added.append(frag.service)

    if not added and not overridden and compose_index is not None:
        return result  # nothing changed

    # Stable order: top-level keys + alphabetised services.
    compose_data = _canonicalize_compose(compose_data)
    rendered = (
        yaml.safe_dump(compose_data, sort_keys=False, default_flow_style=False).rstrip() + "\n"
    )

    new_files = list(result.files)
    if compose_index is not None:
        new_files[compose_index] = GeneratedFile(path=compose_path, content=rendered)
    else:
        new_files.append(GeneratedFile(path="docker-compose.yml", content=rendered))

    return result.model_copy(update={"files": new_files})


def check_frontend_collisions(
    result: GenerationResult,
    stack: ResolvedStack | None,
    *,
    strict: bool = False,
) -> list[str]:
    """Flag model-emitted files matching a frontend capability's ``emit_files`` glob.

    Frontend capabilities ship template trees the scaffold copies verbatim;
    the LLM mustn't author files inside that tree. Returns the list of
    colliding paths. In ``strict`` mode raises :class:`ContractParseError`
    on the first collision; non-strict logs a warning and returns the list
    for the caller's progress display.
    """
    if stack is None:
        return []
    frontend_caps = [c for c in stack.capabilities if c.kind == "frontend"]
    if not frontend_caps:
        return []

    globs: list[tuple[str, str]] = []  # (capability_id, glob)
    for cap in frontend_caps:
        for emit in cap.emit_files:
            normalized = emit.dest.replace("\\", "/").rstrip("/")
            # source: foo/** → dest is a directory; match dest/**
            # source: single file → dest is a single path; match exact
            if emit.source.endswith("**") or emit.source.endswith("/*"):
                globs.append((cap.id, f"{normalized}/**" if normalized else "**"))
            else:
                globs.append((cap.id, normalized))

    colliding: list[str] = []
    for file in result.files:
        path = file.path.replace("\\", "/")
        for cap_id, pattern in globs:
            if _path_matches(path, pattern):
                colliding.append(f"{path} (matches {cap_id} emit_files {pattern!r})")
                break

    if not colliding:
        return []

    if strict:
        raise ContractParseError(
            raw="(files)",
            reason=(
                "model emitted file(s) under a frontend capability's template "
                "tree; templates are copied by the scaffold:\n  - " + "\n  - ".join(colliding)
            ),
        )
    for entry in colliding:
        log.warning("frontend collision: %s", entry)
    return colliding


# ---------------------------------------------------------------------------
# Compose merge internals
# ---------------------------------------------------------------------------


def _find_compose(result: GenerationResult) -> tuple[int | None, str]:
    """Return ``(index_in_files, path)`` for the first compose file, or ``(None, default)``."""
    for idx, file in enumerate(result.files):
        normalized = file.path.replace("\\", "/")
        if normalized in _COMPOSE_FILENAMES or normalized.endswith("/" + _COMPOSE_FILENAMES[0]):
            return idx, normalized
    return None, "docker-compose.yml"


def _parse_compose_yaml(text: str) -> dict[str, Any]:
    if not text.strip():
        return {}
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        log.warning(
            "merge_capability_fragments: existing compose is invalid YAML (%s); replacing", exc
        )
        return {}
    if not isinstance(data, dict):
        log.warning("merge_capability_fragments: existing compose is not a mapping; replacing")
        return {}
    return data


def _fragment_to_compose_block(frag: Any) -> dict[str, Any]:
    """Convert a :class:`DockerFragment` into a compose-shaped dict."""
    block: dict[str, Any] = {"image": frag.image}
    if frag.ports:
        block["ports"] = list(frag.ports)
    if frag.volumes:
        block["volumes"] = list(frag.volumes)
    if frag.environment:
        block["environment"] = dict(frag.environment)
    if frag.healthcheck:
        block["healthcheck"] = dict(frag.healthcheck)
    return block


def _canonicalize_compose(data: dict[str, Any]) -> dict[str, Any]:
    """Stable ordering: top-level keys in canonical order; services alphabetised."""
    canonical_top: list[str] = ["version", "services", "volumes", "networks", "configs", "secrets"]
    ordered: dict[str, Any] = {}
    for key in canonical_top:
        if key in data:
            value = data[key]
            if key == "services" and isinstance(value, dict):
                value = {k: value[k] for k in sorted(value)}
            ordered[key] = value
    for key, value in data.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def _path_matches(path: str, pattern: str) -> bool:
    """Match ``path`` against a glob pattern. ``**`` expands to any depth."""
    if pattern == "**":
        return True
    if "**" not in pattern:
        return fnmatch.fnmatch(path, pattern)
    prefix = pattern.split("/**", 1)[0]
    if not prefix:
        return True
    # path must equal prefix or start with prefix + "/"
    if path == prefix:
        return True
    return path.startswith(prefix + "/")
