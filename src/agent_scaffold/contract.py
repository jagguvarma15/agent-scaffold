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
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError

from agent_scaffold.models import RUNTIME_MODEL_CHOICES, find_unknown_model_ids

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from agent_scaffold.capabilities import ResolvedStack
    from agent_scaffold.discovery import MCPServerSpec

log = logging.getLogger(__name__)

_COMPOSE_FILENAMES = ("docker-compose.yml", "docker-compose.yaml", "compose.yaml")

_FENCE_OPEN_RE = re.compile(r"^```(?:json)?\s*\n", re.IGNORECASE)
_FENCE_CLOSE_RE = re.compile(r"\n```\s*$")


ContractFailureTier = Literal[
    "json",
    "schema",
    "path",
    "required-files",
    "model-id",
    "refusal",
    "truncation",
]


class ContractParseError(Exception):
    """Raised when the LLM response does not satisfy the generation contract.

    Carries a structured ``tier`` so callers (pipeline, repair-prompt builder,
    error rendering) can branch on the failure mode without parsing the
    ``reason`` string. Optional ``field`` names the specific path / schema
    field / required-file that triggered the failure when known.
    """

    def __init__(
        self,
        raw: str,
        reason: str,
        *,
        tier: ContractFailureTier,
        field: str | None = None,
    ) -> None:
        super().__init__(reason)
        self.raw = raw
        self.reason = reason
        self.tier: ContractFailureTier = tier
        self.field = field


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


# The response schema sent as ``output_config.format`` on the generation call.
# Authored by hand rather than derived from :class:`GenerationResult` because
# the structured-outputs subset is restrictive and the constraints must be
# deliberate: every object sets ``additionalProperties: false`` with every key
# in ``required`` (the API mandates both), no recursion, no string-length or
# numeric constraints, and array ``minItems`` limited to 0 or 1. Grammar-
# constrained decoding then guarantees the response parses and matches this
# shape — the ``json``/``schema`` failure tiers cannot occur when it is
# active. Keep field-for-field in sync with :class:`GenerationResult`
# (asserted in tests/test_contract.py).
GENERATION_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "project_name",
        "language",
        "files",
        "post_install",
        "smoke_check",
        "known_limitations",
    ],
    "properties": {
        "project_name": {"type": "string"},
        "language": {"type": "string"},
        "files": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["path", "content"],
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
            },
        },
        "post_install": {"type": "array", "items": {"type": "string"}},
        "smoke_check": {"type": "string"},
        "known_limitations": {"type": "array", "items": {"type": "string"}},
    },
}


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
            tier="json",
        ) from exc

    try:
        return GenerationResult.model_validate(data)
    except ValidationError as exc:
        first_loc: str | None = None
        errors = exc.errors() if hasattr(exc, "errors") else []
        if errors:
            loc = errors[0].get("loc", ())
            first_loc = ".".join(str(p) for p in loc) if loc else None
        raise ContractParseError(
            raw=raw,
            reason=(
                f"Schema validation failed:\n{exc}\n"
                "Hint: The JSON structure didn't match the expected contract. "
                "The repair flow will attempt to fix this automatically."
            ),
            tier="schema",
            field=first_loc,
        ) from exc


class _FilePatch(BaseModel):
    """Shape of a validation-repair response: changed files only."""

    files: list[GeneratedFile] = Field(min_length=1)


def parse_file_patch(
    raw: str,
    dest: Path,
    *,
    allowed_paths: set[str],
) -> list[GeneratedFile]:
    """Parse a validation-repair response — ``{"files": [{path, content}]}``.

    Reuses the generation contract's fence-stripping, JSON, schema, and
    path-safety tiers. Additionally constrains *where* a patch may write:
    a path is accepted if it's one the project already knows (an existing
    project file or recipe-required file) or a new file inside a directory
    the project already populates. A repair response fixes files; it doesn't
    get to spray new directory trees.
    """
    cleaned = _strip_fences(raw)
    try:
        data: Any = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ContractParseError(
            raw=raw,
            reason=f"invalid JSON in repair patch: {exc}",
            tier="json",
        ) from exc
    try:
        patch = _FilePatch.model_validate(data)
    except ValidationError as exc:
        first_loc: str | None = None
        errors = exc.errors() if hasattr(exc, "errors") else []
        if errors:
            loc = errors[0].get("loc", ())
            first_loc = ".".join(str(p) for p in loc) if loc else None
        raise ContractParseError(
            raw=raw,
            reason=f"repair patch failed schema validation:\n{exc}",
            tier="schema",
            field=first_loc,
        ) from exc

    # Reuse the per-path safety rules (relative, no "..", inside dest, unique)
    # via a synthetic GenerationResult wrapper. The known project files are
    # the workflow exceptions: a repair may rewrite a declared CI workflow
    # but never introduce a new one.
    synthetic = GenerationResult(
        project_name="patch",
        language="patch",
        files=patch.files,
        smoke_check="-",
    )
    validate_paths(synthetic, dest, allowed_exceptions=allowed_paths)

    # Every directory (at any depth) that already holds an allowed file is a
    # legitimate home for a new file; anything else is out of bounds.
    allowed_dirs: set[str] = set()
    for known in allowed_paths:
        parts = known.replace("\\", "/").split("/")[:-1]
        for i in range(1, len(parts) + 1):
            allowed_dirs.add("/".join(parts[:i]))
    for entry in patch.files:
        path = entry.path.replace("\\", "/")
        if path in allowed_paths:
            continue
        parent = path.rsplit("/", 1)[0] if "/" in path else ""
        if parent == "" or parent in allowed_dirs:
            continue
        raise ContractParseError(
            raw=raw,
            reason=(
                f"repair patch writes outside the known project structure: {path!r} "
                "(not an existing file, and its directory holds no project files)"
            ),
            tier="path",
            field=path,
        )
    return patch.files


# Destinations a model-authored file may never claim. First-segment entries
# gate whole trees: .git (a written .git/config executes via fsmonitor /
# pager / alias on the next git operation — including the one commit_push
# runs), .scaffold (feeds the manifest smoke_check / entry_point that the
# run pipeline executes), .ssh. Exact entries are credential / runtime-secret
# files — .env.example stays ALLOWED; it is a required emitted file. Prefix
# entries are CI-executed trees, exempted only for paths the recipe itself
# declares (allowed_exceptions).
_DENIED_FIRST_SEGMENTS = frozenset({".git", ".scaffold", ".ssh"})
_DENIED_EXACT = frozenset({".env", ".env.local", ".npmrc", ".pypirc"})
_DENIED_PREFIXES = (".github/workflows/",)


def validate_paths(
    result: GenerationResult,
    dest: Path,
    *,
    canonical_module_name: str | None = None,
    allowed_exceptions: Collection[str] = (),
) -> None:
    """Ensure every emitted path is safe and unique within ``dest``.

    When ``canonical_module_name`` is given (the Python-underscored form of
    the project name), also reject any ``src/<dir>/`` segment that uses the
    hyphenated form of the same name. The LLM occasionally splits the
    project across ``src/foo-bar/`` and ``src/foo_bar/`` — only one can be
    a real Python package, so this is a generation bug. Raising triggers
    the repair loop, which usually self-corrects on retry.

    ``allowed_exceptions`` are paths (typically the recipe's required files)
    exempt from the CI-tree prefix denial — a recipe may legitimately require
    ``.github/workflows/ci.yml``. The hard segment / exact denials have no
    exceptions.
    """
    dest_resolved = dest.resolve()
    seen: set[str] = set()
    exception_set = {p.replace("\\", "/") for p in allowed_exceptions}
    # Python-only: a hyphenated directory can't be imported as a package, so a
    # split across src/foo-bar/ and src/foo_bar/ is a generation bug. In the
    # npm world hyphenated directories are idiomatic — never police them.
    hyphenated_form = (
        canonical_module_name.replace("_", "-")
        if canonical_module_name and "_" in canonical_module_name and result.language == "python"
        else None
    )
    for entry in result.files:
        raw_path = entry.path
        if not raw_path or raw_path != raw_path.strip():
            raise ContractParseError(
                raw=raw_path,
                reason=f"empty or whitespace-padded path: {raw_path!r}",
                tier="path",
                field=raw_path,
            )
        if raw_path.startswith(("/", "\\")):
            raise ContractParseError(
                raw=raw_path,
                reason=f"absolute path not allowed: {raw_path}",
                tier="path",
                field=raw_path,
            )
        normalized = raw_path.replace("\\", "/")
        if any(part == ".." for part in normalized.split("/")):
            raise ContractParseError(
                raw=raw_path,
                reason=f"'..' segment not allowed: {raw_path}",
                tier="path",
                field=raw_path,
            )
        first_segment = normalized.split("/", 1)[0]
        if first_segment in _DENIED_FIRST_SEGMENTS or normalized in _DENIED_EXACT:
            raise ContractParseError(
                raw=raw_path,
                reason=(
                    f"writes to a protected destination: {raw_path} — emit project "
                    "files only; secrets belong in .env.example"
                ),
                tier="path",
                field=raw_path,
            )
        if (
            any(normalized.startswith(prefix) for prefix in _DENIED_PREFIXES)
            and normalized not in exception_set
        ):
            raise ContractParseError(
                raw=raw_path,
                reason=(
                    f"writes a CI workflow the recipe does not declare: {raw_path} — "
                    "drop the file or move its logic into the project"
                ),
                tier="path",
                field=raw_path,
            )
        try:
            candidate = (dest_resolved / normalized).resolve()
            candidate.relative_to(dest_resolved)
        except ValueError as exc:
            # relative_to on an escaping path, or resolve() on a path with an
            # embedded NUL byte — both are contract violations, not crashes.
            raise ContractParseError(
                raw=raw_path,
                reason=f"path escapes destination or is malformed: {raw_path!r}",
                tier="path",
                field=raw_path,
            ) from exc
        folded = normalized.casefold()
        if folded in seen:
            raise ContractParseError(
                raw=raw_path,
                reason=f"duplicate path (case-insensitive): {raw_path}",
                tier="path",
                field=raw_path,
            )
        seen.add(folded)

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
                        tier="path",
                        field=raw_path,
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
        raise ContractParseError(
            raw="(hints)",
            reason="language hints missing 'manifest'",
            tier="required-files",
            field="manifest",
        )
    # Accumulate EVERY required path the model failed to emit and report them in
    # one error. The single repair round is built from ``exc.reason``; if we
    # raised on the first miss, repair would only learn about one file at a time
    # and a recipe missing several files could never be satisfied in one round.
    # ``missing`` holds bare paths (for ``field`` + the repair prompt); ``labels``
    # annotates roles for the human-readable reason.
    missing: list[str] = []
    labels: list[str] = []

    if manifest not in paths:
        missing.append(manifest)
        labels.append(f"{manifest} (manifest)")

    # The language-default entry point (e.g. ``src/<pkg>/main.py``) is only a
    # fallback. A recipe that declares its own application entry in
    # ``required_files`` (e.g. an ``app/`` layout) is authoritative — enforcing
    # the generic location too would double-require two conflicting entries, and
    # the model never sees the language default anyway (it's validation-only).
    entry_template = hints.get("entry_point", "")
    entry_point = entry_template.replace("{project_name}", result.project_name)
    entry_basename = entry_point.rsplit("/", 1)[-1] if entry_point else ""
    recipe_declares_entry = bool(entry_basename) and any(
        req.replace("\\", "/").rsplit("/", 1)[-1] == entry_basename
        for req in (extra_required or [])
    )
    if entry_point and not recipe_declares_entry and entry_point not in paths:
        missing.append(entry_point)
        labels.append(f"{entry_point} (entry point)")

    for required in ("README.md", ".env.example"):
        if required not in paths:
            missing.append(required)
            labels.append(required)

    for required in extra_required or []:
        normalized = required.replace("\\", "/")
        if normalized not in paths and normalized not in missing:
            missing.append(normalized)
            labels.append(normalized)

    if missing:
        raise ContractParseError(
            raw="(files)",
            reason="missing required file(s): " + ", ".join(labels),
            tier="required-files",
            field=missing[0],
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
    capability's data. When the model authored a service with the same name,
    the fragment's keys are overlaid onto it: the fragment is the verified
    infra config, so its image pin, port bindings, volumes, and environment
    win on conflict (a model-authored ``arrowhead`` without the fragment's
    auth env exits at startup), while keys only the model set — such as
    ``depends_on`` — survive. ``environment`` merges per variable.

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
            existing = services[frag.service]
            if isinstance(existing, dict) and _reconcile_fragment_service(existing, block):
                log.info(
                    "merge_capability_fragments: overlaying capability fragment onto "
                    "model-authored service %s",
                    frag.service,
                )
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


# The agent's own key, mirrored from ``auth.ENV_API_KEY`` (inlined to keep the
# parse path off the keyring-importing auth module). Always forwarded to the app
# container — generated agents build an Anthropic client at startup.
_AGENT_KEY_ENV = "ANTHROPIC_API_KEY"

# Conventional backend service names, used only when no service builds locally.
_APP_SERVICE_NAMES = frozenset({"app", "api", "backend", "web", "server"})


def normalize_app_service(
    result: GenerationResult,
    stack: ResolvedStack | None,
) -> GenerationResult:
    """Guarantee the backend (app) compose service can actually boot.

    The LLM-generated ``docker-compose.yml`` reliably forgets two things that
    leave the backend container dead on arrival:

    1. **The Anthropic key never reaches the container.** Generated agents build
       an Anthropic client at startup, but the ``app`` service rarely lists
       ``ANTHROPIC_API_KEY``. We add it (plus any capability secret var) using
       the same ``${VAR:-}`` interpolation the capability fragments already use,
       so ``docker compose`` fills it from the environment it runs in (the
       scaffold's resolved ``runtime_env``) — no plaintext file, host value
       forwarded.
    2. **A dangling ``env_file: .env``.** Compose treats a missing ``env_file``
       as a hard error; scaffold never writes ``.env``. We rewrite each entry to
       the ``{path, required: false}`` long form so a missing file is ignored and
       a present one still loads.

    In-network values the LLM *did* set (``DATABASE_URL: …@postgres``) and env
    keys owned by other services (``POSTGRES_USER`` on the ``postgres`` service)
    are left untouched. No-op when there's no compose file or no app service.
    """
    compose_index, compose_path = _find_compose(result)
    if compose_index is None:
        return result
    compose_data = _parse_compose_yaml(result.files[compose_index].content)
    services = compose_data.get("services")
    if not isinstance(services, dict) or not services:
        return result
    app_names = _app_service_names(services)
    if not app_names:
        return result

    # When the agent ships the runtime key-bootstrap (auth.*), tell its /setup
    # form which env vars to offer (mandatory key + optional services) via a
    # literal AGENT_SETUP_FIELDS JSON the verbatim agent_key_setup.py reads.
    setup_fields_json: str | None = None
    if stack is not None and any(c.kind == "auth" for c in stack.capabilities):
        from agent_scaffold.preflight import build_setup_fields

        setup_fields_json = json.dumps(build_setup_fields(stack), separators=(",", ":"))

    wanted = [_AGENT_KEY_ENV, *(stack.env_vars() if stack is not None else [])]
    changed = False
    for name in app_names:
        svc = services[name]
        if not isinstance(svc, dict):
            continue
        other_keys = _other_service_env_keys(services, exclude=name)
        if _inject_interpolated_env(svc, wanted, other_keys):
            changed = True
        if setup_fields_json is not None and _set_literal_env(
            svc, "AGENT_SETUP_FIELDS", setup_fields_json
        ):
            changed = True
        if _relax_env_file(svc):
            changed = True

    if not changed:
        return result

    compose_data = _canonicalize_compose(compose_data)
    rendered = (
        yaml.safe_dump(compose_data, sort_keys=False, default_flow_style=False).rstrip() + "\n"
    )
    new_files = list(result.files)
    new_files[compose_index] = GeneratedFile(path=compose_path, content=rendered)
    return result.model_copy(update={"files": new_files})


_FRONTEND_SERVICE_NAME = "frontend"
_DEFAULT_FRONTEND_PORT = 3000
_DEFAULT_BACKEND_PORT = 8000


def normalize_frontend_service(
    result: GenerationResult,
    stack: ResolvedStack | None,
    agent_title: str | None = None,
) -> GenerationResult:
    """Add a built ``frontend`` container to the sandbox when the stack ships a UI.

    A frontend capability with ``serve_in_container: true`` emits a ``frontend/``
    template tree including a ``Dockerfile``; this guarantees the generated
    ``docker-compose.yml`` has a matching ``frontend`` service — ``build:
    ./frontend``, the UI port, the backend URL passed as a **build arg** to the
    **host-mapped** backend port (the browser runs on the host, so it reaches the
    backend at ``localhost``, not the in-network service name; and a static Vite
    build bakes the URL in at build time, so a runtime ``environment`` would be
    inert), and ``depends_on`` the backend. One ``docker compose up`` then brings
    up frontend + backend as containers.

    No-op when no frontend capability opts into a container (it runs as a local
    ``pnpm dev`` instead), when there's no compose file, or when a ``frontend``
    service already exists. Stays inert until the deployments template ships a
    Dockerfile and sets ``serve_in_container`` — so it never references a missing
    build.
    """
    if stack is None:
        return result
    frontend_caps = [c for c in stack.capabilities if c.kind == "frontend" and c.serve_in_container]
    if not frontend_caps:
        return result
    compose_index, compose_path = _find_compose(result)
    if compose_index is None:
        return result
    compose_data = _parse_compose_yaml(result.files[compose_index].content)
    services = compose_data.get("services")
    if not isinstance(services, dict):
        return result
    if _FRONTEND_SERVICE_NAME in services:
        return result

    backend_names = _app_service_names(services)
    backend_name = backend_names[0] if backend_names else None
    backend_url = f"http://localhost:{_backend_host_port(services, backend_name)}"
    url_vars = list(dict.fromkeys(v for cap in frontend_caps for v in cap.env_vars))

    service: dict[str, Any] = {
        "build": {"context": "./frontend", "dockerfile": "Dockerfile"},
        "ports": [f"{_DEFAULT_FRONTEND_PORT}:{_DEFAULT_FRONTEND_PORT}"],
    }
    # Both the backend URL and the agent title are BUILD args, not runtime
    # `environment` entries: a static frontend (Vite + nginx) inlines VITE_*
    # values into the bundle at build time, and nginx serving those files ignores
    # runtime env — so `environment:` would silently do nothing. `build.args`
    # feeds the Dockerfile's `ARG VITE_AGENT_URL` / `ARG VITE_AGENT_TITLE`.
    args: dict[str, str] = dict.fromkeys(url_vars, backend_url)
    if agent_title and agent_title.strip():
        args["VITE_AGENT_TITLE"] = agent_title.strip()
    if args:
        service["build"]["args"] = args
    if backend_name:
        service["depends_on"] = [backend_name]
    services[_FRONTEND_SERVICE_NAME] = service

    compose_data = _canonicalize_compose(compose_data)
    rendered = (
        yaml.safe_dump(compose_data, sort_keys=False, default_flow_style=False).rstrip() + "\n"
    )
    new_files = list(result.files)
    new_files[compose_index] = GeneratedFile(path=compose_path, content=rendered)
    return result.model_copy(update={"files": new_files})


# Minimal capability set the official nginx image needs after cap_drop ALL:
# the master chowns cache dirs and setuid/setgids to the worker user, and
# binds port 80 in-container (NET_BIND_SERVICE is gone even for root once
# ALL is dropped).
_FRONTEND_CAP_ADD = ["CHOWN", "SETGID", "SETUID", "NET_BIND_SERVICE"]


def harden_scaffold_services(
    result: GenerationResult,
    stack: ResolvedStack | None,
) -> GenerationResult:
    """Harden the compose services the scaffold itself normalizes.

    A plain container is not a security boundary for generated code; these
    defaults narrow the accident surface without pretending otherwise:

    - ``security_opt: [no-new-privileges:true]`` — no privilege escalation
      via setuid binaries inside the container.
    - ``cap_drop: [ALL]`` — generated Python/node servers need no kernel
      capabilities; the scaffold-added frontend gets the minimal nginx set
      back via ``cap_add`` (chown/setuid/setgid for the worker drop plus
      NET_BIND_SERVICE for port 80 in-container).
    - Port bindings pin to ``127.0.0.1`` — this is a localhost dev tool,
      not a LAN service; an entry that already names a host ip is respected.

    Scope is deliberate: only the app service(s) and the scaffold-added
    ``frontend`` are touched — capability-authored fragments (postgres,
    qdrant, …) keep their authored shape; their hardening belongs in the
    deployments docs. Author-set ``security_opt``/``cap_drop`` values are
    respected (additive only), so a recipe that needs looser settings
    declares them and wins.

    Two knobs are deliberately absent: ``read_only`` rootfs would break the
    in-container ``.agent/runs`` and ``.agent/trace.jsonl`` writes the T2+
    substrates perform, and a pinned ``user:`` assumes a uid the
    model-authored Dockerfile never guaranteed — both need declared
    writable-mount support first (a follow-up, not a default).
    """
    del stack  # scope is name-based; the stack does not alter the policy
    compose_index, compose_path = _find_compose(result)
    if compose_index is None:
        return result
    compose_data = _parse_compose_yaml(result.files[compose_index].content)
    services = compose_data.get("services")
    if not isinstance(services, dict) or not services:
        return result

    targets = list(_app_service_names(services))
    if _FRONTEND_SERVICE_NAME in services and _FRONTEND_SERVICE_NAME not in targets:
        targets.append(_FRONTEND_SERVICE_NAME)

    changed = False
    for name in targets:
        svc = services.get(name)
        if not isinstance(svc, dict):
            continue
        if "security_opt" not in svc:
            svc["security_opt"] = ["no-new-privileges:true"]
            changed = True
        if "cap_drop" not in svc:
            svc["cap_drop"] = ["ALL"]
            if name == _FRONTEND_SERVICE_NAME and "cap_add" not in svc:
                svc["cap_add"] = list(_FRONTEND_CAP_ADD)
            changed = True
        ports = svc.get("ports")
        if isinstance(ports, list):
            rebound: list[Any] = []
            for entry in ports:
                pinned, did_pin = _pin_port_entry(entry)
                rebound.append(pinned)
                changed = changed or did_pin
            svc["ports"] = rebound

    if not changed:
        return result

    compose_data = _canonicalize_compose(compose_data)
    rendered = (
        yaml.safe_dump(compose_data, sort_keys=False, default_flow_style=False).rstrip() + "\n"
    )
    new_files = list(result.files)
    new_files[compose_index] = GeneratedFile(path=compose_path, content=rendered)
    return result.model_copy(update={"files": new_files})


def _pin_port_entry(entry: Any) -> tuple[Any, bool]:
    """Pin one compose port entry to loopback unless it already names a host ip.

    Covers every publishing shape the compose short/long syntax allows —
    ``8080`` (YAML int), ``"8080"``/``"8080/udp"``/``"8000-8010"`` (bare,
    published on all interfaces), ``"H:C"``, and the long-form mapping —
    because ``ports:`` publishes on 0.0.0.0 in ALL of these, not just the
    one-colon string the first pass handled. Returns ``(entry, changed)``;
    anything unrecognized passes through untouched.
    """
    if isinstance(entry, int):
        return f"127.0.0.1:{entry}:{entry}", True
    if isinstance(entry, dict):
        if "host_ip" in entry:
            return entry, False
        pinned = dict(entry)
        pinned["host_ip"] = "127.0.0.1"
        return pinned, True
    if isinstance(entry, str):
        body, slash, proto = entry.partition("/")
        if body.count(":") == 1:
            # "H:C" (optionally "/udp") — prefix the host ip.
            return f"127.0.0.1:{entry}", True
        if body.count(":") == 0 and body and all(c.isdigit() or c == "-" for c in body):
            # Bare container port or range: published on all interfaces with
            # the same host port — make that explicit and pin it.
            return f"127.0.0.1:{body}:{body}{slash}{proto}", True
    return entry, False


def assert_chat_endpoint(result: GenerationResult, stack: ResolvedStack | None) -> None:
    """Backstop the canonical ``POST /chat`` contract when a chat UI ships.

    The default containerized frontend POSTs to ``/chat``; if the stack
    containerizes a frontend but no generated file references a ``/chat`` route,
    raise :class:`ContractParseError` so the generation repair loop adds one
    (request ``{"message": str, "history": [...]}`` → response
    ``{"reply": str}``, non-streaming JSON). No-op when no containerized
    frontend is present — nothing calls ``/chat`` — so non-chat stacks are
    unaffected.
    """
    if stack is None:
        return
    if not any(c.kind == "frontend" and c.serve_in_container for c in stack.capabilities):
        return
    if any("/chat" in f.content for f in result.files):
        return
    raise ContractParseError(
        raw="",
        reason=(
            "the containerized chat frontend calls POST /chat, but no generated file "
            "defines a /chat route. Add a POST /chat endpoint that accepts "
            '{"message": str, "history": [{"role": "user"|"agent", "text": str}, ...]} '
            "(history is optional and oldest-first; tolerate its absence) and returns "
            '{"reply": str} as non-streaming JSON. Trim history to the recipe context '
            "budget (CONTEXT_INPUT_MAX) before the model call."
        ),
        tier="required-files",
    )


# Strings whose presence in any generated file counts as "CORS is configured",
# across the supported backend frameworks: FastAPI (CORSMiddleware /
# allow_origins), Hono (hono/cors), Express / Fastify and generic middleware
# (`cors(`), and hand-set headers (Access-Control-Allow-Origin).
_CORS_MARKERS: tuple[str, ...] = (
    "CORSMiddleware",
    "allow_origins",
    "hono/cors",
    "cors(",
    "Access-Control-Allow-Origin",
)


def assert_cors(result: GenerationResult, stack: ResolvedStack | None) -> None:
    """Backstop CORS when a chat UI ships on a different origin than the backend.

    The containerized frontend (``http://localhost:3000``) calls the backend
    (``http://localhost:8000``) cross-origin, which the browser blocks unless the
    backend sends ``Access-Control-Allow-Origin``. If a containerized frontend is
    present but no generated file configures CORS, raise :class:`ContractParseError`
    so the repair loop adds the middleware — otherwise the chat shows a bare
    "could not reach the agent". No-op when no containerized frontend ships.
    """
    if stack is None:
        return
    if not any(c.kind == "frontend" and c.serve_in_container for c in stack.capabilities):
        return
    # Framework-agnostic recognition: FastAPI middleware, Hono / Express /
    # Fastify middleware imports and calls, or hand-set headers. A valid Hono
    # backend (`import { cors } from "hono/cors"` + `app.use("*", cors())`)
    # once failed here because only the FastAPI tokens were checked.
    if any(marker in f.content for f in result.files for marker in _CORS_MARKERS):
        return
    fix = (
        'Add FastAPI CORSMiddleware with allow_origins=["*"] '
        '(allow_methods=["*"], allow_headers=["*"]).'
        if result.language == "python"
        else 'Add CORS middleware — Hono: `import { cors } from "hono/cors"` then '
        '`app.use("*", cors())`; Express: `app.use(cors())`.'
    )
    raise ContractParseError(
        raw="",
        reason=(
            "a containerized chat frontend (origin http://localhost:3000) calls the backend "
            "cross-origin, but no generated file configures CORS — the browser will block every "
            f"request. {fix}"
        ),
        tier="required-files",
    )


def normalize_package_manager(result: GenerationResult, hints: dict[str, Any]) -> GenerationResult:
    """Pin corepack's package manager in a generated ``package.json``.

    corepack resolves an unpinned manager to its LATEST release at image-build
    time, so the day a new pnpm major ships that needs a newer Node builtin,
    every ``RUN pnpm install`` on an older base image starts failing (seen
    live: pnpm 11 on ``node:20-alpine`` → ``ERR_UNKNOWN_BUILTIN_MODULE``).
    Deterministic injection beats prompting the model: no repair round, no
    drift. No-op for non-TypeScript results, when the language hints carry no
    ``package_manager_pin``, when ``package.json`` is absent or unparseable,
    or when the model already pinned a manager.
    """
    if str(hints.get("language", "")) != "typescript":
        return result
    pin = str(hints.get("package_manager_pin", "") or "").strip()
    if not pin:
        return result
    for i, entry in enumerate(result.files):
        if entry.path.replace("\\", "/") != "package.json":
            continue
        try:
            data = json.loads(entry.content)
        except json.JSONDecodeError:
            return result
        if not isinstance(data, dict) or data.get("packageManager"):
            return result
        data["packageManager"] = pin
        files = list(result.files)
        files[i] = GeneratedFile(path=entry.path, content=json.dumps(data, indent=2) + "\n")
        return result.model_copy(update={"files": files})
    return result


def assert_model_ids(result: GenerationResult) -> None:
    """Backstop against hallucinated Anthropic model ids in generated files.

    The LLM sometimes welds a real alias to a fabricated date suffix
    (e.g. ``claude-sonnet-4-6-20250514``) or emits a retired id. The API
    returns 404 on the generated agent's first model call, which surfaces to
    the user as "the agent is broken" long after generation succeeded. Scan
    every generated file for model-id-shaped strings and raise
    :class:`ContractParseError` on any the API does not serve, so the repair
    loop rewrites them to a valid id.
    """
    findings: dict[str, list[str]] = {}
    for f in result.files:
        unknown = find_unknown_model_ids(f.content)
        if unknown:
            findings[f.path] = unknown
    if not findings:
        return
    located = "; ".join(f"{path}: {', '.join(ids)}" for path, ids in sorted(findings.items()))
    raise ContractParseError(
        raw="",
        reason=(
            f"generated files reference Anthropic model id(s) the API does not serve: {located}. "
            "Replace every occurrence with one of the valid ids "
            f"({', '.join(RUNTIME_MODEL_CHOICES)}). These are complete ids; never append a "
            "date suffix to an alias and never invent new ids."
        ),
        tier="model-id",
    )


# ---------------------------------------------------------------------------
# Compose volumes + MCP registry passes
# ---------------------------------------------------------------------------


def normalize_compose_volumes(result: GenerationResult) -> GenerationResult:
    """Declare a top-level ``volumes:`` entry for every named volume in use.

    ``merge_capability_fragments`` copies a capability fragment's service
    ``volumes`` (e.g. ``arrowhead_corpus:/app/documents``) verbatim, and the
    LLM authors its own — but a named volume referenced by a service without
    a matching top-level declaration is a hard ``docker compose`` error.
    Collect every named-volume source across services and add the missing
    declarations (``name: {}``, sorted), leaving existing definitions
    untouched. Bind mounts (``./x``, ``/x``, ``~/x``, ``$VAR``), anonymous
    volumes, and long-form bind mounts are ignored. Stack-independent so it
    also repairs model-authored volumes. No-op without a compose file.
    """
    compose_index, compose_path = _find_compose(result)
    if compose_index is None:
        return result
    compose_data = _parse_compose_yaml(result.files[compose_index].content)
    services = compose_data.get("services")
    if not isinstance(services, dict) or not services:
        return result
    named = _named_volume_sources(services)
    if not named:
        return result
    raw_volumes = compose_data.get("volumes")
    volumes: dict[str, Any] = dict(raw_volumes) if isinstance(raw_volumes, dict) else {}
    added = False
    for name in sorted(named):
        if name not in volumes:
            volumes[name] = {}
            added = True
    if not added:
        return result
    compose_data["volumes"] = volumes
    compose_data = _canonicalize_compose(compose_data)
    rendered = (
        yaml.safe_dump(compose_data, sort_keys=False, default_flow_style=False).rstrip() + "\n"
    )
    new_files = list(result.files)
    new_files[compose_index] = GeneratedFile(path=compose_path, content=rendered)
    return result.model_copy(update={"files": new_files})


def _named_volume_sources(services: dict[str, Any]) -> list[str]:
    """Named-volume sources referenced by any service's ``volumes`` list.

    A short-form source counts as a named volume when it is not path-like
    (``.`` / ``/`` / ``~`` / ``$``-prefixed, no ``/`` inside) and the entry
    has a ``:`` (a single-segment entry is an anonymous volume). A long-form
    entry counts with ``type: volume``, or with ``type`` omitted and a
    non-path-like ``source``. POSIX sources only — generated compose never
    uses Windows paths.
    """
    named: list[str] = []
    for svc in services.values():
        if not isinstance(svc, dict):
            continue
        for entry in svc.get("volumes") or []:
            source: str | None = None
            if isinstance(entry, str):
                body = entry.strip()
                if ":" not in body:
                    continue
                candidate = body.split(":", 1)[0].strip()
                if candidate and candidate[0] not in "./~$" and "/" not in candidate:
                    source = candidate
            elif isinstance(entry, dict):
                vol_type = entry.get("type")
                src = entry.get("source")
                if isinstance(src, str) and src:
                    if vol_type == "volume" or (
                        vol_type is None and src[0] not in "./~$" and "/" not in src
                    ):
                        source = src
            if source and source not in named:
                named.append(source)
    return named


def normalize_mcp_registry_mount(
    result: GenerationResult,
    stack: ResolvedStack | None,
    mcp_servers: Sequence[MCPServerSpec] = (),
) -> GenerationResult:
    """Deterministically wire the ``mcp.json`` registry into the compose stack.

    Two rewrites, both no-ops without MCP servers, a compose file, or an app
    service:

    1. Bind-mount the step-owned registry read-only into every service built
       from the project root (``./mcp.json:<workdir>/mcp.json:ro``, workdir
       from the root Dockerfile's last ``WORKDIR``, default ``/app``) unless
       a mount already references it — the prompt instructs the model to do
       this, but the containerized backend goes toolless when it forgets.
       Non-root builds (``./frontend``) are skipped: they never read the
       registry and don't share the root Dockerfile's workdir.
    2. Pin recipe-declared env defaults on bound MCP services: a server
       ``env`` value that is not a ``required`` / ``optional`` sentinel is a
       default, written as ``VAR: ${VAR:-value}`` on the bound capability's
       service so the environment can still override it (this is how a
       recipe selects e.g. ``ARROWHEAD_PROFILE: coding``).
    """
    if not mcp_servers:
        return result
    compose_index, compose_path = _find_compose(result)
    if compose_index is None:
        return result
    compose_data = _parse_compose_yaml(result.files[compose_index].content)
    services = compose_data.get("services")
    if not isinstance(services, dict) or not services:
        return result

    changed = False
    workdir = _dockerfile_workdir(result).rstrip("/") or "/app"
    mount = f"./mcp.json:{workdir}/mcp.json:ro"
    for name in _mcp_mount_targets(services):
        svc = services[name]
        if not isinstance(svc, dict):
            continue
        volumes = svc.get("volumes")
        entries = volumes if isinstance(volumes, list) else []
        if any("mcp.json" in str(entry) for entry in entries):
            continue
        svc["volumes"] = [*entries, mount]
        changed = True

    capabilities = {cap.id: cap for cap in stack.capabilities} if stack is not None else {}
    for server in mcp_servers:
        capability = capabilities.get(server.capability)
        docker = getattr(capability, "docker", None)
        service_name = getattr(docker, "service", None)
        if not service_name or service_name not in services:
            continue
        svc = services[service_name]
        if not isinstance(svc, dict):
            continue
        for var, value in server.env.items():
            if value.strip().lower() in ("required", "optional", ""):
                continue
            if _force_literal_env(svc, var, f"${{{var}:-{value}}}"):
                changed = True

    if not changed:
        return result
    compose_data = _canonicalize_compose(compose_data)
    rendered = (
        yaml.safe_dump(compose_data, sort_keys=False, default_flow_style=False).rstrip() + "\n"
    )
    new_files = list(result.files)
    new_files[compose_index] = GeneratedFile(path=compose_path, content=rendered)
    return result.model_copy(update={"files": new_files})


def _dockerfile_workdir(result: GenerationResult) -> str:
    """The last ``WORKDIR`` in the root ``Dockerfile``; ``/app`` when absent."""
    for f in result.files:
        if f.path.replace("\\", "/") != "Dockerfile":
            continue
        workdir = "/app"
        for line in f.content.splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("WORKDIR ") and len(stripped.split(None, 1)) == 2:
                value = stripped.split(None, 1)[1].strip().strip("\"'")
                if value:
                    workdir = value
        return workdir
    return "/app"


def _force_literal_env(svc: dict[str, Any], name: str, value: str) -> bool:
    """Set ``name: value`` on the service env, replacing any existing value.

    Unlike :func:`_set_literal_env`, an existing entry is overwritten — the
    recipe-declared default wins over the fragment's authored one. Returns
    whether the service changed.
    """
    raw = svc.get("environment")
    if raw is None:
        env: dict[str, Any] = {}
    elif isinstance(raw, list):
        env = _env_list_to_dict(raw)
    elif isinstance(raw, dict):
        env = dict(raw)
    else:
        return False
    if env.get(name) == value:
        return False
    env[name] = value
    svc["environment"] = env
    return True


# File extensions that count as generated application source for
# assert_mcp_wiring: compose files, READMEs, and .env examples mentioning
# mcp.json prove nothing about the agent loop actually loading its tools.
_MCP_SOURCE_EXTENSIONS = (".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")


def assert_mcp_wiring(result: GenerationResult, mcp_servers: Sequence[MCPServerSpec]) -> None:
    """Backstop: generated source must read the MCP registry.

    The project binds MCP servers, the registry step writes ``mcp.json``, and
    ``normalize_mcp_registry_mount`` mounts it — but only generated code can
    connect and expose the tools to the agent loop. If no source file
    references ``mcp.json``, raise :class:`ContractParseError` so the repair
    loop wires the client. The marker is deliberately loose (framework MCP
    client APIs vary too much to demand specific imports); a mention in a
    compose file or README does not count. No-op without MCP servers.
    """
    if not mcp_servers:
        return
    for f in result.files:
        path = f.path.replace("\\", "/")
        if path.endswith(_MCP_SOURCE_EXTENSIONS) and "mcp.json" in f.content:
            return
    names = ", ".join(server.id for server in mcp_servers)
    raise ContractParseError(
        raw="",
        reason=(
            f"the project binds MCP server(s) ({names}) but no generated source file "
            "references the mcp.json registry, so the agent never gains its MCP tools. "
            "At startup, read mcp.json from the project root when present, expand ${VAR} "
            "placeholders from the process environment, connect each streamable_http "
            "entry with the framework's MCP client (prefer the entry's containerUrl "
            "when the backend runs inside compose), expose the discovered tools to the "
            "agent loop, and log-and-continue when the file is absent or a server is "
            "unreachable."
        ),
        tier="required-files",
    )


def _backend_host_port(services: dict[str, Any], backend_name: str | None) -> int:
    """Host-mapped backend port (``"8000:8000"`` → 8000); default 8000."""
    if backend_name and isinstance(services.get(backend_name), dict):
        for entry in services[backend_name].get("ports") or []:
            host = str(entry).split(":", 1)[0].strip().strip("\"'")
            if host.isdigit():
                return int(host)
    return _DEFAULT_BACKEND_PORT


def _app_service_names(services: dict[str, Any]) -> list[str]:
    """Backend service(s): those built locally (``build:``), else conventionally named."""
    build_services = [
        name for name, svc in services.items() if isinstance(svc, dict) and "build" in svc
    ]
    if build_services:
        return build_services
    return [name for name in services if name in _APP_SERVICE_NAMES]


def _mcp_mount_targets(services: dict[str, Any]) -> list[str]:
    """Services that get the ``mcp.json`` registry mount: root-context builds.

    The mount's in-container path comes from the ROOT Dockerfile's
    ``WORKDIR``, so only services built from the project root can host it —
    a ``./frontend`` build neither reads the registry nor shares that
    workdir. Falls back to conventional backend names when nothing builds
    from the root (image-only compose files).
    """
    roots = [
        name
        for name, svc in services.items()
        if isinstance(svc, dict) and _is_root_build(svc.get("build"))
    ]
    if roots:
        return roots
    return [name for name in services if name in _APP_SERVICE_NAMES]


def _is_root_build(build: Any) -> bool:
    """Whether a compose ``build`` value points at the project root context."""
    if isinstance(build, str):
        return build.strip() in (".", "./")
    if isinstance(build, dict):
        return str(build.get("context", ".")).strip() in (".", "./")
    return False


def _service_env_keys(svc: dict[str, Any]) -> set[str]:
    """Env var names declared in a service's ``environment`` (dict or list form)."""
    env = svc.get("environment")
    if isinstance(env, dict):
        return {str(k) for k in env}
    if isinstance(env, list):
        return {str(item).split("=", 1)[0] for item in env if isinstance(item, str)}
    return set()


def _other_service_env_keys(services: dict[str, Any], *, exclude: str) -> set[str]:
    """Env keys owned by every service except ``exclude`` (so DB config stays put)."""
    keys: set[str] = set()
    for name, svc in services.items():
        if name != exclude and isinstance(svc, dict):
            keys |= _service_env_keys(svc)
    return keys


def _env_list_to_dict(items: list[Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        if isinstance(item, str):
            key, sep, value = item.partition("=")
            out[key] = value if sep else None
    return out


def _inject_interpolated_env(svc: dict[str, Any], wanted: list[str], other_keys: set[str]) -> bool:
    """Add ``VAR: ${VAR:-}`` entries the app needs but doesn't already set.

    Skips vars already on the service and vars owned by another service (so a
    DB's ``POSTGRES_USER`` isn't duplicated onto the app). A list-form
    ``environment`` is normalized to a dict only when we actually add something.
    """
    raw = svc.get("environment")
    if raw is None:
        env: dict[str, Any] = {}
    elif isinstance(raw, list):
        env = _env_list_to_dict(raw)
    elif isinstance(raw, dict):
        env = dict(raw)
    else:
        return False
    existing = set(env)
    added = False
    for var in dict.fromkeys(wanted):  # dedupe, preserve order
        if var in existing or var in other_keys:
            continue
        env[var] = f"${{{var}:-}}"  # ${VAR:-} — Compose forwards the host value
        added = True
    if added:
        svc["environment"] = env
    return added


def _set_literal_env(svc: dict[str, Any], name: str, value: str) -> bool:
    """Set ``name: value`` (a literal) on the service env; no-op if already set.

    Normalizes a list-form ``environment`` to a dict only when it changes
    something. Returns whether the service was modified.
    """
    raw = svc.get("environment")
    if raw is None:
        env: dict[str, Any] = {}
    elif isinstance(raw, list):
        env = _env_list_to_dict(raw)
    elif isinstance(raw, dict):
        env = dict(raw)
    else:
        return False
    if name in env:
        return False
    env[name] = value
    svc["environment"] = env
    return True


def _relax_env_file(svc: dict[str, Any]) -> bool:
    """Rewrite ``env_file`` entries to ``{path, required: false}`` (missing-file safe)."""
    raw = svc.get("env_file")
    if raw is None:
        return False
    entries = raw if isinstance(raw, list) else [raw]
    normalized: list[Any] = []
    changed = False
    for entry in entries:
        if isinstance(entry, str):
            normalized.append({"path": entry, "required": False})
            changed = True
        elif isinstance(entry, dict) and "required" not in entry:
            normalized.append({**entry, "required": False})
            changed = True
        else:
            normalized.append(entry)
    if changed:
        svc["env_file"] = normalized
    return changed


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
            tier="path",
            field=colliding[0].split(" ", 1)[0] if colliding else None,
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
    if frag.platform:
        # Single-arch images (TEI's CPU builds ship amd64 only) hard-fail the
        # pull on other hosts unless compose is told to run them emulated.
        block["platform"] = frag.platform
    if frag.ports:
        block["ports"] = list(frag.ports)
    if frag.volumes:
        block["volumes"] = list(frag.volumes)
    if frag.environment:
        block["environment"] = dict(frag.environment)
    if frag.healthcheck:
        block["healthcheck"] = dict(frag.healthcheck)
    return block


def _reconcile_fragment_service(existing: dict[str, Any], block: dict[str, Any]) -> bool:
    """Overlay a capability fragment block onto a model-authored service.

    Every key the fragment declares wins wholesale — except ``environment``,
    which merges per variable (model-added vars survive, fragment values win
    on collision) so a model-tuned setting outside the fragment's contract
    isn't dropped. Returns whether the service changed.
    """
    changed = False
    for key, value in block.items():
        if key == "environment":
            raw = existing.get(key)
            current: dict[str, Any] = (
                _env_list_to_dict(raw)
                if isinstance(raw, list)
                else raw
                if isinstance(raw, dict)
                else {}
            )
            merged = {**current, **value}
            if merged != raw:
                existing[key] = merged
                changed = True
            continue
        if existing.get(key) != value:
            existing[key] = value
            changed = True
    return changed


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
