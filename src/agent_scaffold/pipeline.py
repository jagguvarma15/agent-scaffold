"""Post-plan generation pipeline.

Lifted out of :mod:`agent_scaffold.cli` so the same orchestration can serve
``agent-scaffold new`` (one-shot CLI) and the upcoming ``agent-scaffold
scaffold`` REPL. The body is the same sequence the CLI ran inline:

    generate (or load from cache)
    → write
    → gitignore enforcement
    → required-file verification
    → format
    → static validation
    → manifest + template snapshot

The REPL needs failures to surface as recoverable errors (so the shell can
report and continue), so every hard exit the inline body used has been
replaced with :class:`PipelineError`. ``cmd_new`` translates that back into
``typer.Exit(1)``; the REPL prints the error panel and returns to the
prompt.

The other helpers in this module (``_attempt_parse``, ``_generate_with_repair``,
``run_post_gen_formatter``, ``_format_hint``, ``_print_*``) live here rather
than ``cli`` because they're owned by the pipeline. ``cli.cmd_regenerate``
imports them back when it needs to format a single regenerated file.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel

from agent_scaffold._redact import redact
from agent_scaffold.auth import project_namespace
from agent_scaffold.cache import get_cached, save_cache
from agent_scaffold.capabilities import ResolvedStack
from agent_scaffold.capability_emit import copy_capability_templates
from agent_scaffold.config import Config
from agent_scaffold.context import AssembledContext
from agent_scaffold.contract import (
    ContractParseError,
    GeneratedFile,
    GenerationResult,
    assert_chat_endpoint,
    check_frontend_collisions,
    merge_capability_fragments,
    normalize_app_service,
    normalize_frontend_service,
    parse,
    parse_file_patch,
    validate_paths,
    validate_required_files,
)
from agent_scaffold.discovery import Recipe
from agent_scaffold.generator import (
    GenerationRequest,
    generate,
    get_run_usage,
    prompts_signature,
    repair,
    repair_validation,
    reset_run_usage,
)
from agent_scaffold.language_hints import reconcile_entry_point
from agent_scaffold.manifest import (
    Manifest,
    build_file_entries,
    write_manifest,
)
from agent_scaffold.progress import (
    GenerationDisplay,
    ProgressEvent,
)
from agent_scaffold.report import (
    GenerationReport,
    derive_observability,
    print_generation_report,
)
from agent_scaffold.run_summary import write_run_summary
from agent_scaffold.template_snapshot import (
    compute_template_sha,
    prune_snapshots,
    save_generation_snapshot,
    short_sha,
)
from agent_scaffold.topology import Role, Topology
from agent_scaffold.validator import (
    ValidationTier,
    tier_command,
    verify_required_files_on_disk,
)
from agent_scaffold.validator import validate as run_validate
from agent_scaffold.writer import (
    DestinationExistsError,
    DiffPreviewCancelled,
    FileDiff,
    WriteMode,
    WriteReport,
    ensure_gitignore_defaults,
    write_project,
)

# Module-level console mirrors the cli.py pattern so the lifted body prints
# through the same Rich sink it did before. Tests don't need to monkeypatch
# this — they pass their own console when they care.
console = Console()


# ---------------------------------------------------------------------------
# Exceptions / result types
# ---------------------------------------------------------------------------


class PipelineError(Exception):
    """Recoverable failure in the post-plan generation pipeline.

    Carries ``phase`` (which stage hit the issue) and ``hint`` (a one-liner
    suggesting next steps) so callers can render a useful error panel
    instead of dumping a stack trace. The CLI converts this back into
    ``typer.Exit(1)``; the REPL prints and returns to the prompt.
    """

    def __init__(self, message: str, *, phase: str = "", hint: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.phase = phase
        self.hint = hint


@dataclass(frozen=True)
class PipelineInputs:
    """All state needed by :func:`run_generation`.

    Frozen so the caller can build it once after the plan-confirm panel and
    hand the same value to retries / cache-validating code paths without
    worrying about mid-flight mutation.
    """

    cfg: Config
    recipe: Recipe
    language: str
    framework: str
    project_name: str
    """The canonical project name used in the request, dest, and as the
    Python module name. cmd_new produces this via ``_python_module_name``."""
    raw_project_name: str | None
    """The user's original typed name (may differ in casing / hyphens). Goes
    into ``manifest.answers.project_name`` for reproducibility."""
    dest: Path
    deployments: Path
    """Path to the resolved deployments tree — needed for the template SHA."""
    ctx: AssembledContext
    hints: dict[str, Any]
    topology: Topology
    roles: list[Role]
    write_mode: WriteMode
    strict: bool
    format_output: bool
    skip_validation: bool
    no_cache: bool

    # Refinement accumulators from the REPL's free-text interpreter.
    # Defaults keep cmd_new's construction site compatible — only the REPL
    # populates these. All five flow into the GenerationRequest *and* the
    # cache_inputs fingerprint so a refinement actually busts the cache.
    extra_dependencies: dict[str, dict[str, str]] = field(default_factory=dict)
    extra_steps: list[str] = field(default_factory=list)
    removed_steps: set[str] = field(default_factory=set)
    removed_roles: set[str] = field(default_factory=set)
    refinement_notes: list[str] = field(default_factory=list)

    # The agent's role / persona from the "describe your agent" step (or the
    # recipe default). Flows into the GenerationRequest (→ backend system prompt)
    # and the cache key so a role change regenerates. ``None`` for vanilla runs.
    agent_role: str | None = None

    # Short agent title from the describe step → passed as a VITE_AGENT_TITLE
    # build arg to the containerized frontend so the chat UI reflects the agent.
    # ``None`` leaves the template's default ("Agent Chat").
    agent_title: str | None = None

    # Resolved capability stack threaded from cmd_new / cmd_regenerate.
    # ``None`` when the deployments source has no ``docs/capabilities/``
    # tree or the recipe didn't declare any.
    resolved_stack: ResolvedStack | None = None

    # Pre-write hook used by the REPL's ``WriteMode.diff`` flow. When set,
    # the writer computes a full :class:`FileDiff` set, calls this callback
    # once with all of them, and lets the user approve / reject the whole
    # batch — instead of asking per file via the legacy ``confirm_diff``
    # path. Returns ``True`` to proceed (modified files become overwrites),
    # ``False`` to abort the write step. Ignored when ``write_mode != diff``.
    pre_write_confirm: Callable[[list[FileDiff]], bool] | None = None


@dataclass
class RunReport:
    """Outcome of :func:`run_generation`.

    The REPL renders a compact summary from these fields; cmd_new prints
    them with its existing helpers. ``result`` and ``report`` may be ``None``
    if a phase aborted — the caller checks ``next_steps_dest`` to decide
    whether to print a "Next steps" panel.
    """

    result: GenerationResult | None
    report: WriteReport | None
    validation_results: list[Any] = field(default_factory=list)
    wall_seconds: float = 0.0
    cached: bool = False
    manifest_written: bool = False


# ---------------------------------------------------------------------------
# Helpers (lifted from cli.py — same behavior)
# ---------------------------------------------------------------------------


def _save_failure(raw: str, failures_dir: Path) -> Path:
    failures_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    path = failures_dir / f"{ts}.json"
    path.write_text(raw, encoding="utf-8")
    return path


def _format_hint(language: str) -> str:
    if language == "python":
        return "ruff check --fix + ruff format"
    if language == "typescript":
        if shutil.which("prettier"):
            return "prettier --write"
        if shutil.which("biome"):
            return "biome format --write"
    return f"no formatter for {language}"


def _run_subprocess_with_events(
    cmd: list[str],
    on_event: Callable[[ProgressEvent], None] | None,
) -> int:
    if on_event is not None:
        on_event(ProgressEvent(kind="bash_started", payload={"cmd": cmd}))
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if on_event is not None:
        on_event(
            ProgressEvent(
                kind="bash_done",
                payload={
                    "cmd": cmd,
                    "exit_code": proc.returncode,
                    "stdout_tail": (proc.stdout or "")[-200:],
                    "stderr_tail": (proc.stderr or "")[-200:],
                },
            )
        )
    return proc.returncode


def run_post_gen_formatter(
    dest: Path,
    language: str,
    on_event: Callable[[ProgressEvent], None] | None = None,
) -> None:
    """Auto-fix trivial lint + reformat freshly-written files.

    Idempotent and best-effort: a missing formatter or non-zero exit must not
    fail the run, since the static-validation tier will surface anything that
    still matters.

    When the project has a ``frontend/`` subdir (typical for projects that
    pulled in a ``frontend.*`` capability), prettier is also run against it
    so the copied TS/TSX files match the team's house style. Skips silently
    when prettier isn't on PATH — the user can run ``pnpm exec prettier``
    themselves later.
    """
    if language == "python":
        if shutil.which("ruff") is None:
            return
        _run_subprocess_with_events(
            ["ruff", "check", "--fix", "--unsafe-fixes", "--quiet", str(dest)], on_event
        )
        _run_subprocess_with_events(["ruff", "format", "--quiet", str(dest)], on_event)
    elif language == "typescript":
        if shutil.which("prettier"):
            _run_subprocess_with_events(
                ["prettier", "--write", "--log-level", "silent", str(dest)], on_event
            )
        elif shutil.which("biome"):
            _run_subprocess_with_events(["biome", "format", "--write", str(dest)], on_event)

    # Dual-language: format any copied frontend/ subtree with prettier when
    # the primary language wasn't already typescript (otherwise the call
    # above already covered it).
    frontend_dir = dest / "frontend"
    if language != "typescript" and frontend_dir.is_dir() and shutil.which("prettier"):
        _run_subprocess_with_events(
            ["prettier", "--write", "--log-level", "silent", str(frontend_dir)],
            on_event,
        )


def _capabilities_brief(stack: ResolvedStack | None) -> list[dict[str, Any]]:
    """Compact projection of the resolved stack for the user-prompt template.

    Surfaces only what the LLM needs to write compose / .env / overrides:
    id, kind, env vars, docker service name, and any ``emit_files`` dest
    globs (so the model knows which paths the scaffold will copy
    verbatim). Full bodies live in the assembled context tier.
    """
    if stack is None:
        return []
    brief: list[dict[str, Any]] = []
    for cap in stack.capabilities:
        emit_globs: list[str] = []
        for emit in cap.emit_files:
            dest = emit.dest.replace("\\", "/").rstrip("/")
            if emit.source.endswith("**") or emit.source.endswith("/*"):
                emit_globs.append(f"{dest}/**" if dest else "**")
            else:
                emit_globs.append(dest)
        brief.append(
            {
                "id": cap.id,
                "kind": cap.kind,
                "env_vars": list(cap.env_vars),
                "docker_service": cap.docker.service if cap.docker else None,
                "emit_globs": emit_globs,
            }
        )
    return brief


def _attempt_parse(
    raw: str,
    dest: Path,
    hints: dict[str, Any],
    project_name: str,
    extra_required: list[str],
    resolved_stack: ResolvedStack | None = None,
    strict: bool = False,
    agent_title: str | None = None,
    check_chat: bool = True,
) -> GenerationResult:
    result = parse(raw)
    validate_paths(result, dest, canonical_module_name=project_name)
    validate_required_files(result, hints, extra_required)
    # Capability-aware passes: collision check (may raise in strict mode)
    # then deterministic compose merge. Both no-op when resolved_stack is
    # ``None`` or the stack has no relevant capabilities.
    check_frontend_collisions(result, resolved_stack, strict=strict)
    result = merge_capability_fragments(result, resolved_stack)
    # Guarantee the backend service can boot: forward ANTHROPIC_API_KEY (+ secret
    # vars) into the app container and make a dangling env_file non-fatal.
    result = normalize_app_service(result, resolved_stack)
    # Containerize the frontend into the sandbox when a frontend capability opts in
    # (serve_in_container) — adds a built `frontend` service wired to the backend,
    # plus the VITE_AGENT_TITLE build arg so the chat UI reflects the agent.
    result = normalize_frontend_service(result, resolved_stack, agent_title)
    # Backstop the canonical POST /chat contract (skipped on the trusted cache
    # path); a miss raises ContractParseError → the repair loop adds the route.
    if check_chat:
        assert_chat_endpoint(result, resolved_stack)
    if result.project_name != project_name:
        # The LLM sometimes canonicalizes hyphens -> underscores for python.
        result = result.model_copy(update={"project_name": project_name})
    return result


def _generate_with_repair(
    req: GenerationRequest,
    cfg: Config,
    dest: Path,
    hints: dict[str, Any],
    project_name: str,
    extra_required: list[str],
    progress: Callable[[ProgressEvent], None] | None = None,
    resolved_stack: ResolvedStack | None = None,
    agent_title: str | None = None,
) -> tuple[GenerationResult, str]:
    """Return ``(parsed_result, raw_response_text_that_succeeded)``.

    One repair attempt: on first parse failure, the raw response is saved
    to ``cfg.failures_dir`` and a repair prompt is sent to the LLM. If the
    repaired response also fails to parse, raises :class:`PipelineError`.
    """
    raw = generate(req, cfg, progress=progress)
    try:
        return (
            _attempt_parse(
                raw,
                dest,
                hints,
                project_name,
                extra_required,
                resolved_stack,
                req.strict,
                agent_title=agent_title,
            ),
            raw,
        )
    except ContractParseError as exc:
        failure_path = _save_failure(raw, cfg.failures_dir)
        tier_label = _format_contract_failure(exc)
        console.print(
            f"[yellow]Warning:[/] contract parse failed ({tier_label}): {exc.reason}.\n"
            f"Raw response saved to: {failure_path}\n"
            "Attempting repair..."
        )
        repaired = repair(raw, exc.reason, cfg, strict=req.strict, progress=progress)
        try:
            return (
                _attempt_parse(
                    repaired,
                    dest,
                    hints,
                    project_name,
                    extra_required,
                    resolved_stack,
                    req.strict,
                    agent_title=agent_title,
                ),
                repaired,
            )
        except ContractParseError as exc2:
            second_failure = _save_failure(repaired, cfg.failures_dir)
            raise PipelineError(
                f"repair also failed ({_format_contract_failure(exc2)}): {exc2.reason}",
                phase="generate",
                hint=(
                    f"First failure: {_format_contract_failure(exc)} (saved to {failure_path})\n"
                    f"Second failure: {_format_contract_failure(exc2)} "
                    f"(saved to {second_failure})"
                ),
            ) from exc2


def _format_contract_failure(exc: ContractParseError) -> str:
    """Render a ContractParseError's tier + field as a short label.

    Used in pipeline warning/error messages so users (and bug reports) see
    which failure mode tripped, not just the prose message.
    """
    if exc.field:
        return f"tier={exc.tier}, field={exc.field!r}"
    return f"tier={exc.tier}"


def _emit_generation_report(
    *,
    inputs: PipelineInputs,
    cfg: Config,
    report: Any,
    wall_seconds: float,
    cached: bool,
    display: GenerationDisplay,
    repair_rounds: int = 0,
) -> None:
    """Build + print the consolidated post-generation panel from the `finally` block.

    Selections come from ``inputs``; usage summed over every API call in the
    run (generate + repair rounds); file counts from the writer's report;
    phase data from the display. Any of these can be missing if generation
    aborted early — the report silently elides sections with no data.
    """
    usage = get_run_usage()
    files_written = files_overwritten = files_skipped = 0
    top_files: list[str] = []
    if report is not None:
        files_written = len(report.written)
        files_overwritten = len(report.overwritten)
        files_skipped = len(report.skipped)
        top_files = sorted({*report.written, *report.overwritten})
    print_generation_report(
        GenerationReport(
            recipe_slug=inputs.recipe.slug,
            language=inputs.language,
            framework=inputs.framework,
            observability=derive_observability(inputs.resolved_stack),
            model=cfg.model,
            wall_seconds=wall_seconds,
            cached=cached,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_input_tokens,
            cache_creation_tokens=usage.cache_creation_input_tokens,
            files_written=files_written,
            files_overwritten=files_overwritten,
            files_skipped=files_skipped,
            top_files=top_files,
            repair_rounds=repair_rounds,
            phase_durations=dict(getattr(display, "phase_durations", {})),
            warnings=list(getattr(display, "warnings", [])),
            errors=list(getattr(display, "errors", [])),
            run_log_dir=str(getattr(display, "run_log_dir", "") or ""),
        )
    )


def print_usage_summary(model: str, wall_seconds: float, *, cached: bool) -> None:
    """Deprecated — kept as a no-op shim. The consolidated GenerationReport
    panel emitted from ``run_generation``'s finally block now covers what
    this used to print. Removed entirely after one release.
    """
    del model, wall_seconds, cached


def print_phase_summary(
    phase_durations: dict[str, float], warnings: list[str], errors: list[str]
) -> None:
    """Deprecated — kept as a no-op shim. Phase timings are now rendered as a
    section inside the consolidated GenerationReport panel.
    """
    del phase_durations, warnings, errors


def print_next_steps(dest: Path, language: str, smoke_check: str, post_install: list[str]) -> None:
    lines = [f"Project written to: [bold]{dest}[/]\n", "Next steps:", f"  cd {dest}"]
    if post_install:
        for cmd in post_install:
            lines.append(f"  {cmd}")
    elif language == "python":
        lines.append("  uv sync")
    elif language == "typescript":
        lines.append("  pnpm install")
    lines.append(f"  {smoke_check}")
    lines.append(f"\n[dim]Run summary: {dest / '.scaffold' / 'run-summary.md'}[/]")
    console.print(Panel("\n".join(lines), title="Next steps", expand=False))


# ---------------------------------------------------------------------------
# Validation repair loop
# ---------------------------------------------------------------------------

# Bounded by design: research on generate→validate→repair loops shows most of
# the win lands in the first round or two; past that you're paying for noise.
MAX_REPAIR_ROUNDS = 2
_REPAIR_OUTPUT_CHAR_CAP = 6_000
_REPAIR_RECIPE_CHAR_CAP = 40_000
_IMPLICATED_FILE_CHAR_CAP = 16_000
_IMPLICATED_FILES_MAX = 6
_VALIDATION_TIERS = [ValidationTier.static, ValidationTier.build]

_OUTPUT_PATH_RE = re.compile(
    r"(?P<path>[A-Za-z0-9_./\\-]+\."
    r"(?:py|pyi|ts|tsx|js|jsx|mjs|cjs|json|toml|yaml|yml|cfg|ini|sh|md|txt))\b"
)


def _implicated_files(
    output: str,
    dest: Path,
    known_paths: set[str],
    required_files: list[str],
) -> dict[str, str]:
    """Map validation output back to project files; return on-disk bodies.

    Ruff / tsc / uv / pnpm all print file paths in their diagnostics — regex
    them out and resolve against the project's known paths (exact, then
    unique suffix match). When nothing matches (e.g. a resolver error with
    no file in it), fall back to the recipe-required files so the repair
    prompt always has *some* concrete code to look at.
    """
    ordered: list[str] = []

    def _push(rel: str) -> None:
        if rel not in ordered and (dest / rel).is_file():
            ordered.append(rel)

    for match in _OUTPUT_PATH_RE.finditer(output):
        if len(ordered) >= _IMPLICATED_FILES_MAX:
            break
        raw = match.group("path").replace("\\", "/").lstrip("./")
        if raw in known_paths:
            _push(raw)
            continue
        suffix_hits = [k for k in known_paths if k.endswith("/" + raw)]
        if len(suffix_hits) == 1:
            _push(suffix_hits[0])
    if not ordered:
        for rel in required_files[:_IMPLICATED_FILES_MAX]:
            _push(rel)
    files: dict[str, str] = {}
    for rel in ordered[:_IMPLICATED_FILES_MAX]:
        try:
            text = (dest / rel).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        files[rel] = text[:_IMPLICATED_FILE_CHAR_CAP]
    return files


def _recipe_body_for_repair(recipe: Recipe) -> str:
    try:
        body = recipe.path.read_text(encoding="utf-8")
    except OSError:
        return f"(recipe body unavailable; slug: {recipe.slug})"
    return body[:_REPAIR_RECIPE_CHAR_CAP]


def _merge_patch(result: GenerationResult, patch: list[GeneratedFile]) -> GenerationResult:
    """Fold patched files into ``result.files`` (replace or append), keeping order."""
    merged: dict[str, GeneratedFile] = {f.path: f for f in result.files}
    for entry in patch:
        merged[entry.path] = entry
    return result.model_copy(update={"files": list(merged.values())})


def _repair_validation_loop(
    inputs: PipelineInputs,
    result: GenerationResult,
    first_results: list[Any],
    progress_cb: Callable[[ProgressEvent], None],
) -> tuple[GenerationResult, list[Any], int]:
    """Feed validation failures back to the model for targeted fixes.

    Each round: take the first failing tier, hand the model its command +
    output + the implicated files' current bodies, parse the changed-files
    patch, write it atomically, re-format, re-validate. Stops on pass, on
    :data:`MAX_REPAIR_ROUNDS`, or on any repair-side error (the original
    failure stays authoritative — repair must never make things worse).
    """
    language = str(inputs.hints.get("language", inputs.language))
    recipe_body = _recipe_body_for_repair(inputs.recipe)
    required = inputs.recipe.required_files
    results = first_results
    rounds = 0
    while any(not r.passed for r in results) and rounds < MAX_REPAIR_ROUNDS:
        rounds += 1
        failing = next(r for r in results if not r.passed)
        op_name = f"repair {rounds}/{MAX_REPAIR_ROUNDS}"
        progress_cb(
            ProgressEvent(
                kind="operation_started",
                payload={"name": op_name, "hint": f"{failing.tier.value} tier failed"},
            )
        )
        known_paths = {f.path for f in result.files}
        try:
            raw = repair_validation(
                config=inputs.cfg,
                recipe_body=recipe_body,
                language_hints=inputs.hints,
                project_file_list=sorted(known_paths),
                failing_command=tier_command(failing.tier, language, result.smoke_check),
                # Redact before anything leaves the machine: subprocess output
                # can echo env values (defense-in-depth on top of the env-name
                # -only discipline elsewhere).
                validation_output=redact(failing.output[-_REPAIR_OUTPUT_CHAR_CAP:]),
                implicated_files=_implicated_files(
                    failing.output, inputs.dest, known_paths, required
                ),
                language=language,
                progress=progress_cb,
            )
            patch = parse_file_patch(
                raw,
                inputs.dest,
                allowed_paths=known_paths | set(required),
            )
            patch_result = result.model_copy(update={"files": patch})
            write_project(
                patch_result,
                inputs.dest,
                WriteMode.overwrite,
                on_event=progress_cb,
            )
        except ContractParseError as exc:
            progress_cb(
                ProgressEvent(
                    kind="operation_done",
                    payload={"name": op_name, "status": "fail", "summary": exc.reason},
                )
            )
            break
        except Exception as exc:  # noqa: BLE001 — repair must never crash the pipeline
            progress_cb(
                ProgressEvent(
                    kind="operation_done",
                    payload={"name": op_name, "status": "fail", "summary": str(exc)},
                )
            )
            break
        result = _merge_patch(result, patch)
        if inputs.format_output:
            run_post_gen_formatter(inputs.dest, inputs.language, on_event=progress_cb)
        results = run_validate(
            inputs.dest,
            inputs.hints,
            result.smoke_check,
            _VALIDATION_TIERS,
            on_event=progress_cb,
        )
        passed = all(r.passed for r in results)
        progress_cb(
            ProgressEvent(
                kind="operation_done",
                payload={
                    "name": op_name,
                    "status": "ok" if passed else "warn",
                    "summary": (
                        f"{len(patch)} file(s) patched; validation "
                        + ("passed" if passed else "still failing")
                    ),
                },
            )
        )
    return result, results, rounds


def repair_smoke_failure(
    *,
    project_dir: Path,
    manifest: Manifest,
    recipe: Recipe,
    cfg: Config,
    failure_output: str,
    on_event: Callable[[ProgressEvent], None] | None = None,
) -> int:
    """One bounded repair round for a ``smoke_test`` failure during ``up``.

    The post-write repair loop can't cover the smoke tier — most smoke checks
    need the provisioned services running. This is its post-``up`` sibling:
    same prompt, same patch contract, hard-capped at ONE round (so the
    worst-case LLM calls per golden-path run stay at generate + 2 validation
    repairs + 1 smoke repair). Returns the number of files patched; raises on
    any repair-side failure (callers surface it and keep the original smoke
    failure authoritative).
    """
    from agent_scaffold.language_hints import load_language_hints

    hints = load_language_hints(manifest.language)
    known_paths = {f.path for f in manifest.files}
    implicated = _implicated_files(failure_output, project_dir, known_paths, recipe.required_files)
    raw = repair_validation(
        config=cfg,
        recipe_body=_recipe_body_for_repair(recipe),
        language_hints=hints,
        project_file_list=sorted(known_paths),
        failing_command="smoke test (post-provisioning)",
        validation_output=redact(failure_output[-_REPAIR_OUTPUT_CHAR_CAP:]),
        implicated_files=implicated,
        language=manifest.language,
        progress=on_event,
    )
    patch = parse_file_patch(
        raw,
        project_dir,
        allowed_paths=known_paths | set(recipe.required_files),
    )
    patch_result = GenerationResult(
        project_name=manifest.answers.get("project_name") or project_dir.name,
        language=manifest.language,
        files=patch,
        smoke_check="-",
    )
    write_project(patch_result, project_dir, WriteMode.overwrite, on_event=on_event)
    run_post_gen_formatter(project_dir, manifest.language, on_event=on_event)
    return len(patch)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def run_generation(
    inputs: PipelineInputs,
    *,
    display: GenerationDisplay,
) -> RunReport:
    """Execute the post-plan generation pipeline.

    All recoverable failures are raised as :class:`PipelineError` so callers
    (CLI and REPL) can render a useful error panel without seeing tracebacks.
    The token+cost summary is printed in ``finally``, matching the inline
    behavior cmd_new had before this extraction.
    """
    cfg = inputs.cfg
    recipe = inputs.recipe

    # Recipes are authoritative about their source layout via ``required_files``.
    # Rewrite the language-default entry point / layout to match before anything
    # reads ``inputs.hints`` (request, cache key, contract + validation), so the
    # model never sees a ``src/`` hint fighting an ``app/`` required file.
    inputs = replace(inputs, hints=reconcile_entry_point(inputs.hints, recipe.required_files))

    # Sorted set fields so both the prompt and the cache key are deterministic.
    sorted_removed_steps = sorted(inputs.removed_steps)
    sorted_removed_roles = sorted(inputs.removed_roles)

    req = GenerationRequest(
        project_name=inputs.project_name,
        target_language=inputs.language,
        framework=inputs.framework,
        assembled_context=inputs.ctx,
        language_hints=inputs.hints,
        extra_required=recipe.required_files,
        strict=inputs.strict,
        extra_dependencies=inputs.extra_dependencies,
        extra_steps=inputs.extra_steps,
        removed_steps=sorted_removed_steps,
        removed_roles=sorted_removed_roles,
        refinement_notes=inputs.refinement_notes,
        capabilities_brief=_capabilities_brief(inputs.resolved_stack),
        agent_role=inputs.agent_role,
    )

    cache_inputs = {
        "project_name": inputs.project_name,
        "language": inputs.language,
        "framework": inputs.framework,
        "context": inputs.ctx.body,
        "model": cfg.model,
        "hints": inputs.hints,
        "prompts": prompts_signature(),
        "required_files": recipe.required_files,
        "strict": inputs.strict,
        "thinking_budget": cfg.thinking_budget,
        # Refinements bust the cache when present — otherwise a stale
        # pre-refinement cached response would mask the user's edits.
        "extra_dependencies": inputs.extra_dependencies,
        "extra_steps": inputs.extra_steps,
        "removed_steps": sorted_removed_steps,
        "removed_roles": sorted_removed_roles,
        "refinement_notes": inputs.refinement_notes,
        "agent_role": inputs.agent_role,
    }
    cached_raw = None if inputs.no_cache else get_cached(cfg.cache_dir, cache_inputs)

    wall_start = time.time()
    reset_run_usage()
    result: GenerationResult | None = None
    report: Any = None
    validation_results: list[Any] = []
    manifest_written = False
    repair_rounds = 0
    try:
        with display as progress:
            # --- Generate (or load from cache) -------------------------------
            if cached_raw is not None:
                progress.on_event(
                    ProgressEvent(
                        kind="operation_started",
                        payload={"name": "cached lookup", "hint": "skipping LLM call"},
                    )
                )
                result = _attempt_parse(
                    cached_raw,
                    inputs.dest,
                    inputs.hints,
                    inputs.project_name,
                    recipe.required_files,
                    inputs.resolved_stack,
                    inputs.strict,
                    agent_title=inputs.agent_title,
                    # A cached response was valid when stored; don't re-block on
                    # the /chat backstop (use --no-cache to regenerate fresh).
                    check_chat=False,
                )
                progress.on_event(
                    ProgressEvent(
                        kind="operation_done",
                        payload={
                            "name": "cached lookup",
                            "status": "ok",
                            "summary": f"{len(result.files)} files",
                        },
                    )
                )
            else:
                progress.on_event(
                    ProgressEvent(
                        kind="operation_started",
                        payload={"name": "generate", "hint": f"model={cfg.model}"},
                    )
                )
                result, raw_response = _generate_with_repair(
                    req,
                    cfg,
                    inputs.dest,
                    inputs.hints,
                    inputs.project_name,
                    recipe.required_files,
                    progress=progress.on_event,
                    resolved_stack=inputs.resolved_stack,
                    agent_title=inputs.agent_title,
                )
                progress.on_event(
                    ProgressEvent(
                        kind="operation_done",
                        payload={
                            "name": "generate",
                            "status": "ok",
                            "summary": f"{len(result.files)} files",
                        },
                    )
                )
                save_cache(cfg.cache_dir, cache_inputs, raw_response)

            # --- Write to disk -----------------------------------------------
            progress.on_event(
                ProgressEvent(
                    kind="operation_started",
                    payload={"name": "write", "hint": f"{len(result.files)} files"},
                )
            )
            try:
                report = write_project(
                    result,
                    inputs.dest,
                    inputs.write_mode,
                    on_event=progress.on_event,
                    pre_confirm=inputs.pre_write_confirm,
                )
            except DestinationExistsError as exc:
                progress.on_event(
                    ProgressEvent(
                        kind="operation_done",
                        payload={"name": "write", "status": "fail", "summary": str(exc)},
                    )
                )
                raise PipelineError(str(exc), phase="write") from exc
            except DiffPreviewCancelled as exc:
                progress.on_event(
                    ProgressEvent(
                        kind="operation_done",
                        payload={
                            "name": "write",
                            "status": "fail",
                            "summary": "user declined diff preview",
                        },
                    )
                )
                raise PipelineError(str(exc), phase="write") from exc
            except ContractParseError as exc:
                progress.on_event(
                    ProgressEvent(
                        kind="operation_done",
                        payload={"name": "write", "status": "fail", "summary": exc.reason},
                    )
                )
                raise PipelineError(exc.reason, phase="write") from exc
            progress.on_event(
                ProgressEvent(
                    kind="operation_done",
                    payload={
                        "name": "write",
                        "status": "ok",
                        "summary": (
                            f"{len(report.written)} new, "
                            f"{len(report.overwritten)} overwritten, "
                            f"{len(report.skipped)} skipped"
                        ),
                    },
                )
            )

            # --- Copy capability template files ------------------------------
            # For every capability with ``emit_files``, copy the source tree
            # verbatim into the generated project. Runs after write_project
            # so model-emitted files always win on collision.
            if inputs.resolved_stack is not None and any(
                cap.emit_files for cap in inputs.resolved_stack.capabilities
            ):
                progress.on_event(
                    ProgressEvent(
                        kind="operation_started",
                        payload={
                            "name": "templates",
                            "hint": (
                                f"{sum(len(cap.emit_files) for cap in inputs.resolved_stack.capabilities)} "
                                "emit_files entries"
                            ),
                        },
                    )
                )
                emit_result = copy_capability_templates(
                    stack=inputs.resolved_stack,
                    capabilities_root=inputs.deployments / "docs" / "capabilities",
                    project_dir=inputs.dest,
                    write_mode=inputs.write_mode,
                    model_paths={f.path for f in result.files},
                )
                summary_parts: list[str] = []
                if emit_result.written:
                    summary_parts.append(f"{len(emit_result.written)} written")
                if emit_result.overwritten:
                    summary_parts.append(f"{len(emit_result.overwritten)} overwritten")
                if emit_result.skipped_existing:
                    summary_parts.append(f"{len(emit_result.skipped_existing)} skipped")
                if emit_result.skipped_unsafe:
                    summary_parts.append(f"{len(emit_result.skipped_unsafe)} unsafe")
                if emit_result.missing_source:
                    summary_parts.append(f"{len(emit_result.missing_source)} missing")
                progress.on_event(
                    ProgressEvent(
                        kind="operation_done",
                        payload={
                            "name": "templates",
                            "status": "warn" if emit_result.skipped_unsafe else "ok",
                            "summary": ", ".join(summary_parts) or "no files",
                        },
                    )
                )

            # --- Enforce the secret-safety .gitignore block -----------------
            try:
                appended = ensure_gitignore_defaults(inputs.dest)
            except OSError:
                appended = []
            if appended:
                progress.on_event(
                    ProgressEvent(
                        kind="operation_done",
                        payload={
                            "name": "gitignore",
                            "status": "ok",
                            "summary": f"+{len(appended)} entries appended",
                        },
                    )
                )

            # --- Verify required files actually landed on disk --------------
            if recipe.required_files:
                progress.on_event(
                    ProgressEvent(
                        kind="operation_started",
                        payload={
                            "name": "verify",
                            "hint": f"{len(recipe.required_files)} required files",
                        },
                    )
                )
                on_disk_missing = verify_required_files_on_disk(inputs.dest, recipe.required_files)
                if on_disk_missing:
                    summary = f"missing: {', '.join(on_disk_missing)}"
                    progress.on_event(
                        ProgressEvent(
                            kind="operation_done",
                            payload={"name": "verify", "status": "fail", "summary": summary},
                        )
                    )
                    raise PipelineError(
                        "Required files missing after write:\n  " + "\n  ".join(on_disk_missing),
                        phase="verify",
                        hint=(
                            "Likely causes:\n"
                            "  - --write-mode skip with a non-empty destination "
                            "containing colliding paths\n"
                            "  - write permissions / disk full / path-traversal sanitisation\n"
                            "Try: --write-mode overwrite (BE CAREFUL — irreversible)"
                        ),
                    )
                progress.on_event(
                    ProgressEvent(
                        kind="operation_done",
                        payload={
                            "name": "verify",
                            "status": "ok",
                            "summary": f"{len(recipe.required_files)} present",
                        },
                    )
                )

            # --- Format ------------------------------------------------------
            if inputs.format_output:
                progress.on_event(
                    ProgressEvent(
                        kind="operation_started",
                        payload={"name": "format", "hint": _format_hint(inputs.language)},
                    )
                )
                run_post_gen_formatter(inputs.dest, inputs.language, on_event=progress.on_event)
                progress.on_event(
                    ProgressEvent(
                        kind="operation_done",
                        payload={"name": "format", "status": "ok"},
                    )
                )

            # --- Validation (static + build) + bounded repair loop ----------
            if not inputs.skip_validation:
                progress.on_event(
                    ProgressEvent(
                        kind="operation_started",
                        payload={"name": "validate", "hint": "static + build tiers"},
                    )
                )
                validation_results = run_validate(
                    inputs.dest,
                    inputs.hints,
                    result.smoke_check,
                    _VALIDATION_TIERS,
                    on_event=progress.on_event,
                )
                status = "ok" if all(r.passed for r in validation_results) else "fail"
                summary = "; ".join(
                    f"{r.tier.value}={'ok' if r.passed else 'fail'}" for r in validation_results
                )
                progress.on_event(
                    ProgressEvent(
                        kind="operation_done",
                        payload={"name": "validate", "status": status, "summary": summary},
                    )
                )
                if status == "fail":
                    result, validation_results, repair_rounds = _repair_validation_loop(
                        inputs,
                        result,
                        validation_results,
                        progress.on_event,
                    )

            # --- Write .scaffold/manifest.json + template snapshot ----------
            # Runs even when validation is still failing: the project is on
            # disk and the manifest is what makes `validate` / `regenerate` /
            # `update` usable for manual recovery.
            template_sha: str | None = None
            if result is not None and report is not None:
                manifest_written, template_sha = _write_manifest_and_snapshot(
                    inputs, result, progress
                )

            # --- Write .scaffold/run-summary.md ------------------------------
            # The durable record that travels with the project (the report
            # panel scrolls away). Best-effort: a write failure warns only.
            if result is not None and report is not None:
                try:
                    write_run_summary(
                        inputs.dest,
                        recipe=recipe,
                        language=inputs.language,
                        framework=inputs.framework,
                        model=cfg.model,
                        result=result,
                        template_sha=template_sha,
                        validation_results=validation_results,
                        repair_rounds=repair_rounds,
                        resolved_stack=inputs.resolved_stack,
                        run_log_dir=str(getattr(display, "run_log_dir", "") or ""),
                    )
                except OSError as exc:
                    progress.on_event(
                        ProgressEvent(
                            kind="operation_done",
                            payload={
                                "name": "run-summary",
                                "status": "warn",
                                "summary": f"could not write run-summary.md: {exc}",
                            },
                        )
                    )

            # --- Surface unrecovered validation failure ----------------------
            still_failing = [r for r in validation_results if not r.passed]
            if still_failing:
                worst = still_failing[0]
                excerpt = redact(worst.output[-1_500:]).strip()
                raise PipelineError(
                    (
                        f"validation failed ({worst.tier.value} tier) after "
                        f"{repair_rounds} repair round(s):\n{excerpt}"
                    ),
                    phase="validate",
                    hint=(
                        "The project is on disk — inspect it, fix manually, then re-run\n"
                        f"  agent-scaffold validate {inputs.dest} --tier build\n"
                        "or regenerate the offending file with\n"
                        f"  agent-scaffold regenerate {inputs.dest} <path>"
                    ),
                )
    finally:
        _emit_generation_report(
            inputs=inputs,
            cfg=cfg,
            report=report,
            wall_seconds=time.time() - wall_start,
            cached=cached_raw is not None,
            display=display,
            repair_rounds=repair_rounds,
        )

    return RunReport(
        result=result,
        report=report,
        validation_results=validation_results,
        wall_seconds=time.time() - wall_start,
        cached=cached_raw is not None,
        manifest_written=manifest_written,
    )


def _write_manifest_and_snapshot(
    inputs: PipelineInputs,
    result: GenerationResult,
    progress: Any,
) -> tuple[bool, str | None]:
    """Best-effort manifest + template-snapshot write.

    Returns ``(manifest_written, template_sha)`` — the sha feeds the
    run-summary file. Failures here are reported via the progress display
    but don't abort the run — the generated project is on disk and usable
    even without the manifest (just no clean ``update`` / ``regenerate``
    path).
    """
    progress.on_event(
        ProgressEvent(
            kind="operation_started",
            payload={"name": "manifest", "hint": ".scaffold/manifest.json"},
        )
    )
    try:
        template_sha: str | None = None
        snapshot_summary = ""
        try:
            template_sha = compute_template_sha(inputs.deployments)
            # Snapshot the freshly generated files, keyed by the template sha.
            # On the next ``update``, this is the merge base.
            snap = save_generation_snapshot(
                inputs.dest,
                template_sha,
                {f.path.replace("\\", "/"): f.content for f in result.files},
            )
            prune_snapshots(inputs.dest)
            snapshot_summary = f"snapshot {short_sha(template_sha)} ({snap.bytes // 1024} KB)"
        except OSError as snap_exc:
            snapshot_summary = f"snapshot skipped: {snap_exc}"
        manifest = Manifest(
            recipe=inputs.recipe.slug,
            language=inputs.language,
            framework=inputs.framework,
            topology=inputs.topology.value if inputs.topology else None,
            roles=[
                {
                    "name": r.name,
                    "description": r.description,
                    "model_hint": r.model_hint,
                    "tools": list(r.tools),
                }
                for r in inputs.roles
            ],
            model=inputs.cfg.model,
            generated_at=datetime.now(UTC).isoformat(),
            files=build_file_entries(inputs.dest, [f.path for f in result.files]),
            template_snapshot_sha=template_sha,
            answers={
                "recipe": inputs.recipe.slug,
                "language": inputs.language,
                "framework": inputs.framework,
                "project_name": inputs.raw_project_name,
            },
            capabilities=(inputs.resolved_stack.ids() if inputs.resolved_stack is not None else []),
            secrets_namespace=project_namespace(inputs.project_name, inputs.dest),
        )
        write_manifest(inputs.dest, manifest)
        if snapshot_summary:
            progress.on_event(
                ProgressEvent(
                    kind="operation_started",
                    payload={"name": "snapshot", "hint": snapshot_summary},
                )
            )
            progress.on_event(
                ProgressEvent(
                    kind="operation_done",
                    payload={
                        "name": "snapshot",
                        "status": "ok",
                        "summary": snapshot_summary,
                    },
                )
            )
        progress.on_event(
            ProgressEvent(
                kind="operation_done",
                payload={
                    "name": "manifest",
                    "status": "ok",
                    "summary": f"{len(manifest.files)} files indexed",
                },
            )
        )
        return True, template_sha
    except OSError as exc:
        progress.on_event(
            ProgressEvent(
                kind="operation_done",
                payload={
                    "name": "manifest",
                    "status": "warn",
                    "summary": f"could not write manifest: {exc}",
                },
            )
        )
        return False, None


__all__ = [
    "PipelineError",
    "PipelineInputs",
    "RunReport",
    "print_next_steps",
    "print_phase_summary",
    "print_usage_summary",
    "run_generation",
    "run_post_gen_formatter",
]
