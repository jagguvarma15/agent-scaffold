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

import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel

from agent_scaffold.cache import get_cached, save_cache
from agent_scaffold.capabilities import ResolvedStack
from agent_scaffold.capability_emit import copy_capability_templates
from agent_scaffold.config import Config
from agent_scaffold.context import AssembledContext
from agent_scaffold.contract import (
    ContractParseError,
    GenerationResult,
    check_frontend_collisions,
    merge_capability_fragments,
    parse,
    validate_paths,
    validate_required_files,
)
from agent_scaffold.discovery import Recipe
from agent_scaffold.generator import (
    GenerationRequest,
    generate,
    get_last_usage,
    prompts_signature,
    repair,
)
from agent_scaffold.manifest import (
    Manifest,
    build_file_entries,
    write_manifest,
)
from agent_scaffold.progress import (
    NullProgressDisplay,
    ProgressEvent,
    RichProgressDisplay,
)
from agent_scaffold.report import (
    GenerationReport,
    derive_layers,
    derive_observability,
    derive_tier,
    print_generation_report,
)
from agent_scaffold.template_snapshot import (
    compute_template_sha,
    prune_snapshots,
    save_generation_snapshot,
    short_sha,
)
from agent_scaffold.topology import Role, Topology
from agent_scaffold.validator import ValidationTier, verify_required_files_on_disk
from agent_scaffold.validator import validate as run_validate
from agent_scaffold.writer import (
    DestinationExistsError,
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

    # Resolved capability stack threaded from cmd_new / cmd_regenerate.
    # ``None`` when the deployments source has no ``docs/capabilities/``
    # tree or the recipe didn't declare any.
    resolved_stack: ResolvedStack | None = None


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
) -> GenerationResult:
    result = parse(raw)
    validate_paths(result, dest, canonical_module_name=project_name)
    validate_required_files(result, hints, extra_required)
    # Capability-aware passes: collision check (may raise in strict mode)
    # then deterministic compose merge. Both no-op when resolved_stack is
    # ``None`` or the stack has no relevant capabilities.
    check_frontend_collisions(result, resolved_stack, strict=strict)
    result = merge_capability_fragments(result, resolved_stack)
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
                raw, dest, hints, project_name, extra_required, resolved_stack, req.strict
            ),
            raw,
        )
    except ContractParseError as exc:
        failure_path = _save_failure(raw, cfg.failures_dir)
        console.print(
            f"[yellow]Warning:[/] contract parse failed: {exc.reason}.\n"
            f"Raw response saved to: {failure_path}\n"
            "Attempting repair..."
        )
        repaired = repair(raw, exc.reason, cfg, strict=req.strict, progress=progress)
        try:
            return (
                _attempt_parse(
                    repaired, dest, hints, project_name, extra_required, resolved_stack, req.strict
                ),
                repaired,
            )
        except ContractParseError as exc2:
            second_failure = _save_failure(repaired, cfg.failures_dir)
            raise PipelineError(
                f"repair also failed: {exc2.reason}",
                phase="generate",
                hint=(
                    f"Original raw response: {failure_path}\n"
                    f"Repaired raw response: {second_failure}"
                ),
            ) from exc2


def _emit_generation_report(
    *,
    inputs: PipelineInputs,
    cfg: Config,
    report: Any,
    wall_seconds: float,
    cached: bool,
    display: RichProgressDisplay | NullProgressDisplay,
) -> None:
    """Build + print the consolidated post-generation panel from the `finally` block.

    Selections come from ``inputs``; usage from the last Anthropic call;
    file counts from the writer's report; phase data from the display.
    Any of these can be missing if generation aborted early — the report
    silently elides sections with no data.
    """
    usage = get_last_usage()
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
            tier=derive_tier(inputs.recipe),
            layers=derive_layers(inputs.resolved_stack),
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
            phase_durations=dict(getattr(display, "phase_durations", {})),
            warnings=list(getattr(display, "warnings", [])),
            errors=list(getattr(display, "errors", [])),
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
    console.print(Panel("\n".join(lines), title="Next steps", expand=False))


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def run_generation(
    inputs: PipelineInputs,
    *,
    display: RichProgressDisplay | NullProgressDisplay,
) -> RunReport:
    """Execute the post-plan generation pipeline.

    All recoverable failures are raised as :class:`PipelineError` so callers
    (CLI and REPL) can render a useful error panel without seeing tracebacks.
    The token+cost summary is printed in ``finally``, matching the inline
    behavior cmd_new had before this extraction.
    """
    cfg = inputs.cfg
    recipe = inputs.recipe

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
    }
    cached_raw = None if inputs.no_cache else get_cached(cfg.cache_dir, cache_inputs)

    wall_start = time.time()
    result: GenerationResult | None = None
    report: Any = None
    validation_results: list[Any] = []
    manifest_written = False
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
                    result, inputs.dest, inputs.write_mode, on_event=progress.on_event
                )
            except DestinationExistsError as exc:
                progress.on_event(
                    ProgressEvent(
                        kind="operation_done",
                        payload={"name": "write", "status": "fail", "summary": str(exc)},
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

            # --- Static validation ------------------------------------------
            if not inputs.skip_validation:
                progress.on_event(
                    ProgressEvent(
                        kind="operation_started",
                        payload={"name": "validate", "hint": "static tier"},
                    )
                )
                validation_results = run_validate(
                    inputs.dest,
                    inputs.hints,
                    result.smoke_check,
                    [ValidationTier.static],
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

            # --- Write .scaffold/manifest.json + template snapshot ----------
            if result is not None and report is not None:
                manifest_written = _write_manifest_and_snapshot(inputs, result, progress)
    finally:
        _emit_generation_report(
            inputs=inputs,
            cfg=cfg,
            report=report,
            wall_seconds=time.time() - wall_start,
            cached=cached_raw is not None,
            display=display,
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
) -> bool:
    """Best-effort manifest + template-snapshot write.

    Returns ``True`` iff the manifest was written. Failures here are reported
    via the progress display but don't abort the run — the generated project
    is on disk and usable even without the manifest (just no clean
    ``update`` / ``regenerate`` path).
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
        return True
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
        return False


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
