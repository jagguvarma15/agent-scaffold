"""State-tracked idempotent step orchestrator.

Track B's keystone. ``up``, ``update``, and future provisioning verbs all
build on this framework. Each provisioning action is a :class:`Step` with
two methods:

- ``detect(ctx) -> DetectionResult`` — read-only; "what is the current state?"
- ``apply(ctx) -> StepResult`` — idempotent; "mutate to desired state".

A single JSON state file at ``<project>/.scaffold/state.json`` records
``{status, fingerprint, error?, ...}`` per step. From that one design, the
flag set ``--resume / --retry / --skip / --force / --only`` and the
``--plan`` mode fall out essentially for free.

**Anti-patterns to avoid** when authoring a Step:

- Reading live state inside ``apply()`` that you did not include in
  ``fingerprint()``. ``--resume`` will then treat semantically-different
  invocations as identical and skip them.
- Performing a side effect inside ``apply()`` that you cannot ``detect()``.
  If a step writes a file but ``detect()`` returns PENDING regardless, the
  next run will redo the work and may not be idempotent.
- Mutating ``state.json`` directly. Only the orchestrator owns it. Steps
  may stash per-step scratch under ``state.steps[id].metadata`` (future
  field) — never anywhere else.

See ``docs/design/orchestrator.md`` for the longer treatment.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from rich.panel import Panel
from rich.table import Table

from agent_scaffold._scaffold_dir import SCAFFOLD_DIR
from agent_scaffold.manifest import Manifest

log = logging.getLogger(__name__)

STATE_DIR = SCAFFOLD_DIR
STATE_FILENAME = "state.json"
STATE_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Status + state dataclasses
# ---------------------------------------------------------------------------


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"  # survives crashes; resumed as PENDING on next run
    DONE = "done"
    SKIPPED = "skipped"
    FAILED = "failed"
    PARTIAL = "partial"  # detect() may report partial; apply() turns it into DONE


@dataclass
class StepState:
    status: StepStatus = StepStatus.PENDING
    fingerprint: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    stderr_tail: str | None = None
    attempt: int = 0
    reason: str | None = None


@dataclass
class OrchestratorState:
    schema_version: int = STATE_SCHEMA_VERSION
    started_at: str = ""
    last_run_at: str = ""
    steps: dict[str, StepState] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Step inputs / outputs / events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepResult:
    status: StepStatus
    detail: str = ""
    error: str | None = None
    stderr_tail: str | None = None


@dataclass(frozen=True)
class DetectionResult:
    status: StepStatus
    reason: str = ""


@dataclass
class StepContext:
    """Everything a Step needs to introspect or mutate the project.

    The orchestrator constructs one per ``run()`` call and threads it
    through every step. Steps SHOULD treat ``state`` as read-only — the
    orchestrator persists it after each step.

    ``resolved_stack`` carries the capability set the recipe declared.
    Steps that act on capabilities (``bootstrap_vector_db`` etc.) read it
    via ``ctx.resolved_stack``; the field is ``None`` when the project
    doesn't use the catalog (older recipes / older scaffold) and those
    steps SKIP.
    """

    project_dir: Path
    manifest: Manifest
    state: OrchestratorState
    callback: Callable[[StepEvent], None] | None = None
    timeout: float = 600.0
    # Forward-declared as Any to keep this module free of the capabilities
    # import (which itself depends on discovery). Concrete type is
    # ``agent_scaffold.capabilities.ResolvedStack | None``; steps that need
    # it import the symbol locally.
    resolved_stack: Any = None
    # Fully-resolved environment for every subprocess a step spawns:
    # shell env > project secrets vault > .env.local. ``None`` keeps the
    # historical inherit-parent behavior (tests, older callers).
    runtime_env: dict[str, str] | None = None

    def emit(self, event: StepEvent) -> None:
        """Convenience: dispatch ``event`` to ``callback`` if one is set."""
        if self.callback is not None:
            self.callback(event)


def dependency_actually_ran(ctx: StepContext, step_id: str) -> bool:
    """True iff ``step_id`` reached DONE in this orchestrator run.

    A step can be present in ``ctx.state.steps`` with status SKIPPED, FAILED,
    or PARTIAL — none of those mean "the side effect actually happened."
    Downstream steps that depend on a service being started (Grafana, Qdrant,
    Postgres) must check this before polling that service's healthcheck, or
    they'll spin for the full timeout against a port that was never bound.
    """
    state = ctx.state.steps.get(step_id)
    if state is None:
        return False
    return state.status is StepStatus.DONE


# Event hierarchy. Concrete steps emit these via ``ctx.emit(...)`` so a
# top-level display (Rich Live panel, JSON sink, log adapter) can render
# progress uniformly. Cross-reference progress.py's ProgressEvent: this is a
# typed sibling — Q6+ wiring will adapt step events into the existing display.


@dataclass(frozen=True)
class StepEvent:
    step_id: str
    timestamp: float = 0.0


@dataclass(frozen=True)
class StepStarted(StepEvent):
    pass


@dataclass(frozen=True)
class StepProgress(StepEvent):
    message: str = ""
    percent: float | None = None


@dataclass(frozen=True)
class StepLog(StepEvent):
    line: str = ""
    stream: Literal["stdout", "stderr"] = "stdout"


@dataclass(frozen=True)
class StepFinished(StepEvent):
    result: StepResult | None = None


# ---------------------------------------------------------------------------
# Step protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Step(Protocol):
    """Contract for a single orchestrator step.

    All three methods MUST be pure with respect to anything outside
    ``ctx.project_dir``. Concretely:

    - ``detect()`` is read-only and side-effect-free. It runs every plan +
      every run (unless ``--resume`` short-circuits).
    - ``apply()`` is the only place side effects live. It must be idempotent
      and safe to re-run.
    - ``fingerprint()`` is a stable hash of the inputs that would change the
      semantically-correct outcome. Same fingerprint + status DONE means
      ``--resume`` will skip the step.
    """

    id: str
    description: str
    depends_on: tuple[str, ...]

    def detect(self, ctx: StepContext) -> DetectionResult: ...

    def apply(self, ctx: StepContext) -> StepResult: ...

    def fingerprint(self, ctx: StepContext) -> str: ...


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class OrchestratorError(Exception):
    """Base class for orchestrator-framework errors."""


class CycleError(OrchestratorError):
    def __init__(self, cycle: list[str]) -> None:
        super().__init__(f"Cycle detected in step graph: {' -> '.join(cycle)}")
        self.cycle = cycle


class MissingDependencyError(OrchestratorError):
    def __init__(self, step_id: str, missing: str) -> None:
        super().__init__(f"Step {step_id!r} depends on unknown step {missing!r}")
        self.step_id = step_id
        self.missing = missing


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------


def compute_fingerprint(inputs: dict[str, Any]) -> str:
    """Stable SHA-256 of ``inputs`` after canonical JSON serialization.

    Key ordering is normalized so ``{"a": 1, "b": 2}`` and ``{"b": 2, "a": 1}``
    produce the same fingerprint. Non-JSON-serializable values fall through
    to ``str()`` (so ``Path`` works), but step authors should prefer
    explicit string conversion for clarity.
    """
    try:
        canonical = json.dumps(inputs, sort_keys=True, separators=(",", ":"), default=str)
    except TypeError as exc:
        raise ValueError(f"compute_fingerprint: input not JSON-serializable: {exc}") from exc
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


# ---------------------------------------------------------------------------
# State file: atomic read + write with schema migrations
# ---------------------------------------------------------------------------


def state_path(project_dir: Path) -> Path:
    return project_dir / STATE_DIR / STATE_FILENAME


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


# Schema migrations: map ``from_version -> migrate(dict) -> dict``. Each
# migration upgrades a single version; ``read_state`` applies them in order.
MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {}


def _apply_migrations(data: dict[str, Any]) -> dict[str, Any]:
    current = int(data.get("schema_version", 1))
    while current < STATE_SCHEMA_VERSION:
        migrate = MIGRATIONS.get(current)
        if migrate is None:
            raise OrchestratorError(
                f"No migration registered from schema_version {current} -> {STATE_SCHEMA_VERSION}"
            )
        data = migrate(data)
        current = int(data.get("schema_version", current + 1))
    return data


def read_state(project_dir: Path) -> OrchestratorState:
    """Load state from disk. A missing file returns a fresh empty state."""
    target = state_path(project_dir)
    if not target.is_file():
        return OrchestratorState()
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OrchestratorError(f"state.json is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise OrchestratorError("state.json must contain a JSON object")
    raw = _apply_migrations(raw)
    steps_raw = raw.get("steps") or {}
    if not isinstance(steps_raw, dict):
        raise OrchestratorError("state.json: 'steps' must be a JSON object")
    steps: dict[str, StepState] = {}
    for step_id, payload in steps_raw.items():
        if not isinstance(payload, dict):
            log.warning("state.json: dropping malformed step %r (not an object)", step_id)
            continue
        try:
            status = StepStatus(payload.get("status", "pending"))
        except ValueError:
            log.warning(
                "state.json: step %r has unknown status %r; resetting to PENDING",
                step_id,
                payload.get("status"),
            )
            status = StepStatus.PENDING
        steps[step_id] = StepState(
            status=status,
            fingerprint=payload.get("fingerprint"),
            started_at=payload.get("started_at"),
            completed_at=payload.get("completed_at"),
            error=payload.get("error"),
            stderr_tail=payload.get("stderr_tail"),
            attempt=int(payload.get("attempt", 0)),
            reason=payload.get("reason"),
        )
    return OrchestratorState(
        schema_version=int(raw.get("schema_version", STATE_SCHEMA_VERSION)),
        started_at=str(raw.get("started_at", "")),
        last_run_at=str(raw.get("last_run_at", "")),
        steps=steps,
    )


def write_state(project_dir: Path, state: OrchestratorState) -> Path:
    """Persist ``state`` atomically: tmp-write → fsync → ``os.replace()``.

    Every string value in the payload runs through :func:`_redact.redact_obj`
    first. The fields most prone to leakage (``error`` and ``stderr_tail``)
    are populated from subprocess output that may include credentials echoed
    by a chatty tool; the redactor scrubs known shapes before persistence.
    """
    from agent_scaffold._redact import redact_obj

    target = state_path(project_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    raw_payload: dict[str, Any] = {
        "schema_version": state.schema_version,
        "started_at": state.started_at,
        "last_run_at": state.last_run_at,
        "steps": {
            step_id: {k: v for k, v in asdict(step).items() if v is not None or k == "attempt"}
            for step_id, step in state.steps.items()
        },
    }
    # ``StepStatus`` is a str enum; asdict gives the enum value already.
    # Normalize for safety:
    for step in raw_payload["steps"].values():
        if isinstance(step.get("status"), StepStatus):
            step["status"] = step["status"].value
    payload = redact_obj(raw_payload)
    body = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        prefix=STATE_FILENAME + ".",
        suffix=".tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(body)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
    try:
        os.chmod(target, 0o644)
    except OSError:
        pass  # best-effort; rare filesystems don't honor chmod
    return target


# ---------------------------------------------------------------------------
# Topological sort
# ---------------------------------------------------------------------------


def _topo_sort(steps: Sequence[Step]) -> list[str]:
    """Kahn's algorithm. Returns step IDs in execution order.

    Raises :class:`MissingDependencyError` for unknown deps and
    :class:`CycleError` for cycles. Tie-breaking is by step-declaration
    order — the test suite asserts stable diamond layouts.
    """
    by_id = {s.id: s for s in steps}
    indegree: dict[str, int] = {s.id: 0 for s in steps}
    reverse: dict[str, list[str]] = {s.id: [] for s in steps}
    for s in steps:
        for dep in s.depends_on:
            if dep not in by_id:
                raise MissingDependencyError(s.id, dep)
            indegree[s.id] += 1
            reverse[dep].append(s.id)

    # Preserve declaration order in the ready set.
    declaration_order = {s.id: i for i, s in enumerate(steps)}

    def _order_key(sid: str) -> int:
        return declaration_order[sid]

    ready = sorted([sid for sid, n in indegree.items() if n == 0], key=_order_key)
    out: list[str] = []
    while ready:
        sid = ready.pop(0)
        out.append(sid)
        for dependent in reverse[sid]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                # Insert in declaration order.
                ready.append(dependent)
                ready.sort(key=_order_key)
    if len(out) != len(steps):
        unresolved = [sid for sid, n in indegree.items() if n > 0]
        raise CycleError(unresolved)
    return out


# ---------------------------------------------------------------------------
# Plan rows + rendering
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanRow:
    step_id: str
    description: str
    detected: StepStatus
    action: str  # "run", "skip (done)", "skip (--skip)", "force-run", "wait (deps failed)", ...
    reason: str = ""


def render_plan_table(rows: Sequence[PlanRow]) -> Panel:
    """Render a Rich ``Panel`` summarising what the orchestrator will do."""
    table = Table(show_header=True, header_style="bold", expand=False, pad_edge=False)
    table.add_column("Step", style="cyan", no_wrap=True)
    table.add_column("State")
    table.add_column("Action")
    icons = {
        StepStatus.DONE: "[green]done[/]",
        StepStatus.PENDING: "[dim]pending[/]",
        StepStatus.PARTIAL: "[yellow]partial[/]",
        StepStatus.RUNNING: "[yellow]running[/]",
        StepStatus.FAILED: "[red]failed[/]",
        StepStatus.SKIPPED: "[dim cyan]skipped[/]",
    }
    for row in rows:
        action_text = row.action
        if row.reason:
            action_text = f"{row.action}  [dim]({row.reason})[/]"
        table.add_row(row.step_id, icons.get(row.detected, str(row.detected.value)), action_text)
    return Panel(table, title="Provisioning plan", expand=False)


# ---------------------------------------------------------------------------
# RunResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunResult:
    statuses: dict[str, StepStatus]

    @property
    def exit_code(self) -> int:
        return (
            0
            if all(s in {StepStatus.DONE, StepStatus.SKIPPED} for s in self.statuses.values())
            else 1
        )

    @property
    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {s.value: 0 for s in StepStatus}
        for s in self.statuses.values():
            counts[s.value] += 1
        return counts


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@dataclass
class _Decision:
    """What the orchestrator decided to do with a step, before topo sort."""

    run: bool
    reason: str
    initial_status: StepStatus | None = None  # explicit SKIPPED, etc.
    clear_state: bool = False


class Orchestrator:
    """Owns the run loop, the state file, and the dependency graph."""

    def __init__(
        self,
        steps: Sequence[Step],
        project_dir: Path,
        manifest: Manifest,
        callback: Callable[[StepEvent], None] | None = None,
        resolved_stack: Any = None,
        runtime_env: dict[str, str] | None = None,
    ) -> None:
        ids = [s.id for s in steps]
        duplicates = {sid for sid in ids if ids.count(sid) > 1}
        if duplicates:
            raise OrchestratorError(f"duplicate step ids: {sorted(duplicates)}")
        self._steps: dict[str, Step] = {s.id: s for s in steps}
        self._order: list[str] = _topo_sort(steps)
        self.project_dir = project_dir
        self.manifest = manifest
        self.callback = callback
        self.resolved_stack = resolved_stack
        self.runtime_env = runtime_env

    # --- planning -------------------------------------------------------

    def plan(self) -> list[PlanRow]:
        """Detect every step and return a planned action row per step.

        Side-effect-free w.r.t. the project (no ``apply()`` calls) but does
        invoke ``detect()`` on every step.
        """
        state = read_state(self.project_dir)
        ctx = StepContext(
            project_dir=self.project_dir,
            manifest=self.manifest,
            state=state,
            callback=self.callback,
            resolved_stack=self.resolved_stack,
            runtime_env=self.runtime_env,
        )
        rows: list[PlanRow] = []
        for step_id in self._order:
            step = self._steps[step_id]
            stored = state.steps.get(step_id, StepState())
            detection = step.detect(ctx)
            if stored.status == StepStatus.DONE and detection.status == StepStatus.DONE:
                action, reason = "skip", "already done"
            elif detection.status == StepStatus.DONE:
                action, reason = "skip", detection.reason or "already provisioned"
            elif detection.status == StepStatus.PARTIAL:
                action, reason = "run (resume)", detection.reason or "partial state"
            elif stored.status == StepStatus.FAILED:
                action, reason = "retry", stored.error or ""
            else:
                action, reason = "run", detection.reason
            rows.append(
                PlanRow(
                    step_id=step_id,
                    description=step.description,
                    detected=detection.status,
                    action=action,
                    reason=reason,
                )
            )
        return rows

    # --- execution ------------------------------------------------------

    def run(
        self,
        only: Sequence[str] = (),
        skip: Sequence[str] = (),
        force: Sequence[str] = (),
        retry: Sequence[str] = (),
        resume: bool = False,
    ) -> RunResult:
        state = read_state(self.project_dir)
        if not state.started_at:
            state.started_at = _utcnow_iso()
        state.last_run_at = _utcnow_iso()

        self._validate_flag_targets(only, skip, force, retry)
        active_ids = self._select_active(only)
        decisions = self._decide(
            active_ids,
            state=state,
            skip=set(skip),
            force=set(force),
            retry=set(retry),
            resume=resume,
        )

        ctx = StepContext(
            project_dir=self.project_dir,
            manifest=self.manifest,
            state=state,
            callback=self.callback,
            resolved_stack=self.resolved_stack,
            runtime_env=self.runtime_env,
        )
        statuses: dict[str, StepStatus] = {}
        halted = False  # set only when an *essential* step fails (hard stop)
        failed_or_blocked: set[str] = set()  # failed steps + their blocked dependents
        for step_id in self._order:
            if step_id not in decisions:
                continue
            decision = decisions[step_id]
            step = self._steps[step_id]
            if not decision.run:
                final_status = decision.initial_status or StepStatus.SKIPPED
                state.steps[step_id] = StepState(
                    status=final_status,
                    reason=decision.reason,
                )
                statuses[step_id] = final_status
                write_state(self.project_dir, state)
                continue
            if halted:
                # An essential step failed earlier; abandon the rest of the run.
                statuses[step_id] = state.steps.get(step_id, StepState()).status
                continue
            # Dependency-aware skip: a step whose prerequisite failed (or was
            # itself blocked) can't run — but steps that don't depend on the
            # failure still execute, so e.g. launch_backend survives a docker
            # or eval failure.
            blocking = [dep for dep in step.depends_on if dep in failed_or_blocked]
            if blocking:
                reason = f"blocked: {', '.join(blocking)} failed"
                state.steps[step_id] = StepState(status=StepStatus.SKIPPED, reason=reason)
                statuses[step_id] = StepStatus.SKIPPED
                failed_or_blocked.add(step_id)
                write_state(self.project_dir, state)
                ctx.emit(
                    StepFinished(
                        step_id=step_id,
                        result=StepResult(status=StepStatus.SKIPPED, detail=reason),
                    )
                )
                continue
            if decision.clear_state:
                state.steps.pop(step_id, None)
            attempt = state.steps.get(step_id, StepState()).attempt + 1
            state.steps[step_id] = StepState(
                status=StepStatus.RUNNING,
                started_at=_utcnow_iso(),
                attempt=attempt,
                reason=decision.reason,
            )
            write_state(self.project_dir, state)
            ctx.emit(StepStarted(step_id=step_id))
            try:
                result = step.apply(ctx)
                fp: str | None
                try:
                    fp = step.fingerprint(ctx)
                except Exception as exc:  # noqa: BLE001 — fingerprint failure shouldn't kill the run
                    log.warning("step %s: fingerprint failed: %s", step_id, exc)
                    fp = None
                state.steps[step_id] = StepState(
                    status=result.status,
                    fingerprint=fp,
                    started_at=state.steps[step_id].started_at,
                    completed_at=_utcnow_iso(),
                    error=result.error,
                    stderr_tail=result.stderr_tail,
                    attempt=attempt,
                    reason=result.detail or decision.reason,
                )
                statuses[step_id] = result.status
                if result.status == StepStatus.FAILED:
                    failed_or_blocked.add(step_id)
                    # Essential step (install_deps) → stop everything. Optional
                    # step → its dependents skip, but independent steps run on.
                    if not getattr(step, "optional", True):
                        halted = True
            except Exception as exc:  # noqa: BLE001
                log.exception("step %s raised an exception", step_id)
                state.steps[step_id] = StepState(
                    status=StepStatus.FAILED,
                    fingerprint=state.steps[step_id].fingerprint,
                    started_at=state.steps[step_id].started_at,
                    completed_at=_utcnow_iso(),
                    error=f"{type(exc).__name__}: {exc}",
                    attempt=attempt,
                    reason=decision.reason,
                )
                statuses[step_id] = StepStatus.FAILED
                failed_or_blocked.add(step_id)
                if not getattr(step, "optional", True):
                    halted = True
                result = StepResult(status=StepStatus.FAILED, error=str(exc))
            write_state(self.project_dir, state)
            ctx.emit(StepFinished(step_id=step_id, result=result))
        return RunResult(statuses=statuses)

    # --- helpers --------------------------------------------------------

    def _validate_flag_targets(self, *flag_lists: Sequence[str]) -> None:
        unknown: list[str] = []
        for ids in flag_lists:
            for sid in ids:
                if sid not in self._steps:
                    unknown.append(sid)
        if unknown:
            raise OrchestratorError(
                f"unknown step id(s) in flags: {sorted(set(unknown))}; "
                f"known: {sorted(self._steps)}"
            )

    def _select_active(self, only: Sequence[str]) -> set[str]:
        if not only:
            return set(self._steps)
        active: set[str] = set()
        for sid in only:
            self._collect_with_deps(sid, active)
        return active

    def _collect_with_deps(self, sid: str, into: set[str]) -> None:
        if sid in into:
            return
        into.add(sid)
        for dep in self._steps[sid].depends_on:
            self._collect_with_deps(dep, into)

    def _decide(
        self,
        active_ids: set[str],
        *,
        state: OrchestratorState,
        skip: set[str],
        force: set[str],
        retry: set[str],
        resume: bool,
    ) -> dict[str, _Decision]:
        ctx = StepContext(
            project_dir=self.project_dir,
            manifest=self.manifest,
            state=state,
            callback=self.callback,
            resolved_stack=self.resolved_stack,
            runtime_env=self.runtime_env,
        )
        decisions: dict[str, _Decision] = {}
        for step_id in self._order:
            if step_id not in active_ids:
                continue
            step = self._steps[step_id]
            stored = state.steps.get(step_id, StepState())
            if step_id in skip:
                decisions[step_id] = _Decision(
                    run=False, reason="--skip", initial_status=StepStatus.SKIPPED
                )
                continue
            if step_id in force:
                decisions[step_id] = _Decision(run=True, reason="--force", clear_state=True)
                continue
            if step_id in retry:
                if stored.status == StepStatus.FAILED:
                    decisions[step_id] = _Decision(
                        run=True, reason="--retry (was FAILED)", clear_state=True
                    )
                else:
                    # --retry on a non-failed step is a no-op (don't re-run a DONE step).
                    decisions[step_id] = _Decision(
                        run=False,
                        reason=f"--retry skipped (status was {stored.status.value})",
                        initial_status=stored.status,
                    )
                continue
            if stored.status == StepStatus.RUNNING:
                # Mid-run crash recovery — treat as fresh.
                decisions[step_id] = _Decision(
                    run=True, reason="recovering from previous crash", clear_state=False
                )
                continue
            if stored.status == StepStatus.DONE:
                if resume:
                    decisions[step_id] = _Decision(
                        run=False, reason="--resume; DONE", initial_status=StepStatus.DONE
                    )
                    continue
                # Re-detect; if still DONE, skip; otherwise re-run (drift case).
                detection = step.detect(ctx)
                if detection.status == StepStatus.DONE:
                    decisions[step_id] = _Decision(
                        run=False, reason="detected DONE", initial_status=StepStatus.DONE
                    )
                else:
                    decisions[step_id] = _Decision(
                        run=True, reason=f"drift: detected {detection.status.value}"
                    )
                continue
            if stored.status == StepStatus.SKIPPED and resume:
                decisions[step_id] = _Decision(
                    run=False,
                    reason="--resume; previously SKIPPED",
                    initial_status=StepStatus.SKIPPED,
                )
                continue
            # PENDING / FAILED / PARTIAL / fresh — run it.
            decisions[step_id] = _Decision(run=True, reason=stored.status.value)
        return decisions


__all__ = [
    "CycleError",
    "DetectionResult",
    "MissingDependencyError",
    "Orchestrator",
    "OrchestratorError",
    "OrchestratorState",
    "PlanRow",
    "RunResult",
    "STATE_DIR",
    "STATE_FILENAME",
    "STATE_SCHEMA_VERSION",
    "Step",
    "StepContext",
    "StepEvent",
    "StepFinished",
    "StepLog",
    "StepProgress",
    "StepResult",
    "StepStarted",
    "StepState",
    "StepStatus",
    "compute_fingerprint",
    "read_state",
    "render_plan_table",
    "state_path",
    "write_state",
]
