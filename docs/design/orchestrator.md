# State-tracked step orchestrator

Track B's keystone. `agent-scaffold up` (Q6/Q7) and `agent-scaffold update` (Q8) both build on this framework: every provisioning action is a [`Step`](https://github.com/jagguvarma15/agent-scaffold/blob/main/src/agent_scaffold/orchestrator.py) with two methods, and a single JSON state file at `<project>/.scaffold/state.json` records progress.

## Step contract

A `Step` declares an id, a one-line description, an ordered tuple of `depends_on`, and three methods:

| Method | Side effects | When called | Returns |
|--------|--------------|-------------|---------|
| `detect(ctx)` | None (read-only) | Every `plan()` + every `run()` (unless `--resume` short-circuits) | `DetectionResult(status, reason)` |
| `apply(ctx)` | Yes — the only place mutations live | Once per `run()` per step (modulo skipping) | `StepResult(status, detail, error?, stderr_tail?)` |
| `fingerprint(ctx)` | None | After `apply()` succeeds | Stable SHA-256 of the inputs that would invalidate this step's result |

`apply()` MUST be idempotent. The orchestrator may invoke it multiple times across runs (after crashes, on `--force`, on detected drift); the second invocation must produce the same end state as the first.

## State file

```
<project>/.scaffold/state.json   (mode 0644)
```

Schema (`schema_version: 1`):

```json
{
  "schema_version": 1,
  "started_at": "2026-05-24T01:00:00+00:00",
  "last_run_at": "2026-05-24T02:00:00+00:00",
  "steps": {
    "install_deps": {
      "status": "done",
      "fingerprint": "sha256:...",
      "started_at": "...",
      "completed_at": "...",
      "attempt": 1
    },
    "docker_up": {
      "status": "failed",
      "error": "Cannot connect to the Docker daemon",
      "attempt": 2
    }
  }
}
```

Writes are atomic: `tmp` file → `fsync` → `os.replace`. A crash mid-write leaves the prior `state.json` intact. Future schema versions go through `MIGRATIONS[from_version]` on read.

## Status values

| Status | Meaning |
|--------|---------|
| `pending` | Has not run in this state file. |
| `running` | The orchestrator started this step. If the process dies before the step finishes, it stays `running` on disk; the next invocation treats it like a recovery (re-runs from scratch). |
| `done` | Completed successfully. With a fingerprint match, future runs skip it. |
| `partial` | `detect()` reports the step is in-flight (e.g. 2 of 3 services healthy). `apply()` should finish the work. |
| `skipped` | Explicitly skipped via `--skip` or the orchestrator decided not to run it. |
| `failed` | `apply()` returned `FAILED` or raised. Halts downstream steps. |

## Flag decision table

For each step the orchestrator builds a decision using this priority (top wins):

| Stored status | Flag | Decision |
|--------------|------|----------|
| any | `--skip step` | mark SKIPPED, don't run |
| any | `--force step` | clear state, run |
| `FAILED` | `--retry step` | clear state, run |
| non-`FAILED` | `--retry step` | no-op (don't re-run a DONE step) |
| `RUNNING` | (none) | crash recovery — run |
| `DONE` | `--resume` | skip without re-detect |
| `DONE` | (no `--resume`) | re-detect; skip if still DONE, else re-run (drift) |
| `SKIPPED` | `--resume` | stay SKIPPED |
| `PENDING` / `FAILED` / `PARTIAL` / fresh | (no flag) | run |

`--only step` restricts the active set to that step plus its transitive `depends_on`. Sibling branches outside the dep chain are not run.

The `--plan` flag prints the plan table (one row per active step) and exits without calling `apply()`.

## Topology

Steps form a DAG. The orchestrator topologically sorts them (Kahn's algorithm, declaration-order tiebreaker) before running. Two error classes raise before any step executes:

- `MissingDependencyError` — a step depends on an unknown id
- `CycleError` — the graph has a cycle

Execution is strictly **sequential** in v2. Parallel-DAG execution is intentionally deferred.

## Progress events

Steps emit typed `StepEvent`s via `ctx.emit(...)`:

- `StepStarted(step_id)`
- `StepProgress(step_id, message, percent?)`
- `StepLog(step_id, line, stream)`
- `StepFinished(step_id, result)`

The orchestrator itself emits `StepStarted` and `StepFinished` for every step it runs. Concrete steps emit `StepProgress` / `StepLog` as work happens. A `None` callback is a valid no-op.

## Anti-patterns

1. **Don't read live state in `apply()` that you didn't fingerprint.** If `apply()` consults `$DATABASE_URL` but `fingerprint()` doesn't include it, `--resume` will treat semantically-different invocations as identical and skip them.

2. **Don't apply something with side effects you can't detect.** A step that writes a file but whose `detect()` always returns PENDING will redo the work each run and may not be idempotent.

3. **Don't mutate `.scaffold/state.json` directly.** Only the orchestrator owns it. Per-step scratch belongs in `state.steps[id].metadata` (reserved for a future schema version) — never in a sibling file or another part of the state.

4. **Don't raise from `detect()`.** Detection runs on every plan + run; an exception aborts the whole flow. Return `DetectionResult(status=FAILED, reason=...)` instead, and let `apply()` decide whether to recover.

5. **Don't catch exceptions inside `apply()` and return `DONE`.** The orchestrator can't tell that something went wrong. Either return `StepResult(status=FAILED, error=...)` or let the exception propagate (the orchestrator records it as FAILED with the exception type + message).

## Authoring checklist for new steps

- [ ] `id` is short, stable, snake_case (`install_deps`, not `Install Deps`).
- [ ] `depends_on` lists every step whose side effects this one relies on.
- [ ] `detect()` is cheap (sub-second). Slow probes belong in `apply()` with progress events.
- [ ] `fingerprint()` reflects every input that, if changed, should trigger a re-run.
- [ ] `apply()` is idempotent and exception-safe.
- [ ] Tests cover OK / skip / fail paths for both `detect()` and `apply()`.
