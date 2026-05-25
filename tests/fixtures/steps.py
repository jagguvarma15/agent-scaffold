"""Reusable test ``Step`` implementations for the orchestrator suite.

Imported by ``tests/test_orchestrator.py`` and (later) Q6/Q7 tests so the
framework test surface stays consistent with how real steps will plug in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent_scaffold.orchestrator import (
    DetectionResult,
    Step,
    StepContext,
    StepResult,
    StepStatus,
    compute_fingerprint,
)


@dataclass
class NoopStep:
    """Always detects PENDING; ``apply()`` records the call and returns DONE."""

    id: str = "noop"
    description: str = "no-op test step"
    depends_on: tuple[str, ...] = ()
    detect_status: StepStatus = StepStatus.PENDING
    apply_calls: int = field(default=0, init=False)

    def detect(self, ctx: StepContext) -> DetectionResult:
        return DetectionResult(status=self.detect_status, reason="noop")

    def apply(self, ctx: StepContext) -> StepResult:
        self.apply_calls += 1
        return StepResult(status=StepStatus.DONE, detail="noop done")

    def fingerprint(self, ctx: StepContext) -> str:
        return compute_fingerprint({"id": self.id, "calls": self.apply_calls})


@dataclass
class FailingStep:
    """Always fails — used to exercise the halt-on-failure path."""

    id: str = "failing"
    description: str = "always fails"
    depends_on: tuple[str, ...] = ()
    raise_in_apply: bool = False  # if True, raises instead of returning FAILED

    def detect(self, ctx: StepContext) -> DetectionResult:
        return DetectionResult(status=StepStatus.PENDING)

    def apply(self, ctx: StepContext) -> StepResult:
        if self.raise_in_apply:
            raise RuntimeError("simulated crash inside apply()")
        return StepResult(
            status=StepStatus.FAILED, error="simulated failure", stderr_tail="last line of stderr"
        )

    def fingerprint(self, ctx: StepContext) -> str:
        return compute_fingerprint({"id": self.id})


@dataclass
class FlakyStep:
    """Fails the first ``fail_first`` invocations, then succeeds.

    Useful for `--retry` semantics. The instance keeps state across calls so
    a test can re-invoke the orchestrator and watch the step transition.
    """

    id: str = "flaky"
    description: str = "fails then succeeds"
    depends_on: tuple[str, ...] = ()
    fail_first: int = 1
    apply_calls: int = field(default=0, init=False)

    def detect(self, ctx: StepContext) -> DetectionResult:
        return DetectionResult(status=StepStatus.PENDING)

    def apply(self, ctx: StepContext) -> StepResult:
        self.apply_calls += 1
        if self.apply_calls <= self.fail_first:
            return StepResult(status=StepStatus.FAILED, error=f"attempt {self.apply_calls} failed")
        return StepResult(status=StepStatus.DONE, detail=f"succeeded on attempt {self.apply_calls}")

    def fingerprint(self, ctx: StepContext) -> str:
        return compute_fingerprint({"id": self.id})


@dataclass
class AlreadyDoneStep:
    """`detect()` reports DONE immediately — exercises the drift / resume paths."""

    id: str = "already-done"
    description: str = "detects as DONE without running"
    depends_on: tuple[str, ...] = ()
    apply_calls: int = field(default=0, init=False)

    def detect(self, ctx: StepContext) -> DetectionResult:
        return DetectionResult(status=StepStatus.DONE, reason="reported by detect")

    def apply(self, ctx: StepContext) -> StepResult:
        self.apply_calls += 1
        return StepResult(status=StepStatus.DONE)

    def fingerprint(self, ctx: StepContext) -> str:
        return compute_fingerprint({"id": self.id})


@dataclass
class DependentStep:
    """Generic step with a configurable ID + deps for topology tests."""

    id: str
    depends_on: tuple[str, ...] = ()
    description: str = "dependent step"
    apply_calls: int = field(default=0, init=False)

    def detect(self, ctx: StepContext) -> DetectionResult:
        return DetectionResult(status=StepStatus.PENDING)

    def apply(self, ctx: StepContext) -> StepResult:
        self.apply_calls += 1
        return StepResult(status=StepStatus.DONE)

    def fingerprint(self, ctx: StepContext) -> str:
        return compute_fingerprint({"id": self.id})


# Sanity: these classes really do satisfy the Step Protocol at import time.
_: Step = NoopStep()
_ = FailingStep()
_ = FlakyStep()
_ = AlreadyDoneStep()
_ = DependentStep(id="x")
