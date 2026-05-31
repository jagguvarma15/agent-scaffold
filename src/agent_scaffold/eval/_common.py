"""Shared types + helpers for eval plugins.

Every plugin module under ``agent_scaffold.eval`` exposes:

.. code-block:: python

    name: str               # short id, e.g. "promptfoo"
    cli_binary: str         # what shutil.which() should find
    install_hint: str       # printed when cli_binary is missing
    config_file: str | None # canonical config path under <project>/evals/

    def run(project_dir: Path, baseline_total: float | None) -> EvalResult: ...

Eval runs (unlike deploys) complete on their own; the plugin uses
``subprocess.run`` with a timeout and parses JSON output. There's no
streaming-to-terminal UX — the CLI renders the result table afterward.

The regression-detection threshold ``REGRESSION_NOISE_FLOOR`` lives here so
``cmd_eval`` and any future plugins agree on what counts as a real drop
vs. ±1% noise.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

# Per-case score deltas of ±0.01 are dominated by sampling noise on small
# eval suites; only flag drops worse than this as a regression.
REGRESSION_NOISE_FLOOR = 0.01


@dataclass(frozen=True)
class EvalCase:
    """One assertion / test case from the eval run."""

    name: str
    score: float
    """``0.0`` ≤ score ≤ ``1.0``. Plugins normalize their native score shape."""

    passed: bool


@dataclass(frozen=True)
class EvalResult:
    """Outcome of one eval-suite invocation."""

    target: str
    """Plugin name (``"promptfoo"``)."""

    cases: list[EvalCase] = field(default_factory=list)
    total: float = 0.0
    """Weighted average across ``cases``. ``0.0`` when ``cases`` is empty."""

    baseline_total: float | None = None
    delta: float | None = None
    """``total - baseline_total``, or ``None`` if no baseline."""

    cmd_run: list[str] = field(default_factory=list)
    """The shell command the plugin invoked, for the result panel + tests."""

    skipped: bool = False
    skip_reason: str = ""
    error: str | None = None
    """Set when the plugin ran but couldn't produce a result (parse error, ...)."""

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.cases if c.passed)

    @property
    def is_regression(self) -> bool:
        """``True`` iff ``delta`` is more negative than the noise floor."""
        return self.delta is not None and self.delta < -REGRESSION_NOISE_FLOOR


class EvalTarget(Protocol):
    """Loose interface every plugin module satisfies (structural typing)."""

    name: str
    cli_binary: str
    install_hint: str
    config_file: str | None

    def run(self, project_dir: Path, baseline_total: float | None) -> EvalResult: ...


def cli_present(binary: str) -> bool:
    return shutil.which(binary) is not None


def compute_delta(total: float, baseline_total: float | None) -> float | None:
    """Return ``total - baseline_total`` or ``None`` when no baseline."""
    if baseline_total is None:
        return None
    return total - baseline_total


__all__ = [
    "REGRESSION_NOISE_FLOOR",
    "EvalCase",
    "EvalResult",
    "EvalTarget",
    "cli_present",
    "compute_delta",
]
