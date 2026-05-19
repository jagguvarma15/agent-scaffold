"""Post-generation validation tiers.

Run lightweight static checks, full builds, or the smoke check as subprocesses
inside the generated project's directory. Each tier captures stdout+stderr.
"""

from __future__ import annotations

import shutil
import subprocess
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class ValidationTier(str, Enum):
    static = "static"
    build = "build"
    smoke = "smoke"


class ValidationResult(BaseModel):
    tier: ValidationTier
    passed: bool
    output: str


def _run(cmd: list[str], cwd: Path) -> tuple[bool, str]:
    if shutil.which(cmd[0]) is None:
        return False, f"command not found on PATH: {cmd[0]}"
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        return False, f"timeout: {exc}"
    except OSError as exc:
        return False, f"failed to launch {cmd[0]}: {exc}"
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, output


def _run_shell(cmd: str, cwd: Path) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            shell=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        return False, f"timeout: {exc}"
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, output


def _static_command(language: str) -> list[str] | None:
    if language == "python":
        return ["ruff", "check", "."]
    if language == "typescript":
        return ["pnpm", "exec", "tsc", "--noEmit"]
    return None


def _build_command(language: str) -> list[str] | None:
    if language == "python":
        return ["uv", "sync"]
    if language == "typescript":
        return ["pnpm", "install"]
    return None


def validate(
    dest: Path,
    hints: dict[str, Any],
    smoke_check: str,
    tiers: list[ValidationTier],
    continue_on_failure: bool = False,
) -> list[ValidationResult]:
    """Run requested validation tiers in order and return their results."""
    results: list[ValidationResult] = []
    language = str(hints.get("language", "python"))
    for tier in tiers:
        if tier is ValidationTier.static:
            cmd = _static_command(language)
            if cmd is None:
                results.append(
                    ValidationResult(
                        tier=tier,
                        passed=True,
                        output=f"no static check defined for language={language}",
                    )
                )
                continue
            passed, output = _run(cmd, dest)
        elif tier is ValidationTier.build:
            cmd = _build_command(language)
            if cmd is None:
                results.append(
                    ValidationResult(
                        tier=tier,
                        passed=True,
                        output=f"no build command defined for language={language}",
                    )
                )
                continue
            passed, output = _run(cmd, dest)
        elif tier is ValidationTier.smoke:
            if not smoke_check:
                results.append(
                    ValidationResult(tier=tier, passed=True, output="no smoke_check supplied")
                )
                continue
            passed, output = _run_shell(smoke_check, dest)
        else:  # pragma: no cover - exhaustive
            raise ValueError(f"Unknown tier: {tier}")

        results.append(ValidationResult(tier=tier, passed=passed, output=output))
        if not passed and not continue_on_failure:
            break
    return results
