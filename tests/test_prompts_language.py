"""The system prompts must speak both target languages.

The same system blocks are attached to every generation, repair, and
single-file call regardless of language, so any Python-only instruction is
read by TypeScript runs too. A live TS run emitted no Hono CORS because the
prompt taught only FastAPI middleware — and the contract check then failed
the project for it. These guards keep the dual-language sections from
regressing to single-language wording.
"""

from __future__ import annotations

from importlib import resources


def _prompt(name: str) -> str:
    return resources.files("agent_scaffold.prompts").joinpath(name).read_text(encoding="utf-8")


def test_system_prompt_gives_cors_guidance_for_both_languages() -> None:
    text = _prompt("system.md")
    assert "CORSMiddleware" in text
    assert "hono/cors" in text


def test_strict_prompt_has_lint_rules_for_both_languages() -> None:
    text = _prompt("system_strict.md")
    assert "ruff" in text
    assert "tsc --noEmit" in text
    assert "prettier" in text


def test_obs_instrumentation_covers_typescript_in_both_prompts() -> None:
    for name in ("system.md", "system_strict.md"):
        text = _prompt(name)
        assert "traceable" in text
        assert "TypeScript" in text


def test_key_bootstrap_gate_is_marked_python_only() -> None:
    assert "Runtime setup gate (Python backends only)" in _prompt("system.md")
