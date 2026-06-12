"""The "LLMs can never read secrets" guarantee, asserted end-to-end.

Plants secret values in every store the scaffold knows (shell env,
``.env.local``, the project vault) and then records EVERY byte that leaves
through an Anthropic client across the full golden path: generation, a
validation-repair round whose failure output deliberately echoes a secret,
and a REPL refinement where the user pastes a credential. Nothing planted
may appear in any outbound payload, nor in the persistent run artifacts
(``run.log`` / ``events.jsonl``).

Secret values are architecturally confined to subprocess ``env=`` — these
tests are the tripwire that keeps it that way.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from agent_scaffold import generator, pipeline
from agent_scaffold.cli import app
from agent_scaffold.validator import ValidationResult, ValidationTier

# Values shaped like real credentials (so redaction patterns also apply) plus
# one deliberately pattern-free value: the architecture must keep it out of
# prompts even when no regex would catch it. The token-shaped ones are
# assembled at runtime so GitHub push protection doesn't mistake these
# fixtures for real leaked credentials.
PLANTED = {
    "ANTHROPIC_API_KEY": "sk-ant-" + "api03-PLANTEDplantedPLANTED123456",
    "REDIS_URL": "redis://scaffold:hunter2trippedwire@localhost:6379/0",
    "LANGFUSE_SECRET_KEY": "xoxb-" + "0" * 10 + "-plantedslackshape",
    "CUSTOM_PLAIN_TOKEN": "plain-shaped-planted-token-no-regex-match",
}
# Identifiable fragments that must never appear in any outbound/persisted text.
TRIPWIRES = ("PLANTEDplanted", "hunter2trippedwire", "plantedslackshape", "no-regex-match")


class _Block:
    def __init__(self, text: str) -> None:
        self.text = text


class _Response:
    def __init__(self, text: str) -> None:
        self.content = [_Block(text)]


class _StreamCtx:
    def __init__(self, response: Any) -> None:
        self._response = response

    def __enter__(self) -> _StreamCtx:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def __iter__(self) -> Any:
        return iter(())

    def get_final_message(self) -> Any:
        return self._response


class _RecordingClient:
    """Returns canned payloads per call while recording every request."""

    def __init__(self, payloads: list[str]) -> None:
        outer = self

        class _Messages:
            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []

            def stream(self, **kwargs: Any) -> _StreamCtx:
                self.calls.append(kwargs)
                index = min(len(self.calls), len(outer._payloads)) - 1
                return _StreamCtx(_Response(outer._payloads[index]))

        self._payloads = payloads
        self.messages = _Messages()

    @property
    def outbound_text(self) -> str:
        return json.dumps(self.messages.calls, default=str)


def _assert_clean(text: str, where: str) -> None:
    for fragment in TRIPWIRES:
        assert fragment not in text, f"planted secret fragment {fragment!r} leaked into {where}"


def test_full_run_with_repair_never_leaks_planted_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_deployments_path: Path,
    mock_responses_path: Path,
) -> None:
    payload = (mock_responses_path / "valid_python.json").read_text(encoding="utf-8")
    patch_payload = (mock_responses_path / "patch_response.json").read_text(encoding="utf-8")
    client = _RecordingClient([payload, patch_payload])
    monkeypatch.setattr(generator, "_make_client", lambda _cfg: client)

    cache_dir = tmp_path / "cache"
    for name, value in PLANTED.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("AGENT_SCAFFOLD_DEPLOYMENTS_PATH", str(mock_deployments_path))
    monkeypatch.setenv("AGENT_SCAFFOLD_CACHE_DIR", str(cache_dir))
    dest = tmp_path / "out" / "demo_agent"

    # Validation fails once with output that ECHOES planted secrets (the way a
    # chatty subprocess can), then passes after the repair round. The repair
    # prompt must carry only the redacted form.
    script = [False, True]

    def fake_validate(
        _dest: Path,
        _hints: dict[str, Any],
        _smoke: str,
        _tiers: list[ValidationTier],
        continue_on_failure: bool = False,
        on_event: Any = None,
    ) -> list[ValidationResult]:
        passed = script.pop(0) if script else True
        output = (
            ""
            if passed
            else (
                "src/demo_agent/main.py:1:1: F401 unused import\n"
                f"env echo: REDIS_URL={PLANTED['REDIS_URL']} "
                f"key={PLANTED['ANTHROPIC_API_KEY']}"
            )
        )
        return [ValidationResult(tier=ValidationTier.static, passed=passed, output=output)]

    monkeypatch.setattr(pipeline, "run_validate", fake_validate)

    result = CliRunner().invoke(
        app,
        [
            "new",
            "--non-interactive",
            "--recipe",
            "customer-support-triage",
            "--language",
            "python",
            "--framework",
            "langgraph",
            "--project-name",
            "demo_agent",
            "--dest",
            str(dest),
            "--write-mode",
            "overwrite",
            "--no-cache",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert len(client.messages.calls) == 2  # generate + one repair round

    # 1. Nothing planted in any outbound Anthropic payload.
    _assert_clean(client.outbound_text, "an outbound LLM payload")
    # The repair call DID carry the validation output — in redacted form.
    repair_call = json.dumps(client.messages.calls[1], default=str)
    assert "REDACTED" in repair_call

    # 2. Nothing planted in the persistent run artifacts.
    run_dirs = list((cache_dir / "runs").iterdir())
    assert len(run_dirs) == 1
    for artifact in ("run.log", "events.jsonl"):
        _assert_clean((run_dirs[0] / artifact).read_text(encoding="utf-8"), artifact)

    # 3. Nothing planted in the console output either.
    _assert_clean(result.output, "console output")


def test_refinement_free_text_is_redacted_before_haiku(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from agent_scaffold.config import load_config
    from agent_scaffold.repl import refine as refine_mod
    from agent_scaffold.repl.session import SessionState
    from agent_scaffold.sources import DEPLOYMENTS_SPEC, ResolvedSource

    captured: list[dict[str, Any]] = []

    class _HaikuClient:
        class messages:  # noqa: N801 — mimic SDK attribute shape
            @staticmethod
            def create(**kwargs: Any) -> Any:
                captured.append(kwargs)
                return _Response('{"refinement_notes": ["use redis"]}')

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(refine_mod, "_make_haiku_client", lambda _cfg: _HaikuClient())

    src = ResolvedSource(
        spec=DEPLOYMENTS_SPEC,
        path=tmp_path / "deployments",
        label="test",
        kind="explicit-path",
        commit_sha=None,
    )
    cfg = load_config()
    state = SessionState(cfg=cfg, deployments=src, blueprints=src)
    text = (
        "add redis and use this connection: "
        "redis://scaffold:hunter2trippedwire@localhost:6379/0 "
        "with key sk-ant-api03-PLANTEDplantedPLANTED123456"
    )
    refine_mod.interpret_refinement(state, text, cfg)

    outbound = json.dumps(captured, default=str)
    _assert_clean(outbound, "the refinement Haiku payload")
    assert "REDACTED" in outbound
