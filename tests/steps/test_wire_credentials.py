"""Tests for ``agent_scaffold.steps.wire_credentials``."""

from __future__ import annotations

import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from agent_scaffold.discovery import ExternalService
from agent_scaffold.orchestrator import (
    StepContext,
    StepEvent,
    StepProgress,
    StepStatus,
)
from agent_scaffold.steps import wire_credentials as wc_mod
from agent_scaffold.steps.wire_credentials import WireCredentialsStep


def _anth_svc() -> ExternalService:
    return ExternalService(id="anthropic", env_vars=["ANTHROPIC_API_KEY"], required=True)


def _redis_svc() -> ExternalService:
    return ExternalService(id="redis", env_vars=["REDIS_URL"], required=True)


def _opt_svc() -> ExternalService:
    return ExternalService(id="opt", env_vars=["OPTIONAL_TOKEN"], required=False)


def test_detect_done_when_no_services(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
    recipe_factory: Callable[..., Any],
    patch_load_recipe: Callable[[Any], None],
) -> None:
    patch_load_recipe(recipe_factory(external_services=[]))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = WireCredentialsStep().detect(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.DONE


def test_detect_pending_lists_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
    recipe_factory: Callable[..., Any],
    patch_load_recipe: Callable[[Any], None],
) -> None:
    patch_load_recipe(recipe_factory(external_services=[_anth_svc(), _redis_svc()]))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setattr(wc_mod, "load_key", lambda: None)
    result = WireCredentialsStep().detect(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.PENDING
    assert "ANTHROPIC_API_KEY" in result.reason
    assert "REDIS_URL" in result.reason


def test_detect_partial_when_only_optional_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
    recipe_factory: Callable[..., Any],
    patch_load_recipe: Callable[[Any], None],
) -> None:
    patch_load_recipe(recipe_factory(external_services=[_opt_svc()]))
    monkeypatch.delenv("OPTIONAL_TOKEN", raising=False)
    result = WireCredentialsStep().detect(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.PARTIAL


def test_apply_yes_fails_when_required_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
    recipe_factory: Callable[..., Any],
    patch_load_recipe: Callable[[Any], None],
) -> None:
    patch_load_recipe(recipe_factory(external_services=[_anth_svc()]))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(wc_mod, "load_key", lambda: None)
    result = WireCredentialsStep(yes=True).apply(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.FAILED
    assert "ANTHROPIC_API_KEY" in (result.error or "")


def test_apply_yes_ok_when_optional_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
    recipe_factory: Callable[..., Any],
    patch_load_recipe: Callable[[Any], None],
) -> None:
    patch_load_recipe(recipe_factory(external_services=[_opt_svc()]))
    monkeypatch.delenv("OPTIONAL_TOKEN", raising=False)
    result = WireCredentialsStep(yes=True).apply(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.DONE


def test_apply_prompts_and_stores_anthropic_in_keyring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
    event_log: list[StepEvent],
    recipe_factory: Callable[..., Any],
    patch_load_recipe: Callable[[Any], None],
) -> None:
    patch_load_recipe(recipe_factory(external_services=[_anth_svc()]))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(wc_mod, "load_key", lambda: None)
    monkeypatch.setattr(wc_mod.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(wc_mod.getpass, "getpass", lambda _p: "sk-ant-test123456")
    stored: dict[str, Any] = {}

    def fake_store(name: str, value: Any, backend: str = "keyring") -> Any:
        stored["name"] = name
        stored["backend"] = backend
        stored["value"] = value.get_secret_value()
        from agent_scaffold.auth import StoredCredential, mask

        return StoredCredential(name=name, backend=backend, masked_value=mask(stored["value"]))

    monkeypatch.setattr(wc_mod, "store_key", fake_store)
    result = WireCredentialsStep().apply(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.DONE
    assert stored["backend"] == "keyring"
    assert stored["value"] == "sk-ant-test123456"
    # Secret value must NEVER appear in any event payload.
    for ev in event_log:
        if isinstance(ev, StepProgress):
            assert "sk-ant-test123456" not in ev.message


def test_apply_writes_env_local_for_project_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
    recipe_factory: Callable[..., Any],
    patch_load_recipe: Callable[[Any], None],
) -> None:
    patch_load_recipe(recipe_factory(external_services=[_redis_svc()]))
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setattr(wc_mod.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(wc_mod.getpass, "getpass", lambda _p: "redis://localhost:6379/0")
    result = WireCredentialsStep().apply(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.DONE
    env_local = tmp_path / ".env.local"
    assert env_local.is_file()
    text = env_local.read_text(encoding="utf-8")
    assert "REDIS_URL=redis://localhost:6379/0" in text
    # Mode 0600
    mode = stat.S_IMODE(env_local.stat().st_mode)
    assert mode == 0o600, f"expected 0600, got {oct(mode)}"
    # .gitignore got an entry
    gi = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    assert ".env.local" in gi


def test_env_local_existing_key_replaced_not_duplicated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
    recipe_factory: Callable[..., Any],
    patch_load_recipe: Callable[[Any], None],
) -> None:
    (tmp_path / ".env.local").write_text("REDIS_URL=old\nOTHER=keep\n", encoding="utf-8")
    patch_load_recipe(recipe_factory(external_services=[_redis_svc()]))
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setattr(wc_mod.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(wc_mod.getpass, "getpass", lambda _p: "redis://new")
    WireCredentialsStep().apply(ctx_factory(project_dir=tmp_path))
    text = (tmp_path / ".env.local").read_text(encoding="utf-8")
    assert "REDIS_URL=redis://new" in text
    assert "OTHER=keep" in text
    # Single REDIS_URL line, not two.
    assert text.count("REDIS_URL=") == 1


def test_existing_env_var_treated_as_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ctx_factory: Callable[..., StepContext],
    recipe_factory: Callable[..., Any],
    patch_load_recipe: Callable[[Any], None],
) -> None:
    patch_load_recipe(recipe_factory(external_services=[_redis_svc()]))
    monkeypatch.setenv("REDIS_URL", "redis://already-set")
    result = WireCredentialsStep().detect(ctx_factory(project_dir=tmp_path))
    assert result.status is StepStatus.DONE
