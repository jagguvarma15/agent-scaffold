"""Post-generation cloud-integration wiring: the ``connect`` command's engine.

``agent-scaffold connect <integration>`` connects an already-generated project
to a cloud-hosted optional integration: capture (or provision) the credential,
validate it against the provider, store it in the encrypted project vault,
wire companion env vars, repair legacy literal compose entries, recreate the
app container so the env actually lands, and verify end-to-end with the
service probe.

Two integrations ship first:

- ``langsmith`` — hosted tracing. No instance to provision: the LangSmith
  project auto-creates (reusing ``bootstrap_langsmith.ensure_project``); only
  the API key is manual. Keeps the legacy ``LANGCHAIN_*`` env names because
  the deployments capability doc owns naming; migrating to ``LANGSMITH_*``
  is a deliberate follow-up.
- ``redis`` — managed Redis override (the local docker container keeps
  working without it). Supports pasting an existing ``rediss://`` URL or
  instant provisioning via Upstash's no-account ``start-redis`` endpoint
  (72-hour database, claimable into an account).

This module never prints secret values (``auth.mask`` for display, RESP
details come from ``probes.redis_ping_url`` which never includes passwords)
and never imports ``cli`` (cli imports it).
"""

from __future__ import annotations

import getpass
import json
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer
import yaml
from pydantic import SecretStr

from agent_scaffold._redact import redact
from agent_scaffold.auth import AuthError, mask, project_namespace, store_project_secret
from agent_scaffold.auth_browser import browser_available, browser_paste_flow
from agent_scaffold.cli_shared import console
from agent_scaffold.discovery import ExternalService
from agent_scaffold.doctor import CheckResult, CheckStatus
from agent_scaffold.envfile import append_env_local, build_runtime_env
from agent_scaffold.manifest import Manifest
from agent_scaffold.probes import (
    probe_langsmith_workspace,
    probe_redis_ping,
    redis_ping_url,
)
from agent_scaffold.writer import ensure_gitignore_defaults

_LANGSMITH_SETTINGS_URL = "https://smith.langchain.com/settings"
_UPSTASH_START_URL = "https://upstash.com/start-redis"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    detail: str
    # True when the provider explicitly rejected the credential (never offer
    # store-anyway); False for network-ish failures where storing may be fine.
    auth_failure: bool = False


def validate_langsmith_key(candidate: str, timeout: float) -> ValidationResult:
    """Check the key against LangSmith via ``Client.info()``; never raises."""
    _ = timeout  # the SDK manages its own timeouts
    try:
        from langsmith import Client
    except ImportError:
        return ValidationResult(True, "not validated (langsmith SDK not installed)")
    try:
        Client(api_key=candidate).info()
    except Exception as exc:  # noqa: BLE001 - validation must never raise
        detail = f"{type(exc).__name__}: {exc}"
        lowered = detail.lower()
        auth = any(marker in lowered for marker in ("401", "403", "unauthorized", "forbidden"))
        return ValidationResult(False, detail, auth_failure=auth)
    return ValidationResult(True, "key accepted by LangSmith")


def validate_redis_url(candidate: str, timeout: float) -> ValidationResult:
    """PING the candidate URL (TLS/AUTH aware); never raises."""
    ping = redis_ping_url(candidate, timeout)
    detail = f"{ping.summary}: {ping.detail}" if ping.detail else ping.summary
    return ValidationResult(ping.ok, detail, auth_failure=ping.kind == "auth")


# ---------------------------------------------------------------------------
# Integration registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Integration:
    """One connectable cloud integration."""

    id: str
    title: str
    # Any of these capability ids on the manifest makes the project eligible.
    capability_ids: frozenset[str]
    credential_var: str
    # Every env var connect may touch; also the compose literal-repair set.
    managed_vars: tuple[str, ...]
    key_page_url: str | None
    hint: str
    placeholder: str
    validate: Callable[[str, float], ValidationResult]
    probe: Callable[..., CheckResult]


INTEGRATIONS: dict[str, Integration] = {
    "langsmith": Integration(
        id="langsmith",
        title="LangSmith tracing",
        capability_ids=frozenset({"obs.langsmith"}),
        credential_var="LANGCHAIN_API_KEY",
        managed_vars=(
            "LANGCHAIN_API_KEY",
            "LANGCHAIN_TRACING_V2",
            "LANGCHAIN_PROJECT",
            "LANGCHAIN_ENDPOINT",
        ),
        key_page_url=_LANGSMITH_SETTINGS_URL,
        hint="create an API key under Settings",
        placeholder="lsv2_...",
        validate=validate_langsmith_key,
        probe=probe_langsmith_workspace,
    ),
    "redis": Integration(
        id="redis",
        title="managed Redis",
        capability_ids=frozenset({"cache.redis", "queue.redis-streams"}),
        credential_var="REDIS_URL",
        managed_vars=("REDIS_URL",),
        key_page_url=None,
        hint="a managed Redis URL (Upstash / ElastiCache / Redis Cloud)",
        placeholder="rediss://:password@host:port",
        validate=validate_redis_url,
        probe=probe_redis_ping,
    ),
}


# ---------------------------------------------------------------------------
# Upstash instant provisioning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UpstashDatabase:
    url: str  # rediss://:password@host:port
    claim_url: str | None


def parse_upstash_start_response(payload: dict[str, Any]) -> UpstashDatabase | None:
    """Extract a usable database from the start-redis response; None on drift.

    The endpoint is convenience-tier, so field names are matched defensively:
    a full URL under ``url``/``redis_url``, or ``endpoint``/``host`` +
    ``port`` + ``password``/``token`` parts.
    """
    claim_raw = payload.get("claim_url") or payload.get("claimUrl")
    claim = claim_raw if isinstance(claim_raw, str) and claim_raw else None
    url = payload.get("url") or payload.get("redis_url") or ""
    if isinstance(url, str) and url.startswith(("redis://", "rediss://")):
        return UpstashDatabase(url=url, claim_url=claim)
    host = payload.get("endpoint") or payload.get("host") or ""
    password = payload.get("password") or payload.get("token") or ""
    if not (isinstance(host, str) and host and isinstance(password, str) and password):
        return None
    try:
        port = int(payload.get("port") or 6379)
    except (TypeError, ValueError):
        port = 6379
    return UpstashDatabase(url=f"rediss://:{password}@{host}:{port}", claim_url=claim)


def provision_upstash_free(timeout: float = 15.0) -> UpstashDatabase | str:
    """Create an instant no-account Upstash database; str is the error text.

    The database lives 72 hours unless claimed into an Upstash account via
    the returned claim URL. Never raises; never logs the password.
    """
    # S310 suppressed on both calls: the URL is the fixed https constant above,
    # so no file:/custom scheme can reach urlopen.
    request = urllib.request.Request(  # noqa: S310
        _UPSTASH_START_URL,
        method="POST",
        data=b"",
        headers={"User-Agent": "agent-scaffold", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return f"Upstash provisioning failed: {type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return "Upstash provisioning failed: unexpected response shape"
    database = parse_upstash_start_response(payload)
    if database is None:
        return "Upstash provisioning failed: response had no usable endpoint"
    return database


# ---------------------------------------------------------------------------
# Compose literal-env repair
# ---------------------------------------------------------------------------


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _block_end(lines: list[str], start: int) -> int:
    """Index one past the last line of the block opened by ``lines[start]``.

    YAML sequence items may sit at the SAME indent as their parent key
    (``environment:`` followed by ``- VAR=x`` at equal indent), so dash lines
    at the base indent still belong to the block.
    """
    base = _indent(lines[start])
    index = start + 1
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            indent = _indent(line)
            if indent < base or (indent == base and not stripped.startswith("-")):
                break
        index += 1
    return index


def _service_env(data: Any, service: str) -> dict[str, str] | None:
    """The service's environment normalized to a str dict (list form included)."""
    if not isinstance(data, dict):
        return None
    services = data.get("services")
    if not isinstance(services, dict) or not isinstance(services.get(service), dict):
        return None
    env = services[service].get("environment")
    if isinstance(env, dict):
        return {str(k): "" if v is None else str(v) for k, v in env.items()}
    if isinstance(env, list):
        out: dict[str, str] = {}
        for entry in env:
            name, _, value = str(entry).partition("=")
            out[name.strip()] = value
        return out
    return None


def find_literal_env_entries(
    compose_text: str, service: str, candidates: tuple[str, ...]
) -> dict[str, str]:
    """Managed vars whose compose value is a literal (no ``${`` interpolation)."""
    try:
        data = yaml.safe_load(compose_text)
    except yaml.YAMLError:
        return {}
    env = _service_env(data, service)
    if env is None:
        return {}
    return {var: env[var] for var in candidates if var in env and "${" not in env[var]}


def _unquote_scalar(raw: str) -> str:
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        return text[1:-1]
    return text


def rewrite_literal_env(
    compose_text: str, service: str, candidates: tuple[str, ...]
) -> tuple[str, list[str]]:
    """Rewrite literal env entries to ``${VAR:-<old-literal>}`` interpolations.

    Line-targeted (a yaml round-trip would destroy comments and ordering),
    verified by re-parsing: any mismatch reverts to the original text. The old
    literal survives as the interpolation default, so semantics are unchanged
    while the var is unset.
    """
    targets = set(find_literal_env_entries(compose_text, service, candidates))
    if not targets:
        return compose_text, []
    lines = compose_text.split("\n")
    services_idx = next((i for i, line in enumerate(lines) if line.rstrip() == "services:"), None)
    if services_idx is None:
        return compose_text, []
    services_end = _block_end(lines, services_idx)
    service_idx = next(
        (i for i in range(services_idx + 1, services_end) if lines[i].strip() == f"{service}:"),
        None,
    )
    if service_idx is None:
        return compose_text, []
    service_end = _block_end(lines, service_idx)
    env_idx = next(
        (i for i in range(service_idx + 1, service_end) if lines[i].strip() == "environment:"),
        None,
    )
    if env_idx is None:
        return compose_text, []
    env_end = _block_end(lines, env_idx)
    env_entry_indent = None
    rewritten: list[str] = []
    for index in range(env_idx + 1, env_end):
        line = lines[index]
        if not line.strip():
            continue
        # Only direct entries of the environment block are eligible; deeper
        # indentation means a continuation line of a multi-line value.
        if env_entry_indent is None:
            env_entry_indent = _indent(line)
        if _indent(line) != env_entry_indent:
            continue
        for var in targets:
            dict_form = re.match(rf"^(\s+){re.escape(var)}:\s*(\S.*?)\s*$", line)
            if dict_form and "${" not in dict_form.group(2):
                literal = _unquote_scalar(dict_form.group(2))
                lines[index] = f"{dict_form.group(1)}{var}: ${{{var}:-{literal}}}"
                rewritten.append(var)
                break
            list_form = re.match(rf"^(\s+-\s*){re.escape(var)}=(.*)$", line)
            if list_form and "${" not in list_form.group(2):
                literal = _unquote_scalar(list_form.group(2))
                lines[index] = f"{list_form.group(1)}{var}=${{{var}:-{literal}}}"
                rewritten.append(var)
                break
    if not rewritten:
        return compose_text, []
    new_text = "\n".join(lines)
    try:
        env = _service_env(yaml.safe_load(new_text), service)
    except yaml.YAMLError:
        return compose_text, []
    if env is None or any(f"${{{var}:-" not in env.get(var, "") for var in rewritten):
        return compose_text, []
    return new_text, sorted(rewritten)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def persist_project_secret(
    namespace: str, project_dir: Path, env_var: str, secret: SecretStr
) -> str | None:
    """Vault-first storage with plaintext ``.env.local`` fallback.

    Mirrors ``wire_credentials._persist``: the vault path inherits the
    plaintext-keyring refusal; the fallback path gitignores ``.env.local``.
    Returns a backend label for display, or None when nothing could store.
    """
    try:
        stored = store_project_secret(namespace, env_var, secret)
    except AuthError as exc:
        console.print(
            f"[yellow]vault rejected {env_var} ({redact(str(exc))}); "
            "falling back to .env.local[/]"
        )
    else:
        return f"vault ({stored.backend})"
    try:
        append_env_local(project_dir, env_var, secret)
    except OSError as exc:
        console.print(f"[red]failed to write .env.local: {exc}[/]")
        return None
    try:
        ensure_gitignore_defaults(project_dir, extra=(".env.local",))
    except OSError as exc:
        console.print(f"[yellow].env.local stored but .gitignore update failed: {exc}[/]")
    return ".env.local"


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def _confirm(prompt: str, *, default: bool) -> bool:
    try:
        return typer.confirm(prompt, default=default)
    except (typer.Abort, EOFError):
        return False


def _paste_secret(integration: Integration) -> str | None:
    """Browser paste form when available, getpass otherwise; None on abort."""
    if browser_available():
        return browser_paste_flow(
            label=integration.credential_var,
            hint=integration.hint,
            hint_url=integration.key_page_url,
            placeholder=integration.placeholder,
        )
    try:
        value = getpass.getpass(f"Enter {integration.credential_var} (never echoed): ")
    except (EOFError, KeyboardInterrupt):
        return None
    return value.strip() or None


def _capture_langsmith(
    integration: Integration, env_before: Mapping[str, str], *, interactive: bool
) -> str | None:
    existing = (env_before.get(integration.credential_var) or "").strip()
    if not interactive:
        return existing or None
    if existing and _confirm(
        f"{integration.credential_var} already set ({mask(existing)}) - reuse it?", default=True
    ):
        return existing
    return _paste_secret(integration)


def _capture_redis(
    integration: Integration, *, interactive: bool, url: str | None, timeout: float
) -> str | None:
    """Returns the URL, "" to signal keep-local (clean no-op), None on abort."""
    if not interactive:
        return url
    from agent_scaffold.cli_interactive import _interactive_select

    choice = _interactive_select(
        "Managed Redis source?",
        choices=[
            ("paste", "paste an existing managed URL (Upstash / ElastiCache / Redis Cloud)"),
            ("upstash", "create an instant Upstash database (free, 72h unless claimed)"),
            ("local", "keep the local docker redis (no change)"),
        ],
        default="paste",
    )
    if choice == "local":
        return ""
    if choice == "upstash":
        console.print("[dim]Provisioning an instant Upstash database...[/]")
        outcome = provision_upstash_free(timeout=max(timeout, 15.0))
        if isinstance(outcome, str):
            console.print(f"[yellow]{redact(outcome)}[/] - paste a URL instead.")
            return _paste_secret(integration)
        if outcome.claim_url:
            console.print(
                "[bold]Claim this database into your Upstash account within 72 hours:[/] "
                f"{outcome.claim_url}"
            )
        else:
            console.print(
                "[bold]Note:[/] this database expires in 72 hours unless claimed "
                "at https://console.upstash.com"
            )
        if not _confirm("Use this database (acknowledging the 72-hour window)?", default=True):
            return None
        return outcome.url
    return _paste_secret(integration)


# ---------------------------------------------------------------------------
# The connect flow
# ---------------------------------------------------------------------------


def _langsmith_companion(project_dir: Path, manifest: Manifest, api_key: str) -> str | None:
    """Ensure the LangSmith project exists and write tracing env; name or None."""
    from agent_scaffold.steps.bootstrap_langsmith import ensure_project, write_tracing_env

    answers = manifest.answers or {}
    project_name = (
        answers.get("langsmith_project") or answers.get("project_name") or manifest.recipe
    )
    endpoint = "https://api.smith.langchain.com"
    try:
        from langsmith import Client
    except ImportError:
        console.print(
            "[yellow]langsmith SDK not installed - skipping project creation "
            '(pip install "agent-scaffold-cli[obs]")[/]'
        )
    else:
        try:
            client = Client(api_key=api_key, api_url=endpoint)
        except Exception as exc:  # noqa: BLE001 - companion work must not abort the flow
            console.print(f"[yellow]langsmith client init failed: {redact(str(exc))}[/]")
        else:
            action, error = ensure_project(client, project_name)
            if error is not None:
                console.print(f"[yellow]{redact(error)}[/]")
            else:
                console.print(f"LangSmith project {project_name!r} {action}")
    written = write_tracing_env(project_dir, project_name, endpoint)
    if written:
        console.print(f"Wrote {written} tracing var(s) to .env.local")
    return project_name


def _repair_compose_literals(
    compose_path: Path | None, integration: Integration, *, yes: bool
) -> None:
    if compose_path is None or not compose_path.is_file():
        return
    from agent_scaffold.steps.docker_up import _compose_app_service

    app_service = _compose_app_service(compose_path.parent)
    if app_service is None:
        return
    text = compose_path.read_text(encoding="utf-8")
    literals = find_literal_env_entries(text, app_service, integration.managed_vars)
    if not literals:
        return
    console.print(
        f"[yellow]{compose_path.name} pins literal values for: "
        f"{', '.join(sorted(literals))}[/] - env wiring can't override them."
    )
    new_text, rewritten = rewrite_literal_env(text, app_service, integration.managed_vars)
    if not rewritten:
        console.print(
            "[yellow]Couldn't safely rewrite them - edit these entries manually to "
            "${VAR:-default} form.[/]"
        )
        return
    for var in rewritten:
        console.print(f"  {var}: {literals[var]!r} -> ${{{var}:-{literals[var]}}}")
    if not yes and not _confirm(f"Rewrite these entries in {compose_path.name}?", default=True):
        console.print("[dim]Skipped - the stored value won't reach the container.[/]")
        return
    compose_path.write_text(new_text, encoding="utf-8")
    console.print(f"Rewrote {len(rewritten)} entr{'y' if len(rewritten) == 1 else 'ies'}.")


def _recreate_app(compose_path: Path | None, env: dict[str, str]) -> bool:
    if compose_path is None or not compose_path.is_file():
        console.print(
            "[dim]No docker-compose.yml found - restart your stack to pick up the new env.[/]"
        )
        return False
    if shutil.which("docker") is None:
        console.print("[dim]docker not on PATH - run `agent-scaffold up` to apply the env.[/]")
        return False
    from agent_scaffold.steps.docker_up import _compose_app_service

    app_service = _compose_app_service(compose_path.parent)
    cmd = ["docker", "compose", "up", "-d"] + ([app_service] if app_service else [])
    console.print(f"[cyan]Running:[/] {' '.join(cmd)}")
    completed = subprocess.run(cmd, cwd=compose_path.parent, env=env, check=False)
    return completed.returncode == 0


def run_connect(
    project_dir: Path,
    manifest: Manifest,
    integration: Integration,
    compose_path: Path | None,
    *,
    yes: bool,
    url: str | None = None,
    timeout: float = 10.0,
    reset_step_state: Callable[[Path, str], None] | None = None,
) -> int:
    """Drive the full connect flow; returns the shell exit code."""
    capabilities = set(manifest.capabilities or [])
    if not capabilities & integration.capability_ids:
        wanted = " or ".join(sorted(integration.capability_ids))
        console.print(
            f"[red]This project doesn't declare {wanted}[/] - `connect {integration.id}` "
            "needs the capability in the generated stack (pick it at generation time, "
            "e.g. REPL /layer)."
        )
        return 1
    namespace = manifest.secrets_namespace or project_namespace(project_dir.name, project_dir)
    env_before = build_runtime_env(project_dir, namespace)
    interactive = not yes and sys.stdin.isatty()

    if integration.id == "redis":
        candidate = _capture_redis(integration, interactive=interactive, url=url, timeout=timeout)
        if candidate == "":
            console.print("Keeping the local docker redis - nothing to change.")
            return 0
    else:
        candidate = _capture_langsmith(integration, env_before, interactive=interactive)
    if not candidate:
        if not interactive:
            source = (
                "--url rediss://..."
                if integration.id == "redis"
                else (f"export {integration.credential_var}=...")
            )
            console.print(
                f"[red]No value for {integration.credential_var}.[/] Non-interactive runs "
                f"need it supplied up front: {source} then re-run with --yes."
            )
            return 2
        console.print("[yellow]Aborted - nothing stored.[/]")
        return 1

    verdict = integration.validate(candidate, timeout)
    if not verdict.ok:
        console.print(f"[red]Validation failed:[/] {redact(verdict.detail)}")
        if verdict.auth_failure or not interactive:
            console.print("[red]Nothing stored.[/]")
            return 1
        if not _confirm("Store anyway (network problems can be transient)?", default=False):
            return 1
    else:
        console.print(f"Validated: {redact(verdict.detail)}")

    backend = persist_project_secret(
        namespace, project_dir, integration.credential_var, SecretStr(candidate)
    )
    if backend is None:
        return 1
    console.print(f"Stored {integration.credential_var} ({mask(candidate)}) in {backend}")

    import os

    shell_value = (os.environ.get(integration.credential_var) or "").strip()
    if shell_value and shell_value != candidate:
        console.print(
            f"[yellow]Your shell also exports {integration.credential_var} with a different "
            "value - the shell wins at run time; unset it to use the stored value.[/]"
        )

    langsmith_project: str | None = None
    if integration.id == "langsmith":
        langsmith_project = _langsmith_companion(project_dir, manifest, candidate)

    _repair_compose_literals(compose_path, integration, yes=yes)

    env_after = build_runtime_env(project_dir, namespace)
    _recreate_app(compose_path, env_after)

    if reset_step_state is not None and integration.id == "langsmith":
        reset_step_state(project_dir, "bootstrap_langsmith")

    service = ExternalService(
        id=integration.id,
        env_vars=[integration.credential_var],
        probe=integration.id,
        required=True,
    )
    check = integration.probe(service, timeout, env=env_after)
    marker = {"ok": "[green]OK[/]", "fail": "[red]FAIL[/]"}.get(
        "ok" if check.status == CheckStatus.OK else "fail"
    )
    console.print(
        f"Verify: {marker} {check.title}" + (f" - {check.detail}" if check.detail else "")
    )

    if integration.id == "langsmith":
        console.print(
            f"Traces will appear at https://smith.langchain.com under project "
            f"{langsmith_project!r} once the agent handles a request."
        )
    else:
        console.print(
            "Managed Redis wired. The local compose redis container keeps running; "
            "the app now prefers the managed URL."
        )
    return 0 if check.status == CheckStatus.OK else 1


__all__ = [
    "INTEGRATIONS",
    "Integration",
    "UpstashDatabase",
    "ValidationResult",
    "find_literal_env_entries",
    "parse_upstash_start_response",
    "persist_project_secret",
    "provision_upstash_free",
    "rewrite_literal_env",
    "run_connect",
    "validate_langsmith_key",
    "validate_redis_url",
]
