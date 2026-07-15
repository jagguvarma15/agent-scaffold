"""Post-generation stack-option wiring: the ``connect`` command's engine.

``agent-scaffold connect <option>`` connects an already-generated project to
one of its stack options — internal docker services and cloud hosted
platforms alike. The flow per mode:

- ``cloud`` (langsmith, langfuse, any credentialed provider): capture the
  credentials, validate against the provider, store in the encrypted project
  vault, wire companion env vars, repair legacy literal compose entries,
  recreate the app container so the env actually lands, verify with the probe.
- ``internal-overridable`` (redis, postgres, qdrant): choose between keeping
  or starting the local docker container and pointing the same env var at a
  managed instance (paste a URL, or instant-provision where a provider
  supports it — Upstash's no-account 72-hour database for redis).
- ``internal``: ensure the docker container runs and verify with the probe.

Options derive from the manifest's capabilities joined with the deployments
catalog (:mod:`agent_scaffold.stack_options`); provider-specific behavior that
the catalog cannot express (capture menus, provisioning, SDK validation,
companion wiring, closing text) lives in the :data:`PROVIDER_EXTRAS` registry.
LangSmith keeps the legacy ``LANGCHAIN_*`` env names because the deployments
capability doc owns naming; migrating to ``LANGSMITH_*`` is a deliberate
follow-up.

This module never prints secret values (``auth.mask`` for display, RESP
details come from ``probes.redis_ping_url`` which never includes passwords)
and never imports ``cli`` (cli imports it).
"""

from __future__ import annotations

import getpass
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import typer
import yaml
from pydantic import SecretStr

from agent_scaffold._redact import redact
from agent_scaffold.auth import (
    AuthError,
    list_project_secret_names,
    mask,
    project_namespace,
    store_project_secret,
)
from agent_scaffold.auth_browser import browser_available, browser_paste_flow
from agent_scaffold.cli_shared import console
from agent_scaffold.doctor import CheckStatus
from agent_scaffold.envfile import append_env_local, build_runtime_env, read_env_local
from agent_scaffold.manifest import Manifest
from agent_scaffold.probes import redis_ping_url, run_probe
from agent_scaffold.stack_options import (
    MODE_INTERNAL,
    MODE_INTERNAL_OVERRIDABLE,
    OVERRIDABLE_URL_VARS,
    CredentialSpec,
    StackOption,
    service_for_option,
)
from agent_scaffold.writer import ensure_gitignore_defaults

_UPSTASH_START_URL = "https://upstash.com/start-redis"


def find_docker_compose(project_dir: Path) -> Path | None:
    """The project's compose file, checking the conventional locations."""
    for candidate in (
        project_dir / "docker-compose.yml",
        project_dir / "infra" / "docker-compose.yml",
        project_dir / "compose.yaml",
    ):
        if candidate.is_file():
            return candidate
    return None


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
# Capture plumbing + provider extras registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaptureResult:
    """Outcome of the capture stage.

    ``kind`` is ``ok`` (values captured), ``local`` (keep/start the local
    docker container instead of a managed instance), or ``abort``.
    """

    kind: str
    values: dict[str, str] = field(default_factory=dict)


CaptureFn = Callable[..., CaptureResult]


@dataclass(frozen=True)
class ProviderExtras:
    """Provider-specific behavior layered onto the generic connect flow.

    Every field is optional — a provider with no entry here still connects
    through the generic capture / probe-validate / store / verify pipeline.
    ``validate`` receives every captured var so multi-credential providers can
    check the full set; ``companion`` runs after storage (provider-side
    resource creation, companion env writes) and returns a display context
    string that ``closing`` can reference.
    """

    capture: CaptureFn | None = None
    provision: Callable[[float], UpstashDatabase | str] | None = None
    validate: Callable[[dict[str, str], float], ValidationResult] | None = None
    companion: Callable[[Path, Manifest, dict[str, str]], str | None] | None = None
    closing: Callable[[StackOption, str | None], str] | None = None


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


def _stdin_isatty() -> bool:
    """Patchable seam: CliRunner swaps sys.stdin, so tests stub this instead."""
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


def _confirm(prompt: str, *, default: bool) -> bool:
    try:
        return typer.confirm(prompt, default=default)
    except (typer.Abort, EOFError):
        return False


def _paste_value(spec: CredentialSpec, option: StackOption) -> str | None:
    """Capture one credential value; browser form for secrets when available.

    Returns None on abort (or on skip for an optional value).
    """
    if spec.secret and browser_available():
        return browser_paste_flow(
            label=spec.var,
            hint=spec.hint,
            hint_url=option.key_page_url,
            placeholder=spec.placeholder,
        )
    suffix = " (leave empty to skip)" if spec.optional else ""
    if spec.secret:
        try:
            value = getpass.getpass(f"Enter {spec.var} (never echoed){suffix}: ")
        except (EOFError, KeyboardInterrupt):
            return None
        return value.strip() or None
    prompt = f"{spec.var}"
    if spec.hint:
        prompt += f" ({spec.hint})"
    try:
        value = typer.prompt(prompt + suffix, default="", show_default=False)
    except (typer.Abort, EOFError):
        return None
    return str(value).strip() or None


def _capture_credentials(
    option: StackOption,
    env_before: Mapping[str, str],
    *,
    interactive: bool,
    url: str | None = None,
) -> CaptureResult:
    """Generic capture: walk the option's credential specs in order.

    Non-interactive runs never prompt: ``--url`` binds to the first spec,
    everything else must already resolve from the runtime env snapshot.
    """
    values: dict[str, str] = {}
    for index, spec in enumerate(option.credentials):
        existing = (env_before.get(spec.var) or "").strip()
        if not interactive:
            supplied = url.strip() if index == 0 and url else existing
            if supplied:
                values[spec.var] = supplied
            elif not spec.optional:
                return CaptureResult("abort")
            continue
        if existing and _confirm(
            f"{spec.var} already set ({mask(existing)}) - reuse it?", default=True
        ):
            values[spec.var] = existing
            continue
        value = _paste_value(spec, option)
        if value is None:
            if spec.optional:
                continue
            return CaptureResult("abort")
        values[spec.var] = value
    return CaptureResult("ok", values)


def _capture_overridable(
    option: StackOption,
    extras: ProviderExtras,
    env_before: Mapping[str, str],
    *,
    interactive: bool,
    url: str | None,
    timeout: float,
) -> CaptureResult:
    """Menu for options that run in docker by default but accept a managed swap."""
    if not interactive:
        return _capture_credentials(option, env_before, interactive=False, url=url)
    from agent_scaffold.cli_interactive import _interactive_select

    service = option.docker_service or option.id
    choices = [
        ("local", f"keep the local docker {service} (ensure it is running)"),
        ("paste", f"connect a managed {option.title} instead (paste credentials)"),
    ]
    if extras.provision is not None:
        choices.insert(
            1, ("provision", "create an instant Upstash database (free, 72h unless claimed)")
        )
    choice = _interactive_select(
        f"How should {option.title} run?", choices=choices, default="local"
    )
    if choice == "local":
        return CaptureResult("local")
    if choice == "provision" and extras.provision is not None:
        console.print("[dim]Provisioning an instant database...[/]")
        outcome = extras.provision(max(timeout, 15.0))
        if isinstance(outcome, str):
            console.print(f"[yellow]{redact(outcome)}[/] - paste a URL instead.")
            return _capture_credentials(option, env_before, interactive=True)
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
            return CaptureResult("abort")
        primary = option.credentials[0].var if option.credentials else "REDIS_URL"
        return CaptureResult("ok", {primary: outcome.url})
    return _capture_credentials(option, env_before, interactive=True)


# ---------------------------------------------------------------------------
# The connect flow
# ---------------------------------------------------------------------------


def _langsmith_companion(
    project_dir: Path, manifest: Manifest, captured: dict[str, str]
) -> str | None:
    """Ensure the LangSmith project exists and write tracing env; name or None."""
    from agent_scaffold.steps.bootstrap_langsmith import ensure_project, write_tracing_env

    api_key = captured.get("LANGCHAIN_API_KEY", "")
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


def _repair_compose_literals(compose_path: Path | None, option: StackOption, *, yes: bool) -> None:
    if compose_path is None or not compose_path.is_file():
        return
    from agent_scaffold.steps.docker_up import _compose_app_service

    app_service = _compose_app_service(compose_path.parent)
    if app_service is None:
        return
    text = compose_path.read_text(encoding="utf-8")
    literals = find_literal_env_entries(text, app_service, option.managed_vars)
    if not literals:
        return
    console.print(
        f"[yellow]{compose_path.name} pins literal values for: "
        f"{', '.join(sorted(literals))}[/] - env wiring can't override them."
    )
    new_text, rewritten = rewrite_literal_env(text, app_service, option.managed_vars)
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


def _verify_option(option: StackOption, env: Mapping[str, str], timeout: float) -> int:
    """Probe the option with the given env; print the outcome, return exit code."""
    if option.probe is None:
        console.print("[dim]No probe declared for this option - nothing to verify.[/]")
        return 0
    check = run_probe(service_for_option(option), timeout=timeout, env=env)
    failed = check.status == CheckStatus.FAIL
    marker = "[red]FAIL[/]" if failed else "[green]OK[/]"
    console.print(
        f"Verify: {marker} {check.title}" + (f" - {check.detail}" if check.detail else "")
    )
    if failed and check.fix_hint:
        console.print(f"[dim]{check.fix_hint}[/]")
    return 1 if failed else 0


def _ensure_local(
    option: StackOption,
    compose_path: Path | None,
    env: Mapping[str, str],
    timeout: float,
) -> int:
    """Make sure the local docker container runs, then verify with the probe."""
    started = False
    if option.docker_service is None:
        console.print(f"[dim]{option.title} has no docker service to start.[/]")
    elif compose_path is None or not compose_path.is_file():
        console.print(
            "[dim]No docker-compose.yml found - run `agent-scaffold up` to start the stack.[/]"
        )
    elif shutil.which("docker") is None:
        console.print("[dim]docker not on PATH - run `agent-scaffold up` to start the stack.[/]")
    else:
        cmd = ["docker", "compose", "up", "-d", option.docker_service]
        console.print(f"[cyan]Running:[/] {' '.join(cmd)}")
        completed = subprocess.run(cmd, cwd=compose_path.parent, env=dict(env), check=False)
        started = completed.returncode == 0
    code = _verify_option(option, env, timeout)
    if code != 0 and started:
        console.print(
            "[dim]The container may still be starting - check `agent-scaffold status` "
            "in a moment.[/]"
        )
    return code


def _missing_value_guidance(option: StackOption, env: Mapping[str, str]) -> str:
    """How to supply the missing values on a non-interactive run."""
    missing = [
        spec
        for spec in option.credentials
        if not spec.optional and not (env.get(spec.var) or "").strip()
    ]
    if len(missing) == 1 and missing[0].var in OVERRIDABLE_URL_VARS:
        placeholder = missing[0].placeholder or "..."
        return f"--url {placeholder}"
    return " ".join(f"export {spec.var}=..." for spec in missing) or "set the credentials"


def _generic_closing(option: StackOption, companion: str | None) -> str:
    _ = companion
    if option.mode == MODE_INTERNAL_OVERRIDABLE:
        service = option.docker_service or option.id
        return (
            f"Managed {option.title} wired. The local compose {service} container keeps "
            "running; the app now prefers the managed values."
        )
    where = f" Manage it at {option.key_page_url}." if option.key_page_url else ""
    return f"{option.title} connected.{where}"


def _langsmith_closing(option: StackOption, companion: str | None) -> str:
    _ = option
    return (
        f"Traces will appear at https://smith.langchain.com under project "
        f"{companion!r} once the agent handles a request."
    )


def _redis_closing(option: StackOption, companion: str | None) -> str:
    _ = option, companion
    return (
        "Managed Redis wired. The local compose redis container keeps running; "
        "the app now prefers the managed URL."
    )


def _validate_langsmith_captured(captured: dict[str, str], timeout: float) -> ValidationResult:
    return validate_langsmith_key(captured.get("LANGCHAIN_API_KEY", ""), timeout)


def _validate_redis_captured(captured: dict[str, str], timeout: float) -> ValidationResult:
    return validate_redis_url(captured.get("REDIS_URL", ""), timeout)


def _probe_validate(
    option: StackOption,
    captured: dict[str, str],
    env_before: Mapping[str, str],
    timeout: float,
) -> ValidationResult:
    """Generic pre-store validation: run the option's probe with the candidate env."""
    if option.probe is None:
        return ValidationResult(True, "not validated (no probe declared)")
    overlay = dict(env_before)
    overlay.update(captured)
    check = run_probe(service_for_option(option), timeout=timeout, env=overlay)
    detail = check.title + (f": {check.detail}" if check.detail else "")
    return ValidationResult(check.status != CheckStatus.FAIL, detail)


PROVIDER_EXTRAS: dict[str, ProviderExtras] = {
    "langsmith": ProviderExtras(
        validate=_validate_langsmith_captured,
        companion=_langsmith_companion,
        closing=_langsmith_closing,
    ),
    "redis": ProviderExtras(
        provision=provision_upstash_free,
        validate=_validate_redis_captured,
        closing=_redis_closing,
    ),
}


def run_connect(
    project_dir: Path,
    manifest: Manifest,
    option: StackOption,
    compose_path: Path | None,
    *,
    yes: bool,
    url: str | None = None,
    timeout: float = 10.0,
    reset_step_state: Callable[[Path, str], None] | None = None,
) -> int:
    """Drive the full connect flow for one stack option; returns the exit code."""
    namespace = manifest.secrets_namespace or project_namespace(project_dir.name, project_dir)
    if list_project_secret_names(namespace):
        console.print(
            "[dim]Reading stored secrets from the system keychain - macOS may ask to "
            "allow access; that is not a request to re-enter values.[/]"
        )
    env_before = build_runtime_env(project_dir, namespace)
    interactive = not yes and _stdin_isatty()
    extras = PROVIDER_EXTRAS.get(option.id, ProviderExtras())

    if option.mode == MODE_INTERNAL or not option.credentials:
        return _ensure_local(option, compose_path, env_before, timeout)

    if extras.capture is not None:
        result = extras.capture(
            option, env_before, interactive=interactive, url=url, timeout=timeout
        )
    elif option.mode == MODE_INTERNAL_OVERRIDABLE:
        result = _capture_overridable(
            option, extras, env_before, interactive=interactive, url=url, timeout=timeout
        )
    else:
        result = _capture_credentials(option, env_before, interactive=interactive, url=url)

    if result.kind == "local":
        service = option.docker_service or option.id
        console.print(f"Keeping the local docker {service} - nothing to change.")
        return _ensure_local(option, compose_path, env_before, timeout)
    captured = dict(result.values)
    if result.kind != "ok" or not captured:
        if not interactive:
            guidance = _missing_value_guidance(option, env_before)
            first_missing = next(
                (
                    spec.var
                    for spec in option.credentials
                    if not spec.optional and spec.var not in captured
                ),
                option.credentials[0].var if option.credentials else option.id,
            )
            console.print(
                f"[red]No value for {first_missing}.[/] Non-interactive runs "
                f"need it supplied up front: {guidance} then re-run with --yes."
            )
            return 2
        console.print("[yellow]Aborted - nothing stored.[/]")
        return 1

    if extras.validate is not None:
        verdict = extras.validate(captured, timeout)
    else:
        verdict = _probe_validate(option, captured, env_before, timeout)
    if not verdict.ok:
        console.print(f"[red]Validation failed:[/] {redact(verdict.detail)}")
        if verdict.auth_failure or not interactive:
            console.print("[red]Nothing stored.[/]")
            return 1
        if not _confirm("Store anyway (network problems can be transient)?", default=False):
            return 1
    else:
        console.print(f"Validated: {redact(verdict.detail)}")

    specs = {spec.var: spec for spec in option.credentials}
    for var, value in captured.items():
        spec = specs.get(var, CredentialSpec(var=var))
        if spec.secret:
            backend = persist_project_secret(namespace, project_dir, var, SecretStr(value))
            if backend is None:
                return 1
        else:
            try:
                append_env_local(project_dir, var, SecretStr(value))
            except OSError as exc:
                console.print(f"[red]failed to write .env.local: {exc}[/]")
                return 1
            backend = ".env.local"
        console.print(f"Stored {var} ({mask(value)}) in {backend}")

    for var, value in captured.items():
        shell_value = (os.environ.get(var) or "").strip()
        if shell_value and shell_value != value:
            console.print(
                f"[yellow]Your shell also exports {var} with a different value - the "
                "shell wins at run time; unset it to use the stored value.[/]"
            )

    companion_context: str | None = None
    if extras.companion is not None:
        companion_context = extras.companion(project_dir, manifest, captured)

    _repair_compose_literals(compose_path, option, yes=yes)

    # One vault read per run (the keychain consent already fired above). The
    # fresh .env.local re-read picks up companion-written tracing vars; the
    # env_before overlay preserves build_runtime_env precedence (shell/vault
    # beat .env.local); captured wins so just-stored values reach the recreate.
    env_after: dict[str, str] = {**read_env_local(project_dir), **env_before, **captured}
    _recreate_app(compose_path, env_after)

    if reset_step_state is not None and option.bootstrap_step:
        reset_step_state(project_dir, option.bootstrap_step)

    code = _verify_option(option, env_after, timeout)

    closing = (extras.closing or _generic_closing)(option, companion_context)
    console.print(closing)

    if (
        code == 0
        and interactive
        and option.mode == MODE_INTERNAL_OVERRIDABLE
        and option.docker_service
        and compose_path is not None
        and compose_path.is_file()
        and shutil.which("docker") is not None
    ):
        stop_cmd = f"docker compose stop {option.docker_service}"
        if _confirm(
            f"Stop the local {option.docker_service} container now that the managed "
            "instance is wired?",
            default=False,
        ):
            subprocess.run(
                ["docker", "compose", "stop", option.docker_service],
                cwd=compose_path.parent,
                env=env_after,
                check=False,
            )
        else:
            console.print(f"[dim]Stop it later with: {stop_cmd}[/]")

    return code


__all__ = [
    "PROVIDER_EXTRAS",
    "CaptureResult",
    "ProviderExtras",
    "UpstashDatabase",
    "ValidationResult",
    "find_docker_compose",
    "find_literal_env_entries",
    "parse_upstash_start_response",
    "persist_project_secret",
    "provision_upstash_free",
    "rewrite_literal_env",
    "run_connect",
    "validate_langsmith_key",
    "validate_redis_url",
]
