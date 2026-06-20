"""``agent-scaffold secrets`` sub-app: survey + purge stored credentials.

Lifted out of ``cli.py`` so the credentials-rotation flow is one
self-contained module. The Typer sub-app is exported as ``secrets_app``;
``cli.py`` wires it onto the parent ``app`` via
``app.add_typer(secrets_app, name="secrets")``.

The ``purge`` command is the canonical "rotate keys" workflow — it
clears every backend the CLI owns (keyring, mode-0600 file, and the
local ``./.env.local``) in one shot, behind a confirmation prompt.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import typer

from agent_scaffold.auth import (
    StoredCredential,
    delete_key,
    delete_project_secret,
    delete_project_secrets,
    list_credentials,
    list_project_namespaces,
    list_project_secret_names,
    mask,
    project_namespace,
    store_project_secret,
)
from agent_scaffold.cli_shared import console

secrets_app = typer.Typer(
    name="secrets",
    help="Survey, list, and purge stored secrets across keyring + file + project.",
)


@dataclass(frozen=True)
class _PurgeSurvey:
    keyring_names: list[str]
    file_names: list[str]
    env_local_paths: list[Path]
    project_namespaces: list[str]

    def is_empty(self) -> bool:
        return not (
            self.keyring_names or self.file_names or self.env_local_paths or self.project_namespaces
        )

    def render_summary(self) -> str:
        parts: list[str] = []
        if self.keyring_names:
            parts.append(
                f"{len(self.keyring_names)} keyring entr"
                f"{'y' if len(self.keyring_names) == 1 else 'ies'} "
                f"({', '.join(self.keyring_names)})"
            )
        if self.file_names:
            parts.append(
                f"{len(self.file_names)} credentials-file entr"
                f"{'y' if len(self.file_names) == 1 else 'ies'} "
                f"({', '.join(self.file_names)})"
            )
        if self.project_namespaces:
            parts.append(
                f"{len(self.project_namespaces)} project vault"
                f"{'' if len(self.project_namespaces) == 1 else 's'} "
                f"({', '.join(self.project_namespaces)})"
            )
        if self.env_local_paths:
            paths_str = ", ".join(str(p) for p in self.env_local_paths)
            parts.append(
                f"{len(self.env_local_paths)} .env.local file"
                f"{'' if len(self.env_local_paths) == 1 else 's'} "
                f"({paths_str})"
            )
        return "; ".join(parts) if parts else "(nothing)"


def _survey_secrets(*, include_env_local: bool) -> _PurgeSurvey:
    """Enumerate stored credentials across all backends we own.

    Project ``.env.local`` files are discovered via the per-config cache
    directory's tracking — but we don't crawl ``$HOME`` looking for them.
    For v2, the only way a file shows up is if the user lists it via
    ``--project-dir`` on the purge command. Defensive: keep the survey
    explicit and predictable.
    """
    keyring_names: list[str] = []
    file_names: list[str] = []
    try:
        for cred in list_credentials():
            # Project-vault entries are handled per-namespace below, not as
            # flat credential names.
            if cred.name.startswith("project:"):
                continue
            if cred.backend == "keyring":
                keyring_names.append(cred.name)
            elif cred.backend == "file":
                file_names.append(cred.name)
    except Exception:  # noqa: BLE001, S110 — survey must never raise; empty result is the intended failure mode
        pass
    env_local_paths: list[Path] = []
    if include_env_local:
        # Look at ``$PWD/.env.local`` only — predictable, no walk.
        candidate = Path.cwd() / ".env.local"
        if candidate.is_file():
            env_local_paths.append(candidate)
    project_namespaces: list[str] = []
    try:
        project_namespaces = list_project_namespaces()
    except Exception:  # noqa: BLE001, S110 — survey must never raise; empty result is the intended failure mode
        pass
    return _PurgeSurvey(
        keyring_names=sorted(keyring_names),
        file_names=sorted(file_names),
        env_local_paths=env_local_paths,
        project_namespaces=project_namespaces,
    )


def _project_vault_inventory() -> dict[str, dict[str, str]]:
    """``{namespace: {env_var: backend}}`` from the names-only index.

    Never reads a secret value — listing must not trigger keyring consent
    prompts or expose anything beyond names + backend markers.
    """
    inventory: dict[str, dict[str, str]] = {}
    try:
        for namespace in list_project_namespaces():
            names = list_project_secret_names(namespace)
            if names:
                inventory[namespace] = names
    except Exception:  # noqa: BLE001, S110 — survey must never raise; empty result is the intended failure mode
        pass
    return inventory


def _resolve_namespace(project_dir: Path) -> str:
    """Namespace for ``project_dir``: manifest-recorded, else derived."""
    from agent_scaffold.manifest import read_manifest

    resolved = project_dir.expanduser().resolve()
    try:
        recorded = read_manifest(resolved).secrets_namespace
    except Exception:  # noqa: BLE001 — missing/old manifest falls back to derived
        recorded = None
    return recorded or project_namespace(resolved.name, resolved)


@secrets_app.command("list")
def secrets_list(
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Inventory every credential the CLI knows about, masked."""
    creds: list[StoredCredential] = []
    try:
        creds = list_credentials()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]Warning:[/] could not enumerate credentials: {exc}")
    vault = _project_vault_inventory()
    # Project-secret value entries already surface via list_credentials()'s
    # file-backend scan with "project:" names — drop them from the flat list
    # so they only render grouped under their namespace.
    creds = [c for c in creds if not c.name.startswith("project:")]

    if json_output:
        payload = {
            "schema_version": 2,
            "credentials": [
                {
                    "name": c.name,
                    "backend": c.backend,
                    "masked": c.masked_value,
                    "created": c.created,
                }
                for c in creds
            ],
            "projects": [
                {
                    "namespace": namespace,
                    "secrets": [
                        {"name": name, "backend": backend} for name, backend in names.items()
                    ],
                }
                for namespace, names in vault.items()
            ],
        }
        typer.echo(json.dumps(payload, indent=2))
        return

    if not creds and not vault:
        console.print("[dim]No stored credentials.[/]")
        return
    if creds:
        console.print("[bold]Stored credentials:[/]")
        for c in creds:
            created = f"   created {c.created}" if c.created else ""
            console.print(f"  {c.name:<14}  {c.masked_value:<20}  ({c.backend}){created}")
    for namespace, names in vault.items():
        console.print(f"\n[bold]Project vault[/] [cyan]{namespace}[/]:")
        for name, backend in sorted(names.items()):
            console.print(f"  {name:<24}  (encrypted, {backend})")
    console.print("\n[dim]Run `agent-scaffold secrets purge` to remove all stored credentials.[/]")


@secrets_app.command("purge")
def secrets_purge(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
    keep_env_local: bool = typer.Option(
        False,
        "--keep-env-local",
        help="Don't touch ./.env.local in the current directory.",
    ),
) -> None:
    """Remove every stored credential. Surveys first; confirms before deleting.

    Operates on **all** backends the CLI manages:

    - keyring entries written via ``auth login``
    - mode-0600 entries in ``~/.config/agent-scaffold/credentials``
    - ``./.env.local`` in the current directory (unless ``--keep-env-local``)

    Designed for the "I'm rotating keys" workflow: one command, full clean slate.
    """
    survey = _survey_secrets(include_env_local=not keep_env_local)
    console.print(f"[bold]Will remove:[/] {survey.render_summary()}")
    if survey.is_empty():
        console.print("[dim]Nothing to purge.[/]")
        raise typer.Exit(code=0)

    if not yes:
        answer = input("Continue? [y/N]: ").strip().lower()
        if answer not in ("y", "yes"):
            console.print("[yellow]Aborted.[/]")
            raise typer.Exit(code=0)

    removed: list[str] = []
    for name in survey.keyring_names:
        if delete_key(name):
            removed.append(f"keyring/{name}")
    for name in survey.file_names:
        if delete_key(name):
            removed.append(f"file/{name}")
    for namespace in survey.project_namespaces:
        count = delete_project_secrets(namespace)
        if count:
            removed.append(f"vault/{namespace} ({count})")
    for path in survey.env_local_paths:
        try:
            path.unlink()
            removed.append(str(path))
        except OSError as exc:
            console.print(f"[yellow]Could not remove {path}:[/] {exc}")

    if removed:
        console.print(f"[green]Removed:[/] {', '.join(removed)}")
    else:
        console.print("[dim]Nothing was removed.[/]")


_ENV_VAR_ARGUMENT = typer.Argument(..., help="Environment variable name, e.g. QDRANT_URL.")
_PROJECT_OPTION = typer.Option(
    Path("."),
    "--project",
    help="Generated project directory the secret belongs to.",
)


@secrets_app.command("set")
def secrets_set(
    env_var: str = _ENV_VAR_ARGUMENT,
    project: Path = _PROJECT_OPTION,
) -> None:
    """Store one project secret in the encrypted vault (prompted, never echoed)."""
    import getpass

    from pydantic import SecretStr

    namespace = _resolve_namespace(project)
    raw = getpass.getpass(f"Enter value for {env_var}: ").strip()
    if not raw:
        console.print("[yellow]Empty value — nothing stored.[/]")
        raise typer.Exit(code=1)
    try:
        stored = store_project_secret(namespace, env_var, SecretStr(raw))
    except Exception as exc:  # noqa: BLE001 — surface as message, not traceback
        console.print(f"[red]Could not store {env_var}:[/] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        f"[green]Stored[/] {env_var} = {mask(raw)} "
        f"in vault [cyan]{namespace}[/] ({stored.backend})."
    )


@secrets_app.command("unset")
def secrets_unset(
    env_var: str = _ENV_VAR_ARGUMENT,
    project: Path = _PROJECT_OPTION,
) -> None:
    """Remove one project secret from the encrypted vault."""
    namespace = _resolve_namespace(project)
    if delete_project_secret(namespace, env_var):
        console.print(f"[green]Removed[/] {env_var} from vault [cyan]{namespace}[/].")
    else:
        console.print(f"[dim]{env_var} was not in vault {namespace}.[/]")
