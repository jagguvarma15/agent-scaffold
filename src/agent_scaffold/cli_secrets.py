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

from agent_scaffold.auth import StoredCredential, delete_key, list_credentials
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

    def is_empty(self) -> bool:
        return not (self.keyring_names or self.file_names or self.env_local_paths)

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
            if cred.backend == "keyring":
                keyring_names.append(cred.name)
            elif cred.backend == "file":
                file_names.append(cred.name)
    except Exception:  # noqa: BLE001 — survey must never raise
        pass
    env_local_paths: list[Path] = []
    if include_env_local:
        # Look at ``$PWD/.env.local`` only — predictable, no walk.
        candidate = Path.cwd() / ".env.local"
        if candidate.is_file():
            env_local_paths.append(candidate)
    return _PurgeSurvey(
        keyring_names=sorted(keyring_names),
        file_names=sorted(file_names),
        env_local_paths=env_local_paths,
    )


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

    if json_output:
        payload = {
            "schema_version": 1,
            "credentials": [
                {
                    "name": c.name,
                    "backend": c.backend,
                    "masked": c.masked_value,
                    "created": c.created,
                }
                for c in creds
            ],
        }
        typer.echo(json.dumps(payload, indent=2))
        return

    if not creds:
        console.print("[dim]No stored credentials.[/]")
        return
    console.print("[bold]Stored credentials:[/]")
    for c in creds:
        created = f"   created {c.created}" if c.created else ""
        console.print(f"  {c.name:<14}  {c.masked_value:<20}  ({c.backend}){created}")
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
