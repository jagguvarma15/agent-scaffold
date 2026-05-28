"""``agent-scaffold auth`` sub-app: keyring-first credential management.

Lifted out of ``cli.py`` so the ~200 LOC of auth wiring isn't tangled
with the project-generation pipeline. The Typer sub-app is exported as
``auth_app``; ``cli.py`` wires it onto the parent ``app`` via
``app.add_typer(auth_app, name="auth")``.

The commands are thin shims around :mod:`agent_scaffold.auth` — backend
selection, keyring writes, file fallback, and the browser-paste flow for
the standard `login`.
"""

from __future__ import annotations

import json
import sys

import typer

from agent_scaffold.auth import (
    DEFAULT_KEY_NAME,
    AuthError,
    BackendKind,
    StoredCredential,
    delete_key,
    describe_backend,
    detect_backend,
    list_credentials,
    resolve_active,
    store_key,
    validate_anthropic_key,
)
from agent_scaffold.cli_shared import console

auth_app = typer.Typer(
    name="auth",
    help="Manage Anthropic credentials (keyring-first; mode-0600 file fallback).",
)


def _select_backend(use_keyring: bool, use_file: bool, use_env: bool) -> BackendKind:
    chosen = [
        name
        for name, flag in (
            ("keyring", use_keyring),
            ("file", use_file),
            ("env", use_env),
        )
        if flag
    ]
    if len(chosen) > 1:
        raise typer.BadParameter("--use-keyring / --use-file / --use-env are mutually exclusive.")
    if chosen:
        return chosen[0]  # type: ignore[return-value]
    try:
        return detect_backend()
    except AuthError:
        # No native keyring available — degrade to the file backend rather
        # than failing the user mid-flow. This is the explicit v2 fallback.
        console.print(
            "[yellow]Warning:[/] no native keyring backend detected; "
            "falling back to mode-0600 credentials file."
        )
        return "file"


def _prompt_paste(prompt: str = "Paste your Anthropic key (input hidden):") -> str:
    import getpass

    try:
        return getpass.getpass(prompt).strip()
    except (EOFError, KeyboardInterrupt) as exc:
        raise typer.Abort() from exc


@auth_app.command("login")
def auth_login(
    name: str = typer.Option(
        DEFAULT_KEY_NAME, "--name", "-n", help="Credential name (for multi-key setups)."
    ),
    use_keyring: bool = typer.Option(False, "--use-keyring", help="Force keyring backend."),
    use_file: bool = typer.Option(False, "--use-file", help="Force mode-0600 file backend."),
    use_env: bool = typer.Option(
        False, "--use-env", help="Don't store; just print the export line."
    ),
    no_browser: bool = typer.Option(
        False, "--no-browser", help="Skip the browser flow; prompt for paste instead."
    ),
    no_validate: bool = typer.Option(
        False, "--no-validate", help="Skip the models.list() probe before storing."
    ),
) -> None:
    """Capture an Anthropic key (browser or paste), validate, store."""
    from pydantic import SecretStr

    backend = _select_backend(use_keyring, use_file, use_env)

    key_text: str | None = None
    if not no_browser:
        from agent_scaffold.auth_browser import browser_paste_flow

        console.print("Opening your browser to paste your Anthropic key...")
        key_text = browser_paste_flow()
        if not key_text:
            console.print("[yellow]No key captured from browser flow.[/] Falling back to paste.")
    if not key_text:
        key_text = _prompt_paste()
    if not key_text:
        console.print("[red]No key supplied.[/]")
        raise typer.Exit(code=1)

    secret = SecretStr(key_text)
    if not no_validate:
        ok, msg = validate_anthropic_key(secret)
        if not ok:
            console.print(f"[red]Validation failed:[/] {msg}")
            raise typer.Exit(code=1)
        console.print(f"[green]Key {msg}.[/]")

    try:
        stored = store_key(name, secret, backend=backend)
    except AuthError as exc:
        console.print(f"[red]Store failed:[/] {exc}")
        raise typer.Exit(code=1) from exc

    if stored.backend == "env":
        console.print(
            f"[bold]Add to your shell:[/]  export ANTHROPIC_API_KEY='{secret.get_secret_value()}'"
        )
    else:
        console.print(
            f"[green]Stored[/] '{stored.name}' in {stored.backend} ({stored.masked_value})."
        )


@auth_app.command("status")
def auth_status(
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Show backend health, stored credentials, and the active resolution."""
    backend_label: str
    backend_ok: bool
    try:
        detect_backend()
        backend_label = describe_backend()
        backend_ok = True
    except AuthError as exc:
        backend_label = f"{describe_backend()} (refused: {exc})"
        backend_ok = False

    creds = list_credentials()
    active = resolve_active()

    if json_output:
        payload = {
            "schema_version": 1,
            "backend": backend_label,
            "backend_ok": backend_ok,
            "credentials": [
                {
                    "name": c.name,
                    "backend": c.backend,
                    "masked": c.masked_value,
                    "created": c.created,
                }
                for c in creds
            ],
            "active": (
                {"name": DEFAULT_KEY_NAME, "backend": active[1]} if active is not None else None
            ),
            "resolution_order": ["env (ANTHROPIC_API_KEY)", "keyring", "file"],
        }
        typer.echo(json.dumps(payload, indent=2))
        return

    health = "[green]good[/]" if backend_ok else "[red]refused[/]"
    console.print(f"[bold]Backend:[/] {backend_label} {health}")
    if creds:
        console.print("[bold]Stored credentials:[/]")
        for c in creds:
            created = f"   created {c.created}" if c.created else ""
            console.print(f"  {c.name:<14}  {c.masked_value:<18}  ({c.backend}){created}")
    else:
        console.print("[dim]No stored credentials.[/]")
    console.print("[bold]Resolution order:[/] ANTHROPIC_API_KEY (env) > keyring > file")
    if active is not None:
        _, src = active
        console.print(f"[bold]Currently resolved:[/] name={DEFAULT_KEY_NAME} from {src}")
    else:
        console.print("[yellow]No key resolved.[/] Run `agent-scaffold auth login`.")


@auth_app.command("logout")
def auth_logout(
    name: str = typer.Option(DEFAULT_KEY_NAME, "--name", "-n", help="Credential name to remove."),
    all_: bool = typer.Option(
        False, "--all", help="Remove every stored credential, not just --name."
    ),
) -> None:
    """Remove a stored credential from every backend it lives in."""
    if all_:
        creds: list[StoredCredential] = list_credentials()
        names = {c.name for c in creds}
        removed_any = False
        for n in names:
            if delete_key(n):
                removed_any = True
                console.print(f"[green]Removed[/] {n}")
        if not removed_any:
            console.print("[dim]No credentials to remove.[/]")
        return
    if delete_key(name):
        console.print(f"[green]Removed[/] {name}")
    else:
        console.print(f"[yellow]No credential named[/] {name}")
        raise typer.Exit(code=1)


@auth_app.command("setup-token")
def auth_setup_token(
    name: str = typer.Argument(..., help="Token name (e.g. ci-prod)."),
    from_stdin: bool = typer.Option(False, "--stdin", help="Read token from stdin (for CI)."),
) -> None:
    """Store a long-lived token in the mode-0600 file backend (for CI)."""
    from pydantic import SecretStr

    if from_stdin:
        text = sys.stdin.read().strip()
    else:
        text = _prompt_paste("Paste the token:")
    if not text:
        console.print("[red]No token supplied.[/]")
        raise typer.Exit(code=1)
    stored = store_key(name, SecretStr(text), backend="file")
    console.print(f"[green]Stored[/] '{stored.name}' in {stored.backend} ({stored.masked_value}).")
