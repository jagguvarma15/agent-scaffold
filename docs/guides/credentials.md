# Credentials

`agent-scaffold` resolves the Anthropic API key in this order:

1. `ANTHROPIC_API_KEY` environment variable
2. `python-keyring` (macOS Keychain / Windows Credential Manager / Linux Secret Service / KDE Wallet)
3. INI file at `$XDG_CONFIG_HOME/agent-scaffold/credentials` (mode `0600`)

The plaintext keyring backend is **refused**: if `keyring.get_keyring()` reports `PlaintextKeyring` (or any non-OS-native backend), `auth login` falls back to the mode-0600 file backend with a warning. Pass `--use-file` or `--use-env` to override the default.

```bash
agent-scaffold auth login              # browser flow
agent-scaffold auth login --no-browser # paste flow (headless / SSH)
agent-scaffold auth status             # show what's stored where
agent-scaffold auth logout --all       # remove every stored credential
echo "$TOKEN" | agent-scaffold auth setup-token ci-prod --stdin

# Cross-backend revocation (keyring + file + ./.env.local in one go)
agent-scaffold secrets list
agent-scaffold secrets purge --yes
```

Project-scoped secrets (service URLs, passwords a recipe's stack needs) live in an encrypted vault managed with `agent-scaffold secrets set <NAME>` and `secrets unset <NAME>` — prompted, never echoed, never written to tracked files.

## Security model

The CLI follows a nine-point hardening checklist for secret handling: **no secrets in argv, `getpass` instead of `input`, `SecretStr` typing, `shell=False`, mode-0600 credential files, plaintext-keyring refusal, output redaction, enforced `.gitignore`, and first-class revocation via `secrets purge`**. Each rule is locked in by an audit test under `tests/security/` so regressions block CI.

See the [security model design](../design/security.md) for the full rationale and per-rule references.
