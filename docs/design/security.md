# Security model — agent-scaffold

`agent-scaffold` handles three categories of secret material:

1. **The Anthropic API key**, captured by `agent-scaffold auth login` and
   used by every LLM call.
2. **Per-service credentials** (e.g. `DATABASE_URL`, `LANGFUSE_SECRET_KEY`,
   `REDIS_URL` with embedded password), captured by `wire_credentials` and
   stored locally per project.
3. **CI tokens** captured by `auth setup-token`.

Each one is governed by the same nine-point checklist below. The rules
exist because every one of them has historically been the cause of an
incident in some other tool.

## The nine rules

### 1. Never accept secrets as positional/flag arguments

`/proc/<pid>/cmdline` is readable by every process the same user owns; on
some Linux setups, by every user on the box. A flag like
`--api-key sk-ant-...` leaks the credential to anyone who can list
processes for the lifetime of the invocation. We always require a separate
channel: an env var, a file, or `getpass`-style paste.

Enforced by [`tests/security/test_argv_no_secrets.py`](https://github.com/jagguvarma15/agent-scaffold/blob/main/tests/security/test_argv_no_secrets.py).

### 2. Use `getpass.getpass()`, never `input()`

`input()` echoes characters to the terminal. The credential ends up in
scrollback, in any tee'd transcript, in tmux scroll history. `getpass`
disables echo for the duration of the prompt.

Enforced by [`tests/security/test_no_input_for_secrets.py`](https://github.com/jagguvarma15/agent-scaffold/blob/main/tests/security/test_no_input_for_secrets.py).

### 3. Wrap in `pydantic.SecretStr` after capture

`SecretStr.__repr__()` and `__str__()` return `'**********'`. A bare `str`
makes it trivial to leak credentials through a stray `print()`, an exception
formatter, or a structured logger that serializes every attribute.

Enforced by [`tests/security/test_secret_typing.py`](https://github.com/jagguvarma15/agent-scaffold/blob/main/tests/security/test_secret_typing.py).

### 4. `subprocess.run([...], shell=False)` always

`shell=True` interprets the command via `/bin/sh`. Any element derived
from user input is now a command-injection vector. There are no sanctioned
exceptions. The one string that used to reach a shell — the **smoke check**,
which on the generation path is model-authored (`GenerationResult.smoke_check`)
— is now gated by `validator._smoke_argv`: `shlex`-split into argv, executed
with `shell=False`, first token confined to the project runners
(`_SMOKE_RUNNERS`), bare shell-operator tokens rejected outright. Shell
composition (`"pytest -xvs && curl ..."`) is not supported; a smoke check is
a single command.

Enforced by [`tests/security/test_no_shell_true.py`](https://github.com/jagguvarma15/agent-scaffold/blob/main/tests/security/test_no_shell_true.py)
with an empty exemption list.

### 5. `os.umask(0o077)` + `chmod 0o600` for credentials files

The default umask on most systems is `022`, which creates new files as
`0o644` (world-readable). Credential files MUST be `0o600` so even other
users on the same box can't read them. The
[`_filesec.secure_write`](https://github.com/jagguvarma15/agent-scaffold/blob/main/src/agent_scaffold/_filesec.py) helper
centralises the umask + atomic-rename + explicit-chmod dance so every
caller gets it right by construction.

### 6. Refuse plaintext keyring backends

`python-keyring` will silently fall back to a plaintext file backend if no
native backend is available. We detect this and **refuse** rather than
write secrets to a plaintext file under a name that implies encryption.
The user has to opt into the `file` backend (mode-0600 INI) explicitly via
`auth login --use-file`.

Enforced by [`tests/test_auth.py`](https://github.com/jagguvarma15/agent-scaffold/blob/main/tests/test_auth.py) (covered by
Q2's existing suite).

### 7. Strip secrets from telemetry/logs/state

There's no telemetry today, but logs and state files do escape into
disk and stdout. The [`_redact.redact`](https://github.com/jagguvarma15/agent-scaffold/blob/main/src/agent_scaffold/_redact.py)
module knows how to detect every common secret shape (Anthropic / OpenAI /
AWS / Bearer / postgres-URL / GitHub PAT / Slack token) and runs on every
external sink:

- `StepLog.line` before Rich panel rendering
- Failure panel `cause` + `stderr_tail` before display
- `orchestrator.write_state` payload before persist
- `manifest.write_manifest` for the `update_history` field

False positives are explicitly preferred over false negatives. If a
legitimate string looks like an `sk-ant-...` token, we'd rather redact it
than risk missing a real one.

Enforced by [`tests/security/test_redact.py`](https://github.com/jagguvarma15/agent-scaffold/blob/main/tests/security/test_redact.py).

### 8. `.gitignore` enforcement

`agent-scaffold new` writes a `.gitignore` containing `.scaffold/`,
`.env`, `.env.local`, `.env.*.local`, `credentials`. Existing
`.gitignore` files are extended (missing entries appended under a
labelled block); user-authored entries are preserved verbatim. The
`wire_credentials` step calls the same helper so a project that wasn't
born with the secret-safety block gets one before the first `.env.local`
is written.

Enforced by [`tests/security/test_gitignore_enforcement.py`](https://github.com/jagguvarma15/agent-scaffold/blob/main/tests/security/test_gitignore_enforcement.py).

### 9. `agent-scaffold secrets purge` from day one

Revocation is a first-class verb. `secrets purge` surveys every backend
the CLI knows about (keyring, file, project `.env.local`) and removes
them in one go after a confirmation prompt. `--yes` skips the prompt for
CI; `--keep-env-local` scopes the purge to keyring + file backends only.

`secrets list` provides the read-only counterpart so users can audit
what's stored before deciding to purge.

Enforced by [`tests/security/test_secrets_purge.py`](https://github.com/jagguvarma15/agent-scaffold/blob/main/tests/security/test_secrets_purge.py).

## What's explicitly out of scope

- **Hardware-token gating** (YubiKey, Touch ID) for credential reads.
- **Encrypted credentials file** beyond the chmod-0600 contract.
- **Active token rotation** — `auth logout` + `auth login` is enough.
- **Vault / Doppler / 1Password adapters** as built-ins — we expect users
  to wrap the CLI with `op run --` or equivalent.
- **Audit-log persistence** of secret accesses.

If any of these become a real requirement, the design table is here to be
re-opened.

## For contributors

If you're adding a new code path that touches a credential:

1. Read the nine rules above.
2. Use `secure_write` for any file that holds a secret.
3. Run anything user-facing through `redact()` before display.
4. Make sure the new code is covered by an audit test under
   `tests/security/` — if the existing tests don't catch your case,
   add one.
